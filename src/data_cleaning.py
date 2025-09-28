import pandas as pd
import numpy as np

class DataCleaning:
    """Utility class for common data cleaning tasks.

    Methods are chainable: each returns self.
    Use get_cleaned_data() to retrieve the cleaned DataFrame.

    Notes:
    - normalize_columns(): should be applied *after* train/test split to avoid data leakage.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._log = []  # keep a simple log of steps

    def _add_log(self, msg: str):
        self._log.append(msg)
        return self

    def show_log(self):
        """Print cleaning steps log."""
        print("\n".join(self._log))
        return self

    # -------------------------
    # Basic cleaning steps
    # -------------------------

    def remove_duplicates(self):
        before = len(self.df)
        self.df = self.df.drop_duplicates().reset_index(drop=True)
        self._add_log(f"Removed duplicates: {before - len(self.df)} rows dropped.")
        return self

    def fill_numerical_median(self, group_cols=None):
        """Fill numeric NaNs with median.
        If group_cols are provided, use group-wise median first.
        """
        num_cols = self.df.select_dtypes(include=['number']).columns
        if group_cols:
            for col in num_cols:
                self.df[col] = self.df.groupby(group_cols)[col].transform(
                    lambda s: s.fillna(s.median())
                )
        for col in num_cols:
            self.df[col] = self.df[col].fillna(self.df[col].median())
        self._add_log("Filled numeric NaNs with median (group-aware if provided).")
        return self

    def fill_categorical_mode(self):
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            mode = self.df[col].mode(dropna=True)
            if not mode.empty:
                self.df[col] = self.df[col].fillna(mode[0])
        self._add_log("Filled categorical NaNs with mode.")
        return self

    # -------------------------
    # Encoding
    # -------------------------

    def encode_label(self, columns):
        """Stable label encoding for binary categoricals.
        If col is 'sex' or 'smoker', use fixed mapping.
        Otherwise, map categories alphabetically.
        """
        if isinstance(columns, str):
            columns = [columns]

        fixed_maps = {
            "sex": {"female": 0, "male": 1},
            "smoker": {"no": 0, "yes": 1},
        }

        for col in columns:
            if col in fixed_maps:
                self.df[col] = self.df[col].map(fixed_maps[col]).astype("Int64")
            else:
                uniques = sorted(self.df[col].dropna().unique().tolist())
                mapping = {v: i for i, v in enumerate(uniques)}
                self.df[col] = self.df[col].map(mapping).astype("Int64")
        self._add_log(f"Label-encoded: {columns}.")
        return self

    def encode_one_hot(self, columns, drop_first=True):
        """One-hot encode the given column(s)."""
        if isinstance(columns, str):
            columns = [columns]
        self.df = pd.get_dummies(self.df, columns=columns, drop_first=drop_first)
        self._add_log(f"One-hot encoded: {columns} (drop_first={drop_first}).")
        return self

    # -------------------------
    # Scaling & transformation
    # -------------------------

    def normalize_columns(self, columns):
        """Min-max normalize numeric columns to [0, 1].
        Warning: Apply after train/test split to avoid leakage.
        """
        if isinstance(columns, str):
            columns = [columns]
        for col in columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                col_min, col_max = self.df[col].min(), self.df[col].max()
                if pd.notna(col_min) and pd.notna(col_max) and col_max != col_min:
                    self.df[col] = (self.df[col] - col_min) / (col_max - col_min)
        self._add_log(f"Normalized columns: {columns}.")
        return self
    def log_transform(self, columns):
        """Apply log1p transformation to skewed positive columns (e.g., charges)."""
        if isinstance(columns, str):
            columns = [columns]
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                s = self.df[col]
                if (s.dropna() >= 0).all():
                    self.df[col] = np.log1p(s)
        self._add_log(f"Applied log1p transform to: {columns}.")
        return self

    # -------------------------
    # Schema validation & outliers
    # -------------------------

    def canonicalize_text(self):
        """Trim/Lowercase all object columns and normalize region names."""
        obj_cols = self.df.select_dtypes(include=["object", "category"]).columns
        for c in obj_cols:
            self.df[c] = (
                self.df[c].astype(str).str.strip().str.lower().replace({"nan": np.nan})
            )
        if "region" in self.df.columns:
            self.df["region"] = self.df["region"].replace({
                "south west": "southwest", "south-west": "southwest",
                "north west": "northwest", "north-west": "northwest",
                "south east": "southeast", "south-east": "southeast",
                "north east": "northeast", "north-east": "northeast",
            })
        self._add_log("Canonicalized text columns and normalized 'region'.")
        return self
    def validate_schema(self, drop_invalid=True):
        """Validate ranges for numerical/categorical columns."""
        invalid_rows = 0

        def _coerce(name):
            return pd.to_numeric(self.df[name], errors="coerce") if name in self.df.columns else None

        age = _coerce("age")
        if age is not None:
            mask = age.between(0, 120)
            invalid_rows += (~mask).sum()
            if drop_invalid:
                self.df = self.df[mask]

        bmi = _coerce("bmi")
        if bmi is not None:
            mask = bmi > 0
            invalid_rows += (~mask).sum()
            if drop_invalid:
                self.df = self.df[mask]

        children = _coerce("children")
        if children is not None:
            mask = (children >= 0) & (children == children.round())
            invalid_rows += (~mask).sum()
            if drop_invalid:
                self.df = self.df[mask]

        charges = _coerce("charges")
        if charges is not None:
            mask = charges >= 0
            invalid_rows += (~mask).sum()
            if drop_invalid:
                self.df = self.df[mask]

        self.df = self.df.reset_index(drop=True)
        self._add_log(f"Schema validated. Invalid rows handled: {invalid_rows}.")
        return self

    def handle_outliers_iqr(self, columns=None, iqr_factor=1.5, winsorize=True):
        """Handle outliers using IQR rule.
        - If winsorize=True: clip values.
        - Else: drop outlier rows.
        """
        if columns is None:
            columns = [c for c in ["bmi", "charges"] if c in self.df.columns]

        removed = 0
        for col in columns:
            q1, q3 = self.df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            low, high = q1 - iqr_factor * iqr, q3 + iqr_factor * iqr

            if winsorize:
                self.df[col] = self.df[col].clip(lower=low, upper=high)
            else:
                before = len(self.df)
                self.df = self.df[self.df[col].between(low, high)]
                removed += before - len(self.df)

        self.df = self.df.reset_index(drop=True)
        action = "winsorized" if winsorize else f"removed {removed} rows"
        self._add_log(f"Outliers handled for {columns} using IQR ({action}).")
        return self

    # -------------------------
    # Final checks
    # -------------------------

    def validate_data(self):
        print("\nDataset Info:")
        self.df.info()
        print("\nFirst 5 Rows:")
        print(self.df.head())
        return self

    def get_cleaned_data(self):
        return self.df.copy()

