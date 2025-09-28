from typing import Dict, Any, Tuple
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from src.evaluate import rmse, rmsle, mae, r2, print_report

def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create a simple numeric preprocessor with standardization."""
    num_cols = X.columns.tolist()
    numeric = Pipeline([("scaler", StandardScaler())])
    return ColumnTransformer([("num", numeric, num_cols)], remainder="drop",
                             verbose_feature_names_out=False)

def get_linear_pipeline(X: pd.DataFrame) -> Pipeline:
    """Assemble the baseline linear regression pipeline."""
    pre = _build_preprocessor(X)
    reg = LinearRegression()
    return Pipeline([("pre", pre), ("model", reg)])

def train_baseline(
    df: pd.DataFrame,
    target: str = "charges",
    test_size: float = 0.2,
    random_state: int = 42,
    cv_splits: int = 5
) -> Tuple[Pipeline, Dict[str, Any]]:
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not in DataFrame")
    X = df.drop(columns=[target])
    y = df[target].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    pipe = get_linear_pipeline(X_train)
    pipe.fit(X_train, y_train)

    # Holdout evaluation
    y_pred = pipe.predict(X_test)
    holdout = {
        "rmse": rmse(y_test, y_pred),
        "rmsle": rmsle(y_test, y_pred),
        "mae": mae(y_test, y_pred),
        "r2": r2(y_test, y_pred),
    }
    print_report("Baseline (Linear) - Holdout", y_test, y_pred)

    # Cross-validated predictions on training set
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    y_cv_pred = cross_val_predict(pipe, X_train, y_train, cv=kf, n_jobs=-1)
    cv = {
        "rmse": rmse(y_train, y_cv_pred),
        "rmsle": rmsle(y_train, y_cv_pred),
        "mae": mae(y_train, y_cv_pred),
        "r2": r2(y_train, y_cv_pred),
    }
    print_report(f"Baseline (Linear) - {cv_splits}-Fold CV (train)", y_train, y_cv_pred)

    return pipe, {"holdout": holdout, "cv": cv}


