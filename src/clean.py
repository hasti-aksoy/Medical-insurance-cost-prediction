"""CLI to clean the insurance dataset using DataCleaning.

Usage examples:
- python -m src.clean --input data/insurance.csv --output data/insurance_cleaned.csv
- python -m src.clean --no-winsorize  # drop outliers instead of clipping
"""

import argparse
from pathlib import Path
import pandas as pd

from src.data_cleaning import DataCleaning


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Clean the insurance dataset and write a cleaned CSV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", default="data/insurance.csv", help="Path to input CSV")
    p.add_argument(
        "--output",
        default="data/insurance_cleaned.csv",
        help="Path to write cleaned CSV",
    )
    p.add_argument(
        "--no-winsorize",
        action="store_true",
        help="Drop IQR outliers instead of winsorizing (clipping)",
    )
    p.add_argument(
        "--keep-invalid",
        action="store_true",
        help="Keep rows failing schema validation (by default they are dropped)",
    )
    return p


def main():
    args = build_parser().parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_path}")

    df = pd.read_csv(in_path)

    cleaner = (
        DataCleaning(df)
        .canonicalize_text()
        .remove_duplicates()
        .validate_schema(drop_invalid=not args.keep_invalid)
        .fill_categorical_mode()
        .fill_numerical_median(group_cols=["sex", "smoker", "region"])
        .handle_outliers_iqr(columns=["bmi", "charges"], winsorize=not args.no_winsorize)
        .encode_label(["sex", "smoker"])
        .encode_one_hot(["region"])
    )

    clean_df = cleaner.get_cleaned_data()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(out_path, index=False)
    print(f"Wrote cleaned data to {out_path}")


if __name__ == "__main__":
    main()
