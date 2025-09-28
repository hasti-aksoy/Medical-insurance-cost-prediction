import argparse
from pathlib import Path
import pandas as pd
from joblib import load
from src.evaluate import print_report


def main():
    parser = argparse.ArgumentParser(description="Predict medical insurance charges using a saved model.")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--output", required=False, help="Path to output CSV with predictions")
    parser.add_argument("--model", required=False, help="Path to saved model .joblib (defaults to models/best_model.joblib)")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    model_path = Path(args.model) if args.model else (root / "models/best_model.joblib")
    in_path = Path(args.input)
    out_path = Path(args.output) if args.output else in_path.with_name(in_path.stem + "_predictions.csv")

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not in_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_path}")

    print(f"Loading model: {model_path}")
    pipe = load(model_path)

    print(f"Reading input: {in_path}")
    df = pd.read_csv(in_path)

    # Determine required feature columns from the pipeline's preprocessor
    try:
        pre = pipe.named_steps.get("pre")
        if pre is None:
            raise AttributeError
        # Assumes the first transformer is the numeric passthrough with list of columns
        transformers = getattr(pre, "transformers_", [])
        feat_cols = None
        if transformers:
            # Look for the transformer named 'num'
            for name, _, cols in transformers:
                if name == "num":
                    feat_cols = list(cols)
                    break
        if feat_cols is None:
            raise AttributeError
    except AttributeError:
        # Fallback: use all columns except target if present
        feat_cols = [c for c in df.columns if c != "charges"]

    # Drop target if present and align to expected columns
    X = df.drop(columns=["charges"], errors="ignore")
    missing = [c for c in feat_cols if c not in X.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")
    X = X[feat_cols]

    print("Running predictions...")
    preds = pipe.predict(X)
    out_df = df.copy()
    out_df["prediction"] = preds

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote predictions to {out_path}")

    # If ground truth is present, print quick metrics
    if "charges" in df.columns:
        print_report("Evaluation on provided data", df["charges"], preds)


if __name__ == "__main__":
    main()

