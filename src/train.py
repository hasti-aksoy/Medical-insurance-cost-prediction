# train.py
from pathlib import Path
import json
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from src.baseline import get_linear_pipeline
from src.tree import get_tree_pipeline, get_rf_pipeline
from src.xgboost import get_xgb_pipeline
from src.evaluate import print_report, evaluate
from joblib import dump

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/insurance_cleaned.csv"
ARTIFACTS = ROOT / "models"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

def main():
    df = pd.read_csv(DATA)
    X = df.drop(columns=["charges"]) if "charges" in df.columns else df
    y = df["charges"] if "charges" in df.columns else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "LinearRegression": get_linear_pipeline(X_train),
        "DecisionTree": get_tree_pipeline(X_train, max_depth=6),
        "RandomForest": get_rf_pipeline(X_train, n_estimators=300),
    }
    # XGBoost is optional depending on environment
    try:
        models["XGBoost"] = get_xgb_pipeline(X_train)
    except Exception as e:
        print(f"Skipping XGBoost: {e}")

    results = {}
    fitted_models = {}
    for name, pipe in models.items():
        t0 = time.perf_counter()
        pipe.fit(X_train, y_train)
        t1 = time.perf_counter()
        y_pred = pipe.predict(X_test)
        t2 = time.perf_counter()
        print_report(name, y_test, y_pred)
        metrics = evaluate(y_test, y_pred)
        metrics.update({
            "fit_seconds": round(t1 - t0, 4),
            "predict_seconds": round(t2 - t1, 4),
            "total_seconds": round(t2 - t0, 4),
        })
        results[name] = metrics
        fitted_models[name] = pipe
        # Save fitted pipeline
        model_path = ARTIFACTS / f"{name}.joblib"
        dump(pipe, model_path)
        print(f"Saved model to {model_path}")

    # Select and save best model by RMSE
    if results:
        best_name = min(results.keys(), key=lambda k: results[k]["rmse"])
        best_path = ARTIFACTS / "best_model.joblib"
        dump(fitted_models[best_name], best_path)
        print(f"Best model: {best_name} (RMSE={results[best_name]['rmse']:.4f}) -> {best_path}")
        # Write a small manifest for convenience
        with open(ARTIFACTS / "best_model.json", "w") as f:
            json.dump({"name": best_name, "metrics": results[best_name]}, f, indent=2)

    with open(ARTIFACTS / "all_models_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved metrics to {ARTIFACTS / 'all_models_results.json'}")


if __name__ == "__main__":
    main()
