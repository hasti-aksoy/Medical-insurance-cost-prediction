import numpy as np
from typing import Iterable, Dict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def rmse(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    """Root Mean Squared Error."""
    return mean_squared_error(y_true, y_pred, squared=False)


def rmsle(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    """Root Mean Squared Log Error with robust handling.

    - Clips negatives to 0 (RMSLE is defined for non-negative values).
    - Ignores non-finite values by operating on a finite mask.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)

    finite = np.isfinite(yt) & np.isfinite(yp)
    if not np.any(finite):
        return float("nan")

    yt = np.clip(yt[finite], 0, None)
    yp = np.clip(yp[finite], 0, None)

    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.sqrt(mean_squared_error(np.log1p(yt), np.log1p(yp))))


def mae(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    """Mean Absolute Error."""
    return mean_absolute_error(y_true, y_pred)


def r2(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    """R-squared (coefficient of determination)."""
    return r2_score(y_true, y_pred)


def evaluate(y_true: Iterable[float], y_pred: Iterable[float]) -> Dict[str, float]:
    """Compute a standard set of regression metrics and return as a dict."""
    return {
        "rmse": rmse(y_true, y_pred),
        "rmsle": rmsle(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2(y_true, y_pred),
    }


def print_report(name: str, y_true: Iterable[float], y_pred: Iterable[float]) -> None:
    """Pretty-print metrics for a given model/prediction set."""
    metrics = evaluate(y_true, y_pred)
    print(f"\n{name}")
    print("-" * len(name))
    print(f"RMSE : {metrics['rmse']:.4f}")
    print(f"RMSLE: {metrics['rmsle']:.4f}")
    print(f"MAE  : {metrics['mae']:.4f}")
    print(f"R2   : {metrics['r2']:.4f}")


__all__ = [
    "rmse",
    "rmsle",
    "mae",
    "r2",
    "evaluate",
    "print_report",
]
