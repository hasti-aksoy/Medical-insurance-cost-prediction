# XGBoost pipeline (optional dependency)
try:
    from xgboost import XGBRegressor  # type: ignore
    _xgb_import_error = None
except Exception as _e:  # pragma: no cover
    XGBRegressor = None  # type: ignore
    _xgb_import_error = _e

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

def _preprocessor(X):
    num_cols = X.columns.tolist()
    numeric = Pipeline([("scaler", StandardScaler())])
    return ColumnTransformer([("num", numeric, num_cols)], remainder="drop",
                             verbose_feature_names_out=False)

def get_xgb_pipeline(X, random_state=42):
    """Return an XGBoost regression pipeline, if xgboost is available.

    Raises ImportError with a helpful message if xgboost is not installed.
    """
    if XGBRegressor is None:  # pragma: no cover
        raise ImportError(f"xgboost is not installed or failed to import: {_xgb_import_error}")

    pre = _preprocessor(X)
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1,
    )
    return Pipeline([("pre", pre), ("model", model)])
