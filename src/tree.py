# Tree-based model pipelines
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

def _preprocessor(X):
    """Numeric preprocessor applying StandardScaler to all columns in X."""
    num_cols = X.columns.tolist()
    numeric = Pipeline([("scaler", StandardScaler())])
    return ColumnTransformer([("num", numeric, num_cols)], remainder="drop",
                             verbose_feature_names_out=False)

def get_tree_pipeline(X, max_depth=None, random_state=42):
    """DecisionTreeRegressor pipeline with scaling."""
    pre = _preprocessor(X)
    model = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
    return Pipeline([("pre", pre), ("model", model)])

def get_rf_pipeline(X, n_estimators=200, random_state=42):
    """RandomForestRegressor pipeline with scaling."""
    pre = _preprocessor(X)
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )
    return Pipeline([("pre", pre), ("model", model)])
