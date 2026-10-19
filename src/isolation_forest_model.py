"""Isolation Forest training and inference."""
import time
import pandas as pd
from sklearn.ensemble import IsolationForest

def train_isolation_forest(normal_features: pd.DataFrame, contamination: float | str = "auto", random_state: int = 42) -> tuple[IsolationForest, float]:
    start = time.perf_counter()
    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=random_state, n_jobs=-1)
    model.fit(normal_features)
    return model, time.perf_counter() - start

def score_isolation_forest(model: IsolationForest, features: pd.DataFrame) -> pd.Series:
    """Return a score where larger values indicate a more unusual transaction."""
    return pd.Series(-model.score_samples(features), index=features.index)

def predict_isolation_forest(model: IsolationForest, features: pd.DataFrame, threshold: float | None = None) -> tuple[pd.Series, pd.Series]:
    scores = score_isolation_forest(model, features)
    predictions = (scores >= threshold).astype(int) if threshold is not None else (model.predict(features) == -1).astype(int)
    return pd.Series(predictions, index=features.index), scores
