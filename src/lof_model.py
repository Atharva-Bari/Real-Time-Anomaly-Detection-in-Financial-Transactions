"""Novelty-detection Local Outlier Factor implementation."""
import time
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

def train_lof(normal_features: pd.DataFrame, contamination: float | str, n_neighbors: int = 35,
              max_training_rows: int | None = 50000, random_state: int = 42) -> tuple[LocalOutlierFactor, float]:
    """Fit LOF in novelty mode. Sampling protects laptop memory; None uses all normal rows."""
    training = normal_features
    if max_training_rows and len(training) > max_training_rows:
        training = training.sample(max_training_rows, random_state=random_state)
    neighbors = min(n_neighbors, max(2, len(training) - 1))
    start = time.perf_counter()
    model = LocalOutlierFactor(n_neighbors=neighbors, contamination=contamination, novelty=True, n_jobs=-1)
    # Use arrays consistently because sklearn's LOF scorer internally validates arrays.
    model.fit(training.to_numpy())
    return model, time.perf_counter() - start

def score_lof(model: LocalOutlierFactor, features: pd.DataFrame) -> pd.Series:
    values = features.to_numpy()
    return pd.Series(-model.score_samples(values), index=features.index)

def predict_lof(model: LocalOutlierFactor, features: pd.DataFrame, threshold: float | None = None) -> tuple[pd.Series, pd.Series]:
    scores = score_lof(model, features)
    predictions = (scores >= threshold).astype(int) if threshold is not None else (model.predict(features.to_numpy()) == -1).astype(int)
    return pd.Series(predictions, index=features.index), scores
