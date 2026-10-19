"""Single-transaction prediction for the real-time prototype."""
from pathlib import Path
import joblib
import json
import pandas as pd
from .preprocessing import FEATURE_COLUMNS, transform_features
from .isolation_forest_model import predict_isolation_forest

def load_artifacts(models_dir: Path):
    with (models_dir / "operating_thresholds.json").open(encoding="utf-8") as file:
        thresholds = json.load(file)
    return joblib.load(models_dir / "isolation_forest.pkl"), joblib.load(models_dir / "amount_scaler.pkl"), thresholds

def predict_transaction(transaction: dict, model, scaler, threshold: float) -> dict:
    missing = set(FEATURE_COLUMNS) - set(transaction)
    if missing: raise ValueError(f"Transaction is missing fields: {sorted(missing)}")
    features = transform_features(pd.DataFrame([transaction]), scaler)
    prediction, score = predict_isolation_forest(model, features, threshold)
    suspicious = bool(prediction.iloc[0]); return {"anomaly_score": float(score.iloc[0]), "prediction": "SUSPICIOUS TRANSACTION" if suspicious else "NORMAL TRANSACTION", "risk_status": "HIGH" if suspicious else "LOW"}
