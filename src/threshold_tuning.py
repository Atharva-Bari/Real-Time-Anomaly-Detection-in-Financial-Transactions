"""Contamination tuning using evaluation labels held out from fitting."""
import numpy as np
import pandas as pd
from .evaluation import evaluate_predictions

DEFAULT_CONTAMINATIONS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05]

def tune_thresholds(baseline_scores, validation_scores, validation_labels, model_name: str,
                    contaminations=DEFAULT_CONTAMINATIONS) -> pd.DataFrame:
    """Select a score cutoff on validation data; test data remains untouched."""
    rows = []
    for contamination in contaminations:
        threshold = float(np.quantile(baseline_scores, 1 - contamination))
        predictions = (validation_scores >= threshold).astype(int)
        record = evaluate_predictions(validation_labels, predictions, validation_scores, model_name)
        record["Contamination"] = contamination
        record["Score Threshold"] = threshold
        rows.append(record)
    return pd.DataFrame(rows).sort_values("F1 Score", ascending=False).reset_index(drop=True)
