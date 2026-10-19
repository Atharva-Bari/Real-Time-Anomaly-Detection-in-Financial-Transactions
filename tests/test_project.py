import pandas as pd
import pytest
from src.data_loader import load_creditcard_data
from src.preprocessing import prepare_data, prepare_experiment, FEATURE_COLUMNS
from src.isolation_forest_model import train_isolation_forest, predict_isolation_forest, score_isolation_forest
from src.threshold_tuning import tune_thresholds
from src.evaluation import evaluate_predictions

def fixture_data():
    rows=[]
    for n in range(12): rows.append({"Time": float(n), **{f"V{i}": float(n+i) for i in range(1,29)}, "Amount": float(n+1), "Class": 0})
    rows.append({"Time": 99., **{f"V{i}": 99. for i in range(1,29)}, "Amount": 999., "Class": 1})
    return pd.DataFrame(rows)

def test_loader_missing_file_has_helpful_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        load_creditcard_data(tmp_path / "missing.csv")

def test_preprocessing_model_prediction_and_metrics():
    prepared=prepare_data(fixture_data()); assert list(prepared.features.columns) == FEATURE_COLUMNS
    model, _ = train_isolation_forest(prepared.features[prepared.labels == 0], .1)
    pred, scores = predict_isolation_forest(model, prepared.features)
    metrics = evaluate_predictions(prepared.labels, pred, scores, "Test")
    assert set(pred.unique()).issubset({0,1}); assert "F1 Score" in metrics

def test_experiment_partition_and_threshold_tuning_are_leakage_safe():
    data = pd.concat([fixture_data()] * 10, ignore_index=True)
    data["Time"] = range(len(data))
    experiment = prepare_experiment(data)
    baseline_indices = set(experiment.baseline_features.index)
    assert baseline_indices.isdisjoint(set(experiment.validation_features.index))
    assert baseline_indices.isdisjoint(set(experiment.test_features.index))
    model, _ = train_isolation_forest(experiment.baseline_features)
    baseline_scores = score_isolation_forest(model, experiment.baseline_features)
    validation_scores = score_isolation_forest(model, experiment.validation_features)
    tuning = tune_thresholds(baseline_scores, validation_scores, experiment.validation_labels, "Isolation Forest")
    assert {"Contamination", "Score Threshold", "F1 Score"}.issubset(tuning.columns)
