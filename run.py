"""Run the end-to-end financial anomaly-detection experiment."""
from pathlib import Path
import argparse
import json
import time
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.data_loader import load_creditcard_data, dataset_summary
from src.preprocessing import prepare_experiment
from src.eda import save_eda_plots, save_pca_anomalies
from src.threshold_tuning import tune_thresholds
from src.isolation_forest_model import train_isolation_forest, score_isolation_forest, predict_isolation_forest
from src.lof_model import train_lof, score_lof, predict_lof
from src.evaluation import evaluate_predictions, save_confusion_matrix, save_precision_recall_curves

ROOT = Path(__file__).resolve().parent

def make_synthetic_demo_data(rows: int = 400) -> pd.DataFrame:
    """Create an in-memory smoke-test dataset; never use its metrics as real results."""
    rng = np.random.default_rng(42)
    normal_count = rows - max(4, rows // 50)
    normal = pd.DataFrame(rng.normal(0, 1, size=(normal_count, 28)), columns=[f"V{i}" for i in range(1, 29)])
    fraud = pd.DataFrame(rng.normal(5, 1, size=(rows - normal_count, 28)), columns=normal.columns)
    frame = pd.concat([normal, fraud], ignore_index=True)
    frame.insert(0, "Time", rng.uniform(0, 172800, len(frame)))
    frame["Amount"] = np.r_[rng.lognormal(3, .7, normal_count), rng.lognormal(6, .7, len(fraud))]
    frame["Class"] = [0] * normal_count + [1] * len(fraud)
    return frame.sample(frac=1, random_state=42).reset_index(drop=True)

def main() -> None:
    parser = argparse.ArgumentParser(description="Financial anomaly detection prototype")
    parser.add_argument("--data", help="Path to creditcard.csv")
    parser.add_argument("--max-lof-rows", type=int, default=50000, help="Normal baseline rows for LOF; 0 means all")
    parser.add_argument("--synthetic-demo", action="store_true", help="Run an in-memory smoke test; its metrics are not real fraud results")
    args = parser.parse_args()
    plot_dir, model_dir, report_dir = (ROOT / "outputs" / name for name in ("plots", "models", "reports"))
    for folder in (plot_dir, model_dir, report_dir): folder.mkdir(parents=True, exist_ok=True)
    print("=" * 40 + "\nFINANCIAL ANOMALY DETECTION SYSTEM\n" + "=" * 40)
    print("Loading dataset...")
    try: frame = make_synthetic_demo_data() if args.synthetic_demo else load_creditcard_data(args.data)
    except (FileNotFoundError, ValueError) as error: print(f"ERROR: {error}"); return
    if args.synthetic_demo: print("WARNING: using synthetic smoke-test data. Do not interpret its metrics as real results.")
    print(f"Dataset loaded successfully: {frame.shape[0]:,} rows, {frame.shape[1]} columns")
    print("Creating leakage-safe train/validation/test partitions...")
    experiment = prepare_experiment(frame)
    pd.DataFrame([dataset_summary(experiment.original)]).to_json(report_dir / "dataset_summary.json", orient="records", indent=2)
    print("Generating EDA plots..."); save_eda_plots(experiment.original, plot_dir)
    print("Training Isolation Forest on normal training transactions only...")
    if_start = time.perf_counter()
    iforest, _ = train_isolation_forest(experiment.baseline_features)
    if_baseline_scores = score_isolation_forest(iforest, experiment.baseline_features)
    if_validation_scores = score_isolation_forest(iforest, experiment.validation_features)
    if_tuning = tune_thresholds(if_baseline_scores, if_validation_scores, experiment.validation_labels, "Isolation Forest")
    if_selection_time = time.perf_counter() - if_start
    print("Training LOF on normal training transactions only...")
    lof_start = time.perf_counter()
    lof, _ = train_lof(experiment.baseline_features, "auto", max_training_rows=args.max_lof_rows or None)
    lof_baseline_scores = score_lof(lof, experiment.baseline_features)
    lof_validation_scores = score_lof(lof, experiment.validation_features)
    lof_tuning = tune_thresholds(lof_baseline_scores, lof_validation_scores, experiment.validation_labels, "LOF")
    lof_selection_time = time.perf_counter() - lof_start
    tuning = pd.concat([if_tuning, lof_tuning], ignore_index=True)
    tuning.to_csv(report_dir / "threshold_results.csv", index=False)
    if_choice, lof_choice = if_tuning.iloc[0], lof_tuning.iloc[0]
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, result in tuning.groupby("Model"):
        ax.plot(result["Contamination"], result["F1 Score"], marker="o", label=name)
    ax.set(xlabel="Baseline outlier fraction", ylabel="Validation F1 score", title="Operating Threshold Tuning"); ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(plot_dir / "contamination_tuning.png", dpi=150); plt.close(fig)
    print("Evaluating selected configurations once on the untouched test partition...")
    if_test_start = time.perf_counter()
    if_pred, if_scores = predict_isolation_forest(iforest, experiment.test_features, float(if_choice["Score Threshold"]))
    if_total_time = if_selection_time + (time.perf_counter() - if_test_start)
    lof_test_start = time.perf_counter()
    lof_pred, lof_scores = predict_lof(lof, experiment.test_features, float(lof_choice["Score Threshold"]))
    lof_total_time = lof_selection_time + (time.perf_counter() - lof_test_start)
    results = pd.DataFrame([
        evaluate_predictions(experiment.test_labels, if_pred, if_scores, "Isolation Forest", if_total_time),
        evaluate_predictions(experiment.test_labels, lof_pred, lof_scores, "LOF", lof_total_time),
    ])
    results.to_csv(report_dir / "model_comparison.csv", index=False)
    save_confusion_matrix(experiment.test_labels, if_pred, "Isolation Forest", plot_dir); save_confusion_matrix(experiment.test_labels, lof_pred, "LOF", plot_dir)
    save_precision_recall_curves({"Isolation Forest": if_scores, "LOF": lof_scores}, experiment.test_labels, plot_dir); save_pca_anomalies(experiment.test_features, if_pred, plot_dir)
    joblib.dump(iforest, model_dir / "isolation_forest.pkl"); joblib.dump(lof, model_dir / "lof.pkl"); joblib.dump(experiment.scaler, model_dir / "amount_scaler.pkl")
    with (model_dir / "operating_thresholds.json").open("w", encoding="utf-8") as file:
        json.dump({"isolation_forest_threshold": float(if_choice["Score Threshold"]), "lof_threshold": float(lof_choice["Score Threshold"]), "selection_partition": "validation"}, file, indent=2)
    print("Models, reports, and plots saved successfully. Project execution completed.")
    print(results[["Model", "Precision", "Recall", "F1 Score", "False Positive Rate"]].to_string(index=False))

if __name__ == "__main__": main()
