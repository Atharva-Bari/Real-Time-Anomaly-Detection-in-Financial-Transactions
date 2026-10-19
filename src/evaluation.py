"""Evaluation focused on rare-event detection rather than accuracy."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score

def evaluate_predictions(y_true, y_pred, scores, model_name: str, execution_time: float = 0.0) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"Model": model_name, "Precision": precision, "Recall": recall, "F1 Score": f1,
            "False Positive Rate": fp / (fp + tn) if fp + tn else 0.0,
            "False Negative Rate": fn / (fn + tp) if fn + tp else 0.0,
            "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn), "Execution Time (seconds)": execution_time,
            "Average Precision": average_precision_score(y_true, scores)}

def save_confusion_matrix(y_true, y_pred, model_name: str, output_dir: Path) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=[0,1])
    fig, ax = plt.subplots(figsize=(5, 4)); sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["Normal", "Anomaly"], yticklabels=["Normal", "Fraud"])
    ax.set(title=f"{model_name} Confusion Matrix", xlabel="Predicted", ylabel="Actual")
    fig.tight_layout(); fig.savefig(output_dir / f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png", dpi=150); plt.close(fig)

def save_precision_recall_curves(result_map: dict, y_true, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, scores in result_map.items():
        precision, recall, _ = precision_recall_curve(y_true, scores)
        ax.plot(recall, precision, label=name)
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curves"); ax.legend(); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(output_dir / "precision_recall_curves.png", dpi=150); plt.close(fig)
