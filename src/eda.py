"""EDA and output plot generation."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

def save_eda_plots(frame: pd.DataFrame, output_dir: Path, max_rows: int = 30000) -> None:
    output_dir.mkdir(parents=True, exist_ok=True); sample = frame.sample(min(len(frame), max_rows), random_state=42)
    plots = [("class_distribution", lambda ax: sns.countplot(data=frame, x="Class", ax=ax)),
             ("amount_distribution", lambda ax: sns.histplot(data=sample, x="Amount", bins=60, ax=ax)),
             ("amount_by_class", lambda ax: sns.boxplot(data=sample, x="Class", y="Amount", showfliers=False, ax=ax)),
             ("time_distribution", lambda ax: sns.histplot(data=sample, x="Time", hue="Class", bins=60, element="step", ax=ax))]
    for name, drawer in plots:
        fig, ax = plt.subplots(figsize=(8, 5)); drawer(ax); ax.set_title(name.replace("_", " ").title()); fig.tight_layout(); fig.savefig(output_dir / f"{name}.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 9)); sns.heatmap(sample.corr(numeric_only=True), cmap="coolwarm", center=0, ax=ax); ax.set_title("Feature Correlation Heatmap"); fig.tight_layout(); fig.savefig(output_dir / "correlation_heatmap.png", dpi=150); plt.close(fig)
    fraud = sample[sample.Class == 1]; fig, ax = plt.subplots(figsize=(8, 5)); sns.histplot(fraud, x="Amount", bins=30, ax=ax); ax.set_title("Fraud Transaction Amounts"); fig.tight_layout(); fig.savefig(output_dir / "fraud_amounts.png", dpi=150); plt.close(fig)

def save_pca_anomalies(features: pd.DataFrame, predictions, output_dir: Path, max_rows: int = 8000) -> None:
    index = features.sample(min(len(features), max_rows), random_state=42).index; coords = PCA(n_components=2, random_state=42).fit_transform(features.loc[index])
    plot = pd.DataFrame({"PC1": coords[:,0], "PC2": coords[:,1], "Detected anomaly": predictions.loc[index].map({0:"Normal",1:"Anomaly"})})
    fig, ax = plt.subplots(figsize=(8, 6)); sns.scatterplot(plot, x="PC1", y="PC2", hue="Detected anomaly", alpha=.55, s=20, ax=ax); ax.set_title("PCA Visualization of Detected Anomalies"); fig.tight_layout(); fig.savefig(output_dir / "pca_anomalies.png", dpi=150); plt.close(fig)
