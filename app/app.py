"""Streamlit dashboard for the real-time financial anomaly detection prototype."""
from pathlib import Path
import sys
import json

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_loader import load_creditcard_data
from src.prediction import load_artifacts, predict_transaction

ROOT = Path(__file__).resolve().parents[1]
PLOTS, REPORTS, MODELS = ROOT / "outputs" / "plots", ROOT / "outputs" / "reports", ROOT / "outputs" / "models"
MODEL_COMPARISON, TUNING_RESULTS = REPORTS / "model_comparison.csv", REPORTS / "threshold_results.csv"
DATASET_SUMMARY = REPORTS / "dataset_summary.json"

@st.cache_data(show_spinner=False)
def get_dataset() -> pd.DataFrame:
    return load_creditcard_data()

def file_version(path: Path) -> int:
    """Return a cache key that changes whenever an artifact is regenerated."""
    return path.stat().st_mtime_ns

@st.cache_data(show_spinner=False)
def read_csv(path: str, version: int) -> pd.DataFrame:
    """Read a generated CSV, refreshing after a new training run."""
    return pd.read_csv(path)

@st.cache_data(show_spinner=False)
def read_summary(path: str, version: int) -> dict:
    """Load aggregate statistics used when the private raw dataset is absent."""
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)[0]

@st.cache_resource(show_spinner=False)
def get_prediction_artifacts(version: int):
    """Load artifacts, refreshing after a new training run."""
    return load_artifacts(MODELS)

def show_plot(filename: str, caption: str) -> None:
    path = PLOTS / filename
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.info(f"`{filename}` will be created when you run `python run.py`.")

def artifacts_ready() -> bool:
    return all(path.exists() for path in [MODELS / "isolation_forest.pkl", MODELS / "amount_scaler.pkl",
        MODELS / "operating_thresholds.json", MODEL_COMPARISON, TUNING_RESULTS])

st.set_page_config(page_title="Financial Anomaly Detection", layout="wide")
st.title("Real-Time Anomaly Detection in Financial Transactions")
st.caption("Educational real-time simulation/prototype - not production banking infrastructure.")
page = st.sidebar.radio("Navigate", ["Project Overview", "Dataset and Class Imbalance", "Preprocessing and RobustScaler",
    "EDA", "Model Comparison", "Confusion Matrices", "Precision-Recall Analysis", "Threshold Tuning",
    "PCA Anomaly Visualization", "Real-Time Prediction"])

if page == "Project Overview":
    st.subheader("How the prototype works")
    st.write("The system learns normal transaction behavior with Isolation Forest and Local Outlier Factor (LOF). Fraud labels are never passed to either model during fitting; they are reserved for validation-based threshold selection and final held-out test evaluation.")
    st.markdown("""**Evaluation design:** stratified 60% training, 20% validation, 20% test. Models fit only normal training transactions. A candidate threshold is selected on validation F1, then evaluated one time on the test set.

**Model intuition:** Isolation Forest flags observations that are easy to isolate with random splits. LOF flags observations whose local density is unusually low compared with their neighbours.""")
    (st.success if artifacts_ready() else st.warning)("Trained models, preprocessing artifacts, tuning results, and evaluation results are available." if artifacts_ready() else "Run `python run.py` to create the trained-model and report artifacts used by this dashboard.")

elif page == "Dataset and Class Imbalance":
    st.subheader("Dataset loading and class imbalance")
    try:
        data = get_dataset(); normal, fraud = int((data.Class == 0).sum()), int((data.Class == 1).sum())
        total = len(data)
        st.dataframe(data.head(), width="stretch")
    except (FileNotFoundError, ValueError):
        summary = read_summary(str(DATASET_SUMMARY), file_version(DATASET_SUMMARY))
        total, normal, fraud = int(summary["shape"][0]), int(summary["class_distribution"]["0"]), int(summary["class_distribution"]["1"])
        st.info("The private raw CSV is not bundled with this public deployment. The metrics and plots below are precomputed from the original dataset.")
    first, second, third = st.columns(3); first.metric("Transactions", f"{total:,}"); second.metric("Normal transactions", f"{normal:,}"); third.metric("Fraud transactions", f"{fraud:,}", f"{fraud / total * 100:.3f}% of dataset")
    st.write("The target is highly imbalanced, so accuracy would be misleading; this project prioritizes precision, recall, and F1.")
    show_plot("class_distribution.png", "Normal vs. fraud transaction distribution")

elif page == "Preprocessing and RobustScaler":
    st.subheader("Leakage-safe preprocessing")
    st.markdown("""- Exact duplicate transactions are removed and missing values in required fields stop execution with an error.
- Data is split before preprocessing: 60% train, 20% validation, 20% test.
- Both models are fitted only on **normal (`Class = 0`) training transactions**.
- A `RobustScaler` is fitted only on training-normal **`Time` and `Amount`**. In particular, `Amount` is transformed using its median and interquartile range, which is less affected by unusually large transaction values.
- `V1` to `V28` are already PCA-transformed dataset features and are used directly for model fitting.""")
    try:
        values = get_dataset()["Amount"]
        amount_stats = [values.min(), values.median(), values.quantile(.75), values.max()]
    except (FileNotFoundError, ValueError):
        summary = read_summary(str(DATASET_SUMMARY), file_version(DATASET_SUMMARY))["amount_summary"]
        amount_stats = [summary["minimum"], summary["median"], summary["p75"], summary["maximum"]]
        st.info("Amount statistics are precomputed because the private raw CSV is not bundled with this public deployment.")
    st.dataframe(pd.DataFrame({"Amount statistic": ["Minimum", "Median", "75th percentile", "Maximum"], "Original Amount": amount_stats}), width="stretch", hide_index=True)
    st.info("The saved `outputs/models/amount_scaler.pkl` is the fitted RobustScaler used by real-time prediction.")

elif page == "EDA":
    st.subheader("Exploratory data analysis")
    st.write("Charts use original, unscaled values so distributions stay interpretable.")
    left, right = st.columns(2)
    with left:
        show_plot("amount_distribution.png", "Transaction amount distribution"); show_plot("amount_by_class.png", "Transaction amount by class"); show_plot("fraud_amounts.png", "Fraud transaction amount distribution")
    with right:
        show_plot("time_distribution.png", "Transaction time distribution by class"); show_plot("correlation_heatmap.png", "Feature correlation heatmap")

elif page == "Model Comparison":
    st.subheader("Held-out test-set model comparison")
    if MODEL_COMPARISON.exists():
        columns = ["Model", "Precision", "Recall", "F1 Score", "False Positive Rate", "False Negative Rate", "TP", "TN", "FP", "FN", "Execution Time (seconds)"]
        st.write("Values are calculated on the untouched test partition after validation-based threshold selection.")
        st.dataframe(read_csv(str(MODEL_COMPARISON), file_version(MODEL_COMPARISON))[columns], width="stretch", hide_index=True)
        st.caption("Precision: flagged transactions that were fraud. Recall: fraud transactions found. F1: balance of precision and recall. Lower false-positive and false-negative rates are preferred.")
    else: st.info("Run `python run.py` first.")

elif page == "Confusion Matrices":
    st.subheader("Confusion matrices - held-out test partition")
    st.write("Rows are actual classes and columns are predictions. TN/FP are the top row; FN/TP are the bottom row.")
    left, right = st.columns(2)
    with left: show_plot("isolation_forest_confusion_matrix.png", "Isolation Forest confusion matrix")
    with right: show_plot("lof_confusion_matrix.png", "LOF confusion matrix")

elif page == "Precision-Recall Analysis":
    st.subheader("Precision-Recall curve")
    st.write("Each curve shows the trade-off as the anomaly-score threshold changes. Higher recall catches more fraud but can reduce precision and increase false alarms. This is more informative than accuracy for rare fraud events.")
    show_plot("precision_recall_curves.png", "Precision-Recall curves for Isolation Forest and LOF")

elif page == "Threshold Tuning":
    st.subheader("Threshold / contamination tuning on validation data")
    st.write("For each candidate baseline outlier fraction, the application derives a score threshold from normal training scores and calculates validation precision, recall, F1, and false-positive rate. The highest validation F1 setting is persisted for prediction.")
    if TUNING_RESULTS.exists():
        columns = ["Model", "Contamination", "Score Threshold", "Precision", "Recall", "F1 Score", "False Positive Rate", "False Negative Rate", "TP", "TN", "FP", "FN"]
        st.dataframe(read_csv(str(TUNING_RESULTS), file_version(TUNING_RESULTS))[columns], width="stretch", hide_index=True); show_plot("contamination_tuning.png", "Validation F1 across tested baseline outlier fractions")
    else: st.info("Run `python run.py` first.")

elif page == "PCA Anomaly Visualization":
    st.subheader("2D PCA visualization of detected anomalies")
    st.write("`V1` to `V28` are already PCA-transformed in the source dataset and are not transformed again for model fitting. The additional two-component PCA here is used only to display normal transactions and Isolation Forest-detected anomalies on a 2D chart.")
    show_plot("pca_anomalies.png", "Normal transactions and detected anomalies in a 2D PCA projection")

else:
    st.subheader("Real-Time Transaction Prediction")
    st.write("Enter a transaction to score it with the saved Isolation Forest, fitted RobustScaler, and validation-selected score threshold.")
    if not artifacts_ready(): st.warning("Train the model first with `python run.py`.")
    else:
        with st.form("transaction"):
            values = {"Time": st.number_input("Time", min_value=0.0, value=0.0)}; columns = st.columns(4)
            for number in range(1, 29): values[f"V{number}"] = columns[(number - 1) % 4].number_input(f"V{number}", value=0.0)
            values["Amount"] = st.number_input("Amount", min_value=0.0, value=0.0); submitted = st.form_submit_button("Analyze transaction")
        if submitted:
            model, scaler, thresholds = get_prediction_artifacts(file_version(MODELS / "operating_thresholds.json")); outcome = predict_transaction(values, model, scaler, float(thresholds["isolation_forest_threshold"]))
            first, second, third = st.columns(3); first.metric("Calculated anomaly score", f"{outcome['anomaly_score']:.5f}"); second.metric("Prediction", outcome["prediction"]); third.metric("Risk status", outcome["risk_status"])
            st.caption("This calculated prototype score supports demonstration only; it is not a production fraud decision.")
