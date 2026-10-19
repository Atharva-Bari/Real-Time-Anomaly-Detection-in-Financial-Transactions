# Real-Time Anomaly Detection in Financial Transactions

An MCS-friendly, runnable Python project that uses unsupervised Isolation Forest and Local Outlier Factor (LOF) to flag anomalous credit-card transactions. It is a real-time **simulation/prototype**, not banking production infrastructure.

## Dataset setup

Download `creditcard.csv` from Kaggle's [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), accept its terms, and place it here:

`data/creditcard.csv`

The expected fields are `Time`, `V1` through `V28`, `Amount`, and `Class`. `Class=1` is fraud. The CSV is excluded from Git because it is large. If it is absent, `run.py` gives a clear placement message.

## Windows installation and running

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python run.py
python -m streamlit run app/app.py
```

If `pip` shows packages installing under `Python314` rather than `.venv`, do not use bare `pip`; use `python -m pip` exactly as above. It guarantees installation into the active virtual environment.

For a code-only smoke test when the real dataset is unavailable (its generated metrics are not real results):

```powershell
python run.py --synthetic-demo
```

Optional: run LOF against all normal baseline records (slower and more memory intensive):

```powershell
python run.py --max-lof-rows 0
```

Run tests:

```powershell
python -m pytest -q
```

## Project structure

```text
financial-anomaly-detection/
  app/app.py                 Streamlit dashboard and prediction form
  data/creditcard.csv        User-provided dataset
  notebooks/                 Guided end-to-end notebook
  outputs/plots/             Generated EDA, PR, PCA, and matrix charts
  outputs/models/            Generated joblib artifacts
  outputs/reports/           Generated tuning/comparison CSVs and report
  src/                       Modular loader, prep, models, EDA, metrics
  tests/test_project.py      Lightweight synthetic-data tests
  run.py                     Training entry point
```

## How it works

1. The program validates and de-duplicates the data, then creates stratified 60%/20%/20% train/validation/test partitions. `RobustScaler` is fitted only on normal transactions in the training partition; it scales `Time` and `Amount`, and is especially important for extreme transaction amounts.
2. Both detectors learn normal training data only. Fraud labels never enter model fitting; labels are used only for partitioning and offline validation/test evaluation.
3. Isolation Forest flags points that are isolated by random feature splits. LOF flags points whose local density is lower than their neighbors'. LOF runs in novelty mode, allowing predictions for newly entered transactions.
4. Each detector scores the validation partition at candidate baseline outlier fractions (`0.001` to `0.05`). The score threshold maximizing validation F1 is selected, then evaluated once on an untouched test partition. Results go to `outputs/reports/threshold_results.csv`.
5. The generated comparison is an unseen-test report emphasizing precision, recall, F1, false-positive rate, and confusion counts - not accuracy, which is misleading for a heavily imbalanced fraud dataset.

The dashboard presents reports/plots after training and accepts `Time`, `V1-V28`, and `Amount` for a single-transaction scoring simulation.

## Metrics and operating threshold

Fraud is rare, so a model can appear accurate simply by predicting every transaction as normal. This project therefore reports:

- **Precision:** of alerts raised, the fraction that are actual fraud.
- **Recall:** the fraction of known fraud that is detected.
- **F1:** the harmonic balance of precision and recall.
- **False positive rate / false negative rate:** the proportion of legitimate transactions incorrectly alerted and fraud transactions missed.
- **TP, TN, FP, FN:** the counts behind the metrics, shown in both model-specific confusion matrices.

The dashboard's Threshold Tuning page shows every tested candidate outlier fraction and its validation metrics. The selected threshold maximizes validation F1; it is not selected using the test set. In a real bank, business cost, fraud-review capacity, and the cost of missed fraud would also influence the final threshold.

## Why unsupervised anomaly detection?

Fraud labels are scarce, delayed, and patterns change. Isolation Forest and LOF establish a baseline of normal behaviour rather than learning fraud labels directly. Isolation Forest uses random splits to isolate unusual points, while density-based LOF compares an observation with neighbouring transactions. Their scores are signals for review, not proof of fraud.

## Outputs

After a real dataset run, see `outputs/models/isolation_forest.pkl`, `outputs/models/lof.pkl`, `outputs/models/amount_scaler.pkl`, `outputs/reports/model_comparison.csv`, and figures in `outputs/plots/`.

## Limitations

Unsupervised scores are not proof of fraud. Production use needs temporal validation, business-cost thresholding, monitoring for drift, review workflows, security controls, and a retraining policy.

## Author

Prepared as an academic MCS project template. Add your name, institution, and supervisor before submission.
