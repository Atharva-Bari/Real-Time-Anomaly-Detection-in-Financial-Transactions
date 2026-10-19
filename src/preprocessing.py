"""Leakage-aware feature preparation for unsupervised models."""
from dataclasses import dataclass
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

FEATURE_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

@dataclass
class PreparedData:
    features: pd.DataFrame
    labels: pd.Series
    original: pd.DataFrame
    scaler: RobustScaler

@dataclass
class ExperimentData:
    """Leakage-safe partitions. Only baseline_features may be used for fitting."""
    baseline_features: pd.DataFrame
    validation_features: pd.DataFrame
    validation_labels: pd.Series
    test_features: pd.DataFrame
    test_labels: pd.Series
    scaler: RobustScaler
    original: pd.DataFrame

def clean_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate records and reject missing values instead of silently imputing finance data."""
    clean = frame.drop_duplicates().copy()
    if clean[FEATURE_COLUMNS + ["Class"]].isna().any().any():
        raise ValueError("Dataset contains missing values in required fields; clean them before training.")
    return clean

def fit_preprocessor(normal_training: pd.DataFrame) -> RobustScaler:
    """Fit only on normal baseline data. RobustScaler resists extreme transaction amounts."""
    scaler = RobustScaler()
    scaler.fit(normal_training[["Time", "Amount"]])
    return scaler

def transform_features(frame: pd.DataFrame, scaler: RobustScaler) -> pd.DataFrame:
    result = frame[FEATURE_COLUMNS].copy()
    result[["Time", "Amount"]] = scaler.transform(result[["Time", "Amount"]])
    return result

def prepare_data(frame: pd.DataFrame) -> PreparedData:
    clean = clean_data(frame)
    normal = clean.loc[clean["Class"] == 0]
    if normal.empty:
        raise ValueError("No normal (Class=0) transactions were found.")
    scaler = fit_preprocessor(normal)
    return PreparedData(transform_features(clean, scaler), clean["Class"].astype(int), clean, scaler)

def prepare_experiment(frame: pd.DataFrame, random_state: int = 42) -> ExperimentData:
    """Create train/validation/test data without fitting on any evaluation row.

    Stratification is used only to make evaluation partitions representative.  The
    detector itself is fit exclusively on normal transactions from the train split.
    """
    clean = clean_data(frame)
    train, remaining = train_test_split(
        clean, test_size=0.40, stratify=clean["Class"], random_state=random_state
    )
    validation, test = train_test_split(
        remaining, test_size=0.50, stratify=remaining["Class"], random_state=random_state
    )
    normal_train = train.loc[train["Class"] == 0]
    if len(normal_train) < 3:
        raise ValueError("At least three normal training transactions are required.")
    scaler = fit_preprocessor(normal_train)
    return ExperimentData(
        baseline_features=transform_features(normal_train, scaler),
        validation_features=transform_features(validation, scaler),
        validation_labels=validation["Class"].astype(int),
        test_features=transform_features(test, scaler),
        test_labels=test["Class"].astype(int),
        scaler=scaler,
        original=clean,
    )
