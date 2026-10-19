"""Dataset discovery and validation."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "creditcard.csv"
REQUIRED_COLUMNS = {"Time", "Amount", "Class"} | {f"V{i}" for i in range(1, 29)}

def load_creditcard_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load the Kaggle credit-card dataset and validate its expected schema."""
    dataset_path = Path(path) if path else DEFAULT_DATASET
    if not dataset_path.exists():
        raise FileNotFoundError(
            "Dataset not found. Please download the Kaggle 'Credit Card Fraud Detection' "
            f"CSV and place it at: {dataset_path}"
        )
    frame = pd.read_csv(dataset_path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset format is invalid. Required columns are missing: {sorted(missing)}")
    return frame

def dataset_summary(frame: pd.DataFrame) -> dict:
    return {"shape": frame.shape, "columns": list(frame.columns), "dtypes": frame.dtypes.astype(str).to_dict(),
            "missing_values": frame.isna().sum().to_dict(), "duplicate_rows": int(frame.duplicated().sum()),
            "class_distribution": frame["Class"].value_counts().sort_index().to_dict()}
