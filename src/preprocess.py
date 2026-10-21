"""Stage 2 - preprocess.

Validates the raw data and splits it into a training set and a test set.

The split is *stratified*: both sets contain the same proportion of malignant
cases. The test set is locked away until the final evaluation, so the reported
results are an honest estimate of how the model behaves on unseen patients.

(Feature scaling is deliberately NOT done here. It is part of the model pipeline
in train.py, so it is learned from training data only and applied identically at
prediction time. Doing it here would leak information from the test set.)
"""
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

RAW_PATH = Path("data/raw/breast_cancer.csv")
PROCESSED_DIR = Path("data/processed")
TARGET = "malignant"


def validate(df: pd.DataFrame) -> None:
    """Stop early, with a clear message, if the data is not what we expect."""
    assert df.shape[1] == 31, f"Expected 30 features + 1 label, got {df.shape[1]} columns"
    assert not df.isna().any().any(), "The data contains missing values"
    assert set(df[TARGET].unique()) == {0, 1}, "The label must contain only 0 and 1"
    features = df.drop(columns=[TARGET])
    assert (features >= 0).all().all(), "Found negative measurements"


def main() -> None:
    with open("params.yaml") as f:
        params = yaml.safe_load(f)["data"]

    df = pd.read_csv(RAW_PATH)
    validate(df)

    train_df, test_df = train_test_split(
        df,
        test_size=params["test_size"],
        random_state=params["random_state"],
        stratify=df[TARGET],
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)
    print(
        f"Train: {len(train_df)} patients ({int(train_df[TARGET].sum())} malignant) | "
        f"Test: {len(test_df)} patients ({int(test_df[TARGET].sum())} malignant)"
    )


if __name__ == "__main__":
    main()
