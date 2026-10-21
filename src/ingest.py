"""Stage 1 - ingest.

Loads the Wisconsin Diagnostic Breast Cancer dataset that ships with
scikit-learn and saves it as a CSV file.

Two small clean-ups are made on the way:
  * feature names get underscores instead of spaces ("mean radius" -> "mean_radius")
  * the label is flipped so that 1 = malignant (the condition we must not miss).
    scikit-learn's original coding is 0 = malignant, 1 = benign.
"""
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer

RAW_PATH = Path("data/raw/breast_cancer.csv")


def main() -> None:
    bunch = load_breast_cancer(as_frame=True)

    # Safety check: confirm the coding we are about to flip is the one we expect.
    assert list(bunch.target_names) == ["malignant", "benign"], bunch.target_names

    df = bunch.data.copy()
    df.columns = [name.replace(" ", "_") for name in df.columns]
    df["malignant"] = 1 - bunch.target  # 1 = malignant, 0 = benign

    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_PATH, index=False)
    print(f"Saved {len(df)} patients ({int(df['malignant'].sum())} malignant) to {RAW_PATH}")


if __name__ == "__main__":
    main()
