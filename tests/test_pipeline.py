"""Automated checks for the data and the model. Run with:  pytest"""
import json
from pathlib import Path

import pandas as pd


def test_no_patient_appears_in_both_train_and_test():
    train = pd.read_csv("data/processed/train.csv")
    test = pd.read_csv("data/processed/test.csv")
    feature_columns = [c for c in train.columns if c != "malignant"]
    overlap = train.merge(test, on=feature_columns, how="inner")
    assert overlap.empty, "Data leakage: some patients are in both sets"


def test_split_keeps_the_class_balance():
    train = pd.read_csv("data/processed/train.csv")
    test = pd.read_csv("data/processed/test.csv")
    assert abs(train["malignant"].mean() - test["malignant"].mean()) < 0.02


def test_the_model_puts_recall_first():
    metrics = json.loads(Path("metrics/metrics.json").read_text())
    # Lowering the threshold below 0.5 must not have made the model *worse* at catching cancer.
    assert metrics["test"]["recall"] >= metrics["test_at_default_threshold_0.5"]["recall"]
    # The cross-validated recall at the chosen threshold meets the target set in params.yaml.
    assert metrics["cv_recall_at_threshold"] >= metrics["target_recall"]
    # And the held-out test recall stays high.
    assert metrics["test"]["recall"] >= 0.95
