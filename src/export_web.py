"""Export the trained model to a JSON file that the browser version can read.

The live demo (docs/index.html) runs entirely inside the visitor's web browser, so there
is no server to pay for or keep awake. For that to work, the model must be written down
as plain numbers. This script does that translation:

  * the scaler          -> a mean and a scale for each of the 30 measurements
  * logistic regression -> 30 weights and an intercept
  * random forest       -> every decision tree (which feature to test, where to split)
  * gradient boosting   -> every decision tree plus its learning rate and starting value

docs/engine.js then repeats the same arithmetic scikit-learn does. The tests in
tests/test_web_parity.py check that both give the same probabilities.

Run it with:  python -m src.export_web
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODEL_DIR = Path("models")
OUT_FILE = Path("docs/model.json")


def tree_to_dict(tree, leaf_value) -> dict:
    """Flatten one scikit-learn tree into parallel lists (the compact form JavaScript walks)."""
    t = tree.tree_
    return {
        "left": t.children_left.tolist(),
        "right": t.children_right.tolist(),
        "feature": t.feature.tolist(),
        "threshold": t.threshold.tolist(),
        "value": [float(leaf_value(t.value[i])) for i in range(t.node_count)],
    }


def build_export(pipeline, card: dict, background: pd.DataFrame) -> dict:
    names = card["feature_names"]
    scaler = pipeline.named_steps["scaler"]
    ensemble = pipeline.named_steps["ensemble"]
    parts = dict(ensemble.named_estimators_)
    lr, rf, gb = parts["lr"], parts["rf"], parts["gb"]

    assert ensemble.voting == "soft" and ensemble.weights is None, "expected an unweighted soft vote"
    assert list(lr.classes_) == [0, 1], "expected classes 0 (benign) and 1 (malignant)"

    # Random forest: each tree's leaf holds class counts; the forest averages the
    # per-tree probability of class 1, so we store that probability for every node.
    def rf_leaf(value):
        counts = value[0]
        return counts[1] / counts.sum()

    # Gradient boosting: start from the log-odds of the training prior, then add
    # learning_rate * (each tree's leaf value). A sigmoid turns the total into a probability.
    prior = float(gb.init_.class_prior_[1])
    gb_start = float(np.log(prior / (1 - prior)))

    return {
        "model": card["model"],
        "feature_names": names,
        "threshold": card["threshold"],
        "feature_stats": card["feature_stats"],
        "examples": card["examples"],
        "metrics": card["metrics"],
        "scaler": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()},
        "lr": {"coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0])},
        "rf": {"trees": [tree_to_dict(est, rf_leaf) for est in rf.estimators_]},
        "gb": {
            "start": gb_start,
            "learning_rate": float(gb.learning_rate),
            "trees": [tree_to_dict(est[0], lambda v: v[0][0]) for est in gb.estimators_],
        },
        "background": background[names].values.tolist(),
    }


def main(out_file: Path = OUT_FILE) -> None:
    pipeline = joblib.load(MODEL_DIR / "model.joblib")
    card = json.loads((MODEL_DIR / "model_card.json").read_text())
    background = pd.read_csv(MODEL_DIR / "background.csv")
    export = build_export(pipeline, card, background)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(export, separators=(",", ":")))
    print(f"Wrote {out_file} ({out_file.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else OUT_FILE)
