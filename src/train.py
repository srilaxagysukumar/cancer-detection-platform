"""Stage 3 - train.

Builds, tunes and evaluates the diagnostic model, putting RECALL first.

Why recall? In a screening setting the worst mistake is a false negative: telling
someone with a malignant tumour that it is benign. Recall (also called
sensitivity) is the share of truly malignant cases the model catches, so every
decision below is made to protect it:

  1. Model      - a soft-voting ENSEMBLE of three different algorithms
                  (logistic regression, random forest, gradient boosting),
                  preceded by feature scaling, all inside one scikit-learn Pipeline.
  2. Tuning     - RandomizedSearchCV picks hyper-parameters by cross-validated
                  RECALL (ROC-AUC only breaks ties), not by accuracy.
  3. Threshold  - the probability cut-off is lowered until cross-validated recall
                  reaches the target in params.yaml, keeping as much precision as possible.
  4. Evaluation - everything is reported on the untouched test set.

Outputs
  models/model.joblib       the trained pipeline
  models/model_card.json    threshold, feature ranges, example patients, metrics
  models/background.csv     small reference set used by SHAP explanations
  metrics/metrics.json      headline numbers (tracked by DVC and Git)
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from scipy.stats import loguniform
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TRAIN_PATH = Path("data/processed/train.csv")
TEST_PATH = Path("data/processed/test.csv")
MODEL_DIR = Path("models")
METRICS_DIR = Path("metrics")
TARGET = "malignant"


def build_pipeline(seed: int) -> Pipeline:
    ensemble = VotingClassifier(
        estimators=[
            ("lr", LogisticRegression(max_iter=5000, random_state=seed)),
            ("rf", RandomForestClassifier(random_state=seed, n_jobs=1)),
            ("gb", GradientBoostingClassifier(random_state=seed)),
        ],
        voting="soft",  # average the predicted probabilities of the three models
    )
    return Pipeline([("scaler", StandardScaler()), ("ensemble", ensemble)])


SEARCH_SPACE = {
    "ensemble__lr__C": loguniform(1e-2, 1e2),
    "ensemble__lr__class_weight": [None, "balanced"],
    "ensemble__rf__n_estimators": [100, 200, 300],
    "ensemble__rf__max_depth": [None, 4, 6, 8],
    "ensemble__rf__min_samples_leaf": [1, 2, 4],
    "ensemble__rf__class_weight": [None, "balanced"],
    "ensemble__gb__n_estimators": [100, 200, 300],
    "ensemble__gb__learning_rate": [0.03, 0.05, 0.1],
    "ensemble__gb__max_depth": [2, 3, 4],
}


def pick_best_by_recall(cv_results: dict) -> int:
    """Highest mean cross-validated recall wins; ROC-AUC breaks ties."""
    recall = np.asarray(cv_results["mean_test_recall"])
    auc = np.asarray(cv_results["mean_test_roc_auc"])
    return int(np.lexsort((-auc, -recall))[0])


def choose_threshold(y_true, probabilities, target_recall: float):
    """Highest-precision threshold whose recall still meets the target."""
    rows = []
    for t in np.round(np.arange(0.05, 0.951, 0.01), 2):
        predicted = probabilities >= t
        rows.append(
            {
                "threshold": float(t),
                "recall": float(recall_score(y_true, predicted)),
                "precision": float(precision_score(y_true, predicted, zero_division=0)),
            }
        )
    meets_target = [r for r in rows if r["recall"] >= target_recall]
    if meets_target:
        return max(meets_target, key=lambda r: (r["precision"], r["threshold"])), True
    return max(rows, key=lambda r: (r["recall"], r["precision"])), False


def evaluate(y_true, probabilities, threshold: float) -> dict:
    predicted = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted).ravel()
    return {
        "recall": float(recall_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "specificity": float(tn / (tn + fp)),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "f1": float(f1_score(y_true, predicted)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "true_positives": int(tp),
        "false_negatives": int(fn),
        "false_positives": int(fp),
        "true_negatives": int(tn),
    }


def pick_examples(X_test, y_test, probabilities, threshold):
    """Three real held-out patients for the dashboard's 'load example' menu."""

    def record(index, title, description):
        return {
            "id": title.lower().replace(" ", "_"),
            "title": title,
            "description": description,
            "true_label": "Malignant" if int(y_test.iloc[index]) == 1 else "Benign",
            "features": {k: float(v) for k, v in X_test.iloc[index].items()},
        }

    def typical(label):
        idx = np.where((y_test.values == label) & ((probabilities >= threshold) == bool(label)))[0]
        median_p = np.median(probabilities[idx])
        return int(idx[np.argmin(np.abs(probabilities[idx] - median_p))])

    borderline = int(np.argmin(np.abs(probabilities - threshold)))
    return [
        record(typical(0), "Typical benign", "A real benign case from the held-out test set."),
        record(typical(1), "Typical malignant", "A real malignant case from the held-out test set."),
        record(borderline, "Borderline case", "The test-set patient the model finds hardest to call."),
    ]


def main() -> None:
    with open("params.yaml") as f:
        params = yaml.safe_load(f)["train"]
    seed = params["random_state"]

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    X_train, y_train = train_df.drop(columns=[TARGET]), train_df[TARGET]
    X_test, y_test = test_df.drop(columns=[TARGET]), test_df[TARGET]
    cv = StratifiedKFold(n_splits=params["cv_folds"], shuffle=True, random_state=seed)

    # 1 + 2. Tune the ensemble for recall
    print("Tuning the ensemble for recall (this takes a minute or two)...")
    search = RandomizedSearchCV(
        build_pipeline(seed),
        SEARCH_SPACE,
        n_iter=params["n_iter"],
        scoring={"recall": "recall", "roc_auc": "roc_auc"},
        refit=pick_best_by_recall,
        cv=cv,
        random_state=seed,
        n_jobs=1,
    )
    search.fit(X_train, y_train)
    model = search.best_estimator_
    print(f"Best cross-validated recall: {search.cv_results_['mean_test_recall'][search.best_index_]:.4f}")

    # 3. Choose the decision threshold from out-of-fold predictions (training data only)
    oof = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    chosen, reached_target = choose_threshold(y_train, oof, params["target_recall"])
    threshold = chosen["threshold"]
    print(
        f"Decision threshold: {threshold:.2f} "
        f"(cross-validated recall {chosen['recall']:.4f}, precision {chosen['precision']:.4f}, "
        f"target {'reached' if reached_target else 'NOT reached'})"
    )

    # 4. Final evaluation on the untouched test set
    test_prob = model.predict_proba(X_test)[:, 1]
    at_threshold = evaluate(y_test, test_prob, threshold)
    at_default = evaluate(y_test, test_prob, 0.5)

    metrics = {
        "positive_class": "malignant",
        "decision_threshold": threshold,
        "target_recall": params["target_recall"],
        "target_reached_in_cv": reached_target,
        "cv_recall_at_threshold": chosen["recall"],
        "cv_precision_at_threshold": chosen["precision"],
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_malignant_test": int(y_test.sum()),
        "test": at_threshold,
        "test_at_default_threshold_0.5": at_default,
    }

    # Save everything the API and dashboard need
    MODEL_DIR.mkdir(exist_ok=True)
    METRICS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "model.joblib")

    background = X_train.sample(n=params["background_size"], random_state=seed)
    background.to_csv(MODEL_DIR / "background.csv", index=False)

    all_features = pd.concat([X_train, X_test])
    model_card = {
        "model": "Soft-voting ensemble (logistic regression + random forest + gradient boosting)",
        "positive_class": "malignant",
        "threshold": threshold,
        "feature_names": list(X_train.columns),
        "feature_stats": {
            name: {
                "min": float(all_features[name].min()),
                "max": float(all_features[name].max()),
                "median": float(all_features[name].median()),
            }
            for name in X_train.columns
        },
        "examples": pick_examples(X_test, y_test, test_prob, threshold),
        "metrics": metrics,
        "best_params": {
            k.replace("ensemble__", ""): (v.item() if hasattr(v, "item") else v)
            for k, v in search.best_params_.items()
        },
        "scikit_learn_version": sklearn.__version__,
    }
    with open(MODEL_DIR / "model_card.json", "w") as f:
        json.dump(model_card, f, indent=2)
    with open(METRICS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    t = at_threshold
    print(
        f"TEST SET  recall {t['recall']:.4f} | precision {t['precision']:.4f} | "
        f"accuracy {t['accuracy']:.4f} | missed malignant cases (false negatives): {t['false_negatives']}"
    )


if __name__ == "__main__":
    main()
