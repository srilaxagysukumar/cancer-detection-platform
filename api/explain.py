"""SHAP explanations for the diagnostic model.

SHAP (SHapley Additive exPlanations) answers: "how much did each measurement
push this patient's predicted malignancy probability up or down?"

  * a POSITIVE value pushed the prediction towards malignant
  * a NEGATIVE value pushed it towards benign
  * base_value + sum(all values) = the patient's predicted probability

We use SHAP's model-agnostic permutation explainer, which works on any model.
That matters here because our model is an ensemble of three different algorithms,
so explaining the final probability directly is more honest than explaining just
one of its parts. A fresh explainer with a fixed seed is built for every request,
so the same input always gives exactly the same explanation.
"""
import numpy as np
import pandas as pd
import shap


class ShapExplainer:
    def __init__(self, pipeline, background: pd.DataFrame, seed: int = 42, max_evals: int = 300):
        self.feature_names = list(background.columns)
        self.background = background
        self.seed = seed
        self.max_evals = max_evals
        self._pipeline = pipeline

    def _malignant_probability(self, rows) -> np.ndarray:
        frame = pd.DataFrame(rows, columns=self.feature_names)
        return self._pipeline.predict_proba(frame)[:, 1]

    def explain(self, patient: pd.DataFrame) -> tuple[float, np.ndarray]:
        """Return (base_value, one SHAP value per feature) for a single patient."""
        masker = shap.maskers.Independent(self.background, max_samples=len(self.background))
        explainer = shap.Explainer(
            self._malignant_probability, masker, algorithm="permutation", seed=self.seed
        )
        result = explainer(patient, max_evals=self.max_evals, silent=True)
        return float(np.ravel(result.base_values)[0]), np.asarray(result.values[0], dtype=float)
