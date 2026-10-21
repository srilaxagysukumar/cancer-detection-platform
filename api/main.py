"""FastAPI backend: the prediction microservice.

Endpoints
  GET  /health      is the service up and the model loaded?
  GET  /model-info  model description, metrics, feature ranges, example patients
  POST /predict     send 30 measurements, get a diagnosis + SHAP explanation

Interactive documentation is generated automatically at /docs.
"""
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from api.explain import ShapExplainer

MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))
DISCLAIMER = (
    "Educational demonstration built on a public research dataset. "
    "Not a medical device and not for clinical use."
)

# Holds the loaded model so it is read from disk once, at start-up, not on every request.
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = MODEL_DIR / "model.joblib"
    if not model_path.exists():
        raise RuntimeError(
            f"{model_path} not found. Build it first with:  dvc repro   (or: python -m src.train)"
        )
    state["pipeline"] = joblib.load(model_path)
    with open(MODEL_DIR / "model_card.json") as f:
        state["card"] = json.load(f)
    background = pd.read_csv(MODEL_DIR / "background.csv")
    state["explainer"] = ShapExplainer(state["pipeline"], background)
    yield
    state.clear()


app = FastAPI(
    title="Breast Cancer Diagnostic API",
    description=(
        "Predicts whether a breast mass is malignant or benign from 30 cell-nucleus "
        "measurements, using a recall-optimised ensemble model, and explains each "
        "prediction with SHAP. " + DISCLAIMER
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------- request / response shapes (validated automatically by Pydantic) ----------
Measurement = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class PredictRequest(BaseModel):
    features: dict[str, Measurement] = Field(
        description="All 30 measurements, keyed by feature name (see /model-info)."
    )
    explain: bool = Field(True, description="Include the SHAP explanation.")
    top_k: int = Field(10, ge=1, le=30, description="How many features to list in the explanation.")


class Contribution(BaseModel):
    feature: str
    value: float
    shap_value: float


class Explanation(BaseModel):
    base_value: float
    contributions: list[Contribution]


class PredictResponse(BaseModel):
    prediction: str
    malignant_probability: float
    decision_threshold: float
    warnings: list[str]
    explanation: Explanation | None
    disclaimer: str


# ---------- endpoints ----------
@app.get("/")
def root():
    return {"service": "Breast Cancer Diagnostic API", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "pipeline" in state}


@app.get("/model-info")
def model_info():
    return state["card"]


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    card = state["card"]
    names: list[str] = card["feature_names"]

    missing = [n for n in names if n not in request.features]
    unexpected = [n for n in request.features if n not in names]
    if missing or unexpected:
        raise HTTPException(
            status_code=422,
            detail={"message": "Features must match the model exactly.", "missing": missing, "unexpected": unexpected},
        )

    patient = pd.DataFrame([request.features])[names]
    probability = float(state["pipeline"].predict_proba(patient)[0, 1])
    threshold = card["threshold"]

    warnings = []
    for name in names:
        stats = card["feature_stats"][name]
        value = request.features[name]
        if value < stats["min"] or value > stats["max"]:
            warnings.append(
                f"{name}={value:g} is outside the range seen in the training data "
                f"({stats['min']:g} to {stats['max']:g}); treat this prediction with caution."
            )

    explanation = None
    if request.explain:
        base_value, shap_values = state["explainer"].explain(patient)
        ranked = sorted(zip(names, patient.iloc[0], shap_values), key=lambda row: abs(row[2]), reverse=True)
        explanation = Explanation(
            base_value=base_value,
            contributions=[
                Contribution(feature=n, value=float(v), shap_value=float(s)) for n, v, s in ranked[: request.top_k]
            ],
        )

    return PredictResponse(
        prediction="Malignant" if probability >= threshold else "Benign",
        malignant_probability=round(probability, 6),
        decision_threshold=threshold,
        warnings=warnings,
        explanation=explanation,
        disclaimer=DISCLAIMER,
    )
