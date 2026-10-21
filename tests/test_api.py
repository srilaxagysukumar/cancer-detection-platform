"""Automated checks for the API. Run with:  pytest

They need the trained model, so run `dvc repro` first (the CI workflow does this for you).
"""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app

CARD = json.loads(Path("models/model_card.json").read_text())
EXAMPLES = {e["title"]: e for e in CARD["examples"]}


@pytest.fixture(scope="module")
def client():
    # Using TestClient as a context manager runs the app's start-up code (loads the model).
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_model_info_lists_30_features(client):
    info = client.get("/model-info").json()
    assert len(info["feature_names"]) == 30
    assert len(info["examples"]) == 3


def test_malignant_example_is_flagged(client):
    body = {"features": EXAMPLES["Typical malignant"]["features"]}
    result = client.post("/predict", json=body).json()
    assert result["prediction"] == "Malignant"
    assert result["malignant_probability"] >= result["decision_threshold"]


def test_benign_example_is_cleared(client):
    body = {"features": EXAMPLES["Typical benign"]["features"]}
    result = client.post("/predict", json=body).json()
    assert result["prediction"] == "Benign"
    assert result["malignant_probability"] < result["decision_threshold"]


def test_shap_values_add_up_to_the_prediction(client):
    body = {"features": EXAMPLES["Borderline case"]["features"], "top_k": 30}
    result = client.post("/predict", json=body).json()
    explanation = result["explanation"]
    total = explanation["base_value"] + sum(c["shap_value"] for c in explanation["contributions"])
    assert total == pytest.approx(result["malignant_probability"], abs=1e-3)


def test_explanation_can_be_switched_off(client):
    body = {"features": EXAMPLES["Typical benign"]["features"], "explain": False}
    assert client.post("/predict", json=body).json()["explanation"] is None


def test_missing_feature_is_rejected(client):
    features = dict(EXAMPLES["Typical benign"]["features"])
    features.pop("mean_radius")
    response = client.post("/predict", json={"features": features})
    assert response.status_code == 422


def test_unknown_feature_is_rejected(client):
    features = dict(EXAMPLES["Typical benign"]["features"], not_a_feature=1.0)
    response = client.post("/predict", json={"features": features})
    assert response.status_code == 422


def test_negative_value_is_rejected(client):
    features = dict(EXAMPLES["Typical benign"]["features"], mean_radius=-1.0)
    response = client.post("/predict", json={"features": features})
    assert response.status_code == 422


def test_out_of_range_value_gives_a_warning(client):
    features = dict(EXAMPLES["Typical benign"]["features"], mean_radius=500.0)
    result = client.post("/predict", json={"features": features}).json()
    assert any("mean_radius" in w for w in result["warnings"])
