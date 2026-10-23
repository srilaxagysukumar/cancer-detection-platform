"""Checks that the browser version (docs/engine.js) gives the same answers as the Python model.

The live demo runs in the visitor's browser, so these tests run the real JavaScript with Node.js
(pre-installed on GitHub's test machines) and compare it with scikit-learn. They are skipped
if Node.js is not installed on your computer.
"""
import json
import shutil
import subprocess
from pathlib import Path

import joblib
import pandas as pd
import pytest

from src.export_web import build_export

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="Node.js is not installed")

CARD = json.loads(Path("models/model_card.json").read_text())
NAMES = CARD["feature_names"]
RUNNER = """
const E = require(process.argv[1]);
const model = require(process.argv[2]);
const patients = JSON.parse(require("fs").readFileSync(0, "utf8"));
const out = patients.map((x) => {
  const e = E.explain(model, x, { permutations: 10 });
  return { p: E.predictProba(model, x), base: e.base_value, shap: e.shap_values, final: e.probability };
});
console.log(JSON.stringify(out));
"""


@pytest.fixture(scope="module")
def js_results(tmp_path_factory):
    pipeline = joblib.load("models/model.joblib")
    background = pd.read_csv("models/background.csv")
    folder = tmp_path_factory.mktemp("web")
    model_file = folder / "model.json"
    model_file.write_text(json.dumps(build_export(pipeline, CARD, background)))

    test = pd.read_csv("data/processed/test.csv")[NAMES]
    patients = test.iloc[:40]
    run = subprocess.run(
        [NODE, "-e", RUNNER, str(Path("docs/engine.js").resolve()), str(model_file)],
        input=json.dumps(patients.values.tolist()), capture_output=True, text=True, check=True,
    )
    return pipeline, patients, json.loads(run.stdout)


def test_js_probabilities_match_python(js_results):
    pipeline, patients, results = js_results
    expected = pipeline.predict_proba(patients)[:, 1]
    for want, got in zip(expected, results):
        assert got["p"] == pytest.approx(want, abs=1e-9)


def test_js_decisions_match_python(js_results):
    pipeline, patients, results = js_results
    threshold = CARD["threshold"]
    expected = pipeline.predict_proba(patients)[:, 1] >= threshold
    assert [r["p"] >= threshold for r in results] == list(expected)


def test_js_shap_values_add_up_to_the_prediction(js_results):
    _, _, results = js_results
    for r in results:
        assert r["base"] + sum(r["shap"]) == pytest.approx(r["final"], abs=1e-9)


def test_published_model_file_is_in_step_with_the_trained_model():
    """docs/model.json is committed to Git; it must describe the model we just trained."""
    published = json.loads(Path("docs/model.json").read_text())
    pipeline = joblib.load("models/model.joblib")
    test = pd.read_csv("data/processed/test.csv")[NAMES]
    assert published["feature_names"] == NAMES
    assert published["threshold"] == CARD["threshold"]
    run = subprocess.run(
        [NODE, "-e", "const E=require(process.argv[1]);const m=require(process.argv[2]);"
                     "const p=JSON.parse(require('fs').readFileSync(0,'utf8'));"
                     "console.log(JSON.stringify(p.map(x=>E.predictProba(m,x))))",
         str(Path("docs/engine.js").resolve()), str(Path("docs/model.json").resolve())],
        input=json.dumps(test.values.tolist()), capture_output=True, text=True, check=True,
    )
    got = json.loads(run.stdout)
    want = pipeline.predict_proba(test)[:, 1]
    assert max(abs(g - w) for g, w in zip(got, want)) < 1e-3, "Run  python -m src.export_web  and commit docs/model.json"
