# Breast Cancer Diagnostic Platform

An end-to-end machine-learning system that classifies breast masses as **malignant or benign** from 30 cell-nucleus measurements, explains every prediction with **SHAP**, and serves it through a **FastAPI** backend and a **Streamlit** dashboard, both containerised with **Docker**.

> **Educational demonstration only.** Built on a public research dataset. It is not a medical device and must not be used for health decisions.

**Live demo:** _add your GitHub Pages link here_ &nbsp;(runs in your browser, always on, nothing to install)

> The live demo runs the trained model **inside your web browser**, so there is no server to wake up and nothing you type leaves your device. The full-stack version (FastAPI + Streamlit in Docker) is in this repository and can be run with `docker compose up`.

---

## What it does

| Component | What it is |
|---|---|
| **Reproducible pipeline** | A three-stage [DVC](https://dvc.org) pipeline (ingest → preprocess → train). Every data file and model artifact is fingerprinted in `dvc.lock`, so anyone can rebuild identical results with `dvc repro`. |
| **Recall-first model** | A soft-voting ensemble (logistic regression + random forest + gradient boosting) tuned for **recall**, with a decision threshold chosen to protect it. |
| **Explainability** | SHAP values show which measurements pushed each prediction towards malignant or benign. They add up exactly to the predicted probability. |
| **Backend** | FastAPI microservice with input validation, out-of-range warnings and auto-generated docs at `/docs`. |
| **Frontend** | Streamlit dashboard with no ML code in it. It only talks to the API over HTTP. |
| **Browser demo** | `docs/` holds a static copy of the dashboard. The trained model is exported to `docs/model.json` and `docs/engine.js` repeats its arithmetic (including SHAP) in JavaScript. Hosted free on GitHub Pages. Tests prove it matches the Python model. |
| **Containers** | One Dockerfile per service, plus `docker-compose.yml` to run both locally. |
| **Quality checks** | Automated tests (`pytest`) run on every push by GitHub Actions. |

## Architecture

```
  Browser ──►  Streamlit dashboard  ──HTTP/JSON──►  FastAPI backend  ──►  model.joblib + SHAP
              (container 1, ui/)                    (container 2, api/)

  DVC pipeline:  ingest ──► preprocess ──► train ──► models/ + metrics/
                                                  │
                                                  └─► src/export_web.py ──► docs/model.json
                                                      (static browser demo: docs/index.html + engine.js,
                                                       served free by GitHub Pages, no backend)
```

## Results

Evaluated on a held-out test set of 114 patients (42 malignant) that was never used for training or tuning. "Positive" means malignant.

| Metric | At default threshold (0.50) | At tuned threshold (0.23) |
|---|---|---|
| **Recall** (malignant cases caught) | 90.5% (4 missed) | **97.6%** (1 missed) |
| Precision | 100% | 97.6% |
| Accuracy | 96.5% | 98.2% |
| ROC-AUC | 0.998 | 0.998 |

Cross-validated recall on the training data at the tuned threshold: **98.2%**.

**Why recall?** A missed malignant tumour (false negative) is far more costly than a false alarm, so the model is tuned for recall rather than raw accuracy. Lowering the decision threshold caught 3 more malignant cases in the test set, at the cost of one extra false alarm.

**Honest caveats.** The test set holds only 42 malignant cases, so one case is about 2.4 percentage points and the figures carry real uncertainty. The dataset is small and clean, which is why scores are high. This is a learning project, not a validated clinical tool.

## Run it yourself

You need Python 3.13, Git and (optionally) Docker.

```bash
# 1. Set up
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt

# 2. Rebuild the data and the model (about 2 minutes)
dvc repro

# 3. Check everything works
pytest

# 4. Start the backend (terminal 1) ...
uvicorn api.main:app --port 8000
# ... and the dashboard (terminal 2, after activating the environment)
streamlit run ui/app.py
```

Or run both services in Docker: `docker compose up --build`, then open http://localhost:8501.

**Browser demo locally:** `python3 -m http.server --directory docs 8000`, then open http://localhost:8000.
If you retrain the model, refresh the browser copy with `python -m src.export_web` and commit `docs/model.json`.

## Project structure

```
src/            pipeline stages: ingest.py, preprocess.py, train.py (+ export_web.py)
api/            FastAPI backend: main.py, explain.py (SHAP)
ui/             Streamlit dashboard: app.py
docs/           static browser demo: index.html, engine.js (model + SHAP in JavaScript), model.json
tests/          pytest checks for the data, the model, the API and the browser version
dvc.yaml        the pipeline definition
params.yaml     all tunable settings
dvc.lock        fingerprints of every pipeline output (reproducibility record)
Dockerfile.*    one container per service
docker-compose.yml, render.yaml   local and cloud deployment
```

## Design decisions

- **Scaling lives inside the model pipeline**, not in preprocessing, so it is learned from training data only. This prevents test-set information leaking into training.
- **The label is flipped so 1 = malignant**, which makes "recall" mean "share of cancers caught" with no ambiguity.
- **The threshold is chosen on out-of-fold predictions from the training set**, never on the test set.
- **The API validates strictly**: all 30 named features must be present, non-negative and finite, and values outside the training range produce a warning.
- **The live demo is static on purpose.** Hosting a Python server 24/7 costs money and free servers sleep, so the model is exported as plain numbers and evaluated in the browser. `tests/test_web_parity.py` runs the real JavaScript under Node and checks it against scikit-learn (probabilities agree to 9 decimal places, SHAP values add up exactly).
- **The Docker image rebuilds the model with the same scripts DVC runs**, so deployed behaviour always matches the repository.
