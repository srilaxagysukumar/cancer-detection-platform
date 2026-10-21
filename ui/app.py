"""Streamlit dashboard: the front end.

This app holds NO machine-learning code. It only collects measurements, sends them
to the FastAPI backend over HTTP, and draws what comes back. Front end and back end
are separate programs (and separate Docker containers) that talk through the API.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = 90  # seconds. Free hosting can take ~1 minute to wake a sleeping service.

st.set_page_config(page_title="Breast Cancer Diagnostic Platform", page_icon="🔬", layout="wide")

# Plain-English definitions of the ten measurements (from the dataset documentation).
MEASUREMENTS = {
    "radius": "Mean distance from the centre of the nucleus to points on its edge.",
    "texture": "Variation in grey-scale values across the nucleus image.",
    "perimeter": "Length of the nucleus outline.",
    "area": "Area of the nucleus.",
    "smoothness": "Local variation in radius lengths.",
    "compactness": "Perimeter squared divided by area, minus 1.",
    "concavity": "Severity of concave portions of the outline.",
    "concave_points": "Number of concave portions of the outline.",
    "symmetry": "How symmetrical the nucleus is.",
    "fractal_dimension": "'Coastline approximation' - how irregular the outline is.",
}
GROUPS = {
    "mean": ("Mean values", "Average over all cells in the sample."),
    "error": ("Standard error", "How much each measurement varied between cells."),
    "worst": ("Worst values", "Mean of the three largest values in the sample."),
}


def group_of(feature: str) -> str:
    if feature.startswith("mean_"):
        return "mean"
    if feature.startswith("worst_"):
        return "worst"
    return "error"


def base_of(feature: str) -> str:
    if feature.startswith("mean_"):
        return feature[len("mean_"):]
    if feature.startswith("worst_"):
        return feature[len("worst_"):]
    return feature[: -len("_error")]


def pretty(feature: str) -> str:
    return feature.replace("_", " ").capitalize()


@st.cache_data(ttl=600, show_spinner=False)
def fetch_model_info() -> dict:
    response = requests.get(f"{API_URL}/model-info", timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


def key(feature: str) -> str:
    return f"input_{feature}"


def apply_example() -> None:
    """Runs when the example menu changes: fill all 30 boxes."""
    choice = st.session_state["example_choice"]
    for name in info["feature_names"]:
        if choice == "Average patient (median values)":
            st.session_state[key(name)] = info["feature_stats"][name]["median"]
        else:
            example = next(e for e in info["examples"] if e["title"] == choice)
            st.session_state[key(name)] = example["features"][name]
    st.session_state.pop("result", None)


# ---------------------------------------------------------------- header
st.title("🔬 Breast Cancer Diagnostic Platform")
st.warning(
    "**Educational demonstration only.** Built on a public research dataset (Wisconsin Diagnostic "
    "Breast Cancer). It is not a medical device and must never be used to make health decisions."
)

try:
    with st.spinner("Connecting to the prediction service (it can take up to a minute to wake up)..."):
        info = fetch_model_info()
except requests.exceptions.RequestException as error:
    st.error(
        f"Cannot reach the prediction service at {API_URL}. Make sure the API is running, "
        f"then refresh this page.\n\nDetails: {error}"
    )
    st.stop()

# ---------------------------------------------------------------- initial values
for name in info["feature_names"]:
    st.session_state.setdefault(key(name), info["feature_stats"][name]["median"])

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("1. Choose a patient")
    st.selectbox(
        "Load an example",
        ["Average patient (median values)"] + [e["title"] for e in info["examples"]],
        key="example_choice",
        on_change=apply_example,
        help="The examples are real patients from a test set the model never saw during training.",
    )
    st.caption("Then edit any measurement on the right, or just press the button below.")
    run = st.button("2. Run diagnostic", type="primary", width="stretch")

    st.divider()
    st.header("About the model")
    st.write(info["model"])
    metrics = info["metrics"]
    test = metrics["test"]
    st.metric("Recall on unseen patients", f"{test['recall']:.1%}",
              help="Share of truly malignant cases the model catches. The number we optimise.")
    st.caption(
        f"Test set: {metrics['n_test']} patients, {metrics['n_malignant_test']} malignant. "
        f"Missed malignant cases: {test['false_negatives']}. "
        f"Cross-validated recall: {metrics['cv_recall_at_threshold']:.1%}."
    )
    st.caption(
        f"Precision {test['precision']:.1%} · Accuracy {test['accuracy']:.1%} · "
        f"ROC-AUC {test['roc_auc']:.3f} · Decision threshold {metrics['decision_threshold']:.2f}"
    )
    st.caption(f"API: {API_URL}/docs")

# ---------------------------------------------------------------- measurement inputs
st.subheader("Nucleus measurements")
st.caption("Hover over the (?) icons for a plain-English definition. All 30 values are sent to the model.")

tabs = st.tabs([GROUPS[g][0] for g in ("mean", "error", "worst")])
for tab, group in zip(tabs, ("mean", "error", "worst")):
    with tab:
        st.caption(GROUPS[group][1])
        names = [n for n in info["feature_names"] if group_of(n) == group]
        columns = st.columns(5)
        for i, name in enumerate(names):
            stats = info["feature_stats"][name]
            columns[i % 5].number_input(
                pretty(base_of(name)),
                min_value=0.0,
                max_value=float(stats["max"]) * 2,
                step=float(stats["max"]) / 200,
                format="%.5f",
                key=key(name),
                help=MEASUREMENTS[base_of(name)] + f" Typical range: {stats['min']:.4g} to {stats['max']:.4g}.",
            )

# ---------------------------------------------------------------- run the model
if run:
    payload = {
        "features": {name: float(st.session_state[key(name)]) for name in info["feature_names"]},
        "explain": True,
        "top_k": 10,
    }
    try:
        with st.spinner("Asking the model and computing the SHAP explanation..."):
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=TIMEOUT)
        if response.status_code == 200:
            st.session_state["result"] = response.json()
        else:
            st.session_state.pop("result", None)
            st.error(f"The API rejected the request (status {response.status_code}): {response.text}")
    except requests.exceptions.RequestException as error:
        st.session_state.pop("result", None)
        st.error(f"Could not reach the prediction service: {error}")

# ---------------------------------------------------------------- results
result = st.session_state.get("result")
if result:
    st.divider()
    st.subheader("Result")
    left, right = st.columns([1, 2])

    with left:
        probability = result["malignant_probability"]
        if result["prediction"] == "Malignant":
            st.error("### 🚨 Malignant")
        else:
            st.success("### ✅ Benign")
        st.metric("Probability of malignancy", f"{probability:.1%}")
        st.progress(min(max(probability, 0.0), 1.0))
        st.caption(
            f"The model flags a case as malignant when this probability reaches "
            f"{result['decision_threshold']:.0%}. The threshold is set low on purpose, so that "
            f"almost no malignant case is missed (a false negative is the costliest mistake)."
        )
        for warning in result["warnings"]:
            st.warning(warning)

    with right:
        explanation = result["explanation"]
        st.markdown("**Why did the model say this?** (SHAP feature contributions)")
        contributions = explanation["contributions"][::-1]  # largest at the top of the chart
        labels = [f"{pretty(c['feature'])} = {c['value']:.4g}" for c in contributions]
        values = [c["shap_value"] * 100 for c in contributions]
        colours = ["#c0392b" if v > 0 else "#2471a3" for v in values]

        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.barh(labels, values, color=colours)
        ax.axvline(0, color="#555555", linewidth=0.8)
        ax.set_xlabel("Effect on malignancy probability (percentage points)")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "🟥 Red bars pushed the prediction towards **malignant**; 🟦 blue bars pushed it towards "
            f"**benign**. Starting from the average prediction of {explanation['base_value']:.1%}, "
            "the contributions of all 30 features add up to this patient's probability. "
            "Only the 10 largest are shown."
        )
        with st.expander("See the numbers"):
            table = pd.DataFrame(explanation["contributions"])
            table["shap_value"] = (table["shap_value"] * 100).round(2)
            table = table.rename(columns={"shap_value": "effect (percentage points)"})
            st.dataframe(table, hide_index=True, width="stretch")
