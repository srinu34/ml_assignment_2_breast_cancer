from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from model_utils import (
    BASE_DIR,
    TARGET,
    build_model_comparison,
    classify_report,
    confusion_plot_data,
    get_best_model_name,
    load_metadata,
    load_models,
    validate_feature_frame,
    evaluate_model,
)

st.set_page_config(
    page_title="Breast Cancer Classifier Lab", page_icon="🧬", layout="wide"
)


@st.cache_data
def read_sample_csv() -> bytes:
    return (BASE_DIR / "sample_test_data.csv").read_bytes()


@st.cache_resource
def load_app_resources() -> tuple[dict, dict[str, object]]:
    metadata = load_metadata()
    models = load_models()
    return metadata, models


metadata, models = load_app_resources()
expected_features = metadata["feature_columns"]


st.title("🧬 Breast Cancer Classification Model Lab")
st.caption(
    "Compare five saved classification models on uploaded test data. Models are never retrained in this app."
)

with st.sidebar:
    st.header("Experiment controls")
    selected_model = st.selectbox("Select a model", list(models.keys()), index=0)
    uploaded_file = st.file_uploader(
        "Upload test CSV",
        type=["csv"],
        help="CSV must contain all 30 feature columns and the target column.",
    )
    st.download_button(
        "Download sample test CSV",
        data=read_sample_csv(),
        file_name="sample_test_data.csv",
        mime="text/csv",
    )
    st.info("Target mapping: 0 = malignant, 1 = benign")

st.subheader("Dataset and deployment rules")
metric_cols = st.columns(4)
metric_cols[0].metric("Instances", "569")
metric_cols[1].metric("Input features", "30")
metric_cols[2].metric("Classes", "2")
metric_cols[3].metric("Held-out test rows", metadata["test_rows"])
st.write(
    "This application accepts only held-out test data, loads pre-trained pipeline files, and computes evaluation results without fitting any estimator."
)

if uploaded_file is None:
    st.warning(
        "Upload `test_data.csv` or download and upload `sample_test_data.csv` to begin."
    )
    metrics_df = pd.read_csv(BASE_DIR / "model_metrics.csv")
    st.dataframe(metrics_df.style.format(precision=4), use_container_width=True)
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
except Exception as exc:  # pragma: no cover - UI guardrail
    st.error(f"The CSV could not be read: {exc}")
    st.stop()

try:
    X, y = validate_feature_frame(df, expected_features, require_target=True)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

predictions, metrics = evaluate_model(models[selected_model], X, y)

st.success(
    f"Evaluated {selected_model} on {len(df)} uploaded rows. No retraining was performed."
)
value_cols = st.columns(6)
for col, (name, value) in zip(value_cols, metrics.items()):
    col.metric(name, "N/A" if pd.isna(value) else f"{value:.4f}")

left, right = st.columns(2)
with left:
    st.subheader("Confusion matrix")
    confusion_matrix = confusion_plot_data(y, predictions)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Malignant", "Benign"],
        yticklabels=["Malignant", "Benign"],
        ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    st.pyplot(fig, clear_figure=True)

with right:
    st.subheader("Classification report")
    report_df = classify_report(y, predictions)
    st.dataframe(report_df.style.format(precision=4), use_container_width=True)

st.subheader("Comparison of all five saved models")
comparison_df = build_model_comparison(models, X, y)
st.dataframe(
    comparison_df.style.format(
        {
            column: "{:.4f}"
            for column in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
        }
    ).highlight_max(
        subset=["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"], color="#b7f7c2"
    ),
    use_container_width=True,
)
best_model_name = get_best_model_name(comparison_df)
st.info(
    f"For this uploaded data, the highest MCC is from {best_model_name} ({comparison_df.iloc[0]['MCC']:.4f}). MCC is used here because it considers all four confusion-matrix cells."
)

results = df.copy()
results["predicted_target"] = predictions.astype(int).to_numpy()
results["predicted_class"] = results["predicted_target"].map(
    {0: "malignant", 1: "benign"}
)
st.download_button(
    "Download predictions",
    data=results.to_csv(index=False).encode("utf-8"),
    file_name="predictions.csv",
    mime="text/csv",
)
