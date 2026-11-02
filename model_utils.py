"""Reusable ML utilities for training, validation, and inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
TARGET = "target"


def load_metadata() -> dict[str, Any]:
    """Load the persisted project metadata."""
    metadata_path = BASE_DIR / "model_metadata.json"
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def load_models() -> dict[str, Any]:
    """Load all saved models from the model directory."""
    metadata = load_metadata()
    model_map: dict[str, Any] = {}
    for name, filename in metadata["model_files"].items():
        model_map[name] = joblib.load(MODEL_DIR / filename)
    return model_map


def validate_feature_frame(
    df: pd.DataFrame,
    expected_features: list[str],
    require_target: bool = True,
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Validate uploaded data before scoring."""
    required = expected_features + ([TARGET] if require_target else [])
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    extra = [col for col in df.columns if col not in required]
    if extra:
        extra_message = ", ".join(extra)
        # Keep the app user-friendly while still preserving required fields.
        _ = extra_message

    selected = df[required].copy()
    if selected.isnull().any().any():
        raise ValueError(
            "Missing values were found. Remove or impute them before evaluation."
        )

    if require_target:
        if not set(selected[TARGET].unique()).issubset({0, 1}):
            raise ValueError("The target column must contain only 0 and 1.")
        y = selected[TARGET].astype(int)
        return selected[expected_features], y

    return selected[expected_features], None


def evaluate_model(
    model: Any, X: pd.DataFrame, y: pd.Series
) -> tuple[pd.Series, dict[str, float]]:
    """Return predictions and evaluation metrics for a model."""
    predictions = pd.Series(model.predict(X), index=X.index)

    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
    else:
        scores = predictions.astype(float)

    auc_score = roc_auc_score(y, scores) if y.nunique() == 2 else float("nan")
    metrics = {
        "Accuracy": accuracy_score(y, predictions),
        "AUC": auc_score,
        "Precision": precision_score(y, predictions, zero_division=0),
        "Recall": recall_score(y, predictions, zero_division=0),
        "F1": f1_score(y, predictions, zero_division=0),
        "MCC": matthews_corrcoef(y, predictions),
    }
    return predictions, metrics


def build_model_comparison(
    models: dict[str, Any], X: pd.DataFrame, y: pd.Series
) -> pd.DataFrame:
    """Build a comparison table for all saved models."""
    rows: list[dict[str, float | str]] = []
    for name, model in models.items():
        _, metrics = evaluate_model(model, X, y)
        rows.append({"ML Model Name": name, **metrics})
    return pd.DataFrame(rows).sort_values("MCC", ascending=False)


def classify_report(y_true: pd.Series, y_pred: pd.Series) -> pd.DataFrame:
    """Return the standard scikit-learn classification_report as a DataFrame."""
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["malignant", "benign"],
        output_dict=True,
        zero_division=0,
    )
    return pd.DataFrame(report).T


def confusion_plot_data(y_true: pd.Series, y_pred: pd.Series) -> pd.DataFrame:
    """Return confusion matrix values ordered as malignant/benign."""
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return pd.DataFrame(
        matrix,
        index=["Actual Malignant", "Actual Benign"],
        columns=["Predicted Malignant", "Predicted Benign"],
    )


def get_best_model_name(comparison: pd.DataFrame) -> str:
    """Return the best model according to MCC."""
    return str(comparison.iloc[0]["ML Model Name"])
