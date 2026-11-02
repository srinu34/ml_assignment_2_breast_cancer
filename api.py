"""FastAPI service for the breast cancer classification models."""

from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from model_utils import (
    BASE_DIR,
    TARGET,
    load_metadata,
    load_models,
    validate_feature_frame,
)

app = FastAPI(
    title="Breast Cancer Classification API",
    description="Predict malignant vs. benign breast cancer cases using the trained models.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


class PredictionRequest(BaseModel):
    """Request payload for prediction calls."""

    model_name: str = Field(..., description="Name of the saved model to use.")
    data: list[dict[str, Any]] = Field(
        ...,
        description="List of patient records. Each record should include the 30 feature columns. Target is optional for prediction calls.",
    )


class EvaluationRequest(PredictionRequest):
    """Request payload for evaluation calls. The target column is required."""


@app.get("/health")
def health() -> dict[str, Any]:
    """Endpoint to check the service status."""
    metadata = load_metadata()
    return {
        "status": "ok",
        "dataset": metadata["dataset_name"],
        "models": list(metadata["model_files"].keys()),
        "feature_count": len(metadata["feature_columns"]),
        "base_dir": str(BASE_DIR),
    }


@app.get("/models")
def list_models() -> dict[str, list[str]]:
    """List all available model names."""
    metadata = load_metadata()
    return {"models": list(metadata["model_files"].keys())}


@app.post("/predict")
def predict(payload: PredictionRequest) -> dict[str, Any]:
    """Generate predictions for a batch of feature rows."""
    models = load_models()
    if payload.model_name not in models:
        raise HTTPException(
            status_code=404, detail=f"Model '{payload.model_name}' was not found."
        )

    if not payload.data:
        raise HTTPException(
            status_code=400, detail="At least one row is required in the data payload."
        )

    try:
        df = pd.DataFrame(payload.data)
        feature_frame, _ = validate_feature_frame(
            df, load_metadata()["feature_columns"], require_target=False
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    model = models[payload.model_name]
    prediction = model.predict(feature_frame)
    probability = (
        model.predict_proba(feature_frame)[:, 1]
        if hasattr(model, "predict_proba")
        else [None] * len(feature_frame)
    )

    return {
        "model_name": payload.model_name,
        "predicted_labels": [int(value) for value in prediction.tolist()],
        "predicted_classes": [
            "malignant" if int(value) == 0 else "benign"
            for value in prediction.tolist()
        ],
        "probabilities": [float(value) for value in probability],
        "count": len(feature_frame),
    }


@app.post("/evaluate")
def evaluate(payload: EvaluationRequest) -> dict[str, Any]:
    """Evaluate a saved model against labeled data."""
    models = load_models()
    if payload.model_name not in models:
        raise HTTPException(
            status_code=404, detail=f"Model '{payload.model_name}' was not found."
        )

    if not payload.data:
        raise HTTPException(
            status_code=400, detail="At least one row is required in the data payload."
        )

    try:
        df = pd.DataFrame(payload.data)
        feature_frame, y = validate_feature_frame(
            df, load_metadata()["feature_columns"], require_target=True
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    model = models[payload.model_name]
    predictions = model.predict(feature_frame)
    scores = (
        model.predict_proba(feature_frame)[:, 1]
        if hasattr(model, "predict_proba")
        else predictions.astype(float)
    )

    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    metrics = {
        "accuracy": accuracy_score(y, predictions),
        "auc": roc_auc_score(y, scores),
        "precision": precision_score(y, predictions, zero_division=0),
        "recall": recall_score(y, predictions, zero_division=0),
        "f1": f1_score(y, predictions, zero_division=0),
        "mcc": matthews_corrcoef(y, predictions),
    }

    return {
        "model_name": payload.model_name,
        "metrics": metrics,
        "count": len(feature_frame),
        "predicted_labels": [int(value) for value in predictions.tolist()],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
