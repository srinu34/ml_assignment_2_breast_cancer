"""Train and persist the breast cancer classification models used by the app and API."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)


def build_models() -> dict[str, object]:
    """Return the model registry used in the assignment."""
    return {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=2000, random_state=42)),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=4,
            random_state=42,
        ),
        "K-Nearest Neighbors": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", KNeighborsClassifier(n_neighbors=7, weights="distance")),
            ]
        ),
        "Gaussian Naive Bayes": GaussianNB(),
        "Support Vector Machine": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", SVC(C=2.0, probability=True, random_state=42)),
            ]
        ),
    }


def compute_metrics(
    y_true: pd.Series, y_pred: pd.Series, y_score: pd.Series
) -> dict[str, float]:
    """Return the assignment metrics for a trained model."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_score),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def main() -> None:
    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = data.target.rename("target")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model_files = {
        "Logistic Regression": "logistic_regression.joblib",
        "Decision Tree": "decision_tree.joblib",
        "K-Nearest Neighbors": "knn.joblib",
        "Gaussian Naive Bayes": "naive_bayes.joblib",
        "Support Vector Machine": "svm.joblib",
    }

    rows: list[dict[str, float | str]] = []
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probability = model.predict_proba(X_test)[:, 1]

        metrics = compute_metrics(
            y_test, predictions, pd.Series(probability, index=y_test.index)
        )
        rows.append({"ML Model Name": name, **metrics})
        joblib.dump(model, MODEL_DIR / model_files[name])

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(ROOT / "model_metrics.csv", index=False, float_format="%.6f")

    test_df = X_test.copy()
    test_df["target"] = y_test.to_numpy()
    test_df.to_csv(ROOT / "test_data.csv", index=False)

    metadata = {
        "dataset_name": "Breast Cancer Wisconsin (Diagnostic)",
        "source": "UCI Machine Learning Repository; distributed through scikit-learn",
        "target_column": "target",
        "class_names": {"0": "malignant", "1": "benign"},
        "feature_columns": list(X.columns),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "random_state": 42,
        "model_files": model_files,
    }
    (ROOT / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
