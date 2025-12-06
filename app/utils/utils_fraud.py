import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"
LOG_PATH = BASE_DIR / "logs" / "security_logs.csv"

SEVERITY_COLORS = {
    "HIGH": "#DC2626",
    "MEDIUM": "#F59E0B",
    "LOW": "#16A34A",
}


def load_assets() -> Tuple[object, object, object, List[str]]:
    """Load trained models, scaler, and column schema."""
    rf_model = joblib.load(MODEL_DIR / "rf_model.pkl")
    xgb_model = joblib.load(MODEL_DIR / "xgb_model.pkl")
    # Backward compatibility: older saved models may reference this flag.
    if not hasattr(xgb_model, "use_label_encoder"):
        xgb_model.use_label_encoder = False
    # Ensure CPU fallback fields exist for newer xgboost runtime.
    if not hasattr(xgb_model, "gpu_id"):
        xgb_model.gpu_id = None
    if not hasattr(xgb_model, "predictor"):
        xgb_model.predictor = "auto"
    if not hasattr(xgb_model, "classes_"):
        xgb_model.classes_ = np.array([0, 1])
    if not hasattr(xgb_model, "n_classes_"):
        xgb_model.n_classes_ = 2
    if not hasattr(xgb_model, "n_features_in_"):
        # Try to infer from saved columns schema
        cols_path = MODEL_DIR / "columns.json"
        if cols_path.exists():
            with cols_path.open("r", encoding="utf-8") as f:
                cols = json.load(f)
            xgb_model.n_features_in_ = len(cols)
        else:
            xgb_model.n_features_in_ = None
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    with open(MODEL_DIR / "columns.json", "r", encoding="utf-8") as f:
        columns = json.load(f)
    return rf_model, xgb_model, scaler, columns


def load_sample_dataset(limit: int = 1200) -> Optional[pd.DataFrame]:
    """Load a lightweight slice of the credit card dataset for demos."""
    sample_path = Path(__file__).resolve().parents[2] / "creditcard.csv"
    if not sample_path.exists():
        return None
    return pd.read_csv(sample_path, nrows=limit)


def validate_columns(df: pd.DataFrame, required: Iterable[str]) -> List[str]:
    """Return a list of missing columns."""
    missing = [col for col in required if col not in df.columns]
    return missing


def summarize_dataset(df: pd.DataFrame) -> Dict[str, object]:
    summary = {
        "rows": len(df),
        "features": len(df.columns),
        "fraud_count": None,
        "legit_count": None,
    }
    if "Class" in df.columns:
        counts = df["Class"].value_counts().to_dict()
        summary["fraud_count"] = counts.get(1, 0)
        summary["legit_count"] = counts.get(0, 0)
    return summary


def _prepare_features(df: pd.DataFrame, feature_cols: List[str], scaler) -> np.ndarray:
    features = df[feature_cols].copy()
    features = features.apply(pd.to_numeric, errors="coerce")
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return scaler.transform(features)


def run_inference(
    df: pd.DataFrame,
    rf_model,
    xgb_model,
    scaler,
    feature_cols: List[str],
) -> Dict[str, object]:
    """Run RF and XGB predictions, returning enriched data and raw outputs."""
    X_scaled = _prepare_features(df, feature_cols, scaler)

    rf_pred = rf_model.predict(X_scaled)
    rf_prob = rf_model.predict_proba(X_scaled)[:, 1]
    xgb_pred = xgb_model.predict(X_scaled)
    xgb_prob = xgb_model.predict_proba(X_scaled)[:, 1]

    blended_prob = (rf_prob + xgb_prob) / 2
    ensemble_pred = (blended_prob >= 0.5).astype(int)

    results = df.copy()
    results["fraud_probability"] = blended_prob
    results["prediction"] = np.where(ensemble_pred == 1, "Fraudulent", "Legitimate")
    results["rf_probability"] = rf_prob
    results["xgb_probability"] = xgb_prob
    results["rf_prediction"] = rf_pred
    results["xgb_prediction"] = xgb_pred

    return {
        "data": results,
        "rf_prob": rf_prob,
        "xgb_prob": xgb_prob,
        "rf_pred": rf_pred,
        "xgb_pred": xgb_pred,
        "blended_prob": blended_prob,
        "ensemble_pred": ensemble_pred,
    }


def severity_label(probability: float) -> str:
    if probability >= 0.7:
        return "HIGH"
    if probability >= 0.3:
        return "MEDIUM"
    return "LOW"


def log_prediction_event(probability: float, result: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    exists = LOG_PATH.exists()
    with LOG_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["timestamp", "event_type", "result", "probability", "severity"])
        writer.writerow(
            [
                datetime.utcnow().isoformat(),
                "fraud_prediction",
                result,
                f"{probability:.4f}",
                severity_label(probability),
            ]
        )


def compute_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
) -> Dict[str, object]:
    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "auc": roc_auc_score(labels, probabilities) if len(np.unique(labels)) > 1 else 0.0,
        "confusion": confusion_matrix(labels, predictions).tolist(),
        "report": classification_report(labels, predictions, zero_division=0, output_dict=True),
    }
    return metrics


def format_metric_badge(label: str, value: float, threshold: float = 0.9) -> str:
    tone = "success" if value >= threshold else "danger"
    return f"<span class='badge {tone}'>{label}: {value:.2%}</span>"
