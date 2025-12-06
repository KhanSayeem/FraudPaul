import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.svm import OneClassSVM


BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
SVM_PATH = MODELS_DIR / "keystroke_svm.pkl"
PROFILE_PATH = MODELS_DIR / "keystroke_profile.json"


def ensure_dirs() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def train_svm(feature_matrix: np.ndarray) -> OneClassSVM:
    model = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale")
    model.fit(feature_matrix)
    return model


def save_model(model: OneClassSVM) -> None:
    ensure_dirs()
    import joblib

    joblib.dump(model, SVM_PATH)


def load_model() -> OneClassSVM:
    import joblib

    return joblib.load(SVM_PATH)


def save_profile(samples: List[Dict[str, Any]]) -> None:
    ensure_dirs()
    payload = {"samples": samples}
    PROFILE_PATH.write_text(json.dumps(payload, indent=2))


def load_profile() -> Tuple[List[Dict[str, Any]], bool]:
    if not PROFILE_PATH.exists():
        return [], False
    data = json.loads(PROFILE_PATH.read_text())
    return data.get("samples", []), True


def score_sample(model: OneClassSVM, vector: np.ndarray) -> float:
    score = model.decision_function(vector.reshape(1, -1))[0]
    return float(score)
