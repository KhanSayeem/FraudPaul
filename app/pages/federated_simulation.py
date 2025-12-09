from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st


LOG_PATH = Path(__file__).resolve().parents[1] / "logs" / "security_logs.csv"
DATA_PATH = Path(__file__).resolve().parents[2] / "creditcard.csv"
COLUMNS_PATH = Path(__file__).resolve().parents[1] / "models" / "columns.json"


@st.cache_data(show_spinner=False)
def load_dataset(sample_size: int) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load and sample the credit card dataset."""
    if not DATA_PATH.exists():
        raise FileNotFoundError("creditcard.csv not found in project root.")

    df = pd.read_csv(DATA_PATH)
    if "Class" not in df.columns:
        raise ValueError("Dataset missing required 'Class' column.")

    feature_cols = [c for c in df.columns if c != "Class"]
    if COLUMNS_PATH.exists():
        try:
            feature_cols = json.loads(COLUMNS_PATH.read_text())
        except Exception:
            pass

    sample_size = min(sample_size, len(df))
    df_sample = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    X = df_sample[feature_cols].values.astype(float)
    y = df_sample["Class"].values.astype(float)
    return X, y, feature_cols


def train_val_split(X: np.ndarray, y: np.ndarray, val_fraction: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shuffle and split into train/validation sets."""
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(X))
    split = int(len(indices) * (1 - val_fraction))
    train_idx, val_idx = indices[:split], indices[split:]
    return X[train_idx], X[val_idx], y[train_idx], y[val_idx]


def standardize(train_X: np.ndarray, val_X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_X.mean(axis=0)
    std = train_X.std(axis=0) + 1e-8
    return (train_X - mean) / std, (val_X - mean) / std, mean, std


def split_clients(X: np.ndarray, y: np.ndarray, num_clients: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Partition data into client shards."""
    shards_X = np.array_split(X, num_clients)
    shards_y = np.array_split(y, num_clients)
    return list(zip(shards_X, shards_y))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def local_train(X: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float, lr: float, epochs: int) -> Tuple[np.ndarray, float]:
    """Run simple batch gradient descent on a client's shard."""
    if len(X) == 0:
        return weights, bias
    w = weights.copy()
    b = bias
    n = len(X)
    for _ in range(epochs):
        logits = X @ w + b
        preds = sigmoid(logits)
        error = preds - y
        grad_w = (X.T @ error) / n
        grad_b = float(error.mean())
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def fedavg_round(clients: List[Tuple[np.ndarray, np.ndarray]], weights: np.ndarray, bias: float, lr: float, epochs: int) -> Tuple[np.ndarray, float]:
    """Perform one FedAvg round across all clients."""
    total_samples = sum(len(X) for X, _ in clients)
    if total_samples == 0:
        return weights, bias

    agg_w = np.zeros_like(weights)
    agg_b = 0.0
    for X_c, y_c in clients:
        w_c, b_c = local_train(X_c, y_c, weights, bias, lr, epochs)
        weight_factor = len(X_c) / total_samples
        agg_w += weight_factor * w_c
        agg_b += weight_factor * b_c
    return agg_w, agg_b


def evaluate(X: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float) -> float:
    if len(X) == 0:
        return 0.0
    preds = sigmoid(X @ weights + bias) >= 0.5
    return float((preds == y).mean())


def log_simulation(clients: int, rounds: int, accuracy: float) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    exists = LOG_PATH.exists()
    with LOG_PATH.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["timestamp", "event", "result", "reason", "severity", "score", "source_page", "status"])
        writer.writerow(
            [
                pd.Timestamp.utcnow().isoformat(),
                "federated_simulation",
                "success",
                f"clients={clients}; rounds={rounds}",
                "Info",
                f"{accuracy:.4f}",
                "federated_simulation",
                "completed",
            ]
        )


def run_simulation(num_clients: int, rounds: int, local_epochs: int, lr: float, sample_size: int) -> Dict[str, object]:
    X, y, feature_cols = load_dataset(sample_size)
    X_train, X_val, y_train, y_val = train_val_split(X, y)
    X_train, X_val, _, _ = standardize(X_train, X_val)
    clients = split_clients(X_train, y_train, num_clients)

    weights = np.zeros(X_train.shape[1], dtype=float)
    bias = 0.0
    history = []

    for r in range(1, rounds + 1):
        weights, bias = fedavg_round(clients, weights, bias, lr, local_epochs)
        acc = evaluate(X_val, y_val, weights, bias)
        history.append({"round": r, "val_accuracy": acc})

    final_acc = history[-1]["val_accuracy"] if history else 0.0
    log_simulation(num_clients, rounds, final_acc)

    client_sizes = [{"client": idx + 1, "samples": len(clients[idx][0])} for idx in range(len(clients))]
    return {
        "history": history,
        "final_acc": final_acc,
        "client_sizes": client_sizes,
        "feature_count": X_train.shape[1],
        "val_samples": len(X_val),
    }


def main() -> None:
    st.set_page_config(
        page_title="Federated Learning Simulation",
        page_icon="??",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    if not st.session_state.get("authenticated"):
        st.warning("You are not authenticated. Redirecting to login.")
        st.switch_page("pages/login.py")

    st.title("Federated Learning Simulation")
    st.caption("Demonstrate decentralized training via FedAvg across virtual banks.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        num_clients = st.slider("Number of clients", min_value=2, max_value=8, value=4, step=1)
    with col2:
        rounds = st.slider("Federated rounds", min_value=1, max_value=25, value=8, step=1)
    with col3:
        local_epochs = st.slider("Local epochs per round", min_value=1, max_value=10, value=3, step=1)
    with col4:
        lr = st.number_input("Learning rate", min_value=0.0005, max_value=0.5, value=0.02, step=0.005, format="%.4f")

    sample_size = st.number_input(
        "Sample size (rows)",
        min_value=500,
        max_value=20000,
        value=4000,
        step=500,
        help="Subset of creditcard.csv used for the simulation.",
    )

    if st.button("Run simulation", type="primary"):
        try:
            with st.spinner("Running federated training..."):
                result = run_simulation(num_clients, rounds, local_epochs, lr, int(sample_size))
        except FileNotFoundError as exc:
            st.error(str(exc))
            st.info("Place creditcard.csv in the project root to run the simulation.")
            return
        except Exception as exc:
            st.error(f"Simulation failed: {exc}")
            return

        st.success(f"Simulation complete. Final validation accuracy: {result['final_acc']:.3f}")

        history_df = pd.DataFrame(result["history"])
        if not history_df.empty:
            st.subheader("Convergence")
            st.line_chart(history_df.set_index("round"), height=300)

        st.subheader("Client shards")
        st.dataframe(pd.DataFrame(result["client_sizes"]), use_container_width=True, hide_index=True)
        st.caption(
            f"Features: {result['feature_count']} | Validation samples: {result['val_samples']} | "
            "Aggregation uses weighted FedAvg (by shard size)."
        )


if __name__ == "__main__":
    main()
