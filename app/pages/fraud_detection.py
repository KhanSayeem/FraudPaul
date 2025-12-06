from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

from app.utils import ui
from app.utils.keystroke_monitor import run_passive_monitor
from app.utils.utils_fraud import (
    compute_metrics,
    format_metric_badge,
    load_assets,
    load_sample_dataset,
    log_prediction_event,
    run_inference,
    summarize_dataset,
    validate_columns,
)
from app.utils.utils_shap import explain_row


SUCCESS = "#16A34A"
WARNING = "#F59E0B"
DANGER = "#DC2626"


def _load_dataframe(uploaded_file, sample_trigger: bool) -> Optional[pd.DataFrame]:
    if sample_trigger:
        return load_sample_dataset()
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return None


def _probability_style(val: float) -> str:
    if val < 0.2:
        return f"color: {SUCCESS}; font-weight: 600;"
    if val < 0.6:
        return f"color: {WARNING}; font-weight: 600;"
    return f"color: {DANGER}; font-weight: 700;"


def _prediction_style(label: str) -> str:
    if label.lower().startswith("fraud"):
        return f"color: {DANGER}; font-weight: 700;"
    return "color: #FFFFFF; font-weight: 500;"


def _log_top_events(probabilities: np.ndarray, predictions: np.ndarray) -> None:
    if len(probabilities) == 0:
        return
    top_idx = np.argsort(probabilities)[-3:][::-1]
    for i in top_idx:
        prob = float(probabilities[i])
        result = "fraudulent" if predictions[i] == 1 else "legitimate"
        log_prediction_event(prob, result)


def main() -> None:
    st.set_page_config(
        page_title="Fraud Detection & Explainability",
        page_icon="??",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    ui.inject_styles()
    st.title("Fraud Detection & Explainability")
    st.caption("Analyze transactions using machine learning models.")
    st.divider()

    upload_col, _ = st.columns([1.3, 1])
    with upload_col:
        upload_card = st.container(border=True)
        with upload_card:
            st.subheader("Upload CSV File")
            st.caption("Dataset must include columns: Time, V1-V28, Amount.")
            uploaded = st.file_uploader("Drag & drop or browse", type=["csv"])
            sample = st.button("Load Sample Dataset", use_container_width=True)

    data = None
    error_message = None
    try:
        data = _load_dataframe(uploaded, sample)
    except Exception as exc:
        error_message = f"Unable to read dataset: {exc}"
    if sample and data is None and error_message is None:
        error_message = "Sample dataset not found. Place creditcard.csv in the project root."

    if data is None:
        if error_message:
            st.error(error_message)
        return

    st.success("Dataset loaded.")

    try:
        rf_model, xgb_model, scaler, feature_cols = load_assets()
    except Exception as exc:
        st.error(f"Models or scaler could not be loaded: {exc}")
        return

    has_labels = "Class" in data.columns
    labels = data["Class"].astype(int).values if has_labels else None
    feature_frame = data.drop(columns=["Class"]) if has_labels else data

    missing = validate_columns(feature_frame, feature_cols)
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        return

    summary = summarize_dataset(data)
    with st.container(border=True):
        st.subheader("Data Summary")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Rows", f"{summary['rows']:,}")
        s2.metric("Features", summary["features"])
        s3.metric("Fraud", f"{summary['fraud_count']:,}" if summary["fraud_count"] is not None else "—")
        s4.metric("Legitimate", f"{summary['legit_count']:,}" if summary["legit_count"] is not None else "—")

    with st.spinner("Running models..."):
        inference = run_inference(feature_frame, rf_model, xgb_model, scaler, feature_cols)
    _log_top_events(inference["blended_prob"], inference["ensemble_pred"])

    with st.container(border=True):
        st.subheader("Model Results")
        col_rf, col_xgb = st.columns(2)

        if has_labels:
            rf_metrics = compute_metrics(labels, inference["rf_prob"], inference["rf_pred"])
            xgb_metrics = compute_metrics(labels, inference["xgb_prob"], inference["xgb_pred"])
        else:
            rf_metrics = xgb_metrics = None

        with col_rf:
            st.markdown("**Random Forest Results**")
            if rf_metrics:
                st.markdown(format_metric_badge("Accuracy", rf_metrics["accuracy"]), unsafe_allow_html=True)
                st.write(f"Precision: {rf_metrics['precision']:.2%}")
                st.write(f"Recall: {rf_metrics['recall']:.2%}")
                st.write(f"F1 Score: {rf_metrics['f1']:.2%}")
                st.write(f"AUC: {rf_metrics['auc']:.3f}")
            else:
                st.info("Ground truth not provided. Metrics unavailable.")

        with col_xgb:
            st.markdown("**XGBoost Results**")
            if xgb_metrics:
                st.markdown(format_metric_badge("Accuracy", xgb_metrics["accuracy"]), unsafe_allow_html=True)
                st.write(f"Precision: {xgb_metrics['precision']:.2%}")
                st.write(f"Recall: {xgb_metrics['recall']:.2%}")
                st.write(f"F1 Score: {xgb_metrics['f1']:.2%}")
                st.write(f"AUC: {xgb_metrics['auc']:.3f}")
            else:
                st.info("Ground truth not provided. Metrics unavailable.")

    display_df = inference["data"].copy()
    display_df.insert(0, "index", display_df.index)
    display_df = display_df.rename(
        columns={
            "index": "Index",
            "Time": "Time",
            "Amount": "Amount",
            "fraud_probability": "Fraud Probability",
            "prediction": "Prediction",
        }
    )
    display_filtered = display_df

    table_view = display_filtered[["Index", "Time", "Amount", "Fraud Probability", "Prediction"]]
    # To avoid Styler limits on huge datasets, only style a capped view.
    max_render_rows = 1200
    if len(table_view) > max_render_rows:
        st.info(f"Showing first {max_render_rows:,} rows out of {len(table_view):,}.")
        table_view = table_view.head(max_render_rows)
    styled_table = (
        table_view.style.applymap(_probability_style, subset=["Fraud Probability"])
        .applymap(_prediction_style, subset=["Prediction"])
        .format({"Fraud Probability": "{:.2%}", "Amount": "{:.2f}"})
    )

    with st.container(border=True):
        st.subheader("Predictions")
        st.dataframe(styled_table, use_container_width=True, hide_index=True)

    # Suspicious spotlight section
    with st.container(border=True):
        st.subheader("Flagged / High-Risk Transactions")
        st.caption("Transactions predicted as Fraudulent or with fraud probability >= 0.20.")
        flagged = display_filtered[
            (display_filtered["Prediction"] == "Fraudulent") | (display_filtered["Fraud Probability"] >= 0.20)
        ].copy()
        if flagged.empty:
            st.success("No high-risk transactions detected in the current dataset.")
        else:
            flagged = flagged.sort_values("Fraud Probability", ascending=False)
            max_rows = 300
            if len(flagged) > max_rows:
                st.info(f"Showing top {max_rows:,} high-risk rows out of {len(flagged):,}.")
                flagged = flagged.head(max_rows)
            flagged_view = flagged[["Index", "Time", "Amount", "Fraud Probability", "Prediction"]]
            flagged_table = (
                flagged_view.style.applymap(_probability_style, subset=["Fraud Probability"])
                .applymap(_prediction_style, subset=["Prediction"])
                .format({"Fraud Probability": "{:.2%}", "Amount": "{:.2f}"})
            )
            st.dataframe(flagged_table, use_container_width=True, hide_index=True)

    with st.container(border=True):
        st.subheader("Explainability (SHAP)")
        st.caption("Select a transaction to inspect feature impact.")

        if "shap_result" not in st.session_state:
            st.session_state["shap_result"] = None
            st.session_state["shap_error"] = None

        options = list(display_filtered["Index"].values)
        selected = st.selectbox("Select transaction ID", options=options, label_visibility="collapsed", key="shap_selected")

        # Auto-generate on selection to avoid confusion and preserve output after reruns.
        row = feature_frame.loc[selected, feature_cols].values.reshape(1, -1)
        with st.spinner("Generating SHAP explanation..."):
            try:
                shap_html, shap_png = explain_row(xgb_model, row, feature_cols)
                st.session_state["shap_result"] = (shap_html, shap_png)
                st.session_state["shap_error"] = None
            except Exception as exc:
                st.session_state["shap_result"] = None
                st.session_state["shap_error"] = exc

        if st.session_state["shap_error"]:
            st.error(f"Unable to generate SHAP plot: {st.session_state['shap_error']}")
            st.exception(st.session_state["shap_error"])
        elif st.session_state["shap_result"]:
            shap_html, shap_png = st.session_state["shap_result"]
            st.components.v1.html(shap_html, height=320)
            st.image(shap_png, caption="Top feature impacts", use_container_width=True)
            st.success("SHAP explanation generated.")

if __name__ == "__main__":
    main()
