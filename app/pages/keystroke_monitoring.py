from pathlib import Path
from typing import Dict, List

import pandas as pd
import streamlit as st

from app.utils.keystroke_model import PROFILE_PATH, SVM_PATH
from app.utils.keystroke_monitor import run_passive_monitor


LOG_PATH = Path(__file__).resolve().parents[1] / "logs" / "security_logs.csv"


def _load_logs() -> pd.DataFrame:
    if not LOG_PATH.exists():
        return pd.DataFrame(columns=["timestamp", "event", "result", "reason", "severity", "score", "source_page", "status"])

    try:
        df = pd.read_csv(LOG_PATH, on_bad_lines="skip", engine="python")
    except Exception:
        return pd.DataFrame(columns=["timestamp", "event", "result", "reason", "severity", "score", "source_page", "status"])
    if "event" not in df.columns and "event_type" in df.columns:
        df = df.rename(columns={"event_type": "event"})

    unnamed = [c for c in df.columns if c.startswith("Unnamed")]
    rename_map: Dict[str, str] = {}
    if unnamed:
        if "score" not in df.columns:
            rename_map[unnamed[0]] = "score"
        if len(unnamed) > 1 and "source_page" not in df.columns:
            rename_map[unnamed[1]] = "source_page"
        if len(unnamed) > 2 and "status" not in df.columns:
            rename_map[unnamed[2]] = "status"
    if rename_map:
        df = df.rename(columns=rename_map)

    if "score" not in df.columns:
        df["score"] = df.get("reason", "").astype(str).str.extract(r"(-?\d+\.\d+)")[0]
    if "status" not in df.columns:
        df["status"] = df.get("result", "")
    if "source_page" not in df.columns:
        df["source_page"] = "-"

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    return df


def _session_history() -> pd.DataFrame:
    history: List[Dict] = st.session_state.get("global_anomaly_history", [])
    if not history:
        return pd.DataFrame(columns=["timestamp", "score", "status", "page"])
    df = pd.DataFrame(history)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp")


def _status_chip(label: str) -> None:
    tone = "#16A34A" if label == "Normal" else "#EA580C" if label == "Warning" else "#DC2626"
    st.markdown(
        f"<div style='padding:8px 12px; border-radius:6px; display:inline-block;"
        f"border:1px solid {tone}; color:{tone}; font-weight:700;'>{label}</div>",
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Keystroke Monitoring",
        page_icon="???",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    if not st.session_state.get("authenticated"):
        st.warning("You are not authenticated. Redirecting to login.")
        st.switch_page("pages/login.py")

    st.title("Continuous Monitoring")
    st.caption("Passive keystroke analytics, fed by the hidden listener across the app.")
    monitor_result = run_passive_monitor(
        "keystroke_monitoring",
        search={"label": "Search keystroke activity", "placeholder": "Search anomalies, pages, severity..."},
    )
    search_text = (monitor_result or {}).get("input_text", "").strip()
    if monitor_result and monitor_result.get("error"):
        st.warning("Keystroke model not available yet. Complete enrollment to enable monitoring.")

    logs_df = _load_logs()
    filtered_logs = logs_df
    if search_text:
        mask = logs_df.apply(lambda row: row.astype(str).str.contains(search_text, case=False, na=False).any(), axis=1)
        filtered_logs = logs_df[mask]
    key_logs = filtered_logs[filtered_logs["event"].str.contains("keystroke_anomaly", case=False, na=False)].copy()
    session_df = _session_history()

    col1, col2, col3 = st.columns(3)
    latest_status = st.session_state.get("global_monitor_status", "Normal")
    latest_score = st.session_state.get("global_monitor_score")
    latest_page = session_df.iloc[-1]["page"] if not session_df.empty else "-"

    with col1:
        st.metric("Last Status", latest_status)
        _status_chip(latest_status)
    with col2:
        score_text = f"{latest_score:.3f}" if latest_score is not None else "—"
        st.metric("Last Score", score_text)
        st.caption(f"Model: {SVM_PATH.name}")
    with col3:
        st.metric("Last Page", latest_page)
        st.caption(f"Profile: {PROFILE_PATH.name}")

    st.divider()
    st.subheader("Session Trend")
    if session_df.empty:
        st.info("No keystroke activity captured in this session yet.")
    else:
        chart_data = session_df[["timestamp", "score"]].set_index("timestamp")
        st.line_chart(chart_data, height=260)

    st.subheader("Recent Anomalies (Log)")
    if key_logs.empty:
        st.info("No anomalies have been logged yet.")
    else:
        display = key_logs.sort_values("timestamp", ascending=False).head(12).copy()
        display["timestamp"] = display["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
        display["score"] = display["score"].astype(str)
        display = display.rename(
            columns={
                "timestamp": "Timestamp",
                "status": "Status",
                "score": "Score",
                "source_page": "Page",
                "severity": "Severity",
            }
        )
        st.dataframe(
            display[["Timestamp", "Score", "Status", "Page", "Severity"]],
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Raw Log Preview")
    if filtered_logs.empty:
        st.info("Security log file is empty.")
    else:
        preview = filtered_logs.sort_values("timestamp", ascending=False).head(25)
        st.dataframe(preview, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
