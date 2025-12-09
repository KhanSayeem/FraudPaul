from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
import streamlit as st
import altair as alt


LOG_PATH = Path(__file__).resolve().parents[1] / "logs" / "security_logs.csv"


def _load_logs() -> pd.DataFrame:
    if not LOG_PATH.exists():
        return pd.DataFrame(columns=["timestamp", "event", "result", "reason", "severity", "probability", "score", "source_page", "status"])

    try:
        df = pd.read_csv(LOG_PATH, on_bad_lines="skip", engine="python")
    except Exception:
        return pd.DataFrame(columns=["timestamp", "event", "result", "reason", "severity", "probability", "score", "source_page", "status"])

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
        df["status"] = df.get("result", "-")
    if "source_page" not in df.columns:
        df["source_page"] = "-"
    if "probability" not in df.columns and "reason" in df.columns:
        df["probability"] = df["reason"].astype(str).str.extract(r"(-?\d+\.\d+)")[0]

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    return df


def _summary_cards(df: pd.DataFrame) -> None:
    total = len(df)
    logins = len(df[df["event"].str.contains("login_attempt", na=False)])
    anomalies = len(df[df["event"].str.contains("keystroke_anomaly", na=False)])
    fraud_preds = len(df[df["event"].str.contains("fraud_prediction", na=False)])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total entries", f"{total:,}")
    c2.metric("Login attempts", f"{logins:,}")
    c3.metric("Keystroke anomalies", f"{anomalies:,}")
    c4.metric("Fraud predictions", f"{fraud_preds:,}")


def _event_trends(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("No logs to chart yet.")
        return
    daily = df.set_index("timestamp").resample("1D").size().rename("events").to_frame()
    st.line_chart(daily, height=260)


def _severity_bar(df: pd.DataFrame) -> None:
    if df.empty or "severity" not in df:
        st.info("Severity data not available.")
        return
    counts = (
        df["severity"]
        .fillna("unknown")
        .str.title()
        .value_counts()
        .rename_axis("Severity")
        .reset_index(name="Count")
    )
    color_map = {
        "High": "#DC2626",
        "Medium": "#F59E0B",
        "Success": "#16A34A",
        "Passed": "#16A34A",
    }
    counts["Color"] = counts["Severity"].map(color_map).fillna("#6B7280")
    chart = (
        alt.Chart(counts)
        .mark_bar()
        .encode(
            x=alt.X("Severity:N", sort=None),
            y=alt.Y("Count:Q"),
            color=alt.Color("Color:N", scale=None),
        )
    )
    st.altair_chart(chart, use_container_width=True)


def _severity_style(val: str) -> str:
    tone = str(val).lower()
    if tone == "high":
        return "color: #DC2626; font-weight: 700;"
    if tone == "medium":
        return "color: #F59E0B; font-weight: 700;"
    if tone in ("passed", "success"):
        return "color: #16A34A; font-weight: 700;"
    return ""


def _fraud_table(df: pd.DataFrame) -> None:
    fraud_df = df[df["event"].str.contains("fraud_prediction", na=False)].copy()
    fraud_df = fraud_df[fraud_df["result"].str.contains("fraud", case=False, na=False)]
    if fraud_df.empty:
        st.success("No fraudulent predictions logged yet.")
        return
    fraud_df["timestamp"] = fraud_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    cols = ["timestamp", "result", "probability", "severity"]
    cols = [c for c in cols if c in fraud_df.columns]
    styled = fraud_df[cols].style.applymap(_severity_style, subset=["severity"] if "severity" in cols else None)
    st.dataframe(styled, use_container_width=True, hide_index=True)


def _raw_preview(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("Security log file is empty.")
        return
    preview = df.sort_values("timestamp", ascending=False).head(100).copy()
    preview["timestamp"] = preview["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    styled = preview.style.applymap(_severity_style, subset=["severity"] if "severity" in preview.columns else None)
    st.dataframe(styled, use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(
        page_title="Security Logs",
        page_icon="??",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    if not st.session_state.get("authenticated"):
        st.warning("You are not authenticated. Redirecting to login.")
        st.switch_page("pages/login.py")

    st.title("Security Logs")
    st.caption("Explore authentication, fraud, and keystroke events with visuals.")

    logs_df = _load_logs()

    _summary_cards(logs_df)
    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Daily event volume")
        _event_trends(logs_df)
    with c2:
        st.subheader("Severity breakdown")
        _severity_bar(logs_df)

    st.subheader("Fraudulent predictions")
    _fraud_table(logs_df)

    st.subheader("Raw Log Preview")
    _raw_preview(logs_df)


if __name__ == "__main__":
    main()
