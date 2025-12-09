import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import streamlit as st

from app.utils.keystroke_monitor import run_passive_monitor


LOG_PATH = Path(__file__).resolve().parents[1] / "logs" / "security_logs.csv"

COLORS: Dict[str, str] = {
    "BACKGROUND": "#0B1220",
    "CARD_BG": "rgba(255,255,255,0.02)",
    "BORDER": "#1F2937",
    "TEXT": "#FFFFFF",
    "MUTED": "#9CA3AF",
    "SUCCESS": "#10B981",
    "WARNING": "#F59E0B",
    "DANGER": "#EF4444",
}


def _inject_styles() -> None:
    st.markdown(
        f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
            html, body, [class*="css"] {{
                font-family: 'Inter', 'Roboto', sans-serif;
                background: {COLORS["BACKGROUND"]};
                color: {COLORS["TEXT"]};
            }}
            .main .block-container {{
                padding-top: 0px;
                padding-bottom: 12px;
                margin-top: 0px;
            }}
            .dashboard-shell {{
                max-width: 1100px;
                margin: 0 auto;
                padding: 0px 12px 22px 12px;
            }}
            .page-title {{
                font-size: 32px;
                font-weight: 700;
                text-align: center;
                margin: 0;
                color: {COLORS["TEXT"]};
            }}
            .page-subtitle {{
                font-size: 16px;
                text-align: center;
                color: {COLORS["MUTED"]};
                margin-top: 6px;
            }}
            .section-spacer {{ margin-top: 40px; }}
            .card {{
                background: {COLORS["CARD_BG"]};
                padding: 24px;
                border-radius: 8px;
                border: 1px solid {COLORS["BORDER"]};
                backdrop-filter: blur(4px);
                margin-bottom: 20px;
            }}
            .card h3 {{
                margin: 0;
                font-size: 20px;
                font-weight: 600;
                color: {COLORS["TEXT"]};
            }}
            .card .subtitle {{
                margin: 8px 0 12px 0;
                color: {COLORS["MUTED"]};
                font-size: 15px;
            }}
            .badge {{
                padding: 6px 12px;
                border-radius: 4px;
                font-size: 13px;
                font-weight: 600;
                border: 1px solid transparent;
                display: inline-block;
            }}
            .status-success {{ background: #10B98133; color: {COLORS["SUCCESS"]}; border-color: #10B981; }}
            .status-danger {{ background: #EF444433; color: {COLORS["DANGER"]}; border-color: #EF4444; }}
            .status-warning {{ background: #F59E0B33; color: {COLORS["WARNING"]}; border-color: #F59E0B; }}
            .quick-actions .stButton>button {{
                width: 100%;
                padding: 12px 24px;
                border-radius: 6px;
                border: 1px solid {COLORS["BORDER"]};
                background: rgba(255,255,255,0.05);
                color: {COLORS["TEXT"]};
                font-weight: 600;
            }}
            .quick-actions .stButton>button:hover {{
                background: rgba(255,255,255,0.1);
            }}
            .section-card {{
                border: 1px solid {COLORS["BORDER"]};
                border-radius: 8px;
                padding: 18px;
                background: {COLORS["CARD_BG"]};
            }}
            .stDataFrame tbody tr td {{
                padding-top: 10px;
                padding-bottom: 10px;
            }}
            div[data-testid="stDataFrame"] {{
                border: 1px solid {COLORS["BORDER"]};
                border-radius: 8px;
                overflow: hidden;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _load_logs() -> pd.DataFrame:
    if not LOG_PATH.exists():
        return pd.DataFrame(columns=["timestamp", "event", "result", "reason", "severity", "score", "source_page", "status"])

    try:
        df = pd.read_csv(LOG_PATH, on_bad_lines="skip", engine="python")
    except Exception:
        # Fallback to an empty frame if the log is malformed.
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

    if "severity" not in df.columns:
        df["severity"] = "-"
    if "reason" not in df.columns:
        df["reason"] = df.get("probability", "-")
    if "score" not in df.columns:
        df["score"] = df.get("reason", "").astype(str).str.extract(r"(-?\d+\.\d+)")[0]
    if "source_page" not in df.columns:
        df["source_page"] = "-"
    if "status" not in df.columns:
        df["status"] = df.get("result", "-")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    return df


def _format_ts(ts: Optional[datetime]) -> str:
    if not ts:
        return "No data"
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def _last_event(df: pd.DataFrame, event_name: str) -> Optional[pd.Series]:
    if df.empty or "event" not in df.columns:
        return None
    filtered = df[df["event"].str.lower() == event_name.lower()]
    if filtered.empty:
        return None
    return filtered.iloc[-1]


def _parse_score(reason: str) -> Optional[float]:
    if not reason:
        return None
    match = re.search(r"(-?\d+(?:\.\d+)?)", str(reason))
    return float(match.group(1)) if match else None


def _render_card(title: str, subtitle: str, badge_label: str, badge_tone: str) -> None:
    st.markdown(
        f"""
        <div class="card">
            <h3>{title}</h3>
            <div class="subtitle">{subtitle}</div>
            <span class="badge status-{badge_tone}">{badge_label}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _system_overview(df: pd.DataFrame) -> None:
    last_login = _last_event(df, "login_attempt")
    if last_login is None:
        login_sub = "No login attempts recorded."
        login_badge = ("No Data", "warning")
    else:
        login_ts = _format_ts(last_login.get("timestamp"))
        login_result = str(last_login.get("result", "unknown")).lower()
        login_status = "success" if login_result == "success" else "danger"
        login_label = "Authenticated" if login_status == "success" else "Failed Attempt"
        login_sub = f"Last login at {login_ts}"
        login_badge = (login_label, login_status)

    keystroke = _last_event(df, "keystroke_anomaly")
    if keystroke is None:
        key_sub = "Awaiting monitoring data."
        key_badge = ("Normal", "success")
    else:
        score_val = keystroke.get("score")
        try:
            score = float(score_val) if pd.notna(score_val) else None
        except Exception:
            score = None
        if score is None:
            score = _parse_score(str(keystroke.get("reason", "")))
        key_status = "danger" if score is not None and score < 0 else "warning"
        key_label = "Anomaly Detected" if key_status == "danger" else "Monitor"
        score_text = f"{score:.3f}" if score is not None else "n/a"
        key_sub = f"Last score {score_text} at {_format_ts(keystroke.get('timestamp'))}"
        key_badge = (key_label, key_status)

    fraud = _last_event(df, "fraud_prediction")
    if fraud is None:
        fraud_sub = "No fraud predictions logged."
        fraud_badge = ("No Activity", "warning")
    else:
        prob = fraud.get("probability") if "probability" in fraud else fraud.get("reason", "-")
        fraud_ts = _format_ts(fraud.get("timestamp"))
        fraud_result = str(fraud.get("result", "legitimate")).lower()
        fraud_status = "danger" if fraud_result.startswith("fraud") else "success"
        fraud_label = "Fraudulent" if fraud_status == "danger" else "Legitimate"
        fraud_sub = f"Last prediction at {fraud_ts} | {prob}"
        fraud_badge = (fraud_label, fraud_status)

    col1, col2, col3 = st.columns(3, gap="medium")
    with col1:
        _render_card("Last Login", login_sub, login_badge[0], login_badge[1])
    with col2:
        _render_card("Last Keystroke Score", key_sub, key_badge[0], key_badge[1])
    with col3:
        _render_card("Last Fraud Prediction", fraud_sub, fraud_badge[0], fraud_badge[1])


def _quick_actions() -> None:
    st.markdown("### Quick Actions")
    st.markdown("Navigate directly to the core tools.")
    action_cols = st.columns(4, gap="medium")
    with action_cols[0]:
        if st.button("Go to Fraud Detection", key="nav_fraud"):
            st.switch_page("pages/fraud_detection.py")
    with action_cols[1]:
        if st.button("Go to Behavioral Monitoring", key="nav_behavior"):
            st.switch_page("pages/keystroke_monitoring.py")
    with action_cols[2]:
        if st.button("Federated Simulation", key="nav_federated"):
            st.switch_page("pages/federated_simulation.py")
    with action_cols[3]:
        if st.button("View Security Logs", key="nav_logs"):
            st.switch_page("pages/logs.py")


def _recent_events(df: pd.DataFrame, query: str = "") -> None:
    st.markdown('<div id="recent-logs"></div>', unsafe_allow_html=True)
    st.markdown("### Recent Security Events")
    st.markdown("Latest activity across authentication, biometrics, and fraud detection.")
    if df.empty:
        st.info("No security events logged yet.")
        return

    display = df.copy().sort_values("timestamp", ascending=False)
    if query:
        mask = display.apply(lambda row: row.astype(str).str.contains(query, case=False, na=False).any(), axis=1)
        display = display[mask]
        if display.empty:
            st.info("No security events match your search.")
            return
    display = display.head(5)
    display["timestamp"] = display["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    for col in ["event", "result", "reason", "severity"]:
        if col not in display:
            display[col] = "-"
        display[col] = display[col].fillna("-").astype(str)

    def _severity_style(val: str) -> str:
        tone = str(val).lower()
        if tone in ("high", "danger", "critical"):
            return f"color: {COLORS['DANGER']}; font-weight: 700;"
        if tone in ("medium", "warning", "warn"):
            return f"color: {COLORS['WARNING']}; font-weight: 700;"
        return f"color: {COLORS['MUTED']}; font-weight: 600;"

    styled = display[["timestamp", "event", "result", "reason", "severity"]].style.applymap(
        _severity_style, subset=["severity"]
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(
        page_title="Analyst Dashboard",
        page_icon="??",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    if not st.session_state.get("authenticated"):
        st.switch_page("pages/login.py")

    _inject_styles()
    logs_df = _load_logs()
    search_text = ""

    st.markdown('<div class="dashboard-shell">', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Welcome, Analyst</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Your security dashboard overview.</div>', unsafe_allow_html=True)
    monitor_result = run_passive_monitor(
        "dashboard",
        search={"label": "Search dashboard", "placeholder": "Search events, pages, statuses..."},
    )
    search_text = (monitor_result or {}).get("input_text", "").strip()
    st.markdown("<div class='section-spacer'></div>", unsafe_allow_html=True)

    _system_overview(logs_df)
    st.markdown("<div class='section-spacer'></div>", unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="section-card quick-actions">', unsafe_allow_html=True)
        _quick_actions()
        st.markdown("</div>", unsafe_allow_html=True)

    jump_to_logs = st.session_state.pop("jump_to_logs", False) or st.session_state.pop("scroll_logs_now", False)

    st.markdown("<div class='section-spacer'></div>", unsafe_allow_html=True)
    if jump_to_logs:
        st.markdown(
            """
            <script>
                const anchor = document.getElementById('recent-logs');
                if (anchor) { anchor.scrollIntoView({ behavior: 'smooth', block: 'start' }); }
            </script>
            """,
            unsafe_allow_html=True,
        )

    with st.container():
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        _recent_events(logs_df, query=search_text)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
