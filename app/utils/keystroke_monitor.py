import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

from app.utils.keystroke_capture import KeystrokeEvent, reset_state, update_keystrokes
from app.utils.keystroke_features import build_feature_vector
from app.utils.keystroke_model import PROFILE_PATH, SVM_PATH, load_model, score_sample


LOG_PATH = Path(__file__).resolve().parents[1] / "logs" / "security_logs.csv"
MIN_KEYS = 10
HISTORY_LIMIT = 200


@st.cache_resource(show_spinner=False)
def _cached_model():
    return load_model()


def _ensure_monitor_state() -> None:
    defaults = {
        "keystroke_listener": "",
        "global_anomaly_history": [],
        "global_monitor_status": "Normal",
        "global_monitor_score": None,
        "global_monitor_error": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _inject_listener(hidden: bool = True, label: str = "", placeholder: str = "") -> str:
    """Render the listener. Hidden by default, but can be surfaced as a search box."""
    if hidden and not st.session_state.get("_listener_css_injected"):
        st.session_state["_listener_css_injected"] = True
        st.markdown(
            """
            <style>
            /* Push the hidden listener off-screen and remove all footprint */
            input#keystroke_listener {
                position: fixed !important;
                top: -10000px !important;
                left: -10000px !important;
                opacity: 0 !important;
                height: 1px !important;
                width: 1px !important;
                padding: 0 !important;
                margin: 0 !important;
                border: 0 !important;
                box-shadow: none !important;
                outline: none !important;
            }
            /* Collapse the wrapper that Streamlit renders around the input */
            div[data-testid="stTextInput"]:has(input#keystroke_listener),
            div[data-testid="stTextInputRootElement"]:has(input#keystroke_listener) {
                max-height: 0 !important;
                height: 0 !important;
                min-height: 0 !important;
                padding: 0 !important;
                margin: 0 !important;
                overflow: hidden !important;
            }
            </style>
            <script>
            const listener = document.getElementById("keystroke_listener");
            if (listener) {
                const wrapper = listener.closest('div[data-testid="stTextInput"]') || listener.parentElement;
                if (wrapper) {
                    wrapper.style.maxHeight = "0";
                    wrapper.style.height = "0";
                    wrapper.style.overflow = "hidden";
                    wrapper.style.margin = "0";
                    wrapper.style.padding = "0";
                }
                // Keep focus to capture keystrokes without showing the control.
                listener.focus({ preventScroll: true });
            }
            </script>
            """,
            unsafe_allow_html=True,
        )
    # Clear pending reset before widget instantiation to satisfy Streamlit rules.
    if st.session_state.pop("_clear_keystroke_listener", False):
        st.session_state["keystroke_listener"] = ""
    return st.text_input(
        label=label,
        key="keystroke_listener",
        placeholder=placeholder,
        label_visibility="visible" if not hidden else "collapsed",
    )


def _status_from_score(score: float) -> Dict[str, str]:
    if score < -0.05:
        return {"status": "Anomaly Detected", "severity": "High"}
    if score < 0:
        return {"status": "Warning", "severity": "Warning"}
    return {"status": "Normal", "severity": "Info"}


def _log_anomaly(score: float, status: str, severity: str, source_page: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    exists = LOG_PATH.exists()
    with LOG_PATH.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(
                [
                    "timestamp",
                    "event",
                    "result",
                    "reason",
                    "severity",
                    "score",
                    "source_page",
                    "status",
                ]
            )
        writer.writerow(
            [
                datetime.utcnow().isoformat(),
                "keystroke_anomaly",
                status.lower(),
                f"score={score:.4f}",
                severity,
                f"{score:.4f}",
                source_page,
                status,
            ]
        )


def _record_history(score: float, status: str, source_page: str) -> None:
    entry = {
        "timestamp": datetime.utcnow(),
        "score": score,
        "status": status,
        "page": source_page,
    }
    st.session_state["global_anomaly_history"].append(entry)
    if len(st.session_state["global_anomaly_history"]) > HISTORY_LIMIT:
        st.session_state["global_anomaly_history"] = st.session_state["global_anomaly_history"][-HISTORY_LIMIT:]


def _reset_buffer() -> None:
    reset_state("global_keystrokes")
    st.session_state["_clear_keystroke_listener"] = True


def run_passive_monitor(page_name: str, enabled: bool = True, search: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
    """Install listener (hidden or visible search box), score keystrokes, and log anomalies."""
    _ensure_monitor_state()
    listener_value = _inject_listener(
        hidden=search is None,
        label=search.get("label", "") if search else "",
        placeholder=search.get("placeholder", "") if search else "",
    )

    result: Dict[str, Any] = {"input_text": listener_value}

    if not enabled:
        return result

    events: List[KeystrokeEvent] = update_keystrokes(listener_value, bucket="global_keystrokes")
    down_count = len([e for e in events if e.event == "down"])
    if down_count < MIN_KEYS:
        return result

    try:
        model = _cached_model()
    except Exception as exc:  # model not trained yet
        st.session_state["global_monitor_error"] = str(exc)
        _reset_buffer()
        result["error"] = str(exc)
        return result

    vector = build_feature_vector(events)
    score = score_sample(model, vector)
    status_meta = _status_from_score(score)
    st.session_state["global_monitor_error"] = None
    st.session_state["global_monitor_status"] = status_meta["status"]
    st.session_state["global_monitor_score"] = score

    _record_history(score, status_meta["status"], page_name)
    if status_meta["status"] != "Normal":
        _log_anomaly(score, status_meta["status"], status_meta["severity"], page_name)

    _reset_buffer()
    result.update(
        {
            "score": score,
            "status": status_meta["status"],
            "severity": status_meta["severity"],
            "page": page_name,
            "model": SVM_PATH.name,
            "profile": PROFILE_PATH.name,
        }
    )
    return result
