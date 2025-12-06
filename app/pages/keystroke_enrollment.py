import csv
from datetime import datetime
from pathlib import Path
from typing import List

import streamlit as st

from app.utils.keystroke_capture import (
    TARGET_SENTENCE,
    KeystrokeEvent,
    reset_state,
    summarize_attempt,
    update_keystrokes,
    validate_sentence,
)
from app.utils.keystroke_features import build_feature_vector, stack_samples
from app.utils.keystroke_model import PROFILE_PATH, SVM_PATH, save_model, save_profile, train_svm


LOG_PATH = Path(__file__).resolve().parents[1] / "logs" / "security_logs.csv"
REQUIRED_SAMPLES = 5


def _init_state() -> None:
    defaults = {
        "enroll_text": "",
        "enroll_samples": [],
        "enroll_history": [],
        "model_trained": False,
        "reset_enroll_text": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _log_event(result: str, reason: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    exists = LOG_PATH.exists()
    with LOG_PATH.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["timestamp", "event", "result", "reason", "severity"])
        writer.writerow(
            [
                datetime.utcnow().isoformat(),
                "keystroke_enrollment",
                result,
                reason,
                "Info" if result == "success" else "Warning",
            ]
        )


def main() -> None:
    st.set_page_config(
        page_title="Keystroke Enrollment",
        page_icon="⌨️",
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    _init_state()

    if st.session_state.get("reset_enroll_text"):
        st.session_state["enroll_text"] = ""
        st.session_state["reset_enroll_text"] = False

    if not st.session_state.get("authenticated"):
        st.warning("You are not authenticated. Redirecting to login.")
        st.switch_page("pages/login.py")

    st.title("Behavioral Biometrics Enrollment")
    st.caption("Type the sentence below 5 times to build your typing profile.")

    st.subheader("Enrollment prompt")
    st.markdown(f"**{TARGET_SENTENCE}**")
    st.write("Keystrokes are captured automatically. Press Enter or click record after each attempt.")

    user_text = st.text_input(
        "Type the sentence",
        key="enroll_text",
        placeholder="Start typing here...",
    )
    events: List[KeystrokeEvent] = update_keystrokes(user_text, bucket="enroll_events")

    submitted = st.button("Record sample", key="record_sample", help="Save this attempt")

    if submitted:
        if not validate_sentence(user_text):
            st.session_state["enroll_events"] = events
            st.error("Sentence mismatch. Please retype exactly.")
            _log_event("failure", "sentence_mismatch")
        else:
            feature_vector = build_feature_vector(events)
            payload = {
                "feature": feature_vector.tolist(),
                "raw": summarize_attempt(events),
                "typed": user_text,
            }
            st.session_state["enroll_samples"].append(feature_vector)
            st.session_state["enroll_history"].append(payload)
            st.session_state["reset_enroll_text"] = True
            reset_state("enroll_events")
            st.success(f"Sample {len(st.session_state['enroll_samples'])} of {REQUIRED_SAMPLES} recorded.")
            _log_event("success", "sample_recorded")

    progress_fraction = len(st.session_state["enroll_samples"]) / REQUIRED_SAMPLES
    st.progress(progress_fraction)

    st.subheader("Status")
    if len(st.session_state["enroll_samples"]) >= REQUIRED_SAMPLES:
        if not st.session_state["model_trained"]:
            feature_matrix = stack_samples(st.session_state["enroll_samples"])
            model = train_svm(feature_matrix)
            save_model(model)
            save_profile(st.session_state["enroll_history"])
            st.session_state["model_trained"] = True
            st.success("Enrollment completed and model trained.")
            _log_event("success", "model_trained")

        st.info(f"Model saved: {SVM_PATH.name} • Profile saved: {PROFILE_PATH.name}")
        if st.button("Proceed to Monitoring", key="go_monitor"):
            st.switch_page("pages/keystroke_monitoring.py")
    else:
        remaining = REQUIRED_SAMPLES - len(st.session_state["enroll_samples"])
        st.warning(f"{remaining} more sample(s) required.")


if __name__ == "__main__":
    main()
