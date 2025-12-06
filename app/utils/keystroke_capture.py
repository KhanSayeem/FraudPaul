import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import streamlit as st


TARGET_SENTENCE = "The quick brown fox jumps over the lazy dog."
MAX_EVENTS = 500


@dataclass
class KeystrokeEvent:
    key: str
    event: str
    ts: float


@dataclass
class CaptureState:
    last_text: str = ""
    last_ts: float = field(default_factory=time.time)
    events: List[KeystrokeEvent] = field(default_factory=list)


def _get_state(bucket: str) -> CaptureState:
    state = st.session_state.get(bucket)
    if not isinstance(state, CaptureState):
        st.session_state[bucket] = CaptureState()
    return st.session_state[bucket]


def reset_state(bucket: str) -> None:
    st.session_state[bucket] = CaptureState()


def update_keystrokes(current_text: str, bucket: str = "keystrokes") -> List[KeystrokeEvent]:
    """Track text changes and synthesize keydown/keyup events.

    Streamlit reruns on each text change. We infer the delta between the
    previous and current text to approximate keyDown/keyUp timings.
    """
    state = _get_state(bucket)
    now = time.time()
    previous = state.last_text

    # Added characters
    if len(current_text) > len(previous):
        added = current_text[len(previous) :]
        for ch in added:
            down_ts = now
            up_ts = now + 0.02  # minimal hold approximation
            state.events.append(KeystrokeEvent(key=ch, event="down", ts=down_ts))
            state.events.append(KeystrokeEvent(key=ch, event="up", ts=up_ts))

    # Removed characters (treat as backspace activity)
    elif len(current_text) < len(previous):
        state.events.append(KeystrokeEvent(key="backspace", event="down", ts=now))
        state.events.append(KeystrokeEvent(key="backspace", event="up", ts=now + 0.02))

    if len(state.events) > MAX_EVENTS:
        state.events = state.events[-MAX_EVENTS:]

    state.last_text = current_text
    state.last_ts = now
    return state.events


def validate_sentence(user_input: str, sentence: str = TARGET_SENTENCE) -> bool:
    return user_input.strip() == sentence


def summarize_attempt(events: List[KeystrokeEvent]) -> Optional[Dict]:
    if not events:
        return None
    return {
        "events": [e.__dict__ for e in events],
        "captured_at": time.time(),
    }
