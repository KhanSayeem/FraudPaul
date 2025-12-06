PRD — Semi-Continuous Keystroke Monitoring (Option B)
Module Phase: C+ (Enhancement)
Prepared for: Codex
UX Style: Streamlit Native + Custom HTML/CSS
Goal: Enable passive keystroke anomaly detection across the app, without JS-heavy background listeners.
🎯 1. Purpose

This enhancement makes the behavioral biometric system usable in real-world conditions by enabling keystroke monitoring whenever the user interacts with specific input fields across multiple pages — not only inside the monitoring page.

Monitoring becomes:

Passive

Automatic

Seamless

Session-aware

Examples of where monitoring should be active:

Dashboard search bar

Fraud Detection filters

Any text input box

Hidden/passive input field printed on every page

🧱 2. Architecture Overview

Semi-continuous monitoring works by:

✔ Adding a “hidden monitoring textarea” to every main page

This is a minimal, visually invisible element that allows capturing keystrokes.

✔ Streaming keystrokes to a global monitoring function

Feature extraction occurs whenever the user types.

✔ Triggering anomaly prediction automatically

Your One-Class SVM (keystroke_svm.pkl) receives the extracted features.

✔ Logging anomalies silently

Detected anomalies are stored in security_logs.csv.

✔ Displaying analytics only inside the Monitoring Page

This page becomes the observer, not the trigger.

🎨 3. UI Requirements
3.1 Hidden Input Listener (global)

Codex must place this in every user-accessible page:

st.text_input(
    label="",
    key="keystroke_listener",
    placeholder="",
    label_visibility="collapsed"
)


Then hide it with CSS:

st.markdown("""
<style>
input[type="text"][id^="keystroke_listener"] {
    opacity: 0;
    height: 0px;
    width: 0px;
    position: absolute;
    z-index: -1;
}
</style>
""", unsafe_allow_html=True)


This allows keystroke collection every time the user types ANYTHING.

If the user types:

into a form

into a search bar

into this hidden listener
→ keystroke timestamps are captured and passed to the monitoring engine.

3.2 No UI Disruption

The hidden input must:

Not shift UI layout

Not display a cursor

Not visually appear

Not affect user typing flow

🧠 4. Functional Requirements
4.1 Global Keystroke Capture

Codex must implement a global hook that:

Detects every change in st.session_state["keystroke_listener"]

Extracts timing metadata using timestamp deltas

Builds a feature vector identical to the enrollment process

Simplified pseudocode:
if "last_key_time" not in st.session_state:
    st.session_state.last_key_time = None

typed = st.session_state.get("keystroke_listener", "")

current_time = time.time()

if st.session_state.last_key_time:
    flight_time = current_time - st.session_state.last_key_time
    # add flight_time to buffer

st.session_state.last_key_time = current_time


Codex already has code from enrollment → reuse.

4.2 Automatic Anomaly Detection

Whenever enough keystrokes accumulate (e.g., 10+):

Extract features

Load One-Class SVM

Compute anomaly score

Evaluate risk

Log if above threshold

Reset buffer

4.3 Silent Logging

Logs must append silently:

timestamp, score, status, source_page


Where:

status = Normal / Warning / Anomaly Detected

source_page = page where anomaly happened

These logs appear later in:

Continuous Monitoring Page

Security Logs Page

📊 5. Continuous Monitoring Page (View Only)

This page should no longer be used for typing input.

Instead, it becomes:

✔ Analytics dashboard
✔ Trend charts
✔ Recent anomaly table
✔ Current session status
✔ A read-only view into the monitoring engine

No input fields needed except optional demonstration.

📂 6. File Structure Requirements
app/
  pages/
    dashboard.py
    fraud_detection.py
    keystroke_monitoring.py  <-- becomes analytics view only
  utils/
    keystroke_capture.py     <-- new
    keystroke_model.py       <-- existing logic extended
  logs/
    security_logs.csv
  models/
    keystroke_svm.pkl
    keystroke_profile.json

🎯 7. Success Criteria

Semi-continuous monitoring is successful when:

✔ User typing on any major page triggers monitoring
✔ Anomaly scores are computed without user action
✔ Logs are updated silently
✔ Monitoring page displays rolling anomaly history
✔ No UI disruption or visible widgets
✔ No JavaScript required (Streamlit-native only)
✔ System still feels clean, minimal, professional

This gives the user a passive security layer, which is ideal for your research.