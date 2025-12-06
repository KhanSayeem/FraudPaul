import csv
import pickle
import time
from datetime import datetime
from pathlib import Path

import cv2
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "face_model.yml"
LABELS_PATH = BASE_DIR / "models" / "face_labels.pkl"
LOG_PATH = BASE_DIR / "logs" / "security_logs.csv"

CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

SUCCESS = "#16A34A"
ERROR = "#DC2626"
THEMES = {
    "dark": {
        "DOMINANT": "#0D1117",
        "ACCENT": "#E5E7EB",
        "SLATE": "#9CA3AF",
        "BORDER": "#1F2937",
        "SKELETON_BG": "#111827",
    },
    "light": {
        "DOMINANT": "#FFFFFF",
        "ACCENT": "#0F172A",
        "SLATE": "#4B5563",
        "BORDER": "#E5E7EB",
        "SKELETON_BG": "#F1F5F9",
    },
}


def palette() -> dict:
    theme = st.session_state.get("theme", "dark")
    return THEMES.get(theme, THEMES["dark"])


@st.cache_resource(show_spinner=False)
def load_model():
    model = cv2.face.LBPHFaceRecognizer_create()
    model.read(str(MODEL_PATH))
    with open(LABELS_PATH, "rb") as f:
        labels = pickle.load(f)
    return model, labels


def ensure_session_defaults() -> None:
    defaults = {
        "authenticated": False,
        "face_status": "neutral",
        "liveness_status": "neutral",
        "login_status_state": "neutral",
        "login_status_text": "Awaiting Scan.",
        "is_streaming": False,
        "scan_requested": False,
        "theme": "dark",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def log_event(result: str, reason: str, severity: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    exists = LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["timestamp", "event", "result", "reason", "severity"])
        writer.writerow(
            [
                datetime.utcnow().isoformat(),
                "login_attempt",
                result,
                reason,
                severity,
            ]
        )


def inject_styles() -> None:
    colors = palette()
    st.markdown(
        f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
            html, body, [class*="css"] {{
                font-family: 'Inter', 'Roboto', sans-serif;
                color: {colors["ACCENT"]};
                background: {colors["DOMINANT"]};
            }}
            .page-title {{
                font-size: 32px;
                font-weight: 600;
                color: {colors["ACCENT"]};
                text-align: center;
            }}
            .page-subtitle {{
                font-size: 16px;
                color: {colors["SLATE"]};
                text-align: center;
                margin-top: 4px;
            }}
            .login-card-marker {{
                display: none;
            }}
            div:has(> .login-card-marker) {{
                border: none;
                border-radius: 0;
                padding: 0;
                background: transparent;
            }}
            .status-badges {{
                display: flex;
                gap: 12px;
                flex-wrap: wrap;
                margin-top: 12px;
            }}
            .status-badge {{
                font-size: 14px;
                font-weight: 500;
                padding: 6px 10px;
                border-radius: 4px;
                border: 1px solid {colors["BORDER"]};
                background: {colors["DOMINANT"]};
                color: {colors["SLATE"]};
            }}
            .btn-primary button {{
                background: {colors["DOMINANT"]};
                color: {colors["ACCENT"]};
                border: 1.5px solid {colors["ACCENT"]};
                border-radius: 4px;
                padding: 10px 16px;
                font-weight: 600;
            }}
            .btn-primary button:hover {{
                border-color: {colors["ACCENT"]};
                background: {colors["SKELETON_BG"]};
            }}
            .webcam-shell {{
                width: 640px;
                max-width: 100%;
                height: 480px;
                border: 1px solid {colors["BORDER"]};
                border-radius: 4px;
                background: {colors["SKELETON_BG"]};
                display: flex;
                align-items: center;
                justify-content: center;
                color: {colors["SLATE"]};
                font-size: 14px;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def badge(label: str, status: str) -> str:
    colors = palette()
    color_map = {
        "success": (SUCCESS, colors["DOMINANT"], SUCCESS),
        "error": (ERROR, colors["DOMINANT"], ERROR),
        "neutral": (colors["SLATE"], colors["DOMINANT"], colors["BORDER"]),
        "pending": (colors["SLATE"], colors["DOMINANT"], colors["BORDER"]),
    }
    fg, bg, border = color_map.get(status, (colors["SLATE"], colors["DOMINANT"], colors["BORDER"]))
    return (
        f"<span class='status-badge' style='color:{fg}; background:{bg};"
        f" border-color:{border};'>{label}</span>"
    )


def run_auth_flow(model, preview=None) -> None:
    st.session_state["is_streaming"] = True
    st.session_state["login_status_state"] = "pending"
    st.session_state["login_status_text"] = "Authenticating."
    st.session_state["face_status"] = "pending"
    st.session_state["liveness_status"] = "pending"

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        st.session_state["login_status_state"] = "error"
        st.session_state["login_status_text"] = "Access Denied"
        st.session_state["is_streaming"] = False
        log_event("failure", "camera_unavailable", "Danger")
        st.error("Could not access webcam.")
        return

    time.sleep(0.5)

    positions = []
    recognized = False
    liveness_passed = False
    max_frames = 400
    movement_threshold = 20

    for _ in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = CASCADE.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))

        if preview is not None:
            preview.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            face_roi = gray[y : y + h, x : x + w]
            face_roi = cv2.resize(face_roi, (200, 200))

            try:
                label_id, confidence = model.predict(face_roi)
                if confidence < 70:
                    recognized = True
                    st.session_state["face_status"] = "success"
                else:
                    st.session_state["face_status"] = "pending"
            except Exception:
                st.session_state["face_status"] = "error"

            positions.append(x)
            if len(positions) > 30:
                positions.pop(0)

            if recognized and len(positions) >= 6:
                movement_range = max(positions) - min(positions)
                if movement_range > movement_threshold:
                    liveness_passed = True
                    st.session_state["liveness_status"] = "success"
                    st.success("Liveness Passed")
                    break
            elif recognized:
                st.session_state["liveness_status"] = "pending"
        else:
            st.session_state["face_status"] = "pending" if recognized else "neutral"

        time.sleep(0.04)

    cap.release()
    if preview is not None:
        preview.empty()
    st.session_state["is_streaming"] = False
    st.session_state["scan_requested"] = False

    if recognized and liveness_passed:
        st.session_state["authenticated"] = True
        st.session_state["login_status_state"] = "success"
        st.session_state["login_status_text"] = "Access Granted"
        log_event("success", "passed", "Info")
        time.sleep(0.4)
        st.switch_page("pages/dashboard.py")
        return

    if not recognized:
        st.session_state["login_status_state"] = "error"
        st.session_state["login_status_text"] = "Access Denied"
        st.session_state["face_status"] = "error"
        log_event("failure", "face mismatch", "Danger")
        st.error("Face not recognized. Please try again.")
    elif not liveness_passed:
        st.session_state["login_status_state"] = "error"
        st.session_state["login_status_text"] = "Access Denied"
        st.session_state["liveness_status"] = "error"
        log_event("failure", "liveness failed", "Warning")
        st.error("Liveness check failed. Move your head left/right and retry.")


def render_status_row():
    colors = palette()
    face_badge = badge(
        "Face Recognized",
        st.session_state.get("face_status", "neutral"),
    )
    live_badge = badge(
        "Liveness Check",
        st.session_state.get("liveness_status", "neutral"),
    )
    status_state = st.session_state.get("login_status_state", "neutral")
    status_color = (
        SUCCESS
        if status_state == "success"
        else ERROR if status_state == "error" else colors["SLATE"]
    )
    status_text = st.session_state.get("login_status_text", "Awaiting Scan.")
    status_badge = (
        f"<span class='status-badge' style='color:{status_color}; background:{colors['DOMINANT']};'>"
        f"Login Status: {status_text}</span>"
    )

    st.markdown(
        f"<div class='status-badges'>{face_badge}{live_badge}{status_badge}</div>",
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Secure Facial Login",
        page_icon="🔒",
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    ensure_session_defaults()
    theme_toggle = st.toggle(
        "White mode",
        value=st.session_state.get("theme") == "light",
        help="Switch between dark and white modes",
    )
    st.session_state["theme"] = "light" if theme_toggle else "dark"
    inject_styles()

    if st.session_state.get("authenticated"):
        st.switch_page("pages/dashboard.py")

    st.markdown('<div class="page-title">Secure Facial Login</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Please verify your identity to continue</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)

    card = st.container()
    with card:
        st.markdown('<div class="login-card-marker"></div>', unsafe_allow_html=True)
        button_container = st.container()
        with button_container:
            trigger = st.button("Scan Face", use_container_width=True, key="scan_btn")

        render_status_row()

    if trigger:
        st.session_state["scan_requested"] = True

    if st.session_state["scan_requested"]:
        try:
            model, _ = load_model()
        except Exception:
            st.session_state["login_status_state"] = "error"
            st.session_state["login_status_text"] = "Access Denied"
            st.error("Face model or labels could not be loaded.")
            log_event("failure", "model_unavailable", "Danger")
            return
        with st.spinner("Starting camera and scanning..."):
            preview_slot = st.empty()
            run_auth_flow(model, preview_slot)


if __name__ == "__main__":
    main()
