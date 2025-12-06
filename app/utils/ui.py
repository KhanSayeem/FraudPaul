import streamlit as st

COLORS = {
    "WHITE": "#FFFFFF",
    "BLACK": "#000000",
    "SLATE": "#6B7280",
    "BORDER": "#E5E7EB",
    "SUCCESS": "#16A34A",
    "WARNING": "#F59E0B",
    "DANGER": "#DC2626",
}


def inject_styles() -> None:
    st.markdown(
        f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
            html, body, [class*="css"] {{
                font-family: 'Inter', 'Roboto', sans-serif;
                background: {COLORS['WHITE']};
                color: {COLORS['BLACK']};
            }}
            .page-shell {{
                max-width: 760px;
                margin: 0 auto;
                padding: 24px 0 32px 0;
            }}
            .page-title {{
                font-size: 32px;
                font-weight: 600;
                text-align: center;
                color: {COLORS['BLACK']};
            }}
            .page-subtitle {{
                font-size: 16px;
                font-weight: 500;
                text-align: center;
                color: {COLORS['SLATE']};
                margin-top: 6px;
            }}
            .card-frame {{
                border: 1px solid {COLORS['BORDER']};
                padding: 28px;
                background: {COLORS['WHITE']};
                border-radius: 0px;
                box-shadow: none;
            }}
            .stTextInput input, .stTextArea textarea {{
                border: 1px solid {COLORS['BORDER']} !important;
                border-radius: 0px !important;
                background: {COLORS['WHITE']} !important;
                color: {COLORS['BLACK']} !important;
                font-size: 16px !important;
            }}
            .stButton button, .stDownloadButton button, .stLinkButton button {{
                border: 1.5px solid {COLORS['BLACK']};
                background: {COLORS['WHITE']};
                color: {COLORS['BLACK']};
                font-weight: 600;
                border-radius: 0px;
                padding: 12px 16px;
            }}
            .stButton button:hover, .stDownloadButton button:hover, .stLinkButton button:hover {{
                border-color: {COLORS['BLACK']};
                background: #F3F4F6;
            }}
            .badge {{
                display: inline-block;
                padding: 6px 10px;
                font-weight: 600;
                font-size: 13px;
                border-radius: 0px;
                border: 1px solid {COLORS['BORDER']};
                color: {COLORS['SLATE']};
                background: {COLORS['WHITE']};
            }}
            .badge.success {{ border-color:{COLORS['SUCCESS']}; color:{COLORS['SUCCESS']}; }}
            .badge.danger {{ border-color:{COLORS['DANGER']}; color:{COLORS['DANGER']}; }}
            .badge.warning {{ border-color:{COLORS['WARNING']}; color:{COLORS['WARNING']}; }}
            .progress-track {{
                width: 100%;
                height: 8px;
                background: {COLORS['BORDER']};
                border-radius: 0px;
                overflow: hidden;
            }}
            .progress-fill {{
                height: 100%;
                background: {COLORS['SUCCESS']};
                transition: width 0.25s ease;
            }}
            .divider {{
                height: 1px;
                background: {COLORS['BORDER']};
                margin: 24px 0;
            }}
            table tbody tr td {{
                border-bottom: 1px solid {COLORS['BORDER']};
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def badge(label: str, tone: str = "neutral") -> str:
    tone_class = {
        "success": "success",
        "danger": "danger",
        "warning": "warning",
        "neutral": "",
    }.get(tone, "")
    return f"<span class='badge {tone_class}'>{label}</span>"
