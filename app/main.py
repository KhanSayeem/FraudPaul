import streamlit as st
from streamlit.runtime import exists as runtime_exists


def _ensure_session_defaults() -> None:
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False


def main() -> None:
    if not runtime_exists():
        print("Run this app with: streamlit run app/main.py")
        return

    st.set_page_config(
        page_title="Secure Facial Login",
        page_icon="🔒",
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    _ensure_session_defaults()

    if st.session_state.get("authenticated"):
        st.switch_page("pages/dashboard.py")
    else:
        st.switch_page("pages/login.py")


if __name__ == "__main__":
    main()
