import streamlit as st
from models import load_disease_model, load_yield_model
from ui import render_home_page, render_yield_page, render_image_analysis_page, feedback_form

def main():
    """
    Main function to run the Streamlit application.
    It sets up the page configuration, sidebar navigation, and loads models.
    """
    # --- PAGE CONFIGURATION ---
    st.set_page_config(
        page_title="AI Agriculture Assistant",
        page_icon="🌱",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # --- LOAD MODELS (Cached for performance) ---
    image_model = load_disease_model()
    yield_model = load_yield_model()

    # --- SIDEBAR NAVIGATION ---
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", [
        "Home",
        "Crop Yield Prediction",
        "Image Analyzer"  # <-- UPDATED
    ])

    st.sidebar.markdown("---")
    st.sidebar.info(
        "This is an AI-powered assistant to help with modern farming challenges. "
        "Navigate through the pages to access different features."
    )

    # --- PAGE RENDERING ---
    if page == "Home":
        render_home_page()
        feedback_form(form_key="home_feedback")
    elif page == "Crop Yield Prediction":
        render_yield_page(yield_model)
        feedback_form(form_key="yield_feedback")
    elif page == "Image Analyzer": # <-- UPDATED
        render_image_analysis_page(image_model) # <-- UPDATED
        feedback_form(form_key="analysis_feedback")

if __name__ == "__main__":
    main()