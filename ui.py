import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import plotly.graph_objects as go
from utils import save_feedback
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import datetime
import os

def render_home_page():
    """Renders the Home/About page content."""
    st.title("Welcome to the AI Agriculture Assistant 🌱")
    st.markdown("""
        This application is your smart farming partner, designed to empower farmers and agricultural enthusiasts
        with cutting-edge AI technology. Our goal is to enhance crop management, increase yield, and promote
        sustainable farming practices.
    """)
    st.subheader("What This App Offers")
    col1, col2 = st.columns(2)
    with col1:
        st.info("**🌾 Crop Yield Prediction**")
        st.write("Leverage machine learning to forecast crop yields based on historical data, weather patterns, and farming inputs.")
    with col2:
        st.info("**🌿 Image Analyzer**")
        st.write("Upload an image of a plant leaf or any object to get an instant identification from a general-purpose AI model.")

def render_yield_page(yield_model):
    """Renders the Crop Yield Prediction page."""
    st.title("🌾 Crop Yield Prediction")
    st.markdown("Fill in the details below to predict the crop yield.")
    if yield_model is None:
        st.error("The yield prediction model is not available. Please check the application setup.")
        return
    df_demo = pd.DataFrame({
        'Area': ['India', 'USA', 'China', 'Brazil', 'Nigeria'],
        'Item': ['Maize', 'Wheat', 'Rice, paddy', 'Potatoes', 'Soybeans'],
    })
    with st.form("yield_prediction_form"):
        area = st.selectbox("Area", options=df_demo['Area'].unique())
        item = st.selectbox("Crop", options=df_demo['Item'].unique())
        year = st.number_input("Year", min_value=2000, max_value=2050, value=datetime.date.today().year)
        avg_rainfall = st.number_input("Average Rainfall (mm/year)", min_value=0.0, value=1000.0)
        pesticides = st.number_input("Pesticides (tonnes)", min_value=0.0, value=1500.0)
        avg_temp = st.number_input("Average Temperature (°C)", min_value=-20.0, max_value=50.0, value=25.0)
        submit_button = st.form_submit_button("Predict Yield")
    if submit_button:
        input_data = pd.DataFrame(columns=yield_model.feature_names_in_)
        input_data.loc[0] = 0
        input_data['Year'] = year
        input_data['average_rain_fall_mm_per_year'] = avg_rainfall
        input_data['pesticides_tonnes'] = pesticides
        input_data['avg_temp'] = avg_temp
        if f'Area_{area}' in input_data.columns:
            input_data[f'Area_{area}'] = 1
        if f'Item_{item}' in input_data.columns:
            input_data[f'Item_{item}'] = 1
        prediction = yield_model.predict(input_data)[0]
        st.success(f"**Predicted Yield: {prediction:,.2f} hg/ha**")

def render_image_analysis_page(disease_model):
    """Renders the Image Analysis page with an improved layout and explanations."""
    st.title("🌿 Image Analyzer")
    st.markdown("Upload a clear image to identify the object within it.")

    st.info("""
        **Please Note:** This page uses a powerful, general-purpose image recognition model (MobileNetV2) trained by Google.
        It can identify over 1,000 common objects but is **not** trained to recognize specific plant diseases.
        The original custom disease model could not be loaded due to a file error.
    """, icon="ℹ️")

    if disease_model is None:
        st.error("The image analysis model is not available. Please check application setup.")
        return

    col1, col2 = st.columns([1, 1])

    with col1:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image", use_container_width=True)

    if uploaded_file:
        with col2:
            st.subheader("Analysis Results")
            if st.button("Analyze Image", use_container_width=True):
                with st.spinner("🧠 Analyzing..."):
                    img_array = np.array(image.resize((224, 224)))
                    img_array = np.expand_dims(img_array, axis=0)
                    img_array = preprocess_input(img_array)

                    prediction = disease_model.predict(img_array)
                    decoded_preds = decode_predictions(prediction, top=3)[0]

                st.success("**Analysis Complete!**")

                # Display top prediction prominently
                top_label = decoded_preds[0][1].replace('_', ' ').title()
                top_score = f"{decoded_preds[0][2]:.1%}"
                st.metric(label="Top Prediction", value=top_label, delta=top_score)
                
                # Display other predictions in an expander
                with st.expander("See other predictions"):
                    for _, label, score in decoded_preds[1:]:
                        st.write(f"**{label.replace('_', ' ').title()}**: {score:.2%}")

def feedback_form(form_key):
    """Creates a feedback form in an expander."""
    with st.expander("📝 Provide Feedback"):
        with st.form(key=form_key, clear_on_submit=True):
            comments = st.text_area("Your comments or suggestions:")
            submitted = st.form_submit_button("Submit")
            if submitted and comments:
                save_feedback("N/A", "N/A", comments, os.path.join('data', 'feedback.csv'))
                st.success("Thank you for your feedback!")

