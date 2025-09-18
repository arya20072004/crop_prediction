# AI Agriculture Assistant - Streamlit Web App
# To run this app: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image
import plotly.graph_objects as go
from fpdf import FPDF
import datetime
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Agriculture Assistant",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- ASSETS & PLACEHOLDERS ---
#
# --- CRITICAL FIX ---
# The loaded model was trained on 19 classes, but the previous list had 38.
# This list MUST be updated to contain the exact 19 class names from your training data.
DISEASE_CLASSES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Tomato___Septoria_leaf_spot'
]


# --- BACK-END LOGIC ---

def create_disease_model():
    """
    FIX: Recreates the model architecture exactly as it was in the training notebook.
    This is the most robust way to avoid version-related loading errors.
    """
    # Define the input layer
    inputs = tf.keras.Input(shape=(224, 224, 3))

    # Replicate the preprocessing step used in the notebook
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

    # Load the base MobileNetV2 model with pre-trained ImageNet weights
    base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3),
                                                 include_top=False,
                                                 weights='imagenet')
    # Freeze the base model as it was during initial training
    base_model.trainable = False

    # Connect the layers
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(len(DISEASE_CLASSES), activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    return model

@st.cache_resource
def load_disease_model(model_path):
    """
    FIX: Creates a clean model architecture and then loads only the weights
    from the specified .h5 file. This bypasses the problematic architecture
    data stored in the file.
    """
    try:
        # Create a fresh, clean instance of the model architecture
        model = create_disease_model()
        # Load the trained weights from your file into this new model
        model.load_weights(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model weights from {model_path}: {e}")
        st.error("Please ensure the model file is not corrupted and that the architecture in `app.py` perfectly matches your training notebook.")
        return None

@st.cache_data
def load_and_train_yield_model(data_path='yield_df.csv'):
    """Loads crop yield data and trains a RandomForestRegressor model."""
    try:
        df = pd.read_csv(data_path)
        # Prepare data for the model
        X = df[['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']]
        y = df['hg/ha_yield']
        
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        return model
    except FileNotFoundError:
        st.error(f"Error: The data file '{data_path}' was not found. Please ensure it's in the same directory as 'app.py'.")
        return None
    except Exception as e:
        st.error(f"An error occurred while training the yield model: {e}")
        return None

def preprocess_image(image, target_size=(224, 224)):
    """Preprocesses an image for model prediction."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0) # Add batch dimension
    return image_array

class PDF(FPDF):
    """Custom PDF class to create headers and footers."""
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'AI Agriculture Assistant - Prediction Report', 0, 1, 'C')

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_pdf_report(prediction_data):
    """Generates a downloadable PDF report for crop prediction."""
    pdf = PDF()
    pdf.add_page()
    pdf.set_font('Arial', '', 12)
    
    pdf.cell(0, 10, f"Report Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1)
    pdf.ln(10)

    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Input Parameters', 0, 1)
    pdf.set_font('Arial', '', 12)
    for key, value in prediction_data['inputs'].items():
        pdf.cell(0, 10, f"- {key.replace('_', ' ').title()}: {value}", 0, 1)
    
    pdf.ln(10)
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Prediction Result', 0, 1)
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, f"- Predicted Yield: {prediction_data['output']['yield']}", 0, 1)

    return pdf.output(dest='S').encode('latin-1')

def save_feedback(name, email, comments):
    """Saves user feedback to a CSV file."""
    feedback_file = 'feedback.csv'
    new_feedback = {
        'timestamp': [datetime.datetime.now()],
        'name': [name],
        'email': [email],
        'comments': [comments]
    }
    df = pd.DataFrame(new_feedback)
    
    try:
        if not os.path.exists(feedback_file):
            df.to_csv(feedback_file, index=False)
        else:
            df.to_csv(feedback_file, mode='a', header=False, index=False)
        return True
    except Exception:
        return False

# --- UI & THEME ---
def apply_custom_theme():
    """Applies custom CSS for theming."""
    custom_css = """
    <style>
        .stApp {
            background-color: #000000;
            color: #ffffff;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #ffffff;
        }
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #28a745; /* Green for header */
        }
        .card {
            background-color: #1a1a1a; /* Dark gray for cards */
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            transition: transform 0.2s;
            border: 1px solid #28a745;
        }
        .card:hover {
            transform: scale(1.02);
        }
        .form-container {
             background-color: #1a1a1a; /* Dark gray for forms */
             border-radius: 10px;
             padding: 25px;
             box-shadow: 0 4px 8px rgba(0,0,0,0.1);
             border: 1px solid #28a745;
        }
        div[data-testid="stSidebar"] {
            background-color: #1a1a1a;
        }
        .stButton>button {
            background-color: #28a745;
            color: white;
            border-radius: 5px;
            border: none;
            padding: 10px 20px;
        }
        .stButton>button:hover {
            background-color: #218838;
            color: white;
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

# --- MAIN APPLICATION ---
def main():
    apply_custom_theme()
    
    with st.sidebar:
        st.markdown("<h1 style='text-align: center; color: #28a745;'>Agri-AI</h1>", unsafe_allow_html=True)
        
        menu = ["Home", "🌱 Crop Prediction", "🦠 Disease Detection", "About"]
        choice = st.radio("Navigation", menu, label_visibility="collapsed")
        
        st.markdown("---")
        st.info("Your smart farming partner. Get data-driven insights to boost yield and protect crops.")
    
    if choice == "Home":
        render_home_page()
    elif choice == "🌱 Crop Prediction":
        render_crop_prediction_page()
    elif choice == "🦠 Disease Detection":
        render_disease_detection_page()
    elif choice == "About":
        render_about_page()

# --- PAGE RENDERING FUNCTIONS ---
def render_home_page():
    st.markdown("<p class='main-header'>AI Agriculture Assistant</p>", unsafe_allow_html=True)
    st.markdown("#### Smart Farming Made Simple")
    st.write("Welcome! This tool leverages AI to help you make data-driven decisions, increase yields, and protect your crops from diseases.")
    
    if os.path.exists("image_8afd94.jpg"):
        st.image("image_8afd94.jpg", use_container_width=True, caption="Empowering Agriculture with Technology")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="card">
            <h3>🌾 Crop Yield Prediction</h3>
            <p>Enter your farm's data—weather conditions and management practices—to get an estimate of your future crop yield.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <h3>🦠 Plant Disease Detection</h3>
            <p>Upload an image of a plant leaf, and our AI model will analyze it to detect common diseases, helping you protect your harvest.</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.success("Navigate using the menu on the left to get started!")

def render_crop_prediction_page():
    st.header("🌾 Crop Yield Prediction")
    st.markdown("Fill in the details below based on your farm's data to get an estimated yield.")

    yield_model = load_and_train_yield_model()
    if yield_model is None:
        return 

    with st.container():
        st.markdown("<div class='form-container'>", unsafe_allow_html=True)
        with st.form("crop_prediction_form"):
            st.subheader("Farm & Weather Inputs")
            col1, col2 = st.columns(2)
            with col1:
                year = st.number_input("Year", min_value=1990, max_value=2050, value=datetime.date.today().year, help="The year for the prediction.")
                pesticides = st.number_input("Pesticides Used (tonnes)", min_value=0.0, value=100.0, step=10.0)

            with col2:
                rainfall = st.number_input("Average Rainfall (mm/year)", min_value=0.0, value=1500.0, step=50.0)
                temp = st.number_input("Average Temperature (°C)", min_value=-20.0, max_value=50.0, value=22.0, step=0.5)
            
            submitted = st.form_submit_button("Predict Yield")
        st.markdown("</div>", unsafe_allow_html=True)

    if submitted:
        with st.spinner("Analyzing data and predicting yield..."):
            input_data = pd.DataFrame([[year, rainfall, pesticides, temp]], 
                                      columns=['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp'])
            predicted_yield = yield_model.predict(input_data)[0]
            
            st.success("**Prediction Complete!**")
            st.metric(
                label="Predicted Crop Yield",
                value=f"{predicted_yield:,.2f} hg/ha"
            )
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = predicted_yield,
                title = {'text': "Yield (hg/ha)"},
                gauge = {'axis': {'range': [0, np.max([predicted_yield * 1.5, 50000])]}}))
            st.plotly_chart(fig, use_container_width=True)

            prediction_data = {
                'inputs': {'Year': year, 'Pesticides': pesticides, 'Rainfall': rainfall, 'Temperature': temp},
                'output': {'yield': f"{predicted_yield:,.2f} hg/ha"}
            }
            pdf_data = generate_pdf_report(prediction_data)
            st.download_button(
                label="📄 Download Prediction Report (PDF)",
                data=pdf_data,
                file_name=f"crop_yield_report_{datetime.date.today()}.pdf",
                mime="application/pdf"
            )
    
    feedback_form("crop_prediction_feedback")

def render_disease_detection_page():
    st.header("🦠 Plant Disease Detection")
    st.markdown("Upload a clear image of a single plant leaf to check for common diseases.")

    with st.expander("📷 Image Upload Guidelines"):
        st.markdown("""
        - **Use a clear, in-focus image.** Blurry images are difficult to analyze.
        - **Capture a single leaf.** Avoid images with multiple leaves or complex backgrounds.
        - **Good lighting is key.** Use natural daylight if possible and avoid harsh shadows.
        - **Supported formats:** `.jpg`, `.jpeg`, `.png`.
        """)
    
    uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            if st.button("Analyze Disease"):
                with st.spinner("Analyzing image... This may take a moment."):
                    # Load the correct model file
                    model = load_disease_model('disease_classifier_1.h5')
                    if model is not None:
                        try:
                            image = Image.open(uploaded_file)
                            processed_image = preprocess_image(image)
                            
                            prediction = model.predict(processed_image)
                            predicted_class_index = np.argmax(prediction, axis=1)[0]
                            confidence = prediction[0][predicted_class_index]
                            disease_name = DISEASE_CLASSES[predicted_class_index]
    
                            st.success("**Analysis Complete!**")
                            st.metric("Detected Condition", disease_name.replace("___", " - ").replace("_", " "))
                            
                            st.write("Confidence:")
                            st.progress(float(confidence))
                            st.write(f"{confidence:.2%}")
                            
                        except Exception as e:
                            st.error(f"An error occurred during analysis: {e}")
    
    feedback_form("disease_detection_feedback")

def render_about_page():
    st.header("About the AI Agriculture System")
    st.markdown("""
    This platform brings cutting-edge AI technology to the agricultural sector, empowering farmers with tools that enhance productivity, sustainability, and profitability.
    """)
    
    with st.expander("The RIDE Framework"):
        st.markdown("""
        Our system is built upon the **RIDE** framework, ensuring a robust and effective solution:
        - **R**ecognize: Identify challenges in farming where AI can make a difference.
        - **I**nnovate: Develop creative and practical AI models to address these challenges.
        - **D**eploy: Build user-friendly applications like this one to make technology accessible.
        - **E**valuate: Continuously monitor and improve our systems based on feedback.
        """)
    
    with st.expander("Benefits of AI for Farming"):
        st.markdown("""
        - **Precision Agriculture:** Apply resources like water and pesticides exactly where needed.
        - **Early Disease Detection:** Prevent widespread crop damage by identifying diseases early.
        - **Yield Optimization:** Make informed decisions to maximize harvest potential.
        - **Sustainable Practices:** Reduce waste and environmental impact.
        """)
    
    st.subheader("System Workflow")
    st.graphviz_chart('''
        digraph {
            rankdir=LR;
            node [shape=box, style="rounded,filled", fillcolor="#90ee90"];
            Data [label="1. Data Collection"];
            Preprocessing [label="2. Data Preprocessing"];
            Model [label="3. AI Model Training"];
            Deployment [label="4. App Deployment"];
            
            Data -> Preprocessing -> Model -> Deployment;
        }
    ''')

def feedback_form(form_key):
    """Creates a feedback form in an expander."""
    with st.expander("📝 Provide Feedback"):
        with st.form(key=form_key, clear_on_submit=True):
            name = st.text_input("Name (Optional)")
            email = st.text_input("Email (Optional)")
            comments = st.text_area("Your comments or suggestions")
            submitted = st.form_submit_button("Submit Feedback")
            
            if submitted:
                if not comments:
                    st.warning("Please provide a comment before submitting.")
                else:
                    if save_feedback(name, email, comments):
                        st.success("Thank you for your feedback!")
                    else:
                        st.error("Sorry, there was an issue saving your feedback.")

if __name__ == "__main__":
    main()

