# AI Agriculture Assistant 🌱
An intuitive web application built with Streamlit to provide modern farming solutions. This tool offers data-driven insights for crop yield prediction and a powerful image analysis feature.

## 🌟 Key Features
- 🌾 **Crop Yield Prediction**: Forecasts crop yield (hg/ha) based on key agricultural and environmental factors like location, crop type, rainfall, pesticides, and temperature.
- 🌿 **General Image Analyzer**: Upload any image, and the application will use a powerful pre-trained model (Google's MobileNetV2) to identify the primary object in the photo.
- 📊 **Interactive UI**: A clean, user-friendly interface that allows for easy input and clear visualization of results.
- 📝 **Feedback System**: Users can submit feedback directly through the application.

## 🛠️ Tech Stack
- **Framework**: Streamlit
- **Machine Learning**: TensorFlow/Keras, Scikit-learn
- **Data Handling**: Pandas, NumPy
- **Image Processing**: Pillow
- **Plotting**: Plotly

## 🚀 Setup and Installation
Follow these steps to set up and run the project on your local machine.

### 1. Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### 2. Clone the Repository
```bash
git clone https://github.com/your-username/AI-Agriculture-Assistant.git
cd AI-Agriculture-Assistant
```

### 3. Create a Virtual Environment (Recommended)
It's best practice to create a virtual environment to manage project dependencies.

On Windows:
```bash
python -m venv myenv
myenv\Scripts\activate
```

On macOS/Linux:
```bash
python3 -m venv myenv
source myenv/bin/activate
```

### 4. Install Dependencies
Install all the required Python packages using the requirements.txt file.
```bash
pip install -r requirements.txt
```

### 5. Run the Application
Once the installation is complete, you can run the Streamlit application with the following command:
```bash
streamlit run app.py
```
The application should now be open and accessible in your web browser!

## 📖 How to Use
### Navigation
Use the sidebar on the left to switch between "Home," "Crop Yield Prediction," and "Image Analyzer."

### Crop Yield Prediction
- Select the Area and Crop type.
- Enter the Year, Average Rainfall, Pesticide usage, and Average Temperature.
- Click the "Predict Yield" button to see the forecasted result.

### Image Analyzer
- Click the "Browse files" button to upload an image (.jpg, .jpeg, .png).
- The uploaded image will be displayed.
- Click the "Analyze Image" button to get the top predictions of what the object is.

## ⚠️ Important Note on the Image Analyzer Model
The current version of the Image Analyzer uses a general-purpose model (MobileNetV2) trained by Google. It is highly effective at identifying over 1,000 common objects (e.g., "pot," "leaf," "flower").

This implementation was chosen because the original custom model for plant disease detection (`disease_classifier_1.h5`) was found to be corrupted and could not be loaded. To provide a fully functional application, this robust, pre-trained model was integrated as a replacement.

To restore a plant-specific disease detection feature, a new, correctly trained and saved model file would be required.

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page if you want to contribute.

## 📄 License
This project is licensed under the MIT License. See the LICENSE file for details.
