import streamlit as st
import tensorflow as tf
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import os
import json

# --- FINAL WORKING MODEL LOADER ---
@st.cache_resource
def load_disease_model():
    """
    Loads a pre-trained, functional MobileNetV2 model from TensorFlow.
    This model is highly effective for general image classification.
    """
    try:
        model = tf.keras.applications.MobileNetV2(weights='imagenet')
        return model
    except Exception as e:
        st.error(f"Error loading the pre-trained MobileNetV2 model: {e}")
        st.error("Please ensure you have an active internet connection for the first run to download the model.")
        return None

@st.cache_resource
def load_yield_model():
    """
    Loads data, trains a RandomForestRegressor model for yield prediction,
    and returns the trained model. Caching ensures this runs only once.
    """
    data_path = 'yield_df.csv'
    if not os.path.exists(data_path):
        st.error(f"Yield data file not found at '{data_path}'.")
        return None
    try:
        df = pd.read_csv(data_path)
        df.rename(columns={'hg/ha_yield': 'yield'}, inplace=True)
        df = pd.get_dummies(df, columns=['Area', 'Item'], drop_first=True)

        X = df.drop('yield', axis=1)
        y = df['yield']
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        st.error(f"An error occurred during yield model training: {e}")
        return None

