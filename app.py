import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

# Constants
MEDIAN_VALUES = {
    'Glucose': 117.0,
    'BloodPressure': 72.0,
    'SkinThickness': 23.0,
    'Insulin': 30.5,
    'BMI': 32.0
}

MODEL_PATHS = {
    'scaler': 'scaler.pkl',
    'naive_bayes': 'naive_bayes.pkl',
    'random_forest': 'rf_model.pkl',
    'logistic_regression': 'lr_model.pkl',
    'deep_learning': 'diabetes_dl_model.h5'
}

# Helper functions
def check_login():
    """Handle user authentication"""
    if not st.session_state.authenticated:
        st.subheader("🔐 Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if username == "admin" and password == "1234": 
                st.session_state.authenticated = True
                st.success("✅ Logged in successfully!")
                st.experimental_rerun()
            else:
                st.error("❌ Invalid credentials")
        return False
    return True

@st.cache_resource
def load_models():
    """Load all ML models with error handling"""
    models = {}
    try:
        models['scaler'] = joblib.load(MODEL_PATHS['scaler'])
        models['naive_bayes'] = joblib.load(MODEL_PATHS['naive_bayes'])
        models['random_forest'] = joblib.load(MODEL_PATHS['random_forest'])
        models['logistic_regression'] = joblib.load(MODEL_PATHS['logistic_regression'])
        models['deep_learning'] = tf.keras.models.load_model(MODEL_PATHS['deep_learning'])
        return models
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.stop()

def get_input():
    """Collect patient data from sidebar"""
    with st.sidebar:
        st.header("Patient Data")
        return {
            'Pregnancies': st.number_input("Pregnancies", 0, 20, 0),
            'Glucose': st.number_input("Glucose", 0, 300, 120),
            'BloodPressure': st.number_input("BloodPressure", 0, 200, 70),
            'SkinThickness': st.number_input("SkinThickness", 0, 100, 20),
            'Insulin': st.number_input("Insulin", 0, 900, 79),
            'BMI': st.number_input("BMI", 0.0, 70.0, 25.5, step=0.1),
            'DiabetesPedigreeFunction': st.number_input("Pedigree Function", 0.0, 3.0, 0.42, step=0.01),
            'Age': st.number_input("Age", 1, 120, 29)
        }

def preprocess_data(df_input):
    """Handle missing values and scale data"""
    for col in MEDIAN_VALUES:
        if df_input.at[0, col] == 0:
            df_input.at[0, col] = MEDIAN_VALUES[col]
    return df_input

def get_risk_level(prob):
    """Convert probability to risk category"""
    if prob < 0.33:
        return "Low Risk"
    elif prob < 0.66:
        return "Medium Risk"
    else:
        return "High Risk"

def plot_results(preds):
    """Visualize model predictions"""
    names = list(preds.keys())
    probabilities = list(preds.values())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#4caf50', '#2196f3', '#ff9800', '#9c27b0']
    bars = ax.bar(names, probabilities, color=colors)
    
    ax.set_ylim([0, 1])
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_title('Diabetes Risk Probability by Model', fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=10)
    
    st.pyplot(fig)

# Main app logic
if not check_login():
    st.stop()

# Load models
try:
    models = load_models()
except Exception as e:
    st.error(f"Failed to load models: {str(e)}")
    st.stop()

# Main page
st.title("🏥 Diabetes Risk Predictor")
st.write("Enter patient data in the sidebar and click Predict")

# Get patient data
patient_data = get_input()
df_input = pd.DataFrame([patient_data])

# Process data
df_processed = preprocess_data(df_input.copy())

# Make predictions
if st.button("🔮 Predict", type="primary"):
    with st.spinner("Analyzing..."):
        try:
            X = models['scaler'].transform(df_processed)
            
            preds = {
                'Naive Bayes': models['naive_bayes'].predict_proba(X)[0][1],
                'Random Forest': models['random_forest'].predict_proba(X)[0][1],
                'Logistic Regression': models['logistic_regression'].predict_proba(X)[0][1],
                'Deep Learning': models['deep_learning'].predict(X)[0][0]
            }
            
            st.subheader("📊 Prediction Results")
            cols = st.columns(2)
            
            for i, (name, prob) in enumerate(preds.items()):
                risk = get_risk_level(prob)
                with cols[i % 2]:
                    st.metric(
                        label=name,
                        value=f"{prob:.2%}",
                        delta=risk,
                        delta_color="off"
                    )
            
            plot_results(preds)
            
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

# Add logout button
if st.session_state.authenticated:
    if st.sidebar.button("🚪 Logout"):
        st.session_state.authenticated = False
        st.experimental_rerun()
        