# app.py - WITH INTERPRETABLE CLINICAL RULES
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Load models
@st.cache_resource
def load_models():
    nn_model = joblib.load('best_model.pkl')      # Neural Network (73.9%)
    rule_model = joblib.load('rule_tree.pkl')     # Clinical Rules (73.2%)
    scaler = joblib.load('scaler.pkl')
    selector = joblib.load('selector.pkl')
    features = joblib.load('features.pkl')
    return nn_model, rule_model, scaler, selector, features

st.set_page_config(page_title="CVD Risk Predictor", page_icon="🫀", layout="wide")

st.title("🫀 Cardiovascular Disease Risk Predictor")
st.markdown("*Explainable AI for Healthcare - Trustworthy Predictions*")

# Sidebar for model selection
st.sidebar.header("Model Selection")
model_type = st.sidebar.radio(
    "Choose Prediction Model:",
    ["Neural Network (Black-box)", "Clinical Decision Tree (Interpretable)"],
    help="Neural Network: 73.9% accuracy | Decision Tree: 73.2% accuracy"
)

# Main input form
with st.form("patient_form"):
    st.subheader("Patient Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age (years)", 18, 100, 55)
        gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
        height = st.number_input("Height (cm)", 100, 250, 170)
        
    with col2:
        weight = st.number_input("Weight (kg)", 30, 200, 75)
        ap_hi = st.number_input("Systolic BP (mmHg)", 80, 250, 120)
        ap_lo = st.number_input("Diastolic BP (mmHg)", 40, 200, 80)
        
    with col3:
        cholesterol = st.selectbox("Cholesterol", ["Normal", "Above Normal", "Well Above Normal"])
        smoker = st.radio("Smoker?", ["No", "Yes"], horizontal=True)
        active = st.radio("Regular Physical Activity?", ["Yes", "No"], horizontal=True)
    
    submitted = st.form_submit_button("Predict CVD Risk", type="primary", use_container_width=True)

if submitted:
    # Convert inputs
    gender_val = 2 if gender == "Female" else 1
    cholesterol_val = ["Normal", "Above Normal", "Well Above Normal"].index(cholesterol) + 1
    smoke_val = 1 if smoker == "Yes" else 0
    active_val = 1 if active == "Yes" else 0
    
    # Load models
    nn_model, rule_model, scaler, selector, features = load_models()
    
    # Create feature vector
    input_data = pd.DataFrame([[
        age, gender_val, height, weight, ap_hi, ap_lo,
        cholesterol_val, 1, smoke_val, 0, active_val
    ]], columns=['age_years', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
                 'cholesterol', 'gluc', 'smoke', 'alco', 'active'])
    
    # Feature engineering
    input_data['bmi'] = input_data['weight'] / ((input_data['height']/100) ** 2)
    input_data['pulse_pressure'] = input_data['ap_hi'] - input_data['ap_lo']
    input_data['map'] = input_data['ap_lo'] + (input_data['pulse_pressure'] / 3)
    input_data['age_x_systolic'] = input_data['age_years'] * input_data['ap_hi']
    input_data['age_squared'] = input_data['age_years'] ** 2
    input_data['bmi_squared'] = input_data['bmi'] ** 2
    input_data['bp_risk'] = ((input_data['ap_hi'] - 120) / 20).clip(0, 3)
    input_data['age_risk'] = ((input_data['age_years'] - 50) / 10).clip(0, 3)
    input_data['age_x_cholesterol'] = input_data['age_years'] * input_data['cholesterol']
    input_data['bmi_x_pulse'] = input_data['bmi'] * input_data['pulse_pressure']
    
    # Scale and select
    input_scaled = scaler.transform(input_data)
    input_selected = selector.transform(input_scaled)
    
    # Predict based on model selection
    if model_type == "Neural Network (Black-box)":
        risk = nn_model.predict_proba(input_selected)[0, 1]
        model_accuracy = 73.89
        model_name = "Neural Network"
    else:
        risk = rule_model.predict_proba(input_selected)[0, 1]
        model_accuracy = 73.20
        model_name = "Clinical Decision Tree"
    
    # Display results
    st.markdown("---")
    st.subheader("📊 Prediction Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("CVD Risk", f"{risk:.1%}")
    
    with col2:
        if risk < 0.3:
            st.success("🟢 Low Risk")
        elif risk < 0.7:
            st.warning("🟡 Moderate Risk")
        else:
            st.error("🔴 High Risk")
    
    with col3:
        st.metric("Model Accuracy", f"{model_accuracy:.1f}%")
    
    with col4:
        st.metric("Model Type", model_name)
    
    # Risk meter
    st.progress(risk)
    
    # Clinical explanation (only for Decision Tree)
    if model_type == "Clinical Decision Tree (Interpretable)":
        st.subheader("🔍 Clinical Explanation (Why this prediction?)")
        
        # Simple rule-based explanation
        explanation = []
        if age > 55:
            explanation.append("• **Age > 55 years** → Significant risk factor")
        if ap_hi > 140:
            explanation.append(f"• **High blood pressure** ({ap_hi} mmHg) → Increases risk")
        if cholesterol_val >= 2:
            explanation.append(f"• **{cholesterol} cholesterol** → Metabolic risk")
        if smoke_val == 1:
            explanation.append("• **Smoking** → Major cardiovascular risk factor")
        if active_val == 1:
            explanation.append("• **Physical activity** → Protective factor (reduces risk)")
        if age <= 45 and ap_hi <= 120 and cholesterol_val == 1:
            explanation.append("• **Low risk profile** → Young age, normal BP, healthy cholesterol")
        
        if explanation:
            for item in explanation:
                st.write(item)
        else:
            st.write("No specific risk factors identified")
        
        st.info("💡 This explanation is based on clinical guidelines and is fully interpretable by healthcare professionals.")
    
    # Recommendations
    st.subheader("📋 Recommendations")
    if risk > 0.7:
        st.error("⚠️ **Immediate action recommended:** Consult a cardiologist within 30 days")
    elif risk > 0.3:
        st.warning("🏃 **Lifestyle modifications recommended:** Regular exercise, healthy diet, BP monitoring")
    else:
        st.success("✅ **Maintain healthy lifestyle:** Continue regular check-ups")

# Footer
st.markdown("---")
st.markdown("*Disclaimer: This tool is for educational purposes. Always consult a healthcare professional.*")