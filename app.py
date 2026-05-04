# app.py - CORRECTED VERSION
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load models
@st.cache_resource
def load_models():
    model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    selector = joblib.load('selector.pkl')
    features = joblib.load('features.pkl')
    return model, scaler, selector, features

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
    gluc_val = 1  # Default normal glucose
    alco_val = 0  # Default no alcohol
    
    # Load models
    nn_model, rule_model, scaler, selector, features = load_models()
    
    # Create DataFrame with EXACT column names the model expects
    # These must match the features used during training
    input_data = pd.DataFrame([[
        age,                    # age_years
        gender_val,             # gender
        height,                 # height
        weight,                 # weight
        ap_hi,                  # ap_hi
        ap_lo,                  # ap_lo
        cholesterol_val,        # cholesterol
        gluc_val,               # gluc
        smoke_val,              # smoke
        alco_val,               # alco
        active_val              # active
    ]], columns=['age_years', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
                 'cholesterol', 'gluc', 'smoke', 'alco', 'active'])
    
    # Feature engineering (same as in training)
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
    
    # Get the feature names the scaler expects
    # Use all columns except the original 11 we started with
    all_features = input_data.columns.tolist()
    
    # Select only the features that were used during training
    # The scaler and selector were fitted on specific columns
    # We need to ensure the order matches exactly
    
    # Method: Use the scaler's feature names if available
    try:
        # If scaler has feature_names_in_ attribute
        expected_features = scaler.feature_names_in_
        input_data = input_data[expected_features]
    except:
        # Otherwise, use all features we created
        pass
    
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
    
    # Recommendations
    st.subheader("📋 Recommendations")
    if risk > 0.7:
        st.error("⚠️ **Immediate action recommended:** Consult a cardiologist within 30 days")
    elif risk > 0.3:
        st.warning("🏃 **Lifestyle modifications recommended:** Regular exercise, healthy diet, BP monitoring")
    else:
        st.success("✅ **Maintain healthy lifestyle:** Continue regular check-ups")
    
    # Clinical explanation
    st.subheader("🔍 Key Risk Factors Identified")
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
    
    if explanation:
        for item in explanation:
            st.write(item)
    else:
        st.write("No major risk factors identified")

# Footer
st.markdown("---")
st.markdown("*Disclaimer: This tool is for educational purposes. Always consult a healthcare professional.*")ducational purposes. Always consult a healthcare professional.*")
