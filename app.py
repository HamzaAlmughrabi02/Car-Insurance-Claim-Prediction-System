import streamlit as st
import pandas as pd
import joblib
import os

# 1. Page Configuration & Professional Light Theme Styling
st.set_page_config(
    page_title="Car Insurance Claim Prediction System",
    page_icon="🚗",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for a clean, modern, and eye-friendly light theme
st.markdown("""
    <style>
    .stApp {
        background-color: #f8fafc;
        color: #1e293b;
    }
    .main-header {
        font-size: 2.3rem;
        color: #0f172a;
        text-align: center;
        font-weight: 800;
        margin-bottom: 0.3rem;
        letter-spacing: -0.5px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 2.5rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: white;
        font-weight: 700;
        padding: 0.8rem;
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%);
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.35);
        color: white;
    }
    div[data-baseweb="select"] > div {
        background-color: #ffffff;
        color: #1e293b;
        border-radius: 8px;
        border-color: #cbd5e1;
    }
    div[data-testid="stMetricValue"] {
        color: #0f172a;
        font-weight: 800;
    }
    </style>
""", unsafe_allow_html=True)

# 2. Load Trained Model
@st.cache_resource
def load_trained_model():
    model_paths = [
        'models/best_voting_model.pkl',
        '../models/best_voting_model.pkl',
        './best_voting_model.pkl'
    ]
    for path in model_paths:
        if os.path.exists(path):
            return joblib.load(path)
    return None

model = load_trained_model()

# 3. Main Interface Header
st.markdown("<div class='main-header'>🚗 Car Insurance Claim Prediction System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>AI-powered platform to analyze and evaluate driver and vehicle risk profiles</div>", unsafe_allow_html=True)

# Model Status Alert
if model is None:
    st.warning("⚠️ **Warning:** Model file (`best_voting_model.pkl`) not found in the `models` directory.")
else:
    st.info("💡 **System Ready:** AI model connected successfully and ready for real-time predictions.")

st.write("")

# 4. Inputs Section
st.markdown("### 📋 Driver & Vehicle Details")

col1, col2 = st.columns(2, gap="large")

with col1:
    age = st.slider("Driver Age", 18, 100, 35)
    driving_experience = st.selectbox("Driving Experience", ["0-9y", "10-19y", "20-29y", "30y+"])
    annual_mileage = st.number_input("Annual Mileage", value=11000, step=500)
    speeding_violations = st.number_input("Speeding Violations", 0, 50, 0)

with col2:
    past_accidents = st.number_input("Past Accidents", 0, 20, 0)
    credit_score = st.number_input("Credit Score", 0.0, 1.0, 0.5, step=0.01)
    vehicle_year = st.selectbox("Vehicle Year", ["before 2015", "after 2015"])
    vehicle_type = st.selectbox("Vehicle Type", ["sedan", "sports car"])

st.divider()

# 5. Prediction Button & Logic
if st.button("🔮 Run Smart Prediction & Analysis", use_container_width=True):
    if model is None:
        st.error("❌ Error: Cannot perform prediction because the model is not loaded.")
    else:
        try:
            # Build input DataFrame
            input_data = pd.DataFrame({
                'age': [age],
                'driving_experience': [driving_experience],
                'annual_mileage': [annual_mileage],
                'speeding_violations': [speeding_violations],
                'past_accidents': [past_accidents],
                'credit_score': [credit_score],
                'vehicle_year': [vehicle_year],
                'vehicle_type': [vehicle_type]
            })
            
            # Align features with training columns (One-Hot Encoding)
            ref_path = 'data/processed/car_insurance_encoded.csv'
            if not os.path.exists(ref_path):
                ref_path = '../data/processed/car_insurance_encoded.csv'
                
            if os.path.exists(ref_path):
                df_ref = pd.read_csv(ref_path, nrows=1)
                expected_columns = [col for col in df_ref.columns if col not in ['id', 'outcome']]
            else:
                expected_columns = input_data.columns.tolist()
            
            input_encoded = pd.get_dummies(input_data)
            
            for col in expected_columns:
                if col not in input_encoded.columns:
                    input_encoded[col] = 0
            input_encoded = input_encoded[expected_columns]
            
            # Prediction
            prediction = model.predict(input_encoded)
            prediction_proba = model.predict_proba(input_encoded)
            risk_score = prediction_proba[0][1]
            
            st.markdown("### 📊 Analysis Results")
            
            res_col1, res_col2 = st.columns([1, 2], gap="large")
            
            with res_col1:
                st.metric(
                    label="Claim Risk Probability", 
                    value=f"{risk_score:.2%}",
                    delta="High Risk" if risk_score > 0.5 else "Safe",
                    delta_color="inverse" if risk_score > 0.5 else "normal"
                )
                
            with res_col2:
                st.write("**Overall Risk Status:**")
                st.progress(float(risk_score))
                
                if prediction[0] == 1:
                    st.error(f"⚠️ **High Risk Warning:** This driver is **likely** to file an insurance claim with a probability of **{risk_score:.2%}**.")
                else:
                    st.success(f"✅ **Safe Driver:** This driver is **unlikely** to file an insurance claim (Safety level: **{1-risk_score:.2%}**).")
                
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")