import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go

# --- 1. Load Model and Preprocessor ---
@st.cache_resource
def load_model(path='employee_retention_model.pkl'):
    """Load the saved ML pipeline."""
    try:
        pipeline = joblib.load(path)
        return pipeline
    except FileNotFoundError:
        st.error(f"❌ Model file '{path}' not found. Please run 'train_and_save.py' first.")
        return None

model_pipeline = load_model()

# --- 2. Page Config ---
st.set_page_config(page_title="Employee Retention Prediction", layout="wide", page_icon="🧑‍💼")

# --- 3. App Header ---
st.title("🧑‍💼 Employee Retention Prediction")
st.markdown("### Predict the likelihood of an employee leaving the organization")
st.markdown("---")

# --- 4. Sidebar for Inputs ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=120)
    st.header("🔎 Employee Characteristics")

    training_hours = st.slider("📘 Training Hours", 1, 500, 45)

    gender = st.selectbox("👤 Gender", ['Male', 'Female', 'Other', 'Unknown'])
    relevent_experience = st.selectbox("💼 Relevant Experience", ['Has relevent experience', 'No relevent experience'])
    education_level = st.selectbox("🎓 Education Level", ['Graduate', 'Masters', 'Phd', 'High School', 'Primary School', 'Unknown'])
    major_discipline = st.selectbox("📖 Major Discipline", ['STEM', 'Humanities', 'Other', 'Business Degree', 'Arts', 'No Major', 'Unknown'])
    experience = st.selectbox("🧾 Experience (Years)", ['<1'] + [str(i) for i in range(1, 21)] + ['>20', 'Unknown'])
    company_size = st.selectbox("🏢 Company Size", ['<10', '10/49', '50-99', '100-500', '500-999', '1000-4999', '5000-9999', '10000+', 'Unknown'])
    company_type = st.selectbox("🏭 Company Type", ['Pvt Ltd', 'Funded Startup', 'Public Sector', 'Early Stage Startup', 'NGO', 'Other', 'Unknown'])
    last_new_job = st.selectbox("📅 Last New Job (Years)", [1, 2, 3, 4, '>4', 'never', 'Unknown'])
    enrolled_university = st.selectbox("🏫 Enrolled University", ['no_enrollment', 'full time course', 'part time course', 'Unknown'])

    predict_button = st.button("🚀 Predict Retention Risk", use_container_width=True)

# --- 5. Prediction Logic ---
if predict_button and model_pipeline is not None:
    # Create input data
    input_data = pd.DataFrame({
        'gender': [gender],
        'relevent_experience': [relevent_experience],
        'education_level': [education_level],
        'major_discipline': [major_discipline],
        'experience': [experience],
        'company_size': [company_size],
        'company_type': [company_type],
        'last_new_job': [last_new_job],
        'training_hours': [training_hours]
    })

    try:
        prediction_proba = model_pipeline.predict_proba(input_data)[0]
        retention_risk_prob = prediction_proba[1]  # probability of leaving
        prediction = model_pipeline.predict(input_data)[0]

        risk_percentage = retention_risk_prob * 100

        # --- Tabs Layout ---
        tab1, tab2 = st.tabs(["📊 Prediction Result", "👤 Employee Profile"])

        with tab1:
            st.subheader("Prediction Outcome")

            if prediction == 1:
                st.error(f"⚠️ HIGH RISK: {risk_percentage:.2f}% chance employee will **leave**")
            else:
                st.success(f"✅ LOW RISK: {risk_percentage:.2f}% chance employee will **stay**")

            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=risk_percentage,
                title={'text': "Leaving Probability (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "red" if prediction == 1 else "green"},
                    'steps': [
                        {'range': [0, 40], 'color': "lightgreen"},
                        {'range': [40, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "salmon"},
                    ],
                }
            ))
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader("Employee Details Provided")
            st.write(input_data.T.rename(columns={0: "Value"}))

    except Exception as e:
        st.exception(f"An error occurred during prediction: {e}")
        st.warning("⚠️ Likely mismatch between training and app input features.")
