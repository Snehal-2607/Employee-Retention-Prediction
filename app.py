import streamlit as st
import pandas as pd
import joblib

# ===== PAGE CONFIGURATION =====
st.set_page_config(
    page_title="Osteoporosis Risk Predictor",
    page_icon="🦴",
    layout="wide",              # full width page
    initial_sidebar_state="expanded"
)

# ===== LOAD MODEL =====
model = joblib.load("osteoporosis_model.pkl")
feature_names = joblib.load("model_features.pkl")

# ===== HEADER =====
st.markdown("<h1 style='text-align:center; color:#2E86C1;'>🦴 Osteoporosis Risk Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<hr style='border:1px solid #ccc;'>", unsafe_allow_html=True)

# ===== FORM LAYOUT (3 columns for compact view) =====
col1, col2, col3 = st.columns(3)

with col1:
    Id = st.number_input("Patient ID", value=1)
    Age = st.number_input("Age", min_value=0, max_value=120, value=50)
    Gender = st.selectbox("Gender", ['Male', 'Female'])
    Body_Weight = st.number_input("Body Weight (kg)", min_value=20, max_value=150, value=60)
    Calcium_Intake = st.selectbox("Calcium Intake", ['Low', 'Moderate', 'High'])

with col2:
    Vitamin_D = st.selectbox("Vitamin D Intake", ['Low', 'Adequate', 'High'])
    Physical_Activity = st.selectbox("Physical Activity", ['Low', 'Moderate', 'High'])
    Hormonal_Changes = st.selectbox("Hormonal Changes", ['Yes', 'No'])
    Medications = st.selectbox("Medications", ['None', 'Steroids', 'Other'])
    Family_History = st.selectbox("Family History of Osteoporosis", ['Yes', 'No'])

with col3:
    Medical_Conditions = st.selectbox("Medical Conditions", ['Yes', 'No'])
    Alcohol_Consumption = st.selectbox("Alcohol Consumption", ['Yes', 'No'])
    Prior_Fractures = st.selectbox("Prior Fractures", ['Yes', 'No'])
    Race_Ethnicity = st.selectbox("Race/Ethnicity", ['Asian', 'Caucasian', 'African', 'Hispanic', 'Other'])

# ===== CREATE INPUT DATAFRAME =====
user_input = pd.DataFrame([{
    'Id': Id,
    'Age': Age,
    'Gender': Gender,
    'Body Weight': Body_Weight,
    'Calcium Intake': Calcium_Intake,
    'Vitamin D Intake': Vitamin_D,
    'Physical Activity': Physical_Activity,
    'Hormonal Changes': Hormonal_Changes,
    'Medications': Medications,
    'Family History': Family_History,
    'Medical Conditions': Medical_Conditions,
    'Alcohol Consumption': Alcohol_Consumption,
    'Prior Fractures': Prior_Fractures,
    'Race/Ethnicity': Race_Ethnicity
}])

# Align columns with training features
input_df = user_input.reindex(columns=feature_names, fill_value=0)

# ===== PREDICT BUTTON =====
st.markdown("<br>", unsafe_allow_html=True)
center_col = st.columns([1, 1, 1])[1]

with center_col:
    if st.button("🔍 Predict Risk", use_container_width=True):
        try:
            prediction = model.predict(input_df)[0]
            st.markdown("<br>", unsafe_allow_html=True)

            if prediction == 1:
                st.error("⚠️ High Risk of Osteoporosis")
            else:
                st.success("✅ Low Risk of Osteoporosis")
        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")

# ===== FOOTER =====
st.markdown("<hr style='border:1px solid #ccc;'>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; color:gray;'>Developed by Snehal Pitale | Powered by Streamlit</p>",
    unsafe_allow_html=True
)
