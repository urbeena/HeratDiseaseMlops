import numpy as np
import pandas as pd
import joblib
import pickle
import streamlit as st
import os

# -------- Load Required Files --------
# Get folder where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model and other files
model_path = os.path.join(BASE_DIR, "models", "best_model.pkl")
scaler_path = os.path.join(BASE_DIR, "models", "scaler.pkl")
encoders_path = os.path.join(BASE_DIR, "models", "encoders.pkl")
selected_features_path = os.path.join(BASE_DIR, "models", "selected_features.txt")

model = joblib.load(model_path)
scaler = pickle.load(open(scaler_path, "rb"))

with open(selected_features_path, "r") as f:
    selected_features = [line.strip() for line in f]

encoders = pickle.load(open(encoders_path, "rb"))

# Title
st.title("💓 Heart Disease Prediction App (Local Model)")

# Numeric inputs
age = st.number_input("Age", min_value=1, max_value=120)
resting_bp = st.number_input("RestingBP")
chol = st.number_input("Cholesterol")
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
max_hr = st.number_input("MaxHR")
oldpeak = st.number_input("Oldpeak")

# Categorical inputs — corrected to match encoder values
sex = st.selectbox("Sex", ['F', 'M'])
chest_pain = st.selectbox("Chest Pain Type", ['ASY', 'ATA', 'NAP'])
resting_ecg = st.selectbox("Resting ECG", ['Normal', 'LVH', 'ST'])
exercise_angina = st.selectbox("Exercise Angina", ['N', 'Y'])
st_slope = st.selectbox("ST Slope", ['Up', 'Flat', 'Down'])

if st.button("Predict"):
    input_data = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": chol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
        "Sex": sex,
        "ChestPainType": chest_pain,
        "RestingECG": resting_ecg,
        "ExerciseAngina": exercise_angina,
        "ST_Slope": st_slope
    }

    # Separate numeric
    numeric_cols = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
    numeric_values = [input_data[col] for col in numeric_cols]
    numeric_scaled = scaler.transform([numeric_values]).flatten().tolist()

    # Encode categorical
    categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    encoded_values = []
    for col in categorical_cols:
        enc = encoders[col]
        encoded_values.append(int(enc.transform([input_data[col]])[0]))

    # Final complete vector
    final_vector = numeric_scaled + encoded_values

    # Convert to dataframe with correct column order
    df = pd.DataFrame([final_vector], columns=numeric_cols + categorical_cols)

    # Filter selected features
    final_input = df[selected_features].values

    # Predict
    prediction = model.predict(final_input)[0]

    # Output
    st.subheader("🔍 Prediction Result:")
    if prediction == 1:
        st.error("⚠️ HIGH RISK of Heart Disease")
    else:
        st.success("✅ LOW RISK of Heart Disease")
