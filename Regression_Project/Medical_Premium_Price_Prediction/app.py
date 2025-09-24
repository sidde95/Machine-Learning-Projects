import streamlit as st
import pickle
import numpy as np

# Load saved model
with open('Regression_Project/Medical_Premium_Price_Prediction/best_rf_model.pkl', "rb") as f:
    model = pickle.load(f)

with open('Regression_Project/Medical_Premium_Price_Prediction/scaler.pkl', "rb") as f:
    scaler = pickle.load(f)

st.set_page_config(page_title="Insurance Premium Prediction", layout="centered")

st.title("Insurance Premium Prediction App")
st.write("Enter health and personal details below to estimate insurance premium.")

# Collect user input
age = st.number_input("Age", min_value=18, max_value=100, value=30)
diabetes = st.selectbox("Diabetes", [0, 1])
blood_pressure = st.selectbox("Blood Pressure Problems", [0, 1])
transplants = st.selectbox("Any Transplants", [0, 1])
chronic = st.selectbox("Any Chronic Diseases", [0, 1])
height = st.number_input("Height (in cm)", min_value=100, max_value=220, value=170)
weight = st.number_input("Weight (in kg)", min_value=30, max_value=200, value=70)
allergies = st.selectbox("Known Allergies", [0, 1])
cancer_history = st.selectbox("History of Cancer in Family", [0, 1])
surgeries = st.number_input("Number of Major Surgeries", min_value=0, max_value=10, value=0)

# Prepare input for prediction
input_data = np.array([[age, diabetes, blood_pressure, transplants, chronic,
                        height, weight, allergies, cancer_history, surgeries]])

# Scale the input
input_data_scaled = scaler.transform(input_data)

# Prediction button
if st.button("Predict Premium Price"):
    prediction = model.predict(input_data_scaled)[0]
    st.success(f"Estimated Insurance Premium: **₹ {prediction:,.2f}**")