import streamlit as st
import pickle
import pandas as pd
import numpy as np

# ---------------------------
# Load preprocessing & model
# ---------------------------
with open("Regression_Project/Regression_of_Used_Car_Prices/preprocessing.pkl", "rb") as f:
    preprocessing = pickle.load(f)

with open("Regression_Project/Regression_of_Used_Car_Prices/model.pkl", "rb") as f:
    model = pickle.load(f)

# ---------------------------
# Streamlit UI setup
# ---------------------------
st.set_page_config(page_title="Car Price Prediction App", page_icon="🚘", layout="centered")

st.title("🚘 Price My Ride - Used Car Price Estimator")
st.markdown("### Enter the car details below to predict its estimated market value.")

# ---------------------------
# User Input Section
# ---------------------------
col1, col2 = st.columns(2)

with col1:
    brand = st.text_input("Car Brand", placeholder="e.g., Toyota, Maruti, BMW")
    model_year = st.number_input("Model Year", min_value=1990, max_value=2025, value=2015)
    milage = st.number_input("Mileage (in km/l)", min_value=0.0, max_value=50.0, value=15.0)
    fuel_type = st.selectbox("Fuel Type", ['Gasoline', 'Hybrid', 'Diesel'])

with col2:
    transmission = st.selectbox("Transmission Type", ["Manual", "Automatic"])
    ext_col = st.text_input("Exterior Color", placeholder="e.g., White, Red, Black")
    int_col = st.text_input("Interior Color", placeholder="e.g., Beige, Black, Grey")
    accident = st.selectbox("Accident History", ["No", "Yes"])

col3, col4 = st.columns(2)
with col3:
    horsepower = st.number_input("Horsepower (HP)", min_value=30, max_value=1000, value=100)
with col4:
    engine_displacement = st.number_input("Engine Displacement (cc)", min_value=500, max_value=6000, value=1500)
    cylinder_count = st.number_input("Cylinder Count", min_value=2, max_value=12, value=4)

# ---------------------------
# Predict button
# ---------------------------
if st.button("🔍 Predict Car Price"):
    try:
        # Create DataFrame with correct columns
        input_df = pd.DataFrame({
            'brand': [brand],
            'model_year': [model_year],
            'milage': [milage],
            'fuel_type': [fuel_type],
            'transmission': [transmission],
            'ext_col': [ext_col],
            'int_col': [int_col],
            'accident': [accident],
            'horsepower': [horsepower],
            'engine_displacement': [engine_displacement],
            'cylinder_count': [cylinder_count]
        })

        # Apply preprocessing
        processed_input = preprocessing.transform(input_df)

        # Predict price
        predicted_price = model.predict(processed_input)[0]

        st.success(f"💰 **Estimated Car Price: ₹ {predicted_price:,.2f}**")
        st.balloons()

    except Exception as e:
        st.error("Error while predicting. Please check your input values or model files.")
        st.write(e)

