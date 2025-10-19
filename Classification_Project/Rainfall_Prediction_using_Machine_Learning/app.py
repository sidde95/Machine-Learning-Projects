import streamlit as st
import pickle 
import pandas as pd
import numpy as np

## Load preprocessing & model

with open("Classification_Project/Rainfall_Prediction_using_Machine_Learning/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("Classification_Project/Rainfall_Prediction_using_Machine_Learning/model.pkl", "rb") as f:
    model = pickle.load(f)


# Streamlit UI Setup

st.title("Rainfall Prediction Based on Weather Conditions")
st.write("Enter the weather conditions below to predict the likelihood / amount of rainfall.")

# User Input Section
col1, col2, col3 = st.columns(3)

with col1:
    day = st.number_input("Day of Year (1–365)", min_value=1, max_value=365, value=150, step=1)
    pressure = st.number_input("Atmospheric Pressure (hPa)", min_value=900.0, max_value=1100.0, value=1013.0, step=0.1)
    maxtemp = st.number_input("Maximum Temperature (°C)", min_value=-10.0, max_value=50.0, value=30.0, step=0.1)
    temperature = st.number_input("Average Temperature (°C)", min_value=-10.0, max_value=50.0, value=25.0, step=0.1)

with col2:
    mintemp = st.number_input("Minimum Temperature (°C)", min_value=-10.0, max_value=50.0, value=20.0, step=0.1)
    dewpoint = st.number_input("Dew Point (°C)", min_value=-10.0, max_value=35.0, value=15.0, step=0.1)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0, step=0.1)
    cloud = st.number_input("Cloud Cover (%)", min_value=0.0, max_value=100.0, value=40.0, step=0.1)

with col3:
    sunshine = st.number_input("Sunshine (hours)", min_value=0.0, max_value=15.0, value=8.0, step=0.1)
    winddirection = st.number_input("Wind Direction (°)", min_value=0.0, max_value=360.0, value=180.0, step=1.0)
    windspeed = st.number_input("Wind Speed (km/h)", min_value=0.0, max_value=100.0, value=10.0, step=0.1)


# Predict button
if st.button("Predict Rainfall"):
    input_data = np.array([[day, pressure, maxtemp, temperature, mintemp, dewpoint, humidity, cloud, sunshine, winddirection, windspeed]])
    # Preprocessing the data
    input_scaled = scaler.transform(input_data)
    # Predict
    prediction = model.predict(input_scaled)
    #Output Result
    st.subheader("Prediction Result")
    st.write(f"Predicted Rainfall: {prediction[0]:.2f} mm")