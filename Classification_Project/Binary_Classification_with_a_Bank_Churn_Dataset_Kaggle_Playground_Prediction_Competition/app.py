import streamlit as st
import pandas as pd
import numpy as np
import pickle

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('preprocessing.pkl', 'rb') as file:
    preprocessing = pickle.load(file)




st.set_page_config(page_title = "Bank Churn Prediction", layout = "centered")

st.title("Bank Churn Prediction")
st.write("Predict whether a customer will exit the bank based on their details.")

# User Input
with st.form("churn_form"):
    col1, col2 = st.columns(2)

    with col1:
        credit_score = st.number_input("Credit Score", min_value = 300, max_value = 900, value = 600)
        age = st.number_input("Age", min_value = 18, max_value = 100, value = 35)
        tenure = st.slider("Tenure (Years)", 0, 10, 2)
        balance = st.number_input("Account Balance", min_value = 0.0, value = 50000.0, step = 1000.0)

    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"])
        geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
        num_products = st.slider("Number of Products", 1, 4, 1)
        has_cr_card = st.selectbox("Has Credit Card?", [0, 1])
        is_active_member = st.selectbox("Is Active Member?", [0, 1])
        estimated_salary = st.number_input("Estimated Salary", min_value = 0.0, value = 50000.0, step =1000.0 )

    submit = st.form_submit_button("Predict")

if submit:
    gender_encoded = 1 if gender == "Male" else 0

    # Build DataFrame from inputs
    input_data = pd.DataFrame([{
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender_encoded,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": has_cr_card,
        "IsActiveMember": is_active_member,
        "EstimatedSalary": estimated_salary
    }])

    # Applying preprocessing
    processed_data = preprocessing.transform(input_data)

    # Predict Churn Probability
    churn_prob = model.predict_proba(processed_data)[0][1]
    churn_pred = model.predict(processed_data)[0]

    # Result Display
    st.subheader("Preidction Result")
    st.write(f"Churn Probability: {churn_prob}")

    if churn_pred == 1:
        st.error("Customer is likely to churn!")
    else:
        st.success("Customer is likely to stay.")