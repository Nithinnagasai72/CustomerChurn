import streamlit as st
import requests

# Title
st.title("Customer Churn Prediction App")
st.markdown("Enter customer details below to predict whether they will churn.")

# Backend URL (FastAPI)
FASTAPI_URL = "http://127.0.0.1:8000/predict" 

# Form inputs
st.subheader("Customer Information")

tenure = st.number_input("Tenure (in months)", min_value=0, max_value=100, value=12)
monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)
total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=600.0)

contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])

# Prepare input data
input_data = {
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "Contract": contract,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "TechSupport": tech_support,
}

if st.button("Predict Churn"):
    with st.spinner("🔄 Predicting... Please wait..."):
        try:
            # Send data to FastAPI
            response = requests.post(FASTAPI_URL, json=input_data)

            # Handle response
            if response.status_code == 200:
                result = response.json()

                st.success(f"✅ Prediction: {result['prediction_label']}")
                st.metric(
                    label="Churn Probability",
                    value=f"{result['churn_probability'] * 100:.1f}%"
                )
            else:
                st.error(f"🚨 API Error: {response.status_code}")
        except Exception as e:
            st.error(f"❌ Error connecting to FastAPI: {e}")


# Footer
st.markdown("---")
st.caption("Developed by **Nithin Naga Sai** • FastAPI + Streamlit + ML Pipeline")