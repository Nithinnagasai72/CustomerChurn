#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Reload cleaned data
df = pd.read_csv("../data/cleaned_churn_data.csv")
X = df.drop("Churn", axis=1)

# Fit scaler
scaler = StandardScaler()
scaler.fit(X)

# Save scaler
joblib.dump(scaler, "../models/scaler.joblib")
print("Scaler saved successfully!")


# In[7]:


# Day 6: Model Inference (Single Prediction)

# 1. Import libraries
import pandas as pd
import numpy as np
import joblib

# 2. Load saved model & scaler
model = joblib.load("models/random_forest_model.joblib")
scaler = joblib.load("models/scaler.joblib")

# 3. Load cleaned dataset (to reuse feature structure)
df = pd.read_csv("../data/cleaned_churn_data.csv")

# Separate features & target
X = df.drop("Churn", axis=1)

# 4. Define a sample new customer (raw format)
new_customer = {
    "gender_Male": 1,
    "SeniorCitizen": 0,
    "Partner_Yes": 1,
    "Dependents_Yes": 0,
    "tenure": 12,
    "PhoneService_Yes": 1,
    "MultipleLines_No phone service": 0,
    "MultipleLines_Yes": 1,
    "InternetService_Fiber optic": 1,
    "InternetService_No": 0,
    "OnlineSecurity_No internet service": 0,
    "OnlineSecurity_Yes": 1,
    "OnlineBackup_No internet service": 0,
    "OnlineBackup_Yes": 0,
    "DeviceProtection_No internet service": 0,
    "DeviceProtection_Yes": 1,
    "TechSupport_No internet service": 0,
    "TechSupport_Yes": 0,
    "StreamingTV_No internet service": 0,
    "StreamingTV_Yes": 1,
    "StreamingMovies_No internet service": 0,
    "StreamingMovies_Yes": 1,
    "Contract_One year": 0,
    "Contract_Two year": 0,
    "PaperlessBilling_Yes": 1,
    "PaymentMethod_Credit card (automatic)": 0,
    "PaymentMethod_Electronic check": 1,
    "PaymentMethod_Mailed check": 0,
    "MonthlyCharges": 70.35,
    "TotalCharges": 845.5
}

# 5. Convert to DataFrame with correct columns
customer_df = pd.DataFrame([new_customer], columns=X.columns)

# 6. Scale numeric features
customer_scaled = scaler.transform(customer_df)

# 7. Make prediction
prediction = model.predict(customer_scaled)[0]
probability = model.predict_proba(customer_scaled)[0][1]

print("Prediction:", "Churn" if prediction == 1 else "Not Churn")
print("Churn Probability:", round(probability, 2))


# In[ ]:




