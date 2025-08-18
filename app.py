#!/usr/bin/env python
# coding: utf-8

# In[1]:



# In[2]:


import streamlit as st
import joblib
import pandas as pd
import numpy as np


# In[3]:


# Load Random Forest model and scaler
model = joblib.load("models/random_forest_model.joblib")
scaler = joblib.load("models/scaler.joblib")

# Load feature names from cleaned data
feature_names = pd.read_csv("data/cleaned_churn_data.csv").drop("Churn", axis=1).columns


# In[4]:


st.title("Telco Customer Churn Prediction")


# In[5]:


st.sidebar.header("Input Customer Data")
input_data = {}

for feature in feature_names:
    input_data[feature] = st.sidebar.number_input(feature, value=0)

input_df = pd.DataFrame([input_data])


# In[6]:


# Scale input
input_scaled = scaler.transform(input_df)

# Predict button
if st.button("Predict Churn"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.write(f"**Prediction:** {'Churn' if prediction==1 else 'No Churn'}")
    st.write(f"**Churn Probability:** {probability:.2f}")


# In[7]:



# In[ ]:




# Test input: take first row from your data
test_input = pd.read_csv("data/cleaned_churn_data.csv").drop("Churn", axis=1).iloc[0:1]

# Scale and predict
test_scaled = scaler.transform(test_input)
test_pred = model.predict(test_scaled)[0]
test_prob = model.predict_proba(test_scaled)[0][1]

print("Test Prediction:", "Churn" if test_pred==1 else "No Churn")
print("Test Probability:", round(test_prob, 2))
