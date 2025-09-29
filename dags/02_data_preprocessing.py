#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import libraries
import pandas as pd
import numpy as np

# Load dataset
data_path = "../data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(data_path)

# Quick overview
print(df.shape)
df.head()


# In[2]:


df.info()


# In[3]:


# Convert TotalCharges to numeric (force errors to NaN)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Fill missing values with median
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())


# In[4]:


df = df.drop("customerID", axis=1)


# In[5]:


# Convert target column to binary
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# One-hot encode categorical columns
df = pd.get_dummies(df, drop_first=True)


# In[6]:


df.to_csv("../data/cleaned_churn_data.csv", index=False)
df = pd.read_csv(data_path)


# ## Data Preprocessing — Customer Churn Prediction
# 
# **Objective:**  
# Prepare the Telco Customer Churn dataset for machine learning by handling missing values, fixing data types, and encoding categorical features.
# 
# **Steps Covered:**
# 1. Load and explore the dataset
# 2. Check data types and missing values
# 3. Handle missing values
# 4. Remove irrelevant columns
# 5. Encode categorical variables
# 6. Save the cleaned dataset
# 

# ## **Observation:**
# - Most columns are `object` type, meaning they are categorical.
# - `TotalCharges` appears as `object` even though it should be numeric.
# - `Churn` is the target variable (Yes/No).
# 

# ## **Changes Made:**
# - Converted `TotalCharges` to numeric
# - Filled missing values
# - Removed `customerID`
# - Encoded categorical features
# - Saved cleaned dataset
# 
# The cleaned dataset is now ready for feature scaling and model training.
# 

# 
