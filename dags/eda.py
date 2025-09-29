#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data_path = "../data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(data_path)

# Show first 5 rows
df.head()


# In[2]:


# Basic info about dataset
df.info()


# In[3]:


# Check missing values
df.isnull().sum()


# In[4]:


# Describe numerical columns
df.describe()


# In[5]:


# Distribution of target variable 'Churn'
sns.countplot(x='Churn', data=df)
plt.title('Churn Distribution')
plt.show()


# In[6]:


# Explore categorical variables - example: Contract type
sns.countplot(x='Contract', hue='Churn', data=df)
plt.title('Contract Type vs Churn')
plt.show()


# In[ ]:




