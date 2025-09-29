#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[2]:


# Load cleaned dataset
data_path = "../data/cleaned_churn_data.csv"
df = pd.read_csv(data_path)

# Quick overview
print(df.shape)
df.head()


# In[3]:


# Separate features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

print("Features shape:", X.shape)
print("Target shape:", y.shape)


# In[4]:


# Check for missing values
print("Missing values in X:\n", X.isnull().sum().sum())
print("Missing values in y:", y.isnull().sum())


# In[5]:


# Split data into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train set shape:", X_train.shape)
print("Test set shape:", X_test.shape)


# In[6]:


# Initialize scaler
scaler = StandardScaler()

# Fit on train, transform train and test
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Scaled Train shape:", X_train_scaled.shape)
print("Scaled Test shape:", X_test_scaled.shape)


# In[7]:


import numpy as np

# Save processed arrays
np.save("../data/X_train_scaled.npy", X_train_scaled)
np.save("../data/X_test_scaled.npy", X_test_scaled)
np.save("../data/y_train.npy", y_train.to_numpy())
np.save("../data/y_test.npy", y_test.to_numpy())


# In[8]:


# Core libraries
import numpy as np
import pandas as pd

# ML libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib  # for saving/loading models

# SHAP library
import shap


# In[9]:


# Load saved numpy arrays from Day 3
X_train = np.load("../data/X_train_scaled.npy")
X_test = np.load("../data/X_test_scaled.npy")
y_train = np.load("../data/y_train.npy")
y_test = np.load("../data/y_test.npy")

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# In[10]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Initialize Random Forest
rf = RandomForestClassifier(random_state=42)

# Train
rf.fit(X_train_scaled, y_train)

# Predict
y_pred = rf.predict(X_test_scaled)

# Accuracy & classification report
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# In[11]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# feature names from training
feature_names = df.drop('Churn', axis=1).columns

# get feature importances from trained model
importances = rf.feature_importances_

# create a DataFrame for plotting
feat_imp = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# plot top 10 features
plt.figure(figsize=(10,6))
plt.barh(feat_imp['Feature'][:10][::-1], feat_imp['Importance'][:10][::-1])
plt.xlabel('Importance')
plt.title('Top 10 Feature Importances')
plt.show()


# In[16]:


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib  # For saving models


# In[17]:


X_train = np.load("../data/X_train_scaled.npy")
X_test = np.load("../data/X_test_scaled.npy")
y_train = np.load("../data/y_train.npy")
y_test = np.load("../data/y_test.npy")

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# In[18]:


rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


# In[19]:


import matplotlib.pyplot as plt

feature_names = pd.read_csv("../data/cleaned_churn_data.csv").drop('Churn', axis=1).columns
importances = rf.feature_importances_

plt.figure(figsize=(12,6))
plt.barh(feature_names, importances)
plt.title("Random Forest Feature Importance")
plt.show()


# In[3]:


# Day 5: Train Final Model and Save It

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

# 1. Load saved train/test data from Day 3
X_train = np.load("../data/X_train_scaled.npy")
X_test = np.load("../data/X_test_scaled.npy")
y_train = np.load("../data/y_train.npy")
y_test = np.load("../data/y_test.npy")

print("X_train:", X_train.shape, "X_test:", X_test.shape)
print("y_train:", y_train.shape, "y_test:", y_test.shape)

# 2. Train Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
rf.fit(X_train, y_train)

# 3. Evaluate
y_pred = rf.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# 4. Save model
os.makedirs("models", exist_ok=True)
joblib.dump(rf, "models/random_forest_model.joblib")
print("✅ Model saved at ../models/random_forest_model.joblib")


# In[ ]:




