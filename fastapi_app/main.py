import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

# -----------------------------
# Initialize FastAPI App
# -----------------------------
app = FastAPI(title="Customer Churn Prediction API")

# Allow CORS (for Streamlit frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can replace "*" with Streamlit URL if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Load Model and Reference Data
# -----------------------------
try:
    model = joblib.load("notebooks/models/random_forest_model.joblib")

    ref_df = pd.read_csv("./data/cleaned_churn_data.csv",encoding = "latin1")
    ref_columns = ref_df.drop("Churn", axis=1).columns

    print("? Model and reference columns loaded successfully.")
except Exception as e:
    print(f"? Error loading model or data: {e}")
    ref_columns = []

# -----------------------------
# Homepage Route
# -----------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head>
            <title>Customer Churn Prediction API</title>
        </head>
        <body style="font-family: Arial; margin: 40px;">
            <h1>?? Customer Churn Prediction API</h1>
            <p>This API is running successfully!</p>
            <p>Use the <code>/predict</code> endpoint to make churn predictions.</p>
            <hr>
            <p><strong>Example:</strong></p>
            <pre>
POST /predict
{
  "tenure": 12,
  "MonthlyCharges": 70,
  "TotalCharges": 900,
  "Contract": "Month-to-month",
  "PaymentMethod": "Electronic check"
}
            </pre>
        </body>
    </html>
    """

# -----------------------------
# Prediction Endpoint
# -----------------------------
@app.post("/predict")
def predict(data: dict):
    try:
        # Convert input JSON into DataFrame
        input_df = pd.DataFrame([data])

        # One-hot encode and align columns
        input_df = pd.get_dummies(input_df, drop_first=True)
        input_df = input_df.reindex(columns=ref_columns, fill_value=0)

        # Predict churn
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        return {
            "prediction": int(prediction),
            "prediction_label": "Churn" if prediction == 1 else "No Churn",
            "churn_probability": round(float(probability), 3)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
