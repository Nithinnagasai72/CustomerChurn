from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# File paths inside Airflow container
DATA_PATH = "/opt/airflow/data/cleaned_churn_data.csv"
CLEANED_DATA_PATH = "/opt/airflow/data/cleaned_churn_data.csv"
MODEL_PATH = "/opt/airflow/models/churn_model.pkl"
REPORT_PATH = "/opt/airflow/reports/model_report.txt"
    
# -------------------------
# Task 1: Data Ingestion
# -------------------------
def data_ingestion():
    df = pd.read_csv(DATA_PATH)
    print(f"? Data Ingestion Successful! Shape: {df.shape}")

# -------------------------
# Task 2: Data Preprocessing
# -------------------------
def data_preprocessing():
    df = pd.read_csv(DATA_PATH)

    # Remove missing values
    df.dropna(inplace=True)

    # Convert categorical columns to numeric using one-hot encoding
    df = pd.get_dummies(df, drop_first=True)

    # Save cleaned data
    os.makedirs(os.path.dirname(CLEANED_DATA_PATH), exist_ok=True)
    df.to_csv(CLEANED_DATA_PATH, index=False)
    print(f"? Data Preprocessing Done! Cleaned shape: {df.shape}")

# -------------------------
# Task 3: Model Training
# -------------------------
def model_training():
    df = pd.read_csv(CLEANED_DATA_PATH)

    # Split features and target
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train RandomForest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print("? Model Training Completed & Saved!")

# -------------------------
# # -------------------------
# Task 4: Model Evaluation (with Postgres storage)
# -------------------------
from airflow.providers.postgres.hooks.postgres import PostgresHook

def model_evaluation():
    df = pd.read_csv(CLEANED_DATA_PATH)

    # Features & target
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Load trained model
    model = joblib.load(MODEL_PATH)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    precision = report["weighted avg"]["precision"]
    recall = report["weighted avg"]["recall"]
    f1 = report["weighted avg"]["f1-score"]

    # Save report to file
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write(f"Accuracy: {acc}\n\n")
        f.write(classification_report(y_test, y_pred))

    print(f"? Model Evaluation Done! Accuracy: {acc}")

    # Store metrics in Postgres
    hook = PostgresHook(postgres_conn_id="postgres_default")  # Connection ID from Airflow UI
    conn = hook.get_conn()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_metrics (
            run_id SERIAL PRIMARY KEY,
            accuracy FLOAT,
            precision FLOAT,
            recall FLOAT,
            f1 FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    cursor.execute("""
        INSERT INTO model_metrics (accuracy, precision, recall, f1)
        VALUES (%s, %s, %s, %s);
    """, (acc, precision, recall, f1))

    conn.commit()
    cursor.close()
    conn.close()

    print("Metrics stored in Postgres successfully!")

# -------------------------
# Airflow DAG Definition
# -------------------------
with DAG(
    dag_id="day7_customer_churn_pipeline",
    schedule_interval=None,  # Trigger manually
    start_date=datetime(2025, 8, 24),
    catchup=False,
    tags=["customer_churn", "day7"],
) as dag:

    ingest_task = PythonOperator(
        task_id="data_ingestion",
        python_callable=data_ingestion,
    )

    preprocess_task = PythonOperator(
        task_id="data_preprocessing",
        python_callable=data_preprocessing,
    )

    train_task = PythonOperator(
        task_id="model_training",
        python_callable=model_training,
    )

    evaluate_task = PythonOperator(
        task_id="model_evaluation",
        python_callable=model_evaluation,
    )

    # Task Dependencies
    ingest_task >> preprocess_task >> train_task >> evaluate_task
