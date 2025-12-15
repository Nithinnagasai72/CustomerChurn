import pandas as pd

def run_ge_validation():
    print("Running data validation")

    df = pd.read_csv("/opt/airflow/data/cleaned_churn_data.csv")

    # 1️⃣ Schema validation
    required_columns = ["tenure", "MonthlyCharges", "TotalCharges", "Churn"]

    missing_cols = [c for c in required_columns if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # 2️⃣ Business rule validations
    if (df["tenure"] < 0).any():
        raise ValueError("Validation failed: tenure contains negative values")

    if (df["MonthlyCharges"] < 0).any():
        raise ValueError("Validation failed: MonthlyCharges contains negative values")

    if (df["TotalCharges"] < 0).any():
        raise ValueError("Validation failed: TotalCharges contains negative values")

    # 3️⃣ Label validation
    if not set(df["Churn"].unique()).issubset({0, 1}):
        raise ValueError("Validation failed: Churn contains invalid values")

    print("Data validation successful")
    return True
