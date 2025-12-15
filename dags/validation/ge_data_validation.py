import great_expectations as gx
from airflow.exceptions import AirflowFailException
import logging
from pathlib import Path


def run_ge_validation(**context):
    """
    GX-driven validation ONLY.
    No pandas logic.
    Uses persistent Great Expectations project.
    """

    logger = logging.getLogger(__name__)
    logger.info("Running GX data validation")

    data_path = "/opt/airflow/data/cleaned_churn_data.csv"

    # ✅ IMPORTANT: point to existing GX project
    ge_root_dir = Path("/opt/airflow/great_expectations")
    ge_context = gx.get_context(context_root_dir=ge_root_dir)

    # 1️⃣ Get datasource (must exist from ge_setup.py)
    datasource = ge_context.get_datasource("customer_churn_ds")

    # 2️⃣ Get data asset
    asset = datasource.get_asset("cleaned_churn_data")

    # 3️⃣ Build batch request
    batch_request = asset.build_batch_request(path=data_path)

    # 4️⃣ Get validator
    validator = ge_context.get_validator(
        batch_request=batch_request,
        expectation_suite_name="customer_churn_suite"
    )

    # -------------------------------
    # GX EXPECTATIONS (NO PANDAS)
    # -------------------------------

    validator.expect_table_columns_to_match_set(
        column_set=[
            "tenure",
            "MonthlyCharges",
            "TotalCharges",
            "Churn"
        ]
    )

    validator.expect_column_values_to_be_between(
        column="tenure",
        min_value=0
    )

    validator.expect_column_values_to_be_between(
        column="MonthlyCharges",
        min_value=0
    )

    validator.expect_column_values_to_be_between(
        column="TotalCharges",
        min_value=0
    )

    validator.expect_column_values_to_be_in_set(
        column="Churn",
        value_set=[0, 1]
    )

    # 5️⃣ Run validation
    results = validator.validate()

    if not results.success:
        logger.error("GX validation failed ❌")
        raise AirflowFailException("Great Expectations validation failed")

    logger.info("GX validation successful ✅")
    return True
