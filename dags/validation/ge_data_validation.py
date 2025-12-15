import great_expectations as gx
from airflow.exceptions import AirflowFailException
import logging


def run_ge_validation(**context):
    logger = logging.getLogger(__name__)
    logger.info("Running GX data validation")

    data_path = "/opt/airflow/data/cleaned_churn_data.csv"

    # 1️⃣ Load GX context
    ge_context = gx.get_context()

    # 2️⃣ Get or create datasource (IMPORTANT)
    datasource_name = "customer_churn_ds"

    try:
        datasource = ge_context.get_datasource(datasource_name)
    except Exception:
        logger.info("Datasource not found. Creating new Pandas datasource.")
        datasource = ge_context.sources.add_pandas(
            name=datasource_name
        )

    # 3️⃣ Get or create asset
    asset_name = "cleaned_churn_data"

    try:
        asset = datasource.get_asset(asset_name)
    except Exception:
        logger.info("Asset not found. Creating CSV asset.")
        asset = datasource.add_csv_asset(
            name=asset_name,
            filepath_or_buffer=data_path
        )

    # 4️⃣ Build batch request
    batch_request = asset.build_batch_request()

    # 5️⃣ Get or create expectation suite
    suite_name = "customer_churn_suite"

    try:
        validator = ge_context.get_validator(
            batch_request=batch_request,
            expectation_suite_name=suite_name
        )
    except Exception:
        logger.info("Expectation suite not found. Creating new suite.")
        validator = ge_context.add_or_update_expectation_suite(
            expectation_suite_name=suite_name
        )
        validator = ge_context.get_validator(
            batch_request=batch_request,
            expectation_suite_name=suite_name
        )

    # -------------------------------
    # GX EXPECTATIONS (REAL VALIDATION)
    # -------------------------------

    validator.expect_table_columns_to_match_set(
        [
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

    # 6️⃣ Run validation
    results = validator.validate()

    if not results.success:
        logger.error("GX validation failed ❌")
        raise AirflowFailException("Great Expectations validation failed")

    logger.info("GX validation successful ✅")
    return True
