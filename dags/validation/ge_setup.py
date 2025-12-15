from great_expectations.data_context import FileDataContext

# Force file-based context (IMPORTANT)
context = FileDataContext(
    context_root_dir="/opt/airflow/great_expectations"
)

DATASOURCE_NAME = "churn_ds"
SUITE_NAME = "customer_churn_suite"

# Create datasource if not exists
try:
    context.get_datasource(DATASOURCE_NAME)
    print(f"Datasource already exists: {DATASOURCE_NAME}")
except Exception:
    context.add_datasource(
        name=DATASOURCE_NAME,
        class_name="Datasource",
        execution_engine={
            "class_name": "PandasExecutionEngine"
        },
        data_connectors={
            "default_runtime_data_connector_name": {
                "class_name": "RuntimeDataConnector",
                "batch_identifiers": ["default_identifier_name"],
            }
        },
    )
    print(f"Datasource created: {DATASOURCE_NAME}")

# Create expectation suite if not exists
try:
    context.get_expectation_suite(SUITE_NAME)
    print("Expectation suite already exists")
except Exception:
    context.create_expectation_suite(
        expectation_suite_name=SUITE_NAME,
        overwrite_existing=False
    )
    print("Expectation suite created")

context.save_context()
