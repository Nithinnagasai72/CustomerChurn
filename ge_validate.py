import great_expectations as gx
from great_expectations.core.expectation_configuration import ExpectationConfiguration

# ------------------------------------------------
# Load Great Expectations Context
# ------------------------------------------------
context = gx.get_context()

# ------------------------------------------------
# Load Existing Expectation Suite
# ------------------------------------------------
SUITE_NAME = "customer_churn_suite"

try:
    suite = context.get_expectation_suite(SUITE_NAME)
    print("Loaded existing expectation suite.")
except Exception:
    suite = context.add_expectation_suite(SUITE_NAME)
    print("Created new expectation suite.")

# ------------------------------------------------
# ADD EXPECTATIONS (SAFE & GX 1.x COMPATIBLE)
# ------------------------------------------------

suite.add_expectation(
    ExpectationConfiguration(
        expectation_type="expect_column_values_to_not_be_null",
        kwargs={"column": "tenure"}
    )
)

suite.add_expectation(
    ExpectationConfiguration(
        expectation_type="expect_column_values_to_not_be_null",
        kwargs={"column": "MonthlyCharges"}
    )
)

suite.add_expectation(
    ExpectationConfiguration(
        expectation_type="expect_column_values_to_be_between",
        kwargs={
            "column": "MonthlyCharges",
            "min_value": 0,
            "max_value": 500
        }
    )
)

suite.add_expectation(
    ExpectationConfiguration(
        expectation_type="expect_column_values_to_be_in_set",
        kwargs={
            "column": "Churn",
            "value_set": [0, 1]
        }
    )
)

# ------------------------------------------------
# SAVE SUITE
# ------------------------------------------------
context.save_expectation_suite(suite)

print("Expectations added and saved successfully!")
