import great_expectations as gx
from great_expectations.core.expectation_configuration import ExpectationConfiguration
from great_expectations.exceptions import DataContextError

context = gx.get_context()

SUITE_NAME = "customer_churn_suite"

try:
    suite = context.get_expectation_suite(SUITE_NAME)
    print("Loaded existing expectation suite.")
except DataContextError:
    suite = context.create_expectation_suite(
        expectation_suite_name=SUITE_NAME,
        overwrite_existing=False
    )
    print("Created new expectation suite.")

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

context.save_expectation_suite(suite)

print("Expectations added and saved successfully!")
