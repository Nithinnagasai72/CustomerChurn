import great_expectations as gx

# Load GE context
context = gx.get_context()

# Get existing datasource
datasource = context.datasources.get("churn_ds")

# Register CSV correctly using batching regex
asset = datasource.add_csv_asset(
    name="cleaned_churn_asset",
    batching_regex=r"cleaned_churn_data\.csv"
)

print("Datasource:", datasource.name)
print("Asset registered:", asset.name)
