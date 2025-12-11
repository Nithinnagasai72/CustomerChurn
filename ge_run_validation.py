import great_expectations as gx

# --------------------------------------------------
# Load Context
# --------------------------------------------------
context = gx.get_context()

# --------------------------------------------------
# Load Datasource
# --------------------------------------------------
datasource = context.datasources["churn_ds"]

# --------------------------------------------------
# Load Data Asset
# --------------------------------------------------
asset = datasource.get_asset("cleaned_churn_asset")

# --------------------------------------------------
# Build Batch Request
# --------------------------------------------------
batch_request = asset.build_batch_request()

# --------------------------------------------------
# Load Expectation Suite
# --------------------------------------------------
suite = context.get_expectation_suite("customer_churn_suite")

# --------------------------------------------------
# Create Validator (NEW API)
# --------------------------------------------------
validator = context.get_validator(
    batch_request=batch_request,
    expectation_suite=suite
)

# --------------------------------------------------
# Run Validation
# --------------------------------------------------
results = validator.validate()

# --------------------------------------------------
# Print results
# --------------------------------------------------
print("\n===============================")
print("📊 GREAT EXPECTATIONS VALIDATION RESULT")
print("===============================")
print("✅ Success:", results["success"])
print("📈 Statistics:", results["statistics"])

# --------------------------------------------------
# Build Data Docs
# --------------------------------------------------
context.build_data_docs()

print("\n📁 Data Docs Generated Successfully!")
print("Open in browser:")
print("gx/uncommitted/data_docs/local_site/index.html")
