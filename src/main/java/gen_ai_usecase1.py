import PyPDF2
import re
import pandas as pd
from sklearn.ensemble import IsolationForest
from transformers import pipeline

# Step 0: Extract text from the regulatory PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

# Step 1: Generate Data Profiling Rules using Hugging Face
def generate_profiling_rules(document):
    generator = pipeline("text-generation", model="EleutherAI/GPT-Neo-1.3B", device="cpu")
    prompt = f"""
    Analyze the following regulatory document and generate data profiling rules:
    {document}

    Rules:
    """
    response = generator(prompt, max_length=200)
    return response[0]["generated_text"]

# Step 2: Map Rules to Dataset Fields
def map_rules_to_fields(rules):
    field_mapping = {}
    for rule in rules.split("\n"):
        if "Field" in rule:
            field_name = re.search(r"Field '(.*?)'", rule)
            if field_name:
                field_name = field_name.group(1)
                field_mapping[field_name] = rule
    return field_mapping

# Step 3: Validate Data Against Mapped Rules
def validate_data(df, field_mapping):
    validation_results = {}
    for field, rule in field_mapping.items():
        if field in df.columns:
            if "must be a 10-digit number" in rule:
                validation_results[field] = df[field].apply(lambda x: len(str(x)) == 10 and str(x).isdigit())
            elif "must be a positive number" in rule:
                validation_results[field] = df[field].apply(lambda x: x > 0)
            elif "must be in YYYY-MM-DD format" in rule:
                validation_results[field] = df[field].apply(lambda x: re.match(r"\d{4}-\d{2}-\d{2}", str(x)) is not None)
            elif "must be above" in rule:
                threshold = float(re.search(r"above (.*?)%", rule).group(1))
                validation_results[field] = df[field].apply(lambda x: x > threshold)
    return validation_results

# Step 4: Adaptive Risk Scoring using Unsupervised Learning
def risk_scoring(df):
    X = df[["Transaction_Amount", "Capital_Adequacy_Ratio"]].values
    iso_forest = IsolationForest(contamination=0.1)
    df["Risk_Score"] = iso_forest.fit_predict(X)
    return df

# Step 5: Suggest Remediation Actions using Hugging Face
def suggest_remediation(df, field_mapping, validation_results):
    flagged_issues = df[df["Risk_Score"] == -1]
    suggestions = []
    generator = pipeline("text-generation", model="gpt2")

    for _, row in flagged_issues.iterrows():
        for field, rule in field_mapping.items():
            if field in validation_results and not validation_results[field][row.name]:
                prompt = f"""
                The following transaction row has been flagged as anomalous:
                {row.to_dict()}

                The rule violated is:
                {rule}

                The field '{field}' failed validation. Suggest a remediation action based on the above rule.

                Remediation Action:
                """
                response = generator(prompt, max_length=150)
                suggestions.append(response[0]["generated_text"])

    return suggestions

# Main Workflow
if __name__ == "__main__":

    # Sample regulatory document text
    regulatory_text = """
    Regulatory Reporting Requirements:
    1. Field 'Customer_ID' must be a 10-digit number.
    2. Field 'Transaction_Amount' must be a positive number.
    3. Field 'Transaction_Date' must be in YYYY-MM-DD format.
    4. Field 'Capital_Adequacy_Ratio' must be above 8%.
    """

    # Step 1: Generate profiling rules from the regulatory text
    profiling_rules = generate_profiling_rules(regulatory_text)
    print("\nGenerated Profiling Rules:\n", profiling_rules)

    # Step 2: Map rules to dataset fields
    field_mapping = map_rules_to_fields(profiling_rules)
    print("\nField Mapping:\n", field_mapping)

    # Step 3: Validate data against mapped rules
    data = {
        "Customer_ID": [1234567890, 987654321, 123456789, 12345678901],
        "Transaction_Amount": [100.50, -200.00, 300.75, 400.00],
        "Transaction_Date": ["2023-10-01", "2023-10-02", "2023/10/03", "2023-10-04"],
        "Capital_Adequacy_Ratio": [9.5, 7.2, 8.8, 6.5],
    }
    df = pd.DataFrame(data)

    validation_results = validate_data(df, field_mapping)
    print("\nValidation Results:\n", validation_results)

    # Step 4: Perform risk scoring on banking data
    df = risk_scoring(df)
    print("\nData with Risk Scores:\n", df)

    # Step 5: Suggest remediation actions for flagged issues
    remediation_actions = suggest_remediation(df, field_mapping, validation_results)
    print("\nSuggested Remediation Actions:\n", remediation_actions)