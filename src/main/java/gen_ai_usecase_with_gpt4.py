import pdfplumber
import pandas as pd
from openai import OpenAI
import json
import re

# Initialize the OpenAI client
client = OpenAI(api_key="DTsRBXkRcYCAY2BZXZ2Pxf1woFJJA_pxx8iMMQHOM0fXPj4ckM9YdI-cblVRm_lUevrug40YT3BlbkFJziWMdALTmDt6g8sjNnZN44gBnwSAkb_prhJsPofi7D0TQOSh2Zkt3VZJT29MFMIfO39Ipw7iwA")  # Replace with your OpenAI API key

# Step 1: Extract text from the PDF
def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file.
    """
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Step 2: Parse rules from the extracted text using OpenAI GPT
def parse_rules_with_openai(text):
    """
    Uses OpenAI GPT to extract rules from the text and return them in a structured format.
    """
    prompt = f"""
    You are a regulatory compliance assistant. Your task is to extract rules from the following text and return them in a JSON format. The rules may include constraints like digit length, positive numbers, date formats, numeric ranges, or allowed values.

    Rules should be extracted as follows:
    1. For numeric fields: Specify the field name, type (e.g., "min_value", "max_value", "range"), and value(s).
    2. For text fields: Specify the field name, type (e.g., "allowed_values"), and allowed values.
    3. For date fields: Specify the field name, type (e.g., "date_format"), and the required format.

    Text:
    {text}

    Return the rules as a JSON array. For example:
    [
        {{"type": "digit_length", "field": "Customer_ID", "value": 10}},
        {{"type": "positive_number", "field": "Transaction_Amount"}},
        {{"type": "date_format", "field": "Transaction_Date", "value": "YYYY-MM-DD"}},
        {{"type": "range", "field": "Transaction_Amount", "min": 100, "max": 10000}},
        {{"type": "allowed_values", "field": "Account_Type", "values": ["Savings", "Checking", "Loan"]}}
    ]

    IMPORTANT: Your response must be valid JSON. Do not include any additional text or explanations.
    """
    
    # Call OpenAI API
    response = client.chat.completions.create(
        model="gpt-4",  # Use "gpt-4" or "gpt-4-turbo"
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.3
    )
    
    # Extract the generated text
    extracted_rules = response.choices[0].message.content
    
    # Clean up the response to extract valid JSON
    try:
        # Use regex to extract the JSON part from the response
        json_match = re.search(r'\[.*\]', extracted_rules, re.DOTALL)
        if json_match:
            rules = json.loads(json_match.group(0))
            return rules
        else:
            print("Failed to extract JSON from OpenAI response.")
            return []
    except json.JSONDecodeError:
        print("Failed to parse OpenAI response as JSON.")
        return []

# Step 3: Validate a record dynamically using OpenAI GPT
def validate_record_with_openai(record, rules):
    """
    Validates a record against the rules using OpenAI GPT.
    """
    prompt = f"""
    You are a regulatory compliance assistant. Your task is to validate the following record against the given rules. Return a JSON object with the following fields:
    - "is_valid": true or false
    - "violations": a list of rules that were violated (if any)

    Record: {record.to_dict()}
    Rules: {json.dumps(rules, indent=2)}

    IMPORTANT: Your response must be valid JSON. Do not include any additional text or explanations.
    """
    
    # Call OpenAI API
    response = client.chat.completions.create(
        model="gpt-4",  # Use "gpt-4" or "gpt-4-turbo"
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300,
        temperature=0.2
    )
    
    # Extract the generated text
    validation_result = response.choices[0].message.content
    
    # Clean up the response to extract valid JSON
    try:
        # Use regex to extract the JSON part from the response
        json_match = re.search(r'\{.*\}', validation_result, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(0))
            return result
        else:
            print("Failed to extract JSON from OpenAI response.")
            return {"is_valid": False, "violations": ["Validation failed due to an error."]}
    except json.JSONDecodeError:
        print("Failed to parse OpenAI validation response.")
        return {"is_valid": False, "violations": ["Validation failed due to an error."]}

# Step 4: Validate the entire dataset
def validate_dataset_with_openai(dataset, rules):
    """
    Validates a dataset against the rules using OpenAI GPT and returns violations.
    """
    violations = []
    
    for index, record in dataset.iterrows():
        result = validate_record_with_openai(record, rules)
        if not result["is_valid"]:
            violations.append({
                "record_id": index,
                "record": record.to_dict(),
                "violations": result["violations"]
            })
    
    return violations

# Step 5: Display violations in a user-friendly format
def display_violations(violations):
    """
    Displays violations in a user-friendly format.
    """
    if not violations:
        print("No violations found. All records are valid.")
        return
    
    print("Violations Found:")
    for violation in violations:
        print(f"\nRecord ID: {violation['record_id']}")
        print("Record Data:", violation["record"])
        print("Violated Rules:")
        for rule in violation["violations"]:
            print(f"  - {rule}")

# Step 6: Main workflow
def main(pdf_path, dataset_path):
    """
    Main workflow to extract text, parse rules using OpenAI GPT, validate the dataset, and display violations.
    """
    # Step 1: Extract text from the PDF
    text = extract_text_from_pdf(pdf_path)
    print("Extracted Text:\n", text)
    
    # Step 2: Parse rules using OpenAI GPT
    rules = parse_rules_with_openai(text)
    print("Parsed Rules:\n", json.dumps(rules, indent=2))
    
    # Step 3: Load the dataset
    dataset = pd.read_csv(dataset_path)  # Assuming the dataset is in CSV format
    print("\nDataset Sample:\n", dataset.head())
    
    # Step 4: Validate the dataset against the rules
    violations = validate_dataset_with_openai(dataset, rules)
    
    # Step 5: Display violations
    display_violations(violations)

# Example usage
if __name__ == "__main__":
    # Path to the regulatory PDF
    pdf_path = "regulatory_document.pdf"
    
    # Path to the dataset (CSV file)
    dataset_path = "dataset.csv"
    
    # Run the main workflow
    main(pdf_path, dataset_path)
