import pdfplumber
import pandas as pd
from openai import OpenAI
import json
import re
from concurrent.futures import ThreadPoolExecutor
from fuzzywuzzy import process  # Import the process object

# Initialize the OpenAI client
client = OpenAI(api_key="")  # Replace with your OpenAI API key

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
    You are a regulatory compliance assistant. Your task is to extract rules from the following text and return them in a JSON format. The rules may include constraints like digit length, positive numbers, date formats, numeric ranges, allowed values, or aggregated amounts over a time period.

    Rules should be extracted as follows:
    1. For numeric fields: Specify the field name, type (e.g., "min_value", "max_value", "range"), and value(s).
    2. For text fields: Specify the field name, type (e.g., "allowed_values"), and allowed values.
    3. For date fields: Specify the field name, type (e.g., "date_format"), and the required format.
    4. For dependent rules: Specify the field name, type (e.g., "conditional_range"), and conditions.
    5. For aggregated amounts: Specify the field name, type (e.g., "aggregated_amount"), time period (e.g., "last_one_year"), and threshold.
    6. For high-frequency transactions: Specify the field name, type (e.g., "high_frequency"), time period (e.g., "last_one_year"), and threshold.

    Text:
    {text}

    Return the rules as a JSON array. For example:
    [
        {{"type": "digit_length", "field": "Customer_ID", "value": 10}},
        {{"type": "min_value", "field": "Transaction_Amount", "value": 100}},
        {{"type": "aggregated_amount", "field": "Transaction_Amount", "time_period": "last_one_year", "threshold": 2000}},
        {{"type": "high_frequency", "field": "Transaction_Amount", "time_period": "last_one_year", "threshold": 10}}
    ]

    IMPORTANT: Your response must be valid JSON. Do not include any additional text or explanations.
    """
    
    # Call OpenAI API
    response = client.chat.completions.create(
        model="gpt-4-turbo",  # Use "gpt-4-turbo" or "gpt-4o-mini"
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

# Step 3: Pre-aggregate transaction data
def pre_aggregate_data(dataset):
    """
    Pre-aggregates transaction data by customer.
    """
    # Group by customer and calculate total amount and transaction count
    aggregated_data = dataset.groupby('Customer_ID').agg(
        Total_Amount=('Transaction_Amt', 'sum'),
        Transaction_Count=('Transaction_Amt', 'count')
    ).reset_index()
    
    # Merge aggregated data back into the original dataset
    dataset = dataset.merge(aggregated_data, on='Customer_ID', how='left')
    return dataset

# Step 4: Map regulatory rules to dataset columns using fuzzy matching
def map_rules_to_dataset(rules, dataset):
    """
    Maps regulatory rules to the corresponding dataset columns using fuzzy matching.
    """
    # Get the list of columns in the dataset
    dataset_columns = dataset.columns.tolist()
    
    mapped_rules = []
    for rule in rules:
        # Use fuzzy matching to find the best match for the rule's field
        match, score = process.extractOne(rule["field"], dataset_columns)
        
        # If the match score is above a threshold (e.g., 80), use the matched column
        if score >= 80:
            rule["field"] = match
            mapped_rules.append(rule)
        else:
            print(f"Warning: No match found for field '{rule['field']}'. Skipping rule.")
    
    return mapped_rules
# Step 5: Validate a batch of records using OpenAI GPT
def validate_batch_with_openai(batch, rules, start_index):
    """
    Validates a batch of records against the rules using OpenAI GPT.
    """
    violations = []
    
    # Add custom validation for aggregated rules
    for rule in rules:
        if rule["type"] == "aggregated_amount":
            # Group by customer and calculate total amount
            customer_aggregates = batch.groupby("Customer_ID")["Transaction_Amt"].sum().reset_index()
            
            # Check for violations
            for _, row in customer_aggregates.iterrows():
                if row["Transaction_Amt"] > rule["threshold"]:
                    violations.append({
                        "record_id": row["Customer_ID"],
                        "record": row.to_dict(),
                        "violations": [f"Aggregated amount {row['Transaction_Amt']} exceeds threshold {rule['threshold']}"]
                    })
    
    # Rest of the OpenAI validation logic
    prompt = f"""
    You are a regulatory compliance assistant. Your task is to validate the following batch of records against the given rules. Return a JSON object with the following fields:
    - "results": a list of validation results for each record, where each result contains:
        - "record_id": the ID of the record (starting from {start_index})
        - "is_valid": true or false
        - "violations": a list of detailed violation descriptions (if any)

    Batch: {batch.to_dict(orient="records")}
    Rules: {json.dumps(rules, indent=2)}

    IMPORTANT: Your response must be valid JSON. Do not include any additional text or explanations.
    """
    
    # Call OpenAI API
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Use "gpt-4-turbo" or "gpt-4o-mini"
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000,  # Increase max_tokens for larger batches
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
            return {"results": []}
    except json.JSONDecodeError:
        print("Failed to parse OpenAI validation response.")
        return {"results": []}

# Step 6: Validate the entire dataset
def validate_dataset_with_openai(dataset, rules, batch_size=100):
    """
    Validates a dataset against the rules using OpenAI GPT and returns violations.
    """
    violations = []
    
    # Split the dataset into batches
    batches = [dataset[i:i + batch_size] for i in range(0, len(dataset), batch_size)]
    
    # Validate batches in parallel
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for i, batch in enumerate(batches):
            start_index = i * batch_size  # Calculate the starting index for the batch
            futures.append(executor.submit(validate_batch_with_openai, batch, rules, start_index))
        
        for future in futures:
            result = future.result()
            for record_result in result.get("results", []):
                if not record_result["is_valid"]:
                    # Use the correct index to access the record
                    record_index = record_result["record_id"]
                    violations.append({
                        "record_id": record_index,
                        "record": dataset.iloc[record_index].to_dict(),
                        "violations": record_result["violations"]
                    })
    
    return violations

# Step 7: Display violations in a user-friendly format
def display_violations(violations, rules):
    """
    Displays violations in a user-friendly format, including aggregated amounts and thresholds.
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
        
        # Check for aggregated amount violations
        for rule in rules:
            if rule["type"] == "aggregated_amount":
                total_amount = violation["record"].get("Total_Amount")
                if total_amount and total_amount > rule["threshold"]:
                    print(f"  - Aggregated amount {total_amount} exceeds threshold {rule['threshold']}")


# Step 8: Main workflow
def main(pdf_path, dataset_path):
    """
    Main workflow to extract text, parse rules using OpenAI GPT, validate the dataset, and display violations.
    """
    # Step 1: Extract text from the PDF
    text = extract_text_from_pdf(pdf_path)
    print("Extracted Text:\n", text)
    
    # Step 2: Parse rules using OpenAI GPT
    rules = parse_rules_with_openai(text)
    print("\nParsed Rules:\n", json.dumps(rules, indent=2))
    
    # Step 3: Load the dataset
    dataset = pd.read_csv(dataset_path)  # Assuming the dataset is in CSV format
    print("\nDataset:\n", dataset)
    
    # Step 4: Pre-aggregate transaction data
    dataset = pre_aggregate_data(dataset)
    print("\nAggregated Dataset:\n", dataset)
    
    # Step 5: Map regulatory rules to dataset columns
    mapped_rules = map_rules_to_dataset(rules, dataset)
    print("\nMapped Rules:\n", json.dumps(mapped_rules, indent=2))
    
    # Step 6: Validate the dataset against the rules
    violations = validate_dataset_with_openai(dataset, mapped_rules)
    
    # Step 7: Display violations
    display_violations(violations, mapped_rules)

# Example usage
if __name__ == "__main__":
    # Path to the regulatory PDF
    pdf_path = "regulatory_document.pdf"
    
    # Path to the dataset (CSV file)
    dataset_path = "dataset.csv"
    
    # Run the main workflow
    main(pdf_path, dataset_path)
