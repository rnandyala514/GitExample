from fpdf import FPDF

# Create a PDF object
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)

# Add content to the PDF
content = """
Banking Regulatory Compliance Rules

This document outlines the regulatory compliance rules for banking transactions. All transactions must adhere to the following rules to ensure compliance with regulatory standards.

**Rules:**

1. Field 'Customer_ID' must be a 10-digit number.

2. Field 'Transaction_Amount' must be a positive number between $500 and $10,000.

3. Field 'Transaction_Date' must be in YYYY-MM-DD format.

4. Field 'Capital_Adequacy_Ratio' must be above 8%.

5. Field 'Account_Type' must be one of the following: Savings, Checking, or Loan.

6. Field 'Transaction_Type' must be one of the following: Deposit, Withdrawal, or Transfer.

7. Field 'Transaction_Date' shoud not be greater than current date.

8. Field 'Transaction_Date' should be greater than 2024-01-01.


"""

# Write the content to the PDF
pdf.multi_cell(0, 10, content)
pdf.output("regulatory_document.pdf")