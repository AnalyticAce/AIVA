"""
API Documentation module for AIVA.

This module provides enhanced documentation for the AIVA API, including
clear examples, detailed descriptions, and response samples for each endpoint.
"""
from typing import Dict, Any

# Examples for agent parse endpoint
PARSE_REQUEST_EXAMPLE = {
    "text": "I spent $42.50 on groceries at Whole Foods yesterday",
    "context": {
        "user_timezone": "America/New_York",
        "currency": "USD"
    }
}

PARSE_RESPONSE_EXAMPLE = {
    "intent": "add_expense",
    "entities": {
        "amount": 42.50,
        "category": "groceries",
        "date": "2025-05-07",
        "store": "Whole Foods",
        "description": None
    },
    "sql": "INSERT INTO expenses (amount, category, date, store) VALUES (42.50, 'groceries', '2025-05-07', 'Whole Foods');",
    "summary": "Added a $42.50 expense to groceries category at Whole Foods for 2025-05-07."
}

# Examples for receipt parsing endpoint
RECEIPT_REQUEST_EXAMPLE = {
    "receipt_text": """WHOLE FOODS MARKET
123 Main St
New York, NY 10001

Date: 05/07/2025

Organic Bananas    $3.99
Almond Milk        $4.50
Avocado            $2.50
Whole Grain Bread  $5.99

Subtotal:         $16.98
Tax:               $1.52
Total:            $18.50

Thank you for shopping with us!""",
    "context": {
        "user_timezone": "America/New_York",
        "currency": "USD"
    }
}

RECEIPT_RESPONSE_EXAMPLE = {
    "intent": "add_expense",
    "entities": {
        "amount": 18.50,
        "category": "groceries",
        "date": "2025-05-07",
        "store": "WHOLE FOODS MARKET",
        "description": None
    },
    "sql": "INSERT INTO expenses (amount, category, date, store) VALUES (18.50, 'groceries', '2025-05-07', 'WHOLE FOODS MARKET');",
    "summary": "Added a $18.50 expense to groceries category at WHOLE FOODS MARKET for 2025-05-07."
}

# Examples for financial command endpoint
COMMAND_REQUEST_EXAMPLE = {
    "text": "Show me how much I spent on groceries last month",
    "context": {
        "user_timezone": "America/New_York",
        "currency": "USD"
    }
}

COMMAND_RESPONSE_EXAMPLE = {
    "intent": "query_category_expenses",
    "sql": "SELECT * FROM expenses WHERE category = 'groceries' AND date >= DATE('now', '-30 day') ORDER BY date DESC;",
    "summary": "This query retrieves all grocery expenses from the last 30 days, ordered from most recent to oldest."
}

# API operation descriptions
OPERATION_DESCRIPTIONS: Dict[str, Dict[str, str]] = {
    "parse": {
        "summary": "Process Natural Language Text",
        "description": """
        Extracts structured financial information from natural language text.
        
        This endpoint analyzes text like "I spent $42.50 on groceries" and converts it into 
        structured data including the expense amount, category, date, and other relevant entities.
        It also generates an appropriate SQL INSERT statement for storing the information.
        
        Common use cases:
        - Add expense entries from casual text description
        - Extract financial information from notes or messages
        - Generate SQL for adding transactions to a finance database
        """
    },
    "receipt": {
        "summary": "Parse Receipt Text", 
        "description": """
        Extracts structured expense information from receipt OCR text.
        
        This endpoint processes raw text from a receipt image (OCR output) and extracts 
        key financial information such as the total amount, merchant name, date, and 
        generates an appropriate category based on the merchant and items purchased.
        
        Common use cases:
        - Digitize paper receipts for expense tracking
        - Extract data from email or digital receipts
        - Automate expense entry from receipt photos
        """
    },
    "command": {
        "summary": "Process Financial Command",
        "description": """
        Generates SQL queries for financial analysis based on natural language commands.
        
        This endpoint interprets questions or commands about financial data like
        "How much did I spend on groceries last month?" and converts them into appropriate 
        SQL queries. It also provides a plain-language explanation of what the query does.
        
        Common use cases:
        - Generate SQL for financial reports and summaries
        - Answer questions about spending patterns
        - Analyze expenses by category, time period, or merchant
        """
    }
}

# Function to get full API operation info
def get_operation_docs(operation_id: str) -> Dict[str, Any]:
    """
    Get the full documentation dictionary for an API operation.
    
    Args:
        operation_id: The ID of the API operation ('parse', 'receipt', or 'command')
        
    Returns:
        Dict[str, Any]: Operation documentation including description and examples
    """
    docs = OPERATION_DESCRIPTIONS.get(operation_id, {}).copy()
    
    # Add examples based on operation ID
    if operation_id == "parse":
        docs["request_example"] = PARSE_REQUEST_EXAMPLE
        docs["response_example"] = PARSE_RESPONSE_EXAMPLE
    elif operation_id == "receipt":
        docs["request_example"] = RECEIPT_REQUEST_EXAMPLE
        docs["response_example"] = RECEIPT_RESPONSE_EXAMPLE
    elif operation_id == "command":
        docs["request_example"] = COMMAND_REQUEST_EXAMPLE
        docs["response_example"] = COMMAND_RESPONSE_EXAMPLE
        
    return docs