"""
Simplified Pydantic models for the finance agent API with structured output parsing.

This module defines the request and response schemas for the AIVA finance agent API
that processes user prompts about expenses and income, with output format enforcement.
It now supports analytical queries and conversation threading with LangGraph.
"""
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Literal
from datetime import date
from decimal import Decimal

from pydantic import BaseModel, Field, ConfigDict, field_validator


class FinanceActionType(str, Enum):
    """
    Enumeration of possible financial actions for better type safety and validation.
    """
    ADD_EXPENSE = "add_expense"
    REMOVE_EXPENSE = "remove_expense"
    ADD_INCOME = "add_income"
    REMOVE_INCOME = "remove_income"
    UNKNOWN = "unknown"


class QueryType(str, Enum):
    """
    Enumeration of query types for routing in the finance agent.
    """
    DATA_ENTRY = "data_entry"
    ANALYSIS = "analysis" 
    LISTING = "listing"
    UNKNOWN = "unknown"


class Context(BaseModel):
    """
    Context information for API requests.
    """
    user_timezone: Optional[str] = Field(
        default="UTC",
        description="User's timezone in IANA format (e.g., 'America/New_York', 'Europe/London')",
        examples=["America/New_York", "Europe/London", "Asia/Tokyo"]
    )
    currency: Optional[str] = Field(
        default="USD",
        description="User's preferred currency code in ISO 4217 format",
        examples=["USD", "EUR", "GBP", "JPY"]
    )
    locale: Optional[str] = Field(
        default="en-US",
        description="User's locale code for formatting numbers and dates",
        examples=["en-US", "fr-FR", "de-DE"]
    )


class FinanceData(BaseModel):
    """
    Extracted financial information from the prompt with validation.
    """
    action: FinanceActionType = Field(
        ...,
        description="The financial action type",
        examples=["add_expense", "add_income"]
    )
    amount: Decimal = Field(
        ...,
        description="The monetary amount of the transaction",
        examples=[42.50, 19.99, 1250.00],
        gt=0  # Ensure amount is greater than 0
    )
    category: str = Field(
        ...,
        description="Category of the transaction (e.g., groceries, salary, transportation)",
        examples=["groceries", "dining", "salary", "freelance"],
        min_length=1  # Ensure category is not empty
    )
    date: str = Field(
        ...,
        description="Date of the transaction in ISO format (YYYY-MM-DD)",
        examples=["2025-05-07", "2025-04-15"],
        pattern=r"^\d{4}-\d{2}-\d{2}$"  # Validate date format using pattern instead of regex
    )
    description: Optional[str] = Field(
        default=None,
        description="Additional description or notes about the transaction",
        examples=["Weekly grocery shopping", "Monthly paycheck", "Freelance project"]
    )
    
    @field_validator('date')
    @classmethod
    def validate_date_format(cls, v):
        """Validate that the date is in proper ISO format."""
        try:
            year, month, day = map(int, v.split('-'))
            date(year, month, day)  # Will raise ValueError if invalid
            return v
        except (ValueError, TypeError):
            raise ValueError('Invalid date format. Must be YYYY-MM-DD')


class CategorySummary(BaseModel):
    """
    Summary of financial transactions for a single category.
    """
    category: str = Field(..., description="Category name")
    total_amount: float = Field(..., description="Total amount in this category")
    transaction_count: int = Field(..., description="Number of transactions in this category")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "category": "groceries",
                "total_amount": 150.75,
                "transaction_count": 3
            }
        }
    )


class AnalyticsResponse(BaseModel):
    """
    Response model for analytics-type queries.
    Can contain structured analysis data or markdown-formatted text.
    """
    # For structured data responses
    total_income: Optional[float] = Field(None, description="Total income amount")
    total_expenses: Optional[float] = Field(None, description="Total expenses amount")
    net: Optional[float] = Field(None, description="Net income/expense (income - expenses)")
    top_expense_categories: Optional[List[CategorySummary]] = Field(None, description="Top expense categories")
    top_income_categories: Optional[List[CategorySummary]] = Field(None, description="Top income categories")
    period: Optional[str] = Field(None, description="Time period of the analysis")
    
    # For text-based responses
    content: Optional[str] = Field(None, description="Text content of the analysis, typically markdown-formatted")
    format: Optional[str] = Field("markdown", description="Format of the content (e.g., 'markdown', 'plain')")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total_income": 5000.00,
                "total_expenses": 3500.50,
                "net": 1499.50,
                "top_expense_categories": [
                    {"category": "groceries", "total_amount": 450.25, "transaction_count": 5},
                    {"category": "rent", "total_amount": 1500.00, "transaction_count": 1}
                ],
                "top_income_categories": [
                    {"category": "salary", "total_amount": 4500.00, "transaction_count": 1},
                    {"category": "freelance", "total_amount": 500.00, "transaction_count": 2}
                ],
                "period": "May 2025"
            }
        }
    )


# Request Model
class PromptRequest(BaseModel):
    """
    Request for processing a financial prompt.
    """
    prompt: str = Field(
        ..., 
        description="Natural language text describing a financial transaction or action",
        examples=[
            "I spent $42.50 on groceries yesterday",
            "I earned $1500 from my job today",
            "Delete the $25 expense for lunch on May 5",
            "What are my top expense categories?",
            "How much did I spend on groceries this month?"
        ],
        min_length=3  # Ensure prompt has minimal content
    )
    thread_id: Optional[str] = Field(
        default=None,
        description="Optional thread ID for maintaining conversation context"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prompt": "I spent $42.50 on groceries yesterday and $100 on Uber the day before",
                "thread_id": None
            }
        }
    )
    
    @field_validator('prompt')
    @classmethod
    def validate_prompt_content(cls, v):
        """Validate that the prompt content is not empty."""
        if v is None or len(str(v).strip()) < 3:
            raise ValueError('Prompt must contain meaningful content (at least 3 characters)')
        return v


# LLM Output Format - This defines the expected structure for LLM responses
class LLMClassificationOutput(BaseModel):
    """
    Structured output format for LLM classification responses.
    """
    action: FinanceActionType = Field(
        ...,
        description="The classified financial action",
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "action": "add_expense"
            }
        }
    )


class LLMExtractionOutput(BaseModel):
    """
    Structured output format for LLM information extraction responses.
    """
    action: FinanceActionType = Field(
        ...,
        description="The classified financial action",
    )
    amount: Decimal = Field(
        ...,
        description="The monetary amount of the transaction",
    )
    category: str = Field(
        ...,
        description="Category of the transaction"
    )
    date: str = Field(
        ...,
        description="Date of the transaction in ISO format (YYYY-MM-DD)",
    )
    description: Optional[str] = Field(
        default=None,
        description="Additional description or notes about the transaction",
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "action": "add_expense",
                "amount": 42.50,
                "category": "groceries", 
                "date": "2025-05-07",
                "description": "Weekly shopping"
            }
        }
    )


class TransactionListOutput(BaseModel):
    """
    Container model for one or multiple transactions.
    This provides a unified response format regardless of transaction count.
    """
    transactions: List[LLMExtractionOutput] = Field(
        ...,
        description="List of extracted financial transactions",
        min_items=1
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "transactions": [
                    {
                        "action": "add_expense",
                        "amount": 42.50,
                        "category": "groceries",
                        "date": "2025-05-07",
                        "description": "Weekly shopping"
                    },
                    {
                        "action": "add_expense",
                        "amount": 100.00,
                        "category": "transportation",
                        "date": "2025-05-06",
                        "description": "Uber to work"
                    }
                ]
            }
        }
    )


# Response Model
class PromptResponse(BaseModel):
    """
    Response from processing a financial prompt with validated data.
    Can contain either transactions (for data entry queries) or
    analytics results (for analysis queries).
    """
    transactions: List[FinanceData] = Field(
        default=[],
        description="List of extracted financial transactions"
    )
    analytics: Optional[AnalyticsResponse] = Field(
        default=None,
        description="Analytics and insights from the financial data"
    )
    query_type: Optional[str] = Field(
        default=None,
        description="The classified type of the query (data_entry, analysis, listing, unknown)"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if processing failed",
        examples=["Failed to extract required information from prompt"]
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "transactions": [
                    {
                        "action": "add_expense",
                        "amount": 42.50,
                        "category": "groceries",
                        "date": "2025-05-07",
                        "description": "Weekly shopping"
                    },
                    {
                        "action": "add_expense",
                        "amount": 100.00,
                        "category": "transportation",
                        "date": "2025-05-06",
                        "description": "Uber to work"
                    }
                ],
                "analytics": None,
                "query_type": "data_entry",
                "error": None
            }
        }
    )