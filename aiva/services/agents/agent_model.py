"""
Models and data structures for agent components.

This module contains the data models, enums, and type definitions used by the finance agent
and other agent implementations. Centralizing these models helps maintain separation of concerns
and makes the codebase more maintainable.
"""

from typing import Any, Dict, List, Optional, Annotated, Sequence, TypedDict
from typing_extensions import TypedDict
from enum import Enum
from decimal import Decimal

from pydantic import BaseModel, Field, ConfigDict
from langgraph.graph.message import add_messages
from langchain.output_parsers import PydanticOutputParser

# Query type classification
class QueryType(str, Enum):
    """
    Enum for classifying the type of financial query from the user.
    
    This helps in routing the query to the appropriate agent handler.
    """
    DATA_ENTRY = "data_entry"
    ANALYSIS = "analysis"
    LISTING = "listing"
    UNKNOWN = "unknown"

# Define the agent state
class FinanceAgentState(TypedDict):
    """
    State for the finance agent within the LangGraph workflow.
    
    Contains the current state of the conversation, query classification,
    collected transactions, and analysis results.
    """
    query: str
    query_type: QueryType
    messages: Annotated[Sequence[Any], add_messages]
    transactions: List[Dict[str, Any]]
    summary: Optional[Dict[str, Any]]

class FinanceActionType(str, Enum):
    """
    Enum for types of financial actions that can be extracted from user input.
    
    Used to classify the transaction operations requested by the user.
    """
    ADD_EXPENSE = "add_expense"
    REMOVE_EXPENSE = "remove_expense"
    ADD_INCOME = "add_income"
    REMOVE_INCOME = "remove_income"
    UNKNOWN = "unknown"

class LLMExtractionOutput(BaseModel):
    """
    Model for a single financial transaction extracted from user input.
    
    This represents structured data for a transaction after being processed
    by the language model.
    """
    action: FinanceActionType = Field(
        ..., 
        description="The classified financial action"
    )
    amount: Decimal = Field(
        ..., 
        description="The monetary amount of the transaction"
    )
    category: str = Field(
        ..., 
        description="Category of the transaction"
    )
    date: str = Field(
        ..., 
        description="Date of the transaction in ISO format (YYYY-MM-DD)"
    )
    description: Optional[str] = Field(
        default=None, 
        description="Additional description or notes about the transaction"
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
    Container model for a list of extracted financial transactions.
    
    Used to provide a structured output from the LLM transaction extraction process.
    """
    transactions: List[LLMExtractionOutput] = Field(
        ..., 
        description="List of extracted financial transactions, can contain one or multiple items"
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

class CategorySummary(BaseModel):
    """
    Summary of financial data for a specific category.
    
    Used for generating financial reports and analysis summaries.
    """
    category: str = Field(
        ..., 
        description="Category name"
    )
    total_amount: float = Field(
        ..., 
        description="Total amount in this category"
    )
    transaction_count: int = Field(
        ..., 
        description="Number of transactions in this category"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "category": "groceries",
                "total_amount": 150.75,
                "transaction_count": 3
            }
        }
    )

class AnalysisSummary(BaseModel):
    """
    Comprehensive financial analysis summary.
    
    This model provides a complete overview of financial status,
    including income, expenses, and category breakdowns.
    """
    total_income: float = Field(
        ..., 
        description="Total income"
    )
    total_expenses: float = Field(
        ..., 
        description="Total expenses"
    )
    net: float = Field(
        ..., 
        description="Net income/expense (income - expenses)"
    )
    top_expense_categories: List[CategorySummary] = Field(
        ..., 
        description="Top expense categories"
    )
    top_income_categories: List[CategorySummary] = Field(
        ..., 
        description="Top income categories"
    )
    period: str = Field(
        ..., 
        description="Time period of the analysis"
    )
    
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
    
transaction_list_parser = PydanticOutputParser(pydantic_object=TransactionListOutput)
analysis_summary_parser = PydanticOutputParser(pydantic_object=AnalysisSummary)
