from typing import Any, Dict, List, Optional, Annotated, Sequence, TypedDict
from typing_extensions import TypedDict
from enum import Enum
from decimal import Decimal

from pydantic import BaseModel, Field, ConfigDict
from langgraph.graph.message import add_messages
from langchain.output_parsers import PydanticOutputParser

class QueryType(str, Enum):
    DATA_ENTRY = "data_entry"
    ANALYSIS = "analysis"
    LISTING = "listing"
    UNKNOWN = "unknown"

class FinanceAgentState(TypedDict):
    query: str
    query_type: QueryType
    messages: Annotated[Sequence[Any], add_messages]
    transactions: List[Dict[str, Any]]
    summary: Optional[Dict[str, Any]]

class FinanceActionType(str, Enum):
    ADD_EXPENSE = "add_expense"
    REMOVE_EXPENSE = "remove_expense"
    ADD_INCOME = "add_income"
    REMOVE_INCOME = "remove_income"
    UNKNOWN = "unknown"

class LLMExtractionOutput(BaseModel):
    action: FinanceActionType = Field(..., description="The classified financial action")
    amount: Decimal = Field(..., description="The monetary amount of the transaction")
    category: str = Field(..., description="Category of the transaction")
    date: str = Field(..., description="Date of the transaction in ISO format (YYYY-MM-DD)")
    description: Optional[str] = Field(default=None, description="Additional description or notes about the transaction")
    
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
    transactions: List[LLMExtractionOutput] = Field(..., description="List of extracted financial transactions, can contain one or multiple items")
    
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

class AnalysisSummary(BaseModel):
    total_income: float = Field(..., description="Total income")
    total_expenses: float = Field(..., description="Total expenses")
    net: float = Field(..., description="Net income/expense (income - expenses)")
    top_expense_categories: List[CategorySummary] = Field(..., description="Top expense categories")
    top_income_categories: List[CategorySummary] = Field(..., description="Top income categories")
    period: str = Field(..., description="Time period of the analysis")
    
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
