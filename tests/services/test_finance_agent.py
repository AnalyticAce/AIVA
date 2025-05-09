"""
Tests for the finance agent module.
"""
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from aiva.services.agents.finance_agent import (
    process_text,
    process_receipt,
    process_command,
    _extract_entities,
    _determine_query_type
)
from aiva.schemas.agent import Context
from aiva.services.tools import ExpenseInserter, QueryBuilder, ReceiptParser, SQLExplainer


class TestFinanceAgent:
    """Test suite for the finance agent module."""
    
    @patch("aiva.services.agents.finance_agent._call_llm")
    @patch.object(ExpenseInserter, "execute")
    async def test_process_text(self, mock_expense_inserter, mock_call_llm):
        """
        Test processing natural language text.
        
        Args:
            mock_expense_inserter: Mocked ExpenseInserter.execute method
            mock_call_llm: Mocked _call_llm function
        """
        # Setup mocks
        mock_call_llm.return_value = """
        ```json
        {
            "intent": "add_expense",
            "amount": 42.50,
            "category": "groceries",
            "date": "2025-05-07",
            "store": "Whole Foods",
            "description": null
        }
        ```
        """
        
        mock_expense_inserter.return_value = AsyncMock(
            success=True,
            sql="INSERT INTO expenses (amount, category, date, store) VALUES (42.50, 'groceries', '2025-05-07', 'Whole Foods');",
            summary="Added a 42.5 expense to groceries at Whole Foods for 2025-05-07."
        )
        
        # Call function
        result = await process_text("I spent $42.50 on groceries at Whole Foods yesterday")
        
        # Verify results
        assert result.intent == "add_expense"
        assert result.entities.amount == 42.50
        assert result.entities.category == "groceries"
        assert result.entities.date == "2025-05-07"
        assert result.entities.store == "Whole Foods"
        assert result.sql is not None
        assert result.summary is not None
        
        # Verify LLM was called with appropriate prompt
        assert mock_call_llm.called
        args, _ = mock_call_llm.call_args
        assert "Extract structured information" in args[0][1]["content"]
    
    @patch.object(ReceiptParser, "execute")
    async def test_process_receipt(self, mock_receipt_parser):
        """
        Test processing receipt text.
        
        Args:
            mock_receipt_parser: Mocked ReceiptParser.execute method
        """
        # Setup mock
        mock_receipt_parser.return_value = AsyncMock(
            success=True,
            intent="add_expense",
            entities={
                "amount": 18.50,
                "category": "groceries",
                "date": "2025-05-07",
                "store": "WHOLE FOODS MARKET"
            },
            sql="INSERT INTO expenses (amount, category, date, store) VALUES (18.50, 'groceries', '2025-05-07', 'WHOLE FOODS MARKET');",
            summary="Added a 18.5 expense to groceries at WHOLE FOODS MARKET for 2025-05-07."
        )
        
        # Sample receipt text
        receipt_text = """WHOLE FOODS MARKET
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

Thank you for shopping with us!"""
        
        # Call function
        result = await process_receipt(receipt_text)
        
        # Verify results
        assert result.intent == "add_expense"
        assert result.entities.amount == 18.50
        assert result.entities.category == "groceries"
        assert result.entities.date == "2025-05-07"
        assert result.entities.store == "WHOLE FOODS MARKET"
        assert result.sql is not None
        assert result.summary is not None
        
        # Verify ReceiptParser was called with the receipt text
        mock_receipt_parser.assert_called_once()
        args, _ = mock_receipt_parser.call_args
        assert receipt_text in args[0].values()
    
    @patch("aiva.services.agents.finance_agent._determine_query_type")
    @patch.object(QueryBuilder, "execute")
    @patch.object(SQLExplainer, "execute")
    async def test_process_command(self, mock_sql_explainer, mock_query_builder, mock_determine_query_type):
        """
        Test processing a financial command.
        
        Args:
            mock_sql_explainer: Mocked SQLExplainer.execute method
            mock_query_builder: Mocked QueryBuilder.execute method
            mock_determine_query_type: Mocked _determine_query_type function
        """
        # Setup mocks
        mock_determine_query_type.return_value = {
            "query_type": "category_expenses",
            "category": "groceries",
            "time_period": "last month",
            "limit": None
        }
        
        mock_query_builder.return_value = AsyncMock(
            success=True,
            intent="query_category_expenses",
            sql="SELECT * FROM expenses WHERE category = 'groceries' AND date >= DATE('now', '-30 day') ORDER BY date DESC;",
            summary="Looking up grocery expenses from the last 30 days."
        )
        
        mock_sql_explainer.return_value = AsyncMock(
            success=True,
            explanation="This query retrieves all columns from expenses in the groceries category for the last 30 days ordered by date in descending order.",
            summary="A query to find recent grocery expenses."
        )
        
        # Call function
        result = await process_command("Show me how much I spent on groceries last month")
        
        # Verify results
        assert result.intent == "query_category_expenses"
        assert "groceries" in result.sql
        assert "descending order" in result.summary
        
        # Verify the query type was determined
        mock_determine_query_type.assert_called_once_with(
            "Show me how much I spent on groceries last month"
        )
        
        # Verify SQL explanation was generated
        mock_sql_explainer.assert_called_once()
        args, _ = mock_sql_explainer.call_args
        assert "sql" in args[0]