"""
Tests for the agent API endpoints.
"""
import pytest
from unittest.mock import patch, AsyncMock

from aiva.schemas.agent import ParseResponse, ReceiptResponse, CommandResponse, Entity


class TestAgentEndpoints:
    """Test suite for the agent API endpoints."""
    
    @patch("aiva.api.v1.endpoints.agent.finance_agent.process_text")
    async def test_parse_endpoint(self, mock_process_text, client):
        """
        Test the parse endpoint for natural language text.
        
        Args:
            mock_process_text: Mocked process_text function
            client: Test client fixture
        """
        # Setup mock response
        mock_response = ParseResponse(
            intent="add_expense",
            entities=Entity(
                amount=42.50,
                category="groceries",
                date="2025-05-07",
                store="Whole Foods"
            ),
            sql="INSERT INTO expenses (amount, category, date, store) VALUES (42.50, 'groceries', '2025-05-07', 'Whole Foods');",
            summary="Added a 42.5 expense to groceries at Whole Foods for 2025-05-07."
        )
        mock_process_text.return_value = mock_response
        
        # Make request
        response = client.post(
            "/api/v1/agent/parse",
            json={
                "text": "I spent $42.50 on groceries at Whole Foods yesterday"
            }
        )
        
        # Assert response
        assert response.status_code == 200
        assert response.json()["intent"] == "add_expense"
        assert response.json()["entities"]["amount"] == 42.50
        assert "sql" in response.json()
        assert "summary" in response.json()
    
    @patch("aiva.api.v1.endpoints.agent.finance_agent.process_receipt")
    async def test_receipt_endpoint(self, mock_process_receipt, client):
        """
        Test the receipt endpoint for receipt parsing.
        
        Args:
            mock_process_receipt: Mocked process_receipt function
            client: Test client fixture
        """
        # Setup mock response
        mock_response = ReceiptResponse(
            intent="add_expense",
            entities=Entity(
                amount=18.50,
                category="groceries",
                date="2025-05-07",
                store="WHOLE FOODS MARKET"
            ),
            sql="INSERT INTO expenses (amount, category, date, store) VALUES (18.50, 'groceries', '2025-05-07', 'WHOLE FOODS MARKET');",
            summary="Added a 18.5 expense to groceries at WHOLE FOODS MARKET for 2025-05-07."
        )
        mock_process_receipt.return_value = mock_response
        
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
        
        # Make request
        response = client.post(
            "/api/v1/agent/receipt",
            json={
                "receipt_text": receipt_text
            }
        )
        
        # Assert response
        assert response.status_code == 200
        assert response.json()["intent"] == "add_expense"
        assert response.json()["entities"]["amount"] == 18.50
        assert response.json()["entities"]["store"] == "WHOLE FOODS MARKET"
        assert "sql" in response.json()
        assert "summary" in response.json()
    
    @patch("aiva.api.v1.endpoints.agent.finance_agent.process_command")
    async def test_command_endpoint(self, mock_process_command, client):
        """
        Test the command endpoint for financial queries.
        
        Args:
            mock_process_command: Mocked process_command function
            client: Test client fixture
        """
        # Setup mock response
        mock_response = CommandResponse(
            intent="query_category_expenses",
            sql="SELECT * FROM expenses WHERE category = 'groceries' AND date >= DATE('now', '-30 day') ORDER BY date DESC;",
            summary="This query retrieves all columns from expenses in the groceries category for the last 30 days ordered by date in descending order."
        )
        mock_process_command.return_value = mock_response
        
        # Make request
        response = client.post(
            "/api/v1/agent/command",
            json={
                "text": "Show me how much I spent on groceries last month"
            }
        )
        
        # Assert response
        assert response.status_code == 200
        assert response.json()["intent"] == "query_category_expenses"
        assert "sql" in response.json()
        assert "summary" in response.json()