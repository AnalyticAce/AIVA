from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict
import json
from enum import Enum
from datetime import datetime, timedelta
from decimal import Decimal

from sqlalchemy import create_engine, Column, Integer, String, Date, Numeric, func
from sqlalchemy.orm import declarative_base, sessionmaker

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import Tool, StructuredTool
from pydantic import BaseModel, Field, ConfigDict

from aiva.core.config import settings
from aiva.core.logging import logger

# Database setup
Base = declarative_base()
engine = create_engine('sqlite:///transactions.db', echo=True)
Session = sessionmaker(bind=engine)
session = Session()

class Transaction(Base):
    """
    SQLAlchemy model representing a financial transaction in the database.
    """
    __tablename__ = 'transactions'
    
    id = Column("id", Integer, primary_key=True)
    action = Column("action", String, nullable=False)
    amount = Column("amount", Numeric(10, 2), nullable=False)
    category = Column("category", String, nullable=False)
    date = Column("date", Date, nullable=False)
    description = Column("description", String, nullable=True)
    
    def __init__(self, action, amount, category, date, description=None):
        self.action = action
        self.amount = amount
        self.category = category
        self.date = date
        self.description = description
    
    def __repr__(self):
        return f"<Transaction(action={self.action}, amount={self.amount}, category={self.category}, date={self.date}, description={self.description})>"

# Create tables if they don't exist
Base.metadata.create_all(bind=engine)

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

transaction_list_parser = PydanticOutputParser(pydantic_object=TransactionListOutput)

def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
        temperature=0.0
    )

def transaction_extraction_prompt_template(parser: PydanticOutputParser):
    extraction_prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are a financial assistant that analyzes user statements and extracts structured financial data.
        
        Given a user's statement about their finances, you need to:
        1. Determine the type of financial action (add_expense, remove_expense, add_income, remove_income, or unknown)
        2. Extract the monetary amount
        3. Categorize the transaction (e.g., groceries, utilities, salary, etc.)
        4. Identify the date (use today's date if not specified)
        5. Capture any additional description
        
        Format your response according to the following schema:

        {{
            "transactions": [
                {{
                    "action": "action_type",
                    "amount": amount_value,
                    "category": "category_name",
                    "date": "YYYY-MM-DD",
                    "description": "description_text"
                }},
                ...more transactions if present
            ]
        }}

        Always return the transactions in a list even if there's only one transaction.

        Examples:
        "I spent $50 on groceries yesterday" -> 
        {{
            "transactions": [
                {{
                    "action": "add_expense",
                    "amount": 50.00,
                    "category": "groceries",
                    "date": "2023-05-10",  # Use the actual date from yesterday
                    "description": "Grocery shopping"
                }}
            ]
        }}
    
        "Delete the $20 I spent on gas and add $1000 from my salary" -> 
        {{
            "transactions": [
                {{
                    "action": "remove_expense",
                    "amount": 20.00,
                    "category": "transportation",
                    "date": "2023-05-11",  # Use today's date if not specified
                    "description": "Gas payment to be removed"
                }},
                {{
                    "action": "add_income",
                    "amount": 1000.00,
                    "category": "salary",
                    "date": "2023-05-11",  # Use today's date if not specified
                    "description": "Monthly salary"
                }}
            ]
        }}
    
        Available tools:
        1. get_current_date - Use this tool if the user doesn't specify a date to get today's date
        2. get_available_categories - Use this tool to get a list of available transaction categories
        3. insert_transaction - Add a new transaction to the database
        4. get_transaction_by_id - Retrieve a specific transaction using its ID
        5. get_transactions_by_category - Get all transactions in a specific category
        6. get_transactions_by_date_range - Get transactions between two dates
        7. group_transactions_by_category - Get summary of total amounts by category
        8. delete_transaction - Remove a transaction from the database
        9. update_transaction - Change fields of an existing transaction

        User interactions:
        - For user requests about adding expenses/income, first extract the transaction details, then use insert_transaction
        - For user requests about viewing expenses, use get_transactions_by_category or get_transactions_by_date_range
        - For user requests about summaries or reports, use group_transactions_by_category
        - For user requests about deleting transactions, first find the transaction ID, then use delete_transaction
        - For user requests about updating transactions, first get the transaction, then use update_transaction

        Just return the JSON response without any additional text especially comments.
        """),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    return extraction_prompt

def get_current_date(date_diff: Any = 0) -> str:
    if isinstance(date_diff, str):
        try:
            date_diff = int(date_diff)
        except ValueError:
            date_diff = 0
            
    return (datetime.now() - timedelta(days=date_diff)).strftime("%Y-%m-%d")

def get_available_categories(*args) -> List[str]:
    return [
        "groceries",
        "utilities",
        "transportation",
        "salary",
        "dining",
        "entertainment",
        "food",
        "shopping",
        "bills",
        "rent",
        "healthcare",
        "education",
        "freelance"
    ]

def insert_transaction(
    action: str,
    amount: float,
    category: str,
    date: str,
    description: Optional[str] = None
) -> Dict[str, Any]:
    try:
        parsed_date = datetime.strptime(date, '%Y-%m-%d').date()

        new_transaction = Transaction(
            action=action,
            amount=amount,
            category=category,
            date=parsed_date,
            description=description
        )

        session.add(new_transaction)
        session.commit()

        return {
            "id": new_transaction.id,
            "action": new_transaction.action,
            "amount": float(new_transaction.amount),
            "category": new_transaction.category,
            "date": new_transaction.date.isoformat(),
            "description": new_transaction.description,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error inserting transaction: {str(e)}")
        session.rollback()
        return {
            "status": "error",
            "message": str(e)
        }

def get_transaction_by_id(transaction_id: int) -> Dict[str, Any]:
    try:
        transaction = session.query(Transaction).filter(
            Transaction.id == transaction_id
        ).first()
        
        if transaction:
            return {
                "id": transaction.id,
                "action": transaction.action,
                "amount": float(transaction.amount),
                "category": transaction.category,
                "date": transaction.date.isoformat(),
                "description": transaction.description,
                "status": "success"
            }
        else:
            return {
                "status": "error",
                "message": f"Transaction with ID {transaction_id} not found"
            }
    except Exception as e:
        logger.error(f"Error retrieving transaction: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }

def get_transactions_by_category(category: str) -> Dict[str, Any]:
    try:
        transactions = session.query(Transaction).filter(
            Transaction.category == category
        ).all()
        
        result = []
        for transaction in transactions:
            result.append({
                "id": transaction.id,
                "action": transaction.action,
                "amount": float(transaction.amount),
                "category": transaction.category,
                "date": transaction.date.isoformat(),
                "description": transaction.description
            })
            
        return {
            "transactions": result,
            "count": len(result),
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error retrieving transactions by category: {str(e)}")
        return {
            "status": "error",
            "message": str(e),
            "transactions": []
        }

def get_transactions_by_date_range(start_date: str, end_date: str) -> Dict[str, Any]:
    try:
        parsed_start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        parsed_end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
        
        transactions = session.query(Transaction).filter(
            Transaction.date >= parsed_start_date,
            Transaction.date <= parsed_end_date
        ).all()
        
        result = []
        for transaction in transactions:
            result.append({
                "id": transaction.id,
                "action": transaction.action,
                "amount": float(transaction.amount),
                "category": transaction.category,
                "date": transaction.date.isoformat(),
                "description": transaction.description
            })
            
        return {
            "transactions": result,
            "count": len(result),
            "start_date": start_date,
            "end_date": end_date,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error retrieving transactions by date range: {str(e)}")
        return {
            "status": "error",
            "message": str(e),
            "transactions": []
        }

def group_transactions_by_category() -> Dict[str, Any]:
    try:
        results = session.query(
            Transaction.category,
            func.sum(Transaction.amount).label('total_amount')
        ).group_by(Transaction.category).all()
        
        category_totals = []
        for category, total_amount in results:
            category_totals.append({
                "category": category,
                "total_amount": float(total_amount)
            })
            
        return {
            "category_totals": category_totals,
            "count": len(category_totals),
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error grouping transactions by category: {str(e)}")
        return {
            "status": "error",
            "message": str(e),
            "category_totals": []
        }

def delete_transaction(transaction_id: int) -> Dict[str, Any]:
    try:
        transaction = session.query(Transaction).filter(
            Transaction.id == transaction_id
        ).first()
        
        if transaction:
            # Store transaction details before deletion for return value
            transaction_details = {
                "id": transaction.id,
                "action": transaction.action,
                "amount": float(transaction.amount),
                "category": transaction.category,
                "date": transaction.date.isoformat(),
                "description": transaction.description
            }
            
            session.delete(transaction)
            session.commit()
            
            return {
                "deleted_transaction": transaction_details,
                "status": "success",
                "message": f"Transaction with ID {transaction_id} successfully deleted"
            }
        else:
            return {
                "status": "error",
                "message": f"Transaction with ID {transaction_id} not found"
            }
    except Exception as e:
        logger.error(f"Error deleting transaction: {str(e)}")
        session.rollback()
        return {
            "status": "error",
            "message": str(e)
        }

def update_transaction(
    transaction_id: int,
    action: Optional[str] = None,
    amount: Optional[float] = None, 
    category: Optional[str] = None,
    date: Optional[str] = None,
    description: Optional[str] = None
) -> Dict[str, Any]:
    try:
        transaction = session.query(Transaction).filter(
            Transaction.id == transaction_id
        ).first()
        
        if not transaction:
            return {
                "status": "error",
                "message": f"Transaction with ID {transaction_id} not found"
            }

        if action:
            transaction.action = action
            
        if amount is not None:
            transaction.amount = amount
            
        if category:
            transaction.category = category
            
        if date:
            try:
                parsed_date = datetime.strptime(date, '%Y-%m-%d').date()
                transaction.date = parsed_date
            except ValueError:
                return {
                    "status": "error",
                    "message": f"Invalid date format: {date}. Use YYYY-MM-DD."
                }
                
        if description is not None:
            transaction.description = description
            
        session.commit()

        return {
            "id": transaction.id,
            "action": transaction.action,
            "amount": float(transaction.amount),
            "category": transaction.category,
            "date": transaction.date.isoformat(),
            "description": transaction.description,
            "status": "success",
            "message": f"Transaction with ID {transaction_id} successfully updated"
        }
    except Exception as e:
        logger.error(f"Error updating transaction: {str(e)}")
        session.rollback()
        return {
            "status": "error",
            "message": str(e)
        }

def create_tools() -> List[Tool]:
    """
    Create and return a list of tools for the agent to use.
    
    Returns:
        List[Tool]: Collection of tools for database operations and utility functions
    """
    time_tool = Tool(
        name="get_current_date",
        func=get_current_date,
        description="""
        Get the current date in YYYY-MM-DD format, with optional adjustment by days.
        
        Parameters:
        - date_diff (int, optional): Number of days to adjust from current date.
            Positive for future dates, negative for past dates.

        Returns:
        - string: Date in YYYY-MM-DD format
        
        Examples:
        - get_current_date() -> "2023-05-12" (today's date)
        - get_current_date(1) -> "2023-05-13" (tomorrow's date)
        - get_current_date(-2) -> "2023-05-10" (day before yesterday)
        
        Usage examples:
        - "What's the date today?" -> get_current_date()
        - "I spent $30 on lunch yesterday" -> get_current_date(-1) to get yesterday's date
        """
    )
    
    category_tool = Tool(
        name="get_available_categories",
        func=get_available_categories,
        description="""
        Get a list of available categories for financial transactions.
        
        Returns:
        - list: Array of category strings like ["groceries", "utilities", "salary", etc.]
        
        Use this tool when you need to suggest appropriate categories to the user
        or validate if a category is available in the system.
        
        Usage examples:
        - "What categories can I use for my expenses?" -> get_available_categories()
        - "Is there a category for medical expenses?" -> check if "healthcare" exists in get_available_categories()
        """
    )
    
    category_summary_tool = Tool(
        name="group_transactions_by_category",
        func=group_transactions_by_category,
        description="""
        Get a summary of total amounts spent or earned per category across all transactions.
        
        Returns:
        - object: Contains "category_totals" array with each item having:
            - category (string): Category name
            - total_amount (number): Sum of all transaction amounts in that category
            - count (number): Number of categories found
            - status (string): Success or error status
        
        Use this tool when the user wants reports, summaries, or analytics on their spending patterns.
        
        Usage examples:
        - "Show me a breakdown of my spending by category" -> group_transactions_by_category()
        - "What category am I spending the most money on?" -> group_transactions_by_category() and sort results
        """
    )
    
    # Multi-parameter tools requiring StructuredTool
    insert_transaction_tool = StructuredTool.from_function(
        func=insert_transaction,
        name="insert_transaction",
        description="""
        Add a new transaction to the database.
        
        Parameters:
        - action (string, required): The type of transaction (add_expense, remove_expense, add_income, remove_income)
        - amount (number, required): The monetary amount of the transaction (positive decimal number)
        - category (string, required): Category of the transaction (groceries, utilities, salary, etc.)
        - date (string, required): Date of the transaction in YYYY-MM-DD format
        - description (string, optional): Additional description or notes about the transaction
        
        Returns:
        - object: The created transaction with its new ID and status
        
        Use this tool when the user wants to record a new expense or income transaction.
        
        Usage examples:
        - "I spent $42.50 on groceries yesterday" -> insert_transaction(action="add_expense", amount=42.50, category="groceries", date="2023-05-07", description="Grocery shopping")
        - "Add my monthly salary of $3000" -> insert_transaction(action="add_income", amount=3000, category="salary", date="2023-05-01", description="Monthly salary deposit")
        """
    )
    
    get_transaction_tool = StructuredTool.from_function(
        func=get_transaction_by_id,
        name="get_transaction_by_id",
        description="""
        Retrieve a specific transaction by its ID.
        
        Parameters:
        - transaction_id (integer, required): The unique identifier of the transaction to retrieve
        
        Returns:
        - object: The transaction details including action, amount, category, date and description,
            or an error message if the transaction doesn't exist
        
        Use this tool when the user references a specific transaction by ID or needs to check transaction details
        before updating or deleting it.
        
        Usage examples:
        - "Show me transaction #15" -> get_transaction_by_id(transaction_id=15)
        - "What was the transaction with ID 42?" -> get_transaction_by_id(transaction_id=42)
        """
    )
    
    get_category_transactions_tool = StructuredTool.from_function(
        func=get_transactions_by_category,
        name="get_transactions_by_category",
        description="""
        Retrieve all transactions belonging to a specific category.
        
        Parameters:
        - category (string, required): The category to filter transactions by (e.g., "groceries", "utilities", "salary")
        
        Returns:
        - object: Contains "transactions" array with matching transactions, count of transactions found, and status
        
        Use this tool when the user wants to see all their spending or income in a particular category.
        
        Usage examples:
        - "Show me all my grocery expenses" -> get_transactions_by_category(category="groceries")
        - "How much have I spent on utilities?" -> get_transactions_by_category(category="utilities") and sum the amounts
        """
    )
    
    get_date_range_transactions_tool = StructuredTool.from_function(
        func=get_transactions_by_date_range,
        name="get_transactions_by_date_range",
        description="""
        Retrieve all transactions that occurred within a specific date range.
        
        Parameters:
        - start_date (string, required): Beginning date of the range in YYYY-MM-DD format
        - end_date (string, required): Ending date of the range in YYYY-MM-DD format
        
        Returns:
        - object: Contains "transactions" array with transactions in the date range,
            count of transactions found, date range parameters, and status
        
        Use this tool when the user wants to see transactions from a specific time period.
        
        Usage examples:
        - "Show me my expenses from last week" -> get_transactions_by_date_range(start_date="2023-05-01", end_date="2023-05-07")
        - "What did I spend in March?" -> get_transactions_by_date_range(start_date="2023-03-01", end_date="2023-03-31")
        """
    )
    
    delete_transaction_tool = StructuredTool.from_function(
        func=delete_transaction,
        name="delete_transaction",
        description="""
        Delete a transaction from the database by its ID.
        
        Parameters:
        - transaction_id (integer, required): The unique identifier of the transaction to delete
        
        Returns:
        - object: Contains details of the deleted transaction, status message, and confirmation
            or an error message if the transaction doesn't exist
        
        Use this tool when the user wants to remove a specific transaction from their records.
        Always confirm the transaction details with the user before deletion if possible.
        
        Usage examples:
        - "Delete transaction #23" -> delete_transaction(transaction_id=23)
        - "Remove the expense with ID 45 from my records" -> delete_transaction(transaction_id=45)
        """
    )
    
    update_transaction_tool = StructuredTool.from_function(
        func=update_transaction,
        name="update_transaction",
        description="""
        Update an existing transaction's details by its ID.
        
        Parameters:
        - transaction_id (integer, required): The unique identifier of the transaction to update
        - action (string, optional): New action type (add_expense, add_income, etc.)
        - amount (number, optional): New monetary amount
        - category (string, optional): New category
        - date (string, optional): New date in YYYY-MM-DD format
        - description (string, optional): New description
        
        Returns:
        - object: The updated transaction with all current values and status message,
            or an error message if the transaction doesn't exist or the update failed
        
        Use this tool when the user wants to modify details of an existing transaction.
        Only the fields that need to be changed should be specified; other fields will remain unchanged.
        
        Usage examples:
        - "Update transaction #123 to $75.50 for groceries" -> update_transaction(transaction_id=123, amount=75.50, category="groceries")
        - "Change the date of transaction ID 87 to yesterday" -> update_transaction(transaction_id=87, date="2023-05-11")
        """
    )
    
    return [
        time_tool, 
        category_tool,
        insert_transaction_tool,
        get_transaction_tool,
        get_category_transactions_tool,
        get_date_range_transactions_tool,
        category_summary_tool,
        delete_transaction_tool,
        update_transaction_tool
    ]

def setup_agent(llm: ChatOpenAI) -> AgentExecutor:
    prompt = transaction_extraction_prompt_template(transaction_list_parser)
    
    tools = create_tools()
    
    agent = create_tool_calling_agent(
        llm=llm,
        prompt=prompt,
        tools=tools,
    )

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True
    )

def process_agent_response(raw_response: Dict[str, Any]) -> Dict[str, Any]:
    try:
        json_str = raw_response['output']
        if "```json" in json_str:
            json_str = json_str.strip('```json\n').strip('```').strip()
        elif "```" in json_str:
            json_str = json_str.strip('```\n').strip('```').strip()

        parsed_response = json.loads(json_str)

        if "transactions" not in parsed_response:
            logger.warning("Response missing 'transactions' list, converting to new format")
            return {"transactions": [parsed_response]}
            
        return parsed_response
        
    except Exception as e:
        logger.error(f"Error processing agent response: {str(e)}")
        return {"transactions": []}

async def process_prompt(prompt: str) -> Dict[str, Any]:
    try:
        agent = setup_agent(get_llm())
        raw_response = await agent.ainvoke({"query": prompt})

        return process_agent_response(raw_response)
    except Exception as e:
        logger.error(f"Error in finance agent: {str(e)}")
        return {"transactions": [], "error": str(e)}

if __name__ == "__main__":
    start = datetime.now()
    print(f"Starting finance agent at {start}")
    agent = setup_agent(get_llm())
    response = agent.invoke(
        {"query": "I earned $1000 from my salary and spent $50 on groceries yesterday, also today I received a %20 commission from $10000 sell."}
    )
    print(process_agent_response(response))
    end = datetime.now()
    print(f"Finished finance agent at {end}")
    print(f"Total time taken: {end - start}")