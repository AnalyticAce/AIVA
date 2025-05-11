from typing import Any, Dict, Optional, List, Generator
from datetime import datetime
from contextlib import contextmanager

from sqlalchemy import create_engine, Column, Integer, String, Date, Numeric, func
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError

from aiva.core.logging import logger

# Database setup
Base = declarative_base()
engine = create_engine('sqlite:///transactions.db')
SessionFactory = sessionmaker(bind=engine)

@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """
    Context manager for database sessions.
    
    Provides a transactional scope around a series of operations.
    Automatically handles commit/rollback and session closing.
    
    Yields:
        Session: SQLAlchemy session object
        
    Example:
        >>> with session_scope() as session:
        ...     transaction = Transaction(...)
        ...     session.add(transaction)
        ...     # No need to commit - happens automatically if no exceptions
    """
    session = SessionFactory()
    try:
        yield session
        session.commit()
    except SQLAlchemyError as e:
        session.rollback()
        logger.error(f"Database error: {str(e)}")
        raise
    finally:
        session.close()

class Transaction(Base):
    """
    SQLAlchemy model representing a financial transaction in the database.
    
    Attributes:
        id (int): Unique identifier for the transaction (auto-generated)
        action (str): Type of transaction - REQUIRED
            Valid values: "add_expense", "add_income", "remove_expense", "remove_income"
        amount (Decimal): Monetary value of the transaction - REQUIRED
        category (str): Category of the transaction - REQUIRED
            (see get_available_categories() for valid options)
        date (Date): Date when the transaction occurred - REQUIRED
            Format: YYYY-MM-DD
        description (str): Additional details about the transaction - OPTIONAL
    """
    __tablename__ = 'transactions'
    
    id = Column("id", Integer, primary_key=True)
    action = Column("action", String, nullable=False)
    amount = Column("amount", Numeric(10, 2), nullable=False)
    category = Column("category", String, nullable=False)
    date = Column("date", Date, nullable=False)
    description = Column("description", String, nullable=True)
    
    def __init__(self, action, amount, category, date, description=None):
        """
        Initialize a new Transaction instance.
        
        Args:
            action (str): REQUIRED - Type of transaction 
                (add_expense, add_income, remove_expense, remove_income)
            amount (float): REQUIRED - Monetary value of the transaction
            category (str): REQUIRED - Category of the transaction
            date (date): REQUIRED - Date when the transaction occurred
            description (str, optional): Additional details. Defaults to None.
        """
        self.action = action
        self.amount = amount
        self.category = category
        self.date = date
        self.description = description
    
    def __repr__(self):
        return f"<Transaction(action={self.action}, amount={self.amount}, category={self.category}, date={self.date}, description={self.description})>"

# Create tables if they don't exist
Base.metadata.create_all(bind=engine)

def get_current_date() -> str:
    """
    Get the current date in ISO format (YYYY-MM-DD).
    
    Returns:
        str: Current date in ISO format (YYYY-MM-DD)
        
    Example:
        >>> get_current_date()
        '2025-05-12'
    """
    return datetime.now().date().isoformat()

def get_available_categories() -> List[str]:
    """
    Get a list of all valid transaction categories supported by the system.
    
    Returns:
        List[str]: List of available category strings
        
    Example:
        >>> get_available_categories()
        ['groceries', 'transportation', 'utilities', ...]
    """
    return [
        "groceries", "transportation", "utilities", "rent", "dining", 
        "entertainment", "healthcare", "education", "shopping", 
        "travel", "salary", "investment", "gift", "other"
    ]

def insert_transaction(transaction_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Insert a new transaction into the database.
    
    Args:
        transaction_data (Dict[str, Any]): Dictionary containing transaction information
            REQUIRED fields:
                - action (str): Type of transaction (add_expense, add_income, etc.)
                - amount (float): Monetary value
                - category (str): Transaction category
                - date (str): Date in YYYY-MM-DD format
            OPTIONAL fields:
                - description (str): Additional details about the transaction
    
    Returns:
        Dict[str, Any]: Result of the operation
            - On success: {"success": True, "id": transaction_id, "message": "..."}
            - On failure: {"success": False, "error": "error message"}
            
    Example:
        >>> insert_transaction({
        ...     "action": "add_expense",
        ...     "amount": 42.50,
        ...     "category": "groceries",
        ...     "date": "2025-05-10",
        ...     "description": "Weekly shopping"
        ... })
        {"success": True, "id": 1, "message": "Transaction added with ID: 1"}
    """
    try:
        with session_scope() as session:
            date_obj = datetime.strptime(transaction_data["date"], "%Y-%m-%d").date()
            
            new_transaction = Transaction(
                action=transaction_data["action"],
                amount=transaction_data["amount"],
                category=transaction_data["category"],
                date=date_obj,
                description=transaction_data.get("description")
            )
            
            session.add(new_transaction)
            # Commit happens automatically when context exits
            return {"success": True, "id": new_transaction.id}
            
    except Exception as e:
        logger.error(f"Error inserting transaction: {str(e)}")
        return {"success": False, "error": str(e)}

def get_transaction_by_id(transaction_id: int) -> Dict[str, Any]:
    """
    Retrieve a transaction by its ID.
    
    Args:
        transaction_id (int): REQUIRED - The unique identifier of the transaction
    
    Returns:
        Dict[str, Any]: Transaction data or error message
            - On success: Dict with transaction details
            - On failure: {"success": False, "error": "error message"}
            
    Example:
        >>> get_transaction_by_id(1)
        {
            "id": 1,
            "action": "add_expense",
            "amount": 42.5,
            "category": "groceries",
            "date": "2025-05-10",
            "description": "Weekly shopping"
        }
    """
    try:
        with session_scope() as session:
            transaction = session.query(Transaction).filter(Transaction.id == transaction_id).first()
            if transaction:
                return {
                    "id": transaction.id,
                    "action": transaction.action,
                    "amount": float(transaction.amount),
                    "category": transaction.category,
                    "date": transaction.date.isoformat(),
                    "description": transaction.description
                }
            else:
                return {"success": False, "error": f"No transaction found with ID: {transaction_id}"}
    except Exception as e:
        logger.error(f"Error retrieving transaction: {str(e)}")
        return {"success": False, "error": str(e)}

def get_transactions_by_category(category: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> list:
    """
    Retrieve all transactions in a specific category with optional date filtering.
    
    Args:
        category (str): REQUIRED - Category to filter transactions by
        start_date (str, optional): Start date in YYYY-MM-DD format
        end_date (str, optional): End date in YYYY-MM-DD format
    
    Returns:
        list: List of transaction dictionaries or error message
            - On success: List of transaction data
            - On failure: {"success": False, "error": "error message"}
            
    Example:
        >>> get_transactions_by_category("groceries", "2025-05-01", "2025-05-12")
        [
            {
                "id": 1,
                "action": "add_expense",
                "amount": 42.5,
                "category": "groceries",
                "date": "2025-05-10",
                "description": "Weekly shopping"
            },
            ...
        ]
    """
    try:
        with session_scope() as session:
            transactions = session.query(Transaction).filter(Transaction.category == category).all()
            return [
                {
                    "id": t.id,
                    "action": t.action,
                    "amount": float(t.amount),
                    "category": t.category,
                    "date": t.date.isoformat(),
                    "description": t.description
                }
                for t in transactions
            ]
    except Exception as e:
        logger.error(f"Error retrieving transactions by category: {str(e)}")
        return {"success": False, "error": str(e)}

def get_transactions_by_date_range(start_date: str, end_date: str) -> list:
    """
    Retrieve transactions between two dates.
    
    Args:
        start_date (str): REQUIRED - Start date in YYYY-MM-DD format
        end_date (str): REQUIRED - End date in YYYY-MM-DD format
    
    Returns:
        list: List of transaction dictionaries or error message
            - On success: List of transaction data
            - On failure: {"success": False, "error": "error message"}
            
    Example:
        >>> get_transactions_by_date_range("2025-05-01", "2025-05-12")
        [
            {
                "id": 1,
                "action": "add_expense",
                "amount": 42.5,
                "category": "groceries",
                "date": "2025-05-10",
                "description": "Weekly shopping"
            },
            ...
        ]
    """
    try:
        with session_scope() as session:
            start_date_obj = datetime.strptime(start_date, "%Y-%m-%d").date()
            end_date_obj = datetime.strptime(end_date, "%Y-%m-%d").date()
            
            transactions = session.query(Transaction).filter(
                Transaction.date >= start_date_obj,
                Transaction.date <= end_date_obj
            ).all()
            
            return [
                {
                    "id": t.id,
                    "action": t.action,
                    "amount": float(t.amount),
                    "category": t.category,
                    "date": t.date.isoformat(),
                    "description": t.description
                }
                for t in transactions
            ]
    except Exception as e:
        logger.error(f"Error retrieving transactions by date range: {str(e)}")
        return {"success": False, "error": str(e)}

def group_transactions_by_category(include_income: bool = True, include_expenses: bool = True, 
                                start_date: Optional[str] = None, end_date: Optional[str] = None) -> list:
    """
    Get summary of total amounts by category within optional date range.
    
    Args:
        include_income (bool, optional): OPTIONAL - Include income transactions. Defaults to True.
        include_expenses (bool, optional): OPTIONAL - Include expense transactions. Defaults to True.
        start_date (str, optional): OPTIONAL - Start date in YYYY-MM-DD format
        end_date (str, optional): OPTIONAL - End date in YYYY-MM-DD format
    
    Returns:
        list: List of category summary dictionaries or error message
            - On success: List of category summaries with totals
            - On failure: {"success": False, "error": "error message"}
            
    Example:
        >>> group_transactions_by_category(include_income=False, start_date="2025-05-01", end_date="2025-05-12")
        [
            {
                "category": "groceries",
                "total_amount": 87.50,
                "transaction_count": 2
            },
            ...
        ]
    """
    try:
        with session_scope() as session:
            query = session.query(
                Transaction.category,
                func.sum(Transaction.amount).label('total_amount'),
                func.count(Transaction.id).label('count')
            )
            
            # Filter by action type if specified
            filters = []
            if include_income and not include_expenses:
                filters.append(Transaction.action.in_(['add_income']))
            elif include_expenses and not include_income:
                filters.append(Transaction.action.in_(['add_expense']))
            
            # Apply date range if specified
            if start_date:
                start_date_obj = datetime.strptime(start_date, "%Y-%m-%d").date()
                filters.append(Transaction.date >= start_date_obj)
            if end_date:
                end_date_obj = datetime.strptime(end_date, "%Y-%m-%d").date()
                filters.append(Transaction.date <= end_date_obj)
                
            # Apply filters and group by category
            if filters:
                query = query.filter(*filters)
            
            result = query.group_by(Transaction.category).all()
            
            return [
                {
                    "category": item.category,
                    "total_amount": float(item.total_amount),
                    "transaction_count": item.count
                }
                for item in result
            ]
    except Exception as e:
        logger.error(f"Error grouping transactions by category: {str(e)}")
        return {"success": False, "error": str(e)}

def delete_transaction(transaction_id: int) -> Dict[str, Any]:
    """
    Remove a transaction from the database.
    
    Args:
        transaction_id (int): REQUIRED - The unique identifier of the transaction to delete
    
    Returns:
        Dict[str, Any]: Result of the operation
            - On success: {"success": True, "message": "Transaction X deleted successfully"}
            - On failure: {"success": False, "error": "error message"}
            
    Example:
        >>> delete_transaction(1)
        {"success": True, "message": "Transaction 1 deleted successfully"}
    """
    try:
        with session_scope() as session:
            transaction = session.query(Transaction).filter(Transaction.id == transaction_id).first()
            if transaction:
                session.delete(transaction)
                return {"success": True, "message": f"Transaction {transaction_id} deleted successfully"}
            else:
                return {"success": False, "error": f"No transaction found with ID: {transaction_id}"}
    except Exception as e:
        logger.error(f"Error deleting transaction: {str(e)}")
        return {"success": False, "error": str(e)}

def update_transaction(transaction_id: int, updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update fields of an existing transaction.
    
    Args:
        transaction_id (int): REQUIRED - The unique identifier of the transaction to update
        updates (Dict[str, Any]): REQUIRED - Dictionary containing fields to update
            OPTIONAL update fields (include only fields that need updating):
                - action (str): Type of transaction (add_expense, add_income, etc.)
                - amount (float): Monetary value
                - category (str): Transaction category
                - date (str): Date in YYYY-MM-DD format
                - description (str): Additional details about the transaction
    
    Returns:
        Dict[str, Any]: Result of the operation
            - On success: {"success": True, "message": "Transaction X updated successfully"}
            - On failure: {"success": False, "error": "error message"}
            
    Example:
        >>> update_transaction(1, {"amount": 45.75, "description": "Updated grocery purchase"})
        {"success": True, "message": "Transaction 1 updated successfully"}
    """
    try:
        with session_scope() as session:
            transaction = session.query(Transaction).filter(Transaction.id == transaction_id).first()
            if transaction:
                # Update fields that are present in the updates dict
                if "action" in updates:
                    transaction.action = updates["action"]
                if "amount" in updates:
                    transaction.amount = updates["amount"]
                if "category" in updates:
                    transaction.category = updates["category"]
                if "date" in updates:
                    transaction.date = datetime.strptime(updates["date"], "%Y-%m-%d").date()
                if "description" in updates:
                    transaction.description = updates["description"]
                    
                return {"success": True, "message": f"Transaction {transaction_id} updated successfully"}
            else:
                return {"success": False, "error": f"No transaction found with ID: {transaction_id}"}
    except Exception as e:
        logger.error(f"Error updating transaction: {str(e)}")
        return {"success": False, "error": str(e)}


# insert transaction

if __name__ == "__main__":
    # Example usage
    transaction_data = {
        "action": "add_expense",
        "amount": 50.00,
        "category": "groceries",
        "date": "2025-05-11",
        "description": "Weekly grocery shopping"
    }
    
    result = insert_transaction(transaction_data)
    print(result)

    transactions = get_transactions_by_category("groceries")
    print(transactions)