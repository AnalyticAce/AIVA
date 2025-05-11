from typing import Any, Dict, Optional
from datetime import datetime

from sqlalchemy import create_engine, Column, Integer, String, Date, Numeric, func
from sqlalchemy.orm import declarative_base, sessionmaker

from aiva.core.logging import logger
from langchain_core.tools import tool

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

@tool
def get_current_date() -> str:
    """Get the current date in ISO format (YYYY-MM-DD)."""
    return datetime.now().date().isoformat()

@tool
def get_available_categories() -> list:
    """Get a list of available transaction categories."""
    return [
        "groceries", "transportation", "utilities", "rent", "dining", 
        "entertainment", "healthcare", "education", "shopping", 
        "travel", "salary", "investment", "gift", "other"
    ]

@tool
def insert_transaction(transaction_data: Dict[str, Any]) -> Dict[str, Any]:
    """Insert a new transaction into the database."""
    try:
        date_obj = datetime.strptime(transaction_data["date"], "%Y-%m-%d").date()
        
        new_transaction = Transaction(
            action=transaction_data["action"],
            amount=transaction_data["amount"],
            category=transaction_data["category"],
            date=date_obj,
            description=transaction_data.get("description")
        )

        session.add(new_transaction)
        session.commit()
        
        return {"success": True, "id": new_transaction.id, "message": f"Transaction added with ID: {new_transaction.id}"}
    except Exception as e:
        session.rollback()
        logger.error(f"Error inserting transaction: {str(e)}")
        return {"success": False, "error": str(e)}

@tool
def get_transaction_by_id(transaction_id: int) -> Dict[str, Any]:
    """Retrieve a transaction by its ID."""
    try:
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

@tool
def get_transactions_by_category(category: str) -> list:
    """Retrieve all transactions in a specific category."""
    try:
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

@tool
def get_transactions_by_date_range(start_date: str, end_date: str) -> list:
    """Retrieve transactions between two dates."""
    try:
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

@tool
def group_transactions_by_category(include_income: bool = True, include_expenses: bool = True, 
                                start_date: Optional[str] = None, end_date: Optional[str] = None) -> list:
    """Get summary of total amounts by category within optional date range."""
    try:
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

@tool
def delete_transaction(transaction_id: int) -> Dict[str, Any]:
    """Remove a transaction from the database."""
    try:
        transaction = session.query(Transaction).filter(Transaction.id == transaction_id).first()
        if transaction:
            session.delete(transaction)
            session.commit()
            return {"success": True, "message": f"Transaction {transaction_id} deleted successfully"}
        else:
            return {"success": False, "error": f"No transaction found with ID: {transaction_id}"}
    except Exception as e:
        session.rollback()
        logger.error(f"Error deleting transaction: {str(e)}")
        return {"success": False, "error": str(e)}

@tool
def update_transaction(transaction_id: int, updates: Dict[str, Any]) -> Dict[str, Any]:
    """Update fields of an existing transaction."""
    try:
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
                
            session.commit()
            return {"success": True, "message": f"Transaction {transaction_id} updated successfully"}
        else:
            return {"success": False, "error": f"No transaction found with ID: {transaction_id}"}
    except Exception as e:
        session.rollback()
        logger.error(f"Error updating transaction: {str(e)}")
        return {"success": False, "error": str(e)}
