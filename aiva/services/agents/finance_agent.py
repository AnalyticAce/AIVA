from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict
import json
from enum import Enum
from datetime import datetime, timedelta
from decimal import Decimal

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import Tool
from pydantic import BaseModel, Field, ConfigDict

from aiva.core.config import settings
from aiva.core.logging import logger

class FinanceActionType(str, Enum):
    """
    Enumeration of possible financial actions for better type safety and validation.
    """
    ADD_EXPENSE = "add_expense"
    REMOVE_EXPENSE = "remove_expense"
    ADD_INCOME = "add_income"
    REMOVE_INCOME = "remove_income"
    UNKNOWN = "unknown"

class LLMExtractionOutput(BaseModel):
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
    Container model for one or multiple transactions.
    This provides a unified response format regardless of transaction count.
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
    
        List of available categories and tools:
        - Always use the get_current_date tool if the user doesn't specify a date.
        - Use the get_available_categories tool to get a list of available categories.

        Just return the JSON response without any additional text.
        """),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    return extraction_prompt

def get_current_date(date_diff: Any = 0) -> str:
    """
    Get the current date in YYYY-MM-DD format, adjusted by a specified number of days.
    
    Args:
        date_diff (Any): Number of days to adjust the current date by.
            Positive values for future dates, negative for past dates.
            Can be passed as string or int.
            
    Returns:
        str: Current date in YYYY-MM-DD format, adjusted by the specified number of days.
    
    Example:
        if today is 2023-05-12 then:
        get_current_date() -> "2023-05-12"
        get_current_date(1) -> "2023-05-13"
        get_current_date(-2) -> "2023-05-10"
        get_current_date(0) -> "2023-05-12"
    """
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

def create_tools() -> List[Tool]:
    time_tool = Tool(
        name="get_current_date",
        func=get_current_date,
        description="Get the current date in YYYY-MM-DD format."
    )
    
    category_tool = Tool(
        name="get_available_categories",
        func=get_available_categories,
        description="Get a list of available categories for financial transactions."
    )
    
    return [time_tool, category_tool]

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
    agent = setup_agent(get_llm())
    response = agent.invoke({"query": "I spent $100 on Uber to go to work the day before yesterday and $50 on groceries yesterday"})
    print(process_agent_response(response))