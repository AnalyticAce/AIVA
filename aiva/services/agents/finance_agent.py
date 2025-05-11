from typing import Optional
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langgraph.checkpoint.memory import MemorySaver

from aiva.core.config import settings
from aiva.services.agents.agent_tools import (
    # Data entry tools
    get_current_date,
    get_available_categories,
    insert_transaction,
    delete_transaction,
    update_transaction,
    
    # Analysis tools
    get_transactions_by_category,
    get_transactions_by_date_range,
    group_transactions_by_category
)
from aiva.services.agents.prompts import (
    _TRANSACTION_EXTRACTION_PROMPT, _FINANCIAL_ANALYST_PROMPT
)

def get_llm(temperature=0.0):
    """Create an OpenAI LLM instance with specified settings"""
    return ChatOpenAI(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
        temperature=temperature
    )

def create_finance_system():
    """Create the finance system with a supervisor managing specialized agents"""
    
    # Create memory saver for persistence
    memory = MemorySaver()
    
    # Create LLM for agents
    llm = get_llm()
    
    # Enhanced prompt for data entry agent with explicit tool calling examples
    data_entry_prompt = f"""
{_TRANSACTION_EXTRACTION_PROMPT}

You are specialized in extracting and managing financial transactions from user input.

IMPORTANT: For each transaction mentioned by the user, you MUST use the insert_transaction tool.
DO NOT just return a JSON structure - you must actually call the tools to modify the database.

Follow this process for each interaction:
1. Extract transaction details from user input
2. For EACH transaction, use insert_transaction tool individually
3. Confirm all transactions have been processed

Example:
User: "I spent $25 on lunch yesterday and $50 on gas today"

Thought: I need to extract and add two transactions: lunch and gas.
Action: get_current_date
Action Input: {{}}

Observation: 2025-05-11

Thought: Now I'll add the lunch transaction from yesterday (2025-05-10)
Action: insert_transaction
Action Input: {{"amount": 25.00, "category": "food", "date": "2025-05-10", "description": "Lunch"}}

Observation: Transaction added successfully.

Thought: Now I'll add the gas transaction from today
Action: insert_transaction
Action Input: {{"amount": 50.00, "category": "transportation", "date": "2025-05-11", "description": "Gas"}}

Observation: Transaction added successfully.

Thought: Both transactions have been added to the database. I'll confirm this to the user.
"""
    
    # Enhanced prompt for analysis agent with explicit tool calling examples
    analysis_prompt = f"""
{_FINANCIAL_ANALYST_PROMPT}

You are specialized in analyzing financial data and providing insights.

IMPORTANT: You MUST use the provided tools to retrieve actual data from the database.
DO NOT make up results - always use tools to get real data before offering analysis.

Follow this process for each interaction:
1. Determine what financial information the user is requesting
2. Use appropriate tools to retrieve the necessary data
3. Analyze the data and provide insights

Example:
User: "What did I spend on groceries last month?"

Thought: I need to find grocery expenses for the past month.
Action: get_current_date
Action Input: {{}}

Observation: 2025-05-11

Thought: Now I'll retrieve all grocery transactions from April 11 to May 11
Action: get_transactions_by_category
Action Input: {{"category": "groceries", "start_date": "2025-04-11", "end_date": "2025-05-11"}}

Observation: [List of transactions...]

Thought: I'll now calculate the total amount spent on groceries
Action: group_transactions_by_category
Action Input: {{"start_date": "2025-04-11", "end_date": "2025-05-11"}}

Observation: {{"groceries": 342.50, "restaurants": 156.78...}}

Thought: I have the data now. I'll provide an analysis of the grocery spending for the past month.
"""
    
    # Create data entry agent
    data_entry_agent = create_react_agent(
        model=llm,
        tools=[
            get_current_date,
            get_available_categories,
            insert_transaction,
            delete_transaction,
            update_transaction,
        ],
        prompt=data_entry_prompt,
        name="data_entry_specialist",
        checkpointer=memory,
        debug=True,
    )
    
    # Create analysis agent
    analysis_agent = create_react_agent(
        model=llm,
        tools=[
            get_current_date,
            get_available_categories,
            get_transactions_by_category,
            get_transactions_by_date_range,
            group_transactions_by_category,
        ],
        prompt=analysis_prompt,
        name="financial_analyst",
        checkpointer=memory,
        debug=True,
    )
    
    # Create supervisor with custom system prompt
    supervisor_prompt = """
You are a financial assistant supervisor coordinating between two specialist agents:

1. data_entry_specialist: Handles recording transactions, adding, updating or deleting financial entries
2. financial_analyst: Handles analyzing financial data, querying transactions, and providing insights

Based on the user's query, decide which specialist is most appropriate to handle the request.
- For queries about adding, updating, or removing transactions, choose the data_entry_specialist
- For queries about analyzing spending patterns, getting reports, or financial insights, choose the financial_analyst

Make your decision based only on the capabilities of each specialist.

When a specialist has completed their task, you should:
1. Acknowledge the work completed
2. Provide a helpful summary if appropriate
3. Ask if the user needs anything else
"""
    
    # Create supervisor workflow
    workflow = create_supervisor(
        agents=[data_entry_agent, analysis_agent],
        model=get_llm(temperature=0),
        prompt=supervisor_prompt,
    )
    
    # Compile the workflow
    finance_system = workflow.compile()
    
    return finance_system

def process_prompt(query: str, thread_id: Optional[str] = None):
    """Process a user query through the finance system"""
    
    # Create the finance system
    finance_system = create_finance_system()
    
    # Set up initial state with the user query
    initial_state = {
        "messages": [
            {
                "role": "user",
                "content": query
            }
        ]
    }
    
    # Configure thread ID if provided
    config = {}
    if thread_id:
        config["configurable"] = {"thread_id": thread_id}
    
    # Invoke the system and return the result
    result = finance_system.invoke(initial_state, config)
    
    return result

if __name__ == "__main__":
    query = "I spent $42.50 on groceries yesterday and $100 on Uber the day before"
    result = process_prompt(query)
    print(result)