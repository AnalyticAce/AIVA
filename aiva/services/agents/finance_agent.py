from typing import Optional, Literal

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.tools import StructuredTool

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent, ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from aiva.core.config import settings
from aiva.services.agents.agent_model import (
    QueryType, FinanceAgentState,
)
from aiva.services.agents.agent_tools import (
    get_current_date,
    get_available_categories,
    insert_transaction,
    delete_transaction,
    update_transaction,
    get_available_categories,
    get_transactions_by_category,
    get_transactions_by_date_range,
    group_transactions_by_category
)
from aiva.services.agents.prompts import (
    _TRANSACTION_EXTRACTION_PROMPT, _FINANCIAL_ANALYST_PROMPT, _QUERY_CLASSIFICATION_PROMPT
)

def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
        temperature=0.0
    )

def prompt_builder(
    prompt: str,
) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", prompt),
        ("placeholder", "{agent_scratchpad}"),
    ])

def classify_query(state: FinanceAgentState):
    query = state["query"]
    llm = get_llm()

    classification_prompt = ChatPromptTemplate.from_messages([
        ("system", _QUERY_CLASSIFICATION_PROMPT),
        ("human", "{query}")
    ])
    
    chain = classification_prompt | llm
    result = chain.invoke({"query": query})
    
    result_text = result.content.strip().upper()
    print("LLM Classification Response:", result_text)
    query_type = QueryType.UNKNOWN
    if "DATA_ENTRY" in result_text:
        query_type = QueryType.DATA_ENTRY
    elif "ANALYSIS" in result_text:
        query_type = QueryType.ANALYSIS

    return {"query_type": query_type}

def route_query(state: FinanceAgentState) -> Literal["data_entry_agent", "analysis_agent"]:
    query_type = state["query_type"]
    
    if query_type == QueryType.DATA_ENTRY:
        return "data_entry_agent"
    elif query_type == QueryType.ANALYSIS:
        return "analysis_agent"
    else:
        return "data_entry_agent"

def modify_query_state(state: FinanceAgentState):
    query = state["query"]

    human_message = {"type": "human", "content": query}
    
    if not any(msg.get("content") == query and msg.get("type") == "human" 
            for msg in state.get("messages", [])):
        return {"messages": [human_message]}
    
    return {}

def create_finance_agents():
    llm = get_llm()
    
    memory = MemorySaver()
    
    data_entry_prompt = prompt_builder(prompt=_TRANSACTION_EXTRACTION_PROMPT)
    analysis_prompt = prompt_builder(prompt=_FINANCIAL_ANALYST_PROMPT)
    
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
        checkpointer=memory,
        debug=True,
    )
    
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
        checkpointer=memory,
        debug=True,
    )
    
    return data_entry_agent, analysis_agent

def create_finance_agent_graph():
    data_entry_agent, analysis_agent = create_finance_agents()

    data_entry_tools = ToolNode([
        get_current_date,
        get_available_categories,
        insert_transaction,
        delete_transaction,
        update_transaction,
    ])
    
    analysis_tools = ToolNode([
        get_current_date,
        get_available_categories,
        get_transactions_by_category,
        get_transactions_by_date_range,
        group_transactions_by_category,
    ])

    workflow = StateGraph(FinanceAgentState)

    workflow.add_node("modify_query_state", modify_query_state)
    workflow.add_node("classify_query", classify_query)
    workflow.add_node("data_entry_agent", data_entry_agent)
    workflow.add_node("analysis_agent", analysis_agent)
    workflow.add_node("data_entry_tools", data_entry_tools)
    workflow.add_node("analysis_tools", analysis_tools)

    workflow.add_edge(START, "modify_query_state")
    workflow.add_edge("modify_query_state", "classify_query")
    workflow.add_conditional_edges(
        "classify_query",
        route_query,
        {
            "data_entry_agent": "data_entry_agent",
            "analysis_agent": "analysis_agent",
        }
    )

    workflow.add_conditional_edges(
        "data_entry_agent",
        tools_condition,
        {
            "tools": "data_entry_tools",
            "__end__": END
        }
    )
    
    workflow.add_conditional_edges(
        "analysis_agent",
        tools_condition,
        {
            "tools": "analysis_tools",
            "__end__": END
        }
    )

    workflow.add_edge("data_entry_tools", "data_entry_agent")
    workflow.add_edge("analysis_tools", "analysis_agent")
    
    finance_graph = workflow.compile()
    
    return finance_graph

def process_prompt(query: str, thread_id: Optional[str] = None):
    finance_agent = create_finance_agent_graph()
    
    initial_state = {
        "query": query,
        "query_type": QueryType.UNKNOWN,
        "messages": [],
        "transactions": [],
        "summary": None
    }

    config = {}
    if thread_id:
        config["configurable"] = {"thread_id": thread_id}

    result = finance_agent.invoke(initial_state, config)
    
    return result

if __name__ == "__main__":
    query = "I spent $42.50 on groceries yesterday and $100 on Uber the day before"
    result = process_prompt(query)
    print(result)