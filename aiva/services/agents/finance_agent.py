from typing import Optional, Literal
import json

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.tools import StructuredTool

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from aiva.core.config import settings
from aiva.services.agents.agent_model import (
    QueryType, FinanceAgentState,
)
from aiva.services.agents.agent_tools import data_tools, report_tools, analysis_tools
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

data_entry_toolkit = [StructuredTool.from_function(tool) for tool in data_tools]
analysis_toolkit = [StructuredTool.from_function(tool) for tool in analysis_tools]

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
        tools=data_entry_toolkit,
        prompt=data_entry_prompt,
        checkpointer=memory,
        debug=True,
    )
    
    analysis_agent = create_react_agent(
        model=llm,
        tools=analysis_toolkit,
        prompt=analysis_prompt,
        checkpointer=memory,
        debug=True,
    )
    
    return data_entry_agent, analysis_agent

def create_finance_agent_graph():
    data_entry_agent, analysis_agent = create_finance_agents()

    workflow = StateGraph(FinanceAgentState)

    workflow.add_node("modify_query_state", modify_query_state)
    workflow.add_node("classify_query", classify_query)
    workflow.add_node("data_entry_agent", data_entry_agent)
    workflow.add_node("analysis_agent", analysis_agent)

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

    workflow.add_edge("data_entry_agent", END)
    workflow.add_edge("analysis_agent", END)
    
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