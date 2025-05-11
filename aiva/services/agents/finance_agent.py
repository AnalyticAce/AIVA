from datetime import datetime
from typing import Dict, Optional
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
    TRANSACTION_EXTRACTION_PROMPT,
    FINANCIAL_ANALYST_PROMPT,
    SUPERVISOR_PROMPT
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
        prompt=TRANSACTION_EXTRACTION_PROMPT,
        name="data_entry_specialist",
        checkpointer=memory,
        # debug=True,
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
        prompt=FINANCIAL_ANALYST_PROMPT,
        name="financial_analyst",
        checkpointer=memory,
        # debug=True,
    )
    
    # Create supervisor workflow
    workflow = create_supervisor(
        agents=[data_entry_agent, analysis_agent],
        model=get_llm(temperature=0),
        prompt=SUPERVISOR_PROMPT,
    )
    
    # Compile the workflow
    finance_system = workflow.compile()
    
    return finance_system

from textwrap import wrap

def format_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def print_ascii_conversation(messages: list):
    """Prints the conversation in a beautiful ASCII format"""
    print("\n" + "═" * 80)
    print(" FINANCIAL ASSISTANT CONVERSATION ".center(80, ' '))
    print("═" * 80 + "\n")
    
    for msg in messages:
        if not hasattr(msg, 'content'):
            continue
            
        # Determine message type and formatting
        if msg.__class__.__name__ == "HumanMessage":
            prefix = "👤 USER: "
            box_char = "─"
            indent = " " * 9
            width = 70
        elif msg.__class__.__name__ == "AIMessage":
            prefix = "🤖 ASSISTANT: "
            box_char = "═"
            indent = " " * 14
            width = 65
        elif msg.__class__.__name__ == "ToolMessage":
            prefix = "🛠️ TOOL: "
            box_char = "─"
            indent = " " * 9
            width = 70
        else:
            continue
            
        # Format the content
        content = msg.content if msg.content else ""
        
        # Handle tool calls if present
        tool_calls = ""
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            tool_calls = "\n" + indent + "Tool Calls:\n"
            for call in msg.tool_calls:
                tool_calls += indent + f"- {call.get('name', 'unknown')}\n"
                args = call.get('args', {})
                if args:
                    tool_calls += indent + "  Arguments:\n"
                    for k, v in args.items():
                        tool_calls += indent + f"  • {k}: {v}\n"
        
        # Handle usage metadata if present
        usage_info = ""
        if hasattr(msg, 'usage_metadata'):
            usage = getattr(msg, 'usage_metadata', {})
            if usage:
                usage_info = "\n" + indent + "Token Usage:\n"
                usage_info += indent + f"- Input: {usage.get('input_tokens', 0)}\n"
                usage_info += indent + f"- Output: {usage.get('output_tokens', 0)}\n"
                usage_info += indent + f"- Total: {usage.get('total_tokens', 0)}\n"
        
        # Print the message
        print(f"╭{box_char * (len(prefix) + width)}╮")
        print(f"│{prefix}{content.ljust(width)}│")
        
        if tool_calls:
            for line in tool_calls.split('\n'):
                print(f"│{line.ljust(width + len(prefix))}│")
        
        if usage_info:
            for line in usage_info.split('\n'):
                print(f"│{line.ljust(width + len(prefix))}│")
        
        print(f"╰{box_char * (len(prefix) + width)}╯\n")

def process_prompt(query: str, thread_id: Optional[str] = None):
    """Process a user query and print beautiful ASCII output"""
    finance_system = create_finance_system()
    initial_state = {"messages": [{"role": "user", "content": query}]}
    config = {"configurable": {"thread_id": thread_id}} if thread_id else {}
    
    print("\n" + "★" * 40)
    print(f" PROCESSING QUERY ".center(40, '★'))
    print("★" * 40 + "\n")
    print(f"📝 Query: {query}")
    print(f"⏱️ Start time: {format_timestamp()}\n")
    
    try:
        result = finance_system.invoke(initial_state, config)
        print_ascii_conversation(result.get("messages", []))
        
        # Extract final answer
        final_messages = [m for m in result.get("messages", []) 
                        if hasattr(m, 'content') and m.content and 
                        not getattr(m, 'tool_calls', None)]
        
        if final_messages:
            print("\n" + "✔" * 40)
            print(f" FINAL ANSWER ".center(40, '✔'))
            print("✔" * 40 + "\n")
            print(final_messages[-1].content)
        
        print(f"\n⏱️ End time: {format_timestamp()}")
        
    except Exception as e:
        print("\n" + "⚠" * 40)
        print(f" ERROR ".center(40, '⚠'))
        print("⚠" * 40 + "\n")
        print(f"Error processing query: {str(e)}")

if __name__ == "__main__":
    query = "How much did I spend on groceries this month?"
    process_prompt(query)