from datetime import datetime
from typing import Dict, Optional
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langgraph.checkpoint.memory import MemorySaver

from aiva.core.config import settings
from aiva.services.agents.agent_tools import (
    get_current_date,
    get_available_categories,
    insert_transaction,
    delete_transaction,
    update_transaction,
    update_multiple_transactions,
    get_transaction_by_id,
    get_transactions_by_description,
    get_all_transactions,
    get_transactions_by_category,
    get_transactions_by_date_range,
    group_transactions_by_category
)
from aiva.services.agents.prompts import (
    TRANSACTION_EXTRACTION_PROMPT,
    FINANCIAL_ANALYST_PROMPT,
    SUPERVISOR_PROMPT
)

from datetime import datetime
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.box import ROUNDED
from rich.markdown import Markdown

console = Console()


def get_llm(temperature=0.0):
    return ChatOpenAI(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
        temperature=temperature
    )

def create_finance_system():
    memory = MemorySaver()

    llm = get_llm()
    
    data_entry_agent = create_react_agent(
        model=llm,
        tools=[
            get_current_date,
            get_available_categories,
            insert_transaction,
            delete_transaction,
            update_transaction,
            get_transactions_by_description,
            update_multiple_transactions,
            get_transaction_by_id,
            get_all_transactions,
        ],
        prompt=TRANSACTION_EXTRACTION_PROMPT,
        name="data_entry_specialist",
        checkpointer=memory,
    )
    
    analysis_agent = create_react_agent(
        model=llm,
        tools=[
            get_current_date,
            get_all_transactions,
            get_transaction_by_id,
            get_available_categories,
            get_transactions_by_category,
            get_transactions_by_description,
            get_transactions_by_date_range,
            group_transactions_by_category,
        ],
        prompt=FINANCIAL_ANALYST_PROMPT,
        name="financial_analyst",
        checkpointer=memory,
    )

    workflow = create_supervisor(
        agents=[data_entry_agent, analysis_agent],
        model=get_llm(temperature=0),
        prompt=SUPERVISOR_PROMPT,
    )
    
    finance_system = workflow.compile()
    
    return finance_system


def format_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def print_ascii_conversation(messages: list):
    console.rule("[bold blue]FINANCIAL ASSISTANT CONVERSATION", style="blue")

    for msg in messages:
        if not hasattr(msg, 'content'):
            continue

        role = msg.__class__.__name__
        content = msg.content or ""
        prefix = ""
        color = "white"
        title = ""
        padding = (1, 2)

        if role == "HumanMessage":
            prefix = "👤 USER"
            color = "green"
        elif role == "AIMessage":
            prefix = "🤖 ASSISTANT"
            color = "cyan"
        elif role == "ToolMessage":
            prefix = "🛠️ TOOL"
            color = "yellow"
        else:
            continue

        tool_calls_text = ""
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            tool_calls_text += "\n[bold yellow]Tool Calls:[/bold yellow]\n"
            for call in msg.tool_calls:
                name = call.get("name", "unknown")
                tool_calls_text += f" • [bold]{name}[/bold]\n"
                args = call.get("args", {})
                if args:
                    for k, v in args.items():
                        tool_calls_text += f"   - {k}: {v}\n"

        usage_text = ""
        if hasattr(msg, 'usage_metadata'):
            usage = getattr(msg, 'usage_metadata', {})
            if usage:
                usage_text += "\n[bold magenta]Token Usage:[/bold magenta]\n"
                usage_text += f" - Input: {usage.get('input_tokens', 0)}\n"
                usage_text += f" - Output: {usage.get('output_tokens', 0)}\n"
                usage_text += f" - Total: {usage.get('total_tokens', 0)}\n"

        full_text = content + tool_calls_text + usage_text

        panel = Panel(
            full_text.strip(),
            title=f"[{color}]{prefix}",
            border_style=color,
            box=ROUNDED,
            padding=padding
        )
        console.print(panel)

def process_prompt(query: str, thread_id: Optional[str] = None, debug=False):
    finance_system = create_finance_system()
    initial_state = {"messages": [{"role": "user", "content": query}]}
    config = {"configurable": {"thread_id": thread_id}} if thread_id else {}
    config = {**config, "recursion_limit": 200}
    
    try:
        result = finance_system.invoke(initial_state, config)
        
        if debug:
            console.rule("[bold green]PROCESSING QUERY", style="green")
            console.print(f"[bold]📝 Query:[/bold] {query}")
            console.print(f"[bold]⏱️ Start time:[/bold] {format_timestamp()}\n")
            
            print_ascii_conversation(result.get("messages", []))
            final_messages = [
                m for m in result.get("messages", [])
                if hasattr(m, 'content') and m.content and not getattr(m, 'tool_calls', None)
            ]

            if final_messages:
                console.rule("[bold green]FINAL ANSWER", style="green")
                console.print(Markdown(final_messages[-1].content.strip()))
                console.print(f"\n[bold]⏱️ End time:[/bold] {format_timestamp()}")
        else:
            print(result)

    except Exception as e:
        console.rule("[bold red]ERROR", style="red")
        console.print(f"[bold red]❌ Error processing query:[/bold red] {e}")

if __name__ == "__main__":
    query = "I spent $1.95 for food today"
    print(process_prompt(query, debug=True))