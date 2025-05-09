import json
from datetime import datetime
from typing import Dict, Any, List

from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.utilities import WikipediaAPIWrapper
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import Tool
from pydantic import BaseModel

class Response(BaseModel):
    topic: str
    summary: str
    source: List[str]
    tool_used: List[str]

def save_to_txt(data: str, filename: str = "research_output.txt") -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"

    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)

    return f"Data successfully saved to {filename}"


def create_tools() -> List[Tool]:
    save_tool = Tool(
        name="save_text_to_file",
        func=save_to_txt,
        description="Saves structured research data to a text file."
    )

    search = DuckDuckGoSearchRun()
    search_tool = Tool(
        name="search",
        func=search.run,
        description="Useful for when you need to answer questions about current events. You should ask targeted questions"
    )

    api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
    wiki_tool = Tool(
        name="wikipedia",
        func=WikipediaQueryRun(api_wrapper=api_wrapper).run,
        description="Useful for getting information from Wikipedia."
    )

    return [search_tool, wiki_tool, save_tool]

def setup_llm(api_config: Dict[str, str]) -> ChatOpenAI:
    return ChatOpenAI(
        model=api_config["model_name"],
        openai_api_key=api_config["deepseek_api_key"],
        openai_api_base=api_config["deepseek_api_base"]
    )


def create_prompt_template(parser: PydanticOutputParser) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a research assistant that will help generate a research paper.
                Answer the user query and use necessary tools.
                Wrap the output in the following format and provide no other text\n{instructions}
                """
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{query}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    ).partial(instructions=parser.get_format_instructions())


def setup_agent(llm: ChatOpenAI, tools: List[Tool]) -> AgentExecutor:
    parser = PydanticOutputParser(pydantic_object=Response)
    prompt = create_prompt_template(parser)

    agent = create_tool_calling_agent(
        llm=llm,
        prompt=prompt,
        tools=tools
    )

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True
    )
    
def process_agent_response(raw_response: Dict[str, Any]) -> Dict[str, Any]:
    """Process and parse the agent's response."""
    json_str = raw_response['output'].strip('```json\n').strip('```').strip()
    return json.loads(json_str)

def main():
    try:
        api_config = setup_api_credentials()
        tools = create_tools()
        llm = setup_llm(api_config)
        agent_executor = setup_agent(llm, tools)

        query = input("What can I help you research? ")

        raw_response = agent_executor.invoke({"query": query})

        structured_data = process_agent_response(raw_response)
        print(structured_data)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()