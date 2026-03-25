from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
#from tavily import TravlyClient

@tool
def search(query: str) -> str:
    """
    Tool that searches over internet

    Args:
        query (str): The query to search for

    Returns:
        The search result.
    """
    print(f"Searching for {query}")
    return "Tokyo weather is sunny."
llm = ChatOpenAI()
tools = [search]
agent = create_agent(model=llm, tools=tools)
def main():
    print("Hello, VS Code Python Project!")
    result = agent.invoke({"messages":HumanMessage(content="What is the weather in Tokoyo?")})
    
if __name__ == "__main__":
    main()