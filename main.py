from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from tavily import TravlyClient
from langchain_tavily import TavilyResearch

tavily = TravlyClient()

@tool  # custom tool from vendor
def search(query: str) -> str:
    """
    Tool that searches over internet

    Args:
        query (str): The query to search for

    Returns:
        The search result.
    """
    print(f"Searching for {query}")
    return tavily.search(query=query)

class Source(BaseModel):
    """Schema for a source used by the agent"""
    
    url:str = Field(description="The URL of source")
    
class AgentReponse(BaseModel):
    """Schema fo agent response with answer and sources"""
    
    answer:str = Field(description="The agent's answer to the query")
    source: List[Source] = Field(default_factory=list, description="List of sources used to generate the answer.")


llm = ChatOpenAI(model="gpt-5")
#tools = [search]
tools = [TavilyResearch()]
#agent = create_agent(model=llm, tools=tools)
agent = create_agent(model=llm, tools=tools, response_format=AgentReponse)

def main():
    print("Hello, VS Code Python Project!")
    result = agent.invoke({"messages":HumanMessage(content="What is the weather in Tokoyo?")})
    
if __name__ == "__main__":
    main()