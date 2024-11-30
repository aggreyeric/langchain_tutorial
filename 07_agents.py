from langchain_community.tools import Tool
from langchain_community.utilities.serpapi import SerpAPIWrapper
from langchain_openai import ChatOpenAI
from langchain.agents import AgentType, initialize_agent, load_tools
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def demonstrate_agents():
    print("=== Basic Agent Demo ===")
    
    # Initialize the language model
    llm = ChatOpenAI(temperature=0)
    
    # Load some basic tools
    tools = load_tools(["llm-math"], llm=llm)
    
    # Initialize the agent
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    
    print("\nMath Problem Solving:")
    result = agent.invoke(
        {"input": "What is the square root of 256 multiplied by the sum of 13 and 27?"}
    )
    print("\nResult:", result["output"])
    
    print("\n=== Custom Tool Agent ===")
    
    def get_word_length(word: str) -> int:
        """Returns the length of a word."""
        return len(word)
    
    # Create a list of tools
    custom_tools = [
        Tool(
            name="WordLength",
            func=get_word_length,
            description="Useful for counting the number of characters in a word"
        )
    ]
    
    # Initialize agent with custom tool
    custom_agent = initialize_agent(
        custom_tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    
    print("\nCustom Tool Usage:")
    result = custom_agent.invoke(
        {"input": "How many characters are in the word 'pneumonoultramicroscopicsilicovolcanoconiosis'?"}
    )
    print("\nResult:", result["output"])
    
    print("\n=== Chain of Thought Agent ===")
    # Initialize agent with chain of thought prompting
    cot_agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    
    print("\nSolving a Complex Problem:")
    result = cot_agent.invoke(
        {"input": """
        If I have a square with area 64 square meters:
        1. What is the length of one side?
        2. If I double this length, what is the new area?
        """}
    )
    print("\nResult:", result["output"])

if __name__ == "__main__":
    demonstrate_agents()
