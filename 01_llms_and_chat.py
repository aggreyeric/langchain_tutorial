from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def demonstrate_llm():
    print("=== Basic LLM Usage ===")
    
    # Initialize the ChatOpenAI model
    chat = ChatOpenAI(
        temperature=0.7,
        model="gpt-3.5-turbo"
    )
    
    # Simple chat example
    messages = [
        SystemMessage(content="You are a helpful assistant that speaks like a pirate."),
        HumanMessage(content="Tell me about the weather today.")
    ]
    
    response = chat.invoke(messages)
    print("\nPirate Chat Response:")
    print(response.content)
    
    # Demonstrate different temperature settings
    print("\n=== Temperature Comparison ===")
    
    # Low temperature (more focused/deterministic)
    chat_precise = ChatOpenAI(temperature=0.1)
    response_precise = chat_precise.invoke("Write a one-sentence story about a cat.")
    print("\nLow Temperature (0.1) - More Focused:")
    print(response_precise.content)
    
    # High temperature (more creative/random)
    chat_creative = ChatOpenAI(temperature=0.9)
    response_creative = chat_creative.invoke("Write a one-sentence story about a cat.")
    print("\nHigh Temperature (0.9) - More Creative:")
    print(response_creative.content)

if __name__ == "__main__":
    demonstrate_llm()
