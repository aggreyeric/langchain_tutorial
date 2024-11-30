from langchain_core.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def demonstrate_memory():
    llm = ChatOpenAI(temperature=0.7)
    
    print("=== Conversation Buffer Memory ===")
    # Simple conversation memory
    conversation = ConversationChain(
        llm=llm,
        memory=ConversationBufferMemory(),
        verbose=True
    )
    
    # Have a conversation
    print("\nFirst Interaction:")
    response1 = conversation.invoke({"input": "Hi! My name is Alice."})
    print(response1["response"])
    
    print("\nSecond Interaction:")
    response2 = conversation.invoke({"input": "What's my name?"})
    print(response2["response"])
    
    print("\n=== Conversation Summary Memory ===")
    # Memory that maintains a summary of the conversation
    summary_memory = ConversationSummaryMemory(llm=llm)
    conversation_with_summary = ConversationChain(
        llm=llm,
        memory=summary_memory,
        verbose=True
    )
    
    # Have a longer conversation
    print("\nFirst Interaction:")
    response1 = conversation_with_summary.invoke(
        {"input": "Hi! I'm a software developer working on a new project."}
    )
    print(response1["response"])
    
    print("\nSecond Interaction:")
    response2 = conversation_with_summary.invoke(
        {"input": "The project is a machine learning application for healthcare."}
    )
    print(response2["response"])
    
    print("\nThird Interaction:")
    response3 = conversation_with_summary.invoke(
        {"input": "Can you remind me what we were discussing?"}
    )
    print(response3["response"])
    
    # Get the current summary
    print("\nCurrent Conversation Summary:")
    print(summary_memory.load_memory_variables({})["history"])

if __name__ == "__main__":
    demonstrate_memory()
