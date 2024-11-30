from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def demonstrate_prompts():
    print("=== Basic Prompt Templates ===")
    
    # Simple prompt template
    basic_prompt = PromptTemplate(
        input_variables=["product"],
        template="What are 3 creative names for a {product}?"
    )
    
    # Create a chain with the prompt
    llm = ChatOpenAI(temperature=0.7)
    name_chain = LLMChain(llm=llm, prompt=basic_prompt)
    
    # Generate names
    result = name_chain.invoke({"product": "time travel machine"})
    print("\nCreative Names Generator:")
    print(result["text"])
    
    # Few-shot prompt example
    print("\n=== Few-Shot Prompting ===")
    
    examples = [
        {"word": "happy", "antonym": "sad"},
        {"word": "tall", "antonym": "short"},
        {"word": "rich", "antonym": "poor"}
    ]
    
    example_formatter_template = """Word: {word}
Antonym: {antonym}"""
    
    example_prompt = PromptTemplate(
        input_variables=["word", "antonym"],
        template=example_formatter_template
    )
    
    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="Give the antonym for each word.\n\n",
        suffix="\nWord: {input}\nAntonym:",
        input_variables=["input"],
        example_separator="\n\n"
    )
    
    # Create a chain with few-shot prompt
    antonym_chain = LLMChain(llm=llm, prompt=few_shot_prompt)
    result = antonym_chain.invoke({"input": "strong"})
    
    print("\nFew-Shot Antonym Generator:")
    print(f"Word: strong\nAntonym: {result['text']}")
    
    # Demonstrate complex prompt template
    print("\n=== Complex Prompt Template ===")
    complex_prompt = PromptTemplate(
        input_variables=["topic", "tone", "length"],
        template="Write a {length} explanation about {topic} in a {tone} tone."
    )
    
    explanation_chain = LLMChain(llm=llm, prompt=complex_prompt)
    result = explanation_chain.invoke({
        "topic": "quantum computing",
        "tone": "humorous",
        "length": "brief"
    })
    
    print("\nComplex Prompt Result:")
    print(result["text"])

if __name__ == "__main__":
    demonstrate_prompts()
