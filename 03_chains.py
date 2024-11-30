from langchain.chains import SimpleSequentialChain, SequentialChain, LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def demonstrate_chains():
    llm = ChatOpenAI(temperature=0.7)
    
    print("=== Simple Sequential Chain ===")
    # First chain: Generate a story idea
    story_prompt = PromptTemplate(
        input_variables=["topic"],
        template="Generate a one-sentence story idea about {topic}."
    )
    story_chain = LLMChain(llm=llm, prompt=story_prompt)
    
    # Second chain: Generate a title
    title_prompt = PromptTemplate(
        input_variables=["story"],
        template="Generate a creative title for this story: {story}"
    )
    title_chain = LLMChain(llm=llm, prompt=title_prompt)
    
    # Combine chains
    story_title_chain = SimpleSequentialChain(
        chains=[story_chain, title_chain],
        verbose=True
    )
    
    # Run the chain
    result = story_title_chain.invoke("a magical library")
    print("\nFinal Title:", result["output"])
    
    print("\n=== Multiple Input Sequential Chain ===")
    # First chain: Generate character
    character_prompt = PromptTemplate(
        input_variables=["profession", "hobby"],
        template="Create a character who is a {profession} and loves {hobby}."
    )
    character_chain = LLMChain(
        llm=llm,
        prompt=character_prompt,
        output_key="character"
    )
    
    # Second chain: Generate a scenario
    scenario_prompt = PromptTemplate(
        input_variables=["character", "setting"],
        template="Create a scenario for this character: {character}\nThe story is set in {setting}."
    )
    scenario_chain = LLMChain(
        llm=llm,
        prompt=scenario_prompt,
        output_key="scenario"
    )
    
    # Final chain: Generate a resolution
    resolution_prompt = PromptTemplate(
        input_variables=["scenario"],
        template="Write a resolution for this scenario: {scenario}"
    )
    resolution_chain = LLMChain(
        llm=llm,
        prompt=resolution_prompt,
        output_key="resolution"
    )
    
    # Combine all chains
    full_story_chain = SequentialChain(
        chains=[character_chain, scenario_chain, resolution_chain],
        input_variables=["profession", "hobby", "setting"],
        output_variables=["character", "scenario", "resolution"],
        verbose=True
    )
    
    # Run the chain
    result = full_story_chain.invoke({
        "profession": "chef",
        "hobby": "skydiving",
        "setting": "ancient Rome"
    })
    
    print("\nFull Story Generation:")
    print("\nCharacter:", result["character"])
    print("\nScenario:", result["scenario"])
    print("\nResolution:", result["resolution"])

if __name__ == "__main__":
    demonstrate_chains()
