from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def demonstrate_vectorstores():
    print("=== Vector Store Demo ===")
    
    # Create a sample text file
    sample_text = """
    LangChain is a framework for developing applications powered by language models.
    It enables applications that are:
    1. Data-aware: connect language models to other sources of data
    2. Agentic: allow language models to interact with their environment
    
    The main value props of LangChain are:
    1. Components: abstractions for working with language models, along with a collection of implementations
    2. Off-the-shelf chains: structured assemblies of components for accomplishing specific higher-level tasks
    
    LangChain's main value props are:
    - Components: LangChain provides many components that can be used to build language model applications
    - Use-case specific chains: LangChain provides chains for common use cases like summarization, Q&A, and more
    """
    
    # Create a temporary file
    with open("sample.txt", "w") as f:
        f.write(sample_text)
    
    # Load the document
    loader = TextLoader("sample.txt")
    documents = loader.load()
    
    # Split text into chunks
    text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    docs = text_splitter.split_documents(documents)
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings()
    
    # Create vector store
    db = Chroma.from_documents(docs, embeddings)
    
    print("\n=== Similarity Search ===")
    # Perform similarity search
    query = "What are the main benefits of LangChain?"
    results = db.similarity_search(query)
    
    print(f"\nQuery: {query}")
    print("\nTop relevant document chunks:")
    for i, doc in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(doc.page_content.strip())
    
    print("\n=== Similarity Search with Score ===")
    # Perform similarity search with score
    results_with_scores = db.similarity_search_with_score(query)
    
    print(f"\nQuery: {query}")
    print("\nTop relevant document chunks with similarity scores:")
    for i, (doc, score) in enumerate(results_with_scores, 1):
        print(f"\nResult {i} (Score: {score:.4f}):")
        print(doc.page_content.strip())
    
    # Clean up
    os.remove("sample.txt")
    
if __name__ == "__main__":
    demonstrate_vectorstores()
