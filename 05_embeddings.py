from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from dotenv import load_dotenv
import numpy as np

# Load environment variables
load_dotenv()

def demonstrate_embeddings():
    embeddings = OpenAIEmbeddings()
    
    print("=== Text Embeddings ===")
    # Generate embeddings for simple texts
    texts = [
        "The quick brown fox jumps over the lazy dog",
        "A lazy dog sleeps all day long",
        "The brown fox is quick and clever"
    ]
    
    # Get embeddings
    text_embeddings = embeddings.embed_documents(texts)
    
    print(f"\nGenerated {len(text_embeddings)} embeddings.")
    print(f"Each embedding has {len(text_embeddings[0])} dimensions.")
    
    # Demonstrate similarity comparison
    print("\n=== Document Similarity ===")
    
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    # Compare similarities between documents
    print("\nSimilarity scores between documents:")
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            similarity = cosine_similarity(text_embeddings[i], text_embeddings[j])
            print(f"\nSimilarity between text {i+1} and text {j+1}: {similarity:.4f}")
            print(f"Text {i+1}: {texts[i]}")
            print(f"Text {j+1}: {texts[j]}")
    
    print("\n=== Document Splitting and Embedding ===")
    # Example with longer text and splitting
    long_text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, 
    as opposed to natural intelligence displayed by animals including humans. 
    AI research has been defined as the field of study of intelligent agents, 
    which refers to any system that perceives its environment and takes actions 
    that maximize its chance of achieving its goals.
    """
    
    # Split text into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=100,
        chunk_overlap=20,
        length_function=len
    )
    
    chunks = text_splitter.split_text(long_text)
    
    print(f"\nSplit long text into {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i}:")
        print(chunk.strip())
        
    # Generate embeddings for chunks
    chunk_embeddings = embeddings.embed_documents(chunks)
    print(f"\nGenerated embeddings for {len(chunk_embeddings)} chunks")
    print(f"Each embedding has {len(chunk_embeddings[0])} dimensions")

if __name__ == "__main__":
    demonstrate_embeddings()
