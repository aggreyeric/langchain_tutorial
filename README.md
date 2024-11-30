# LangChain Concepts Tutorial

This repository contains examples and demonstrations of various LangChain concepts. Each file focuses on a specific concept with clear examples and explanations.

## Setup
1. Install required dependencies:
```bash
pip install langchain openai python-dotenv
```

2. Create a `.env` file in the root directory and add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Concepts Covered

1. **LLMs and Chat Models** (`01_llms_and_chat.py`)
   - Basic usage of LLMs
   - Chat model interactions
   - Model parameters and settings

2. **Prompts and PromptTemplates** (`02_prompts.py`)
   - Creating prompt templates
   - Working with variables
   - Few-shot prompting

3. **Chains** (`03_chains.py`)
   - Simple chains
   - Sequential chains
   - Custom chains

4. **Memory** (`04_memory.py`)
   - Conversation memory
   - Buffer memory
   - Different memory types

5. **Embeddings** (`05_embeddings.py`)
   - Text embeddings
   - Document similarity
   - Vector operations

6. **Vector Stores** (`06_vectorstores.py`)
   - Document storage
   - Similarity search
   - Vector databases

7. **Agents** (`07_agents.py`)
   - Basic agents
   - Tools and toolkits
   - Custom agents

Each file contains detailed comments and examples to help you understand the concepts better.

## Usage
Run each file individually to see the demonstrations:
```bash
python 01_llms_and_chat.py
```
