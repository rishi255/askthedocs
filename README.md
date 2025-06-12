# AskTheDocs

A RAG-based chatbot that can answer questions about your documents using LangGraph and Weaviate.

## Features

-   📄 Supports PDF and TXT documents
-   🔍 Handles multi-column PDFs
-   🤖 Interactive chat interface
-   📚 Automatic document processing
-   🔐 Secure API key management

## Quick Start

1. **Setup Environment**

    ```bash
    # Create and activate virtual environment
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

    # Install dependencies
    pip install -r requirements.txt

    # Configure environment variables
    cp .env.example .env
    # Edit .env with your API keys
    ```

2. **Add Documents**

    - Place your PDF or TXT files in the `data/` directory
    - The system will automatically process them

3. **Run the Chatbot**
    ```bash
    python rag_chatbot.py
    ```

## Required API Keys

-   Weaviate API Key
-   Weaviate Cluster URL
-   HuggingFace API Key
-   OpenAI API Key (for future use)

## Project Structure

```
askthedocs/
├── README.md               # This file
├── src/                    # Source code
├── data/                   # Document storage
├── .env.example           # Environment template
└── requirements.txt       # Dependencies
```

## Dependencies

-   langchain
-   langgraph
-   unstructured
-   sentence-transformers
-   weaviate-client
-   python-dotenv
