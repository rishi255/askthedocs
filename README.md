# AskTheDocs

A RAG-based chatbot that can answer questions about your documentation using LangGraph and Weaviate.

## Features

-   📄 Supports PDF and TXT documents, with support for more formats on the way!
-   🔍 Handles multi-column PDFs
-   🤖 Interactive chat interface
-   📚 Automatic document processing
-   🔐 Secure API key management
-   🔗 [future scope] Auto-scrape documentation from URL

## Setup

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/askthedocs.git
    cd askthedocs
    ```

2. Create and activate a virtual environment:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Configure environment variables

    ```bash
        cp .env.example .env
        # Edit .env with your API keys
    ```

5. Place your documents in the `data/` directory

    - Place your PDF or TXT files in the `data/` directory
    - The system will automatically process them

6. Run the chatbot:

    ```bash
    python rag_chatbot.py
    ```

Type your questions and get answers based on your documentation! Type 'quit' to exit.

## Project Structure

```
askthedocs/
├── README.md               # Project documentation
├── src/                    # Source code
├── data/                   # Document storage
├── .env.example           # Environment template
└── requirements.txt       # Dependencies
```
