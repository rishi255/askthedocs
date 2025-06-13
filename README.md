# AskTheDocs

A smart, local-first RAG-based chatbot that answers your questions using only the documentation you provide. Built with LangGraph and Weaviate.

## 🧠 Use Case

**Are you a developer working with complex technical documentation?**

Whether you're exploring new frameworks, integrating databases, or debugging APIs - AskTheDocs helps you query your docs directly using natural language.

Instead of scouring long documentation pages or risking hallucinated LLM answers, just drop the official docs into a folder and ask away!

This chatbot helps you:

-   🔍 Search and compare framework features
-   🧑‍💻 Debug issues with factual info from your actual docs
-   🧱 Get accurate code usage and examples from the right source
-   🤖 Avoid hallucinations by grounding responses in your local technical docs

> **Example:** Drop in Flask + Django docs and ask:
> _“What’s the equivalent of `@app.route` in Django?”_

---

## Features

-   ⚡ Quickly query your docs with natural language
-   📄 Simply drop PDF or TXT documentation into a folder — no config needed
-   🛠️ Process once, reuse anytime for efficient Q&A
-   💬 Interactive command-line chat interface
-   🧠 RAG pipeline powered by LangGraph and Weaviate
-   🌐 \[Coming soon] Auto-scrape docs from URLs
-   🖥️ \[Coming soon] Modern UI for chatting

---

## Setup

1. **Clone the repo**

    ```bash
    git clone https://github.com/rishi255/askthedocs
    cd askthedocs
    ```

2. **Create and activate a virtual environment**

    ```bash
    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    ```

3. **Install dependencies**

    ```bash
    pip install -r requirements.txt
    ```

4. **Configure environment variables**

    ```bash
    cp .env.example .env
    # Then edit .env to set your API keys (OpenAI etc.)
    ```

5. **Add your documentation**

    Place your `.pdf` or `.txt` files inside the `data/` directory.
    These will be automatically parsed and indexed for Q\&A.

> [!TIP]
> Keep your docs organized folder-wise for best results.  
> eg. `data/flask_docs/flask-getting-started.pdf`, `data/django_docs/django-getting-started.pdf`

6. **Run the chatbot**

    ```bash
    python rag_chatbot.py
    ```

    Ask your questions based on the loaded documentation!
    Type `'quit'` to exit.

---

## Project Structure

```
askthedocs/
├── README.md             # Project info
├── src/                  # Core RAG logic and orchestration
├── data/                 # Drop your docs here
├── .env.example          # Sample env config
└── requirements.txt      # Python dependencies
```
