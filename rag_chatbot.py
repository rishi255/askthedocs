import logging
import os
import warnings
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, TypedDict

import weaviate
import weaviate.classes as wvc
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langgraph.graph import END, StateGraph
from tqdm import tqdm
from weaviate.classes.init import Auth

# Load environment variables
load_dotenv()

# Configure logging
logging.getLogger("weaviate").setLevel(logging.WARNING)


def clean_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Clean metadata to ensure valid Weaviate property names."""
    # Only keep metadata with valid property names
    valid_metadata = {}
    for key, value in metadata.items():
        # Check if key is a valid GraphQL name
        if re.match(r"^[_A-Za-z][_0-9A-Za-z]{0,230}$", key):
            valid_metadata[key] = value
    return valid_metadata


def init_weaviate_client():
    """Initialize and return a Weaviate client."""
    auth_config = Auth.api_key(os.getenv("WEAVIATE_API_KEY"))
    return weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_CLUSTER_URL"),
        auth_credentials=auth_config,
    )


def create_schema_if_not_exists(client: weaviate.Client, collection_name: str):
    """Create Weaviate schema if it doesn't exist."""
    if not client.collections.exists(collection_name):
        client.collections.create(
            name=collection_name,
            properties=[
                wvc.config.Property(name="text", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="source", data_type=wvc.config.DataType.TEXT),
            ],
            vectorizer_config=wvc.config.Configure.Vectorizer.none(),
        )
        print(f"Created new collection: {collection_name}")


def get_document_count(client: weaviate.Client, collection_name: str) -> int:
    """Get the total number of documents in the collection."""
    result = client.collections.get(collection_name).aggregate.over_all(
        total_count=True
    )
    return result.total_count


def get_loader(filepath: str):
    """Get the appropriate loader based on file extension."""
    file_extension = Path(filepath).suffix.lower()
    if file_extension == ".pdf":
        return PyPDFLoader(filepath, extract_images=False)
    elif file_extension == ".txt":
        return TextLoader(filepath, autodetect_encoding=True)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")


class DocumentLoader:
    """Handles document loading and processing into Weaviate."""

    def __init__(self, collection_name: str = "DocumentChunk"):
        self.collection_name = collection_name
        self.client = init_weaviate_client()
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        create_schema_if_not_exists(self.client, self.collection_name)

    def load_documents(self, filepath: str) -> None:
        """Load and process a single document into the vector store."""
        try:
            loader = get_loader(filepath)
            print(f"Using loader: {loader.__class__.__name__} for {filepath}")

            # Load and split document into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=400, chunk_overlap=50
            )
            print(f"Loading and chunking documents from: {filepath}...")
            docs = loader.load_and_split(text_splitter)

            if not docs:
                print(f"No chunks created from {filepath}. Check the file content.")
                return

            # Clean metadata for each document
            for doc in docs:
                doc.metadata = clean_metadata(doc.metadata)
                # Ensure source is always present
                if "source" not in doc.metadata:
                    doc.metadata["source"] = str(filepath)

            # Add document to vector store using Weaviate's built-in batching
            vectorstore = WeaviateVectorStore.from_documents(
                documents=docs,
                client=self.client,
                index_name=self.collection_name,
                text_key="text",
                embedding=self.embeddings,
                by_text=False,
                batch_size=100,  # Weaviate's default batch size
                batch_concurrency=5,  # Number of concurrent batches
            )

            print(
                f"Total documents in vector store: {get_document_count(self.client, self.collection_name)}"
            )

        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            raise

    def load_documents_from_directory(self, directory: str) -> None:
        """Load all supported documents from a directory."""
        supported_extensions = {".pdf", ".txt"}
        directory_path = Path(directory)

        if not directory_path.exists():
            print(f"Directory {directory} does not exist. Creating it...")
            directory_path.mkdir(parents=True)
            return

        files_processed = 0
        for file_path in directory_path.rglob("*"):
            if file_path.suffix.lower() in supported_extensions:
                try:
                    self.load_documents(str(file_path))
                    files_processed += 1
                except Exception as e:
                    print(f"Failed to process {file_path}: {e}")
                    continue

        if files_processed == 0:
            print(f"No supported documents found in {directory}")
        else:
            print(f"Successfully processed {files_processed} documents")

    def close(self):
        """Close the Weaviate client connection."""
        self.client.close()


class RAGChatbot:
    def __init__(self, collection_name: str = "DocumentChunk"):
        self.collection_name = collection_name
        self.client = init_weaviate_client()
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.llm = self._init_llm()
        self._ollama_client = None  # Store Ollama client reference

        # #! TODO: Remove this line in production
        # # clear this collection before creating it
        # print(f"Clearing collection {self.collection_name} if it exists...")
        # if self.client.collections.exists(self.collection_name):
        #     print("Exists! Deleting collection...")
        #     self.client.collections.delete(self.collection_name)

        # Check if collection exists and has documents
        if not self.client.collections.exists(self.collection_name):
            print(
                f"Collection {self.collection_name} does not exist. Loading documents..."
            )
            loader = DocumentLoader(self.collection_name)
            loader.load_documents_from_directory("data")
            loader.close()
        else:
            doc_count = get_document_count(self.client, self.collection_name)
            print(f"Found {doc_count} documents in collection {self.collection_name}")

        # Initialize vector store and retriever
        self.vectorstore = WeaviateVectorStore(
            client=self.client,
            index_name=self.collection_name,
            text_key="text",
            embedding=self.embeddings,
            # by_text=False,
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})

        # Initialize prompts
        self.qa_prompt = ChatPromptTemplate.from_template(
            """You are an assistant for question-answering tasks. 
            Use the following pieces of retrieved context to answer the question. 
            If you don't know the answer, just say that you don't know. 
            Question: {question} 
            Context Summary: {context_summary}
            Context: {context} 
            Answer:
            """
        )

        self.summarizer_prompt = ChatPromptTemplate.from_template(
            """Summarise the following context in a compact way while also providing all important and relevant information.
            Context: {context}"""
        )

    def _init_llm(self) -> ChatOpenAI | ChatOllama:
        """Get the appropriate LLM based on environment variable."""
        model_type = os.getenv("MODEL_PROVIDER")
        if not model_type:
            raise ValueError("MODEL_PROVIDER environment variable is not set.")

        if model_type == "ollama":
            self._ollama_client = ChatOllama(model="llama3.1:8b", temperature=0)
            return self._ollama_client
        elif model_type == "openai":
            return ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def create_graph(self) -> StateGraph:
        """Create the RAG workflow graph."""

        # Define the graph nodes
        def retrieve(state: Dict) -> Dict:
            """Retrieve relevant documents."""
            question = state["question"]
            docs = self.retriever.invoke(question)
            return {"context": docs}

        def summarize(state: Dict) -> Dict:
            """Summarize the retrieved context."""
            context = state["context"]

            # Format context for prompt
            context_text = "\n\n".join([doc.page_content for doc in context])

            # Generate summary using LLM
            summary = self.llm.invoke(
                self.summarizer_prompt.format(context=context_text)
            )

            return {"context_summary": summary.content}

        def generate_response(state: Dict) -> Dict:
            """Generate the final response."""
            question = state["question"]
            context = state["context"]
            context_summary = state["context_summary"]

            # Format context for prompt
            context_text = "\n\n".join([doc.page_content for doc in context])

            # Generate response using LLM
            response = self.llm.invoke(
                self.qa_prompt.format(
                    question=question,
                    context=context_text,
                    context_summary=context_summary,
                )
            )

            return {"response": response.content}

        # Define state schema
        class RAGState(TypedDict):
            question: str
            context: List[Document]
            context_summary: str
            response: str

        # Create the graph
        workflow = StateGraph(RAGState)

        # Add nodes
        workflow.add_node("retrieve", retrieve)
        workflow.add_node("summarize", summarize)
        workflow.add_node("generate", generate_response)

        # Add edges
        workflow.add_edge("retrieve", "summarize")
        workflow.add_edge("summarize", "generate")
        workflow.add_edge("generate", END)

        # Set entry point
        workflow.set_entry_point("retrieve")

        return workflow.compile()

    def chat(self, question: str) -> str:
        """Process a single question through the RAG pipeline."""
        graph = self.create_graph()
        result = graph.invoke({"question": question})
        return result["response"]

    def close(self):
        """Close all client connections."""
        self.client.close()
        if self._ollama_client:
            self._ollama_client._client._client.close()  # Close Ollama client connection


def main():
    """Main function to run the chatbot."""
    chatbot = RAGChatbot()

    # client.collections.delete("DocumentChunks")

    print("RAG Chatbot initialized. Type 'quit' to exit.")
    while True:
        user_input = input("\nYour question: ")
        if user_input.lower() == "quit":
            break

        response = chatbot.chat(user_input)
        print(f"\nResponse: {response}")

    chatbot.close()


if __name__ == "__main__":
    main()
