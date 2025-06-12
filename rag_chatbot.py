import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, TypedDict

import weaviate
import weaviate.classes as wvc
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader, UnstructuredPDFLoader
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_weaviate import WeaviateVectorStore
from langgraph.graph import END, StateGraph
from weaviate.classes.init import Auth

# Load environment variables
load_dotenv()


class RAGChatbot:
    def __init__(self):
        # Initialize Weaviate client
        auth_config = Auth.api_key(os.getenv("WEAVIATE_API_KEY"))
        # Auth.api_key(os.getenv("WEAVIATE_API_KEY"))

        self.client = weaviate.connect_to_weaviate_cloud(
            cluster_url=os.getenv("WEAVIATE_CLUSTER_URL"),
            auth_credentials=auth_config,
            # headers={"X-HuggingFace-Api-Key": os.getenv("HUGGINGFACE_API_KEY")},
            # skip_init_checks=True
        )

        # Create schema if it doesn't exist
        if not self.client.collections.exists("DocumentChunk"):
            self.client.collections.create(
                name="DocumentChunk",
                properties=[
                    wvc.config.Property(name="text", data_type=wvc.config.DataType.TEXT)
                ],
                vectorizer_config=wvc.config.Configure.Vectorizer.none(),  # we provide our own vectors
                # vector_index_config=wvc.config.Configure.VectorIndex.hnsw(
                #     distance_metric=wvc.config.VectorDistances.COSINE
                # ),
            )

        print(f"Client Ready?: {self.client.is_ready()}")

        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # Initialize LLM
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

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

        # Initialize vector store
        self.vectorstore = None
        self.retriever = None

    def _get_loader(self, filepath: str):
        """Get the appropriate loader based on file extension."""
        file_extension = Path(filepath).suffix.lower()
        if file_extension == ".pdf":
            return UnstructuredPDFLoader(
                filepath,
                mode="multi",  # This preserves document structure
                strategy="fast",  # Use fast strategy for better performance
            )
        elif file_extension == ".txt":
            return TextLoader(filepath, autodetect_encoding=True)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

    def load_documents(self, filepath: str) -> None:
        """Load and process documents into the vector store."""
        # try:
        loader = self._get_loader(filepath)
        print(f"Using loader: {loader.__class__.__name__} for {filepath}")
        documents = loader.load()

        # Split documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)

        print(f"Chunks: {len(chunks)} created from {filepath}")
        if not chunks:
            print(f"No chunks created from {filepath}. Check the file content.")
            return

        # Create vector store
        self.vectorstore = WeaviateVectorStore(
            client=self.client,
            index_name="Document",
            text_key="text",
            embedding=self.embeddings,
        )

        # Add documents to vector store
        self.vectorstore.add_documents(chunks)
        self.retriever = self.vectorstore.as_retriever()
        print(f"Successfully loaded and processed: {filepath}")
        # except Exception as e:
        #     print(f"Error processing {filepath}: {e}")
        #     raise

    def load_documents_from_directory(self, directory: str) -> None:
        """Load all supported documents from a directory."""
        supported_extensions = {".pdf", ".txt"}
        directory_path = Path(directory)

        if not directory_path.exists():
            print(f"Directory {directory} does not exist. Creating it...")
            directory_path.mkdir(parents=True)
            return

        files_processed = 0
        for file_path in directory_path.glob("*"):
            if file_path.suffix.lower() in supported_extensions:
                # try:
                self.load_documents(str(file_path))
                files_processed += 1
                # except Exception as e:
                #     print(f"Failed to process {file_path}: {e}")
                #     continue

        if files_processed == 0:
            print(f"No supported documents found in {directory}")
        else:
            print(f"Successfully processed {files_processed} documents")

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


def main():
    """Main function to run the chatbot."""
    chatbot = RAGChatbot()

    # Load documents from data directory
    chatbot.load_documents_from_directory("data")

    print("RAG Chatbot initialized. Type 'quit' to exit.")
    while True:
        user_input = input("\nYour question: ")
        if user_input.lower() == "quit":
            break

        response = chatbot.chat(user_input)
        print(f"\nResponse: {response}")

    chatbot.client.close()


if __name__ == "__main__":
    main()
