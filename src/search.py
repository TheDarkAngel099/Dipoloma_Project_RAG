"""
RAG Search Module

This module provides the main RAG (Retrieval-Augmented Generation) search functionality
that combines vector similarity search with LLM-based summarization using Groq.
"""

import os
from dotenv import load_dotenv  # Load environment variables from .env file
from .vectorstore import FaiassVectorStore
from langchain_groq import ChatGroq  # Groq's fast LLM inference API

# Load environment variables (GROQ_API_KEY) from .env file
load_dotenv()

class RAGSearch:
    """
    Main RAG search class that orchestrates document retrieval and LLM-based summarization.
    
    This class manages the vector store for document retrieval and the Groq LLM for 
    generating contextual summaries based on retrieved documents.
    """
    
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = "llama-3.1-8b-instant", build_if_missing: bool = True):
        """
        Initialize the RAG search system with vector store and LLM.
        
        Args:
            persist_dir (str): Directory to store/load the FAISS vector index.
            embedding_model (str): Name of the SentenceTransformer model for embeddings.
            llm_model (str): Name of the Groq LLM model for text generation.
        """
        # Resolve absolute paths relative to project root
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        persist_path = persist_dir if os.path.isabs(persist_dir) else os.path.join(base_dir, persist_dir)
        data_dir = os.path.join(base_dir, "data")

        # Initialize the FAISS vector store
        self.vectorstore = FaiassVectorStore(persist_path, embedding_model)
        
        # Define paths for FAISS index and metadata files
        faiss_path = os.path.join(persist_path, "faiss_index.bin")
        meta_path = os.path.join(persist_path, "metadata.pkl")
        
        # Check if vector store already exists; if not, build it from documents
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            if not build_if_missing:
                raise FileNotFoundError(
                    f"FAISS index not found at {faiss_path}. Build the index first or enable build_if_missing."
                )
            from .data_loader import load_all_documents
            # Load all documents from the data directory
            docs = load_all_documents(data_dir)
            # Build and persist the vector index
            self.vectorstore.build_from_documents(docs)
        else:
            # Load existing vector store from disk
            self.vectorstore.load_index()
        
        # Initialize the Groq LLM with API key from environment
        groq_api_key = os.getenv("GROQ_API_KEY")
        self.llm = ChatGroq(groq_api_key=groq_api_key, model_name=llm_model)
        print(f"[INFO] Groq LLM initialized: {llm_model}")
        
        
    def search_and_sumarize(self, query: str, top_k: int = 5) -> str:
        """
        Search for relevant documents and generate an AI-powered summary.
        
        This method performs two key steps:
        1. Semantic search: Retrieve the most relevant document chunks from vector store
        2. LLM summarization: Generate a concise summary using retrieved context
        
        Args:
            query (str): The user's question or search query.
            top_k (int): Number of most relevant document chunks to retrieve.
            
        Returns:
            str: AI-generated summary based on retrieved context, or a message if no results found.
        """
        # Step 1: Retrieve top-k most relevant document chunks using vector similarity
        results = self.vectorstore.query(query, top_k=top_k)
        
        # Step 2: Extract text content from metadata of retrieved documents
        texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
        
        # Step 3: Concatenate all retrieved texts into a single context string
        context = "\n\n".join(texts)
        
        # Handle case where no relevant documents are found
        if not context.strip():
            return "No relevant information found in the documents."
        
        # Step 4: Create a prompt for the LLM with the query and retrieved context
        prompt = f"Summarize the following information in relation to the query: '{query}'.\n\nContext:\n{context}\n\nSummary:"
        
        # Step 5: Send prompt to Groq LLM and get the response
        response = self.llm.invoke([prompt])
        
        # Return the generated summary
        return response.content
    
    
# Example usage: Run this script directly to test the RAG search system
if __name__ == "__main__":
    # Initialize the RAG search system (will build index if not exists)
    rag_search = RAGSearch()
    
    # Define a sample query
    query = "which model performed the best?"
    
    # Perform semantic search and get AI-generated summary
    summary = rag_search.search_and_sumarize(query, top_k=3)
    
    # Display the result
    print("Summary:", summary)  
    