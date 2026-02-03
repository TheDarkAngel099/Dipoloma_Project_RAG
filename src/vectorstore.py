"""
FAISS Vector Store Module

This module provides a vector store implementation using FAISS (Facebook AI Similarity Search)
for efficient similarity search and document retrieval in RAG applications.
"""

import os
import faiss  # Facebook AI Similarity Search library for fast vector search
import numpy as np
import pickle  # For serializing metadata
from typing import List, Any
from sentence_transformers import SentenceTransformer
from .embedding import EmbeddingGenerator


class FaiassVectorStore:
    """
    FAISS-based vector store for storing and retrieving document embeddings.
    
    This class manages the entire lifecycle of a vector store: building from documents,
    saving to disk, loading from disk, and performing similarity searches.
    """
    
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the FAISS vector store.
        
        Args:
            persist_dir (str): Directory path where the FAISS index and metadata will be saved.
            embedding_model (str): Name of the SentenceTransformer model for generating embeddings.
            chunk_size (int): Maximum size of text chunks in characters.
            chunk_overlap (int): Number of overlapping characters between chunks.
        """
        self.persist_dir = persist_dir
        # Initialize the embedding generator for converting text to vectors
        self.embedding_generator = EmbeddingGenerator(model_name=embedding_model, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.index = None  # FAISS index will be initialized when first embeddings are added
        self.metadata = []  # Stores text content and other info for each embedding
        
        
    def build_from_documents(self, documents: List[Any]):
        """
        Build the FAISS index from a list of documents.
        
        This method performs the complete pipeline: document chunking, embedding generation,
        index building, and persistence to disk.
        
        Args:
            documents (List[Any]): List of LangChain Document objects to index.
            
        Raises:
            ValueError: If no documents are provided or no chunks can be created.
        """
        # Validate input documents
        if not documents:
            raise ValueError("No documents provided. Ensure the data directory contains files to load.")
        
        # Step 1: Split documents into smaller chunks
        text_chunks = self.embedding_generator.split_documents(documents)
        if not text_chunks:
            raise ValueError("No text chunks were created. Check chunking settings and document contents.")
        
        # Step 2: Generate embeddings for all chunks
        embeddings = self.embedding_generator.generate_embeddings(text_chunks)
        
        # Step 3: Create metadata dictionary for each chunk (stores original text)
        metadata = [{"text": chunk.page_content} for chunk in text_chunks]
        
        # Step 4: Add embeddings and metadata to the FAISS index
        self.add_embeddings(embeddings, metadata)
        
        # Step 5: Persist the index to disk
        self.save_index()
        print(f"Built Faiss index with {len(text_chunks)} vectors.")
    
    
    def add_embeddings(self, embeddings: np.ndarray, metadata: List[dict]):
        """
        Add embeddings to the FAISS index.
        
        Creates the FAISS index on first call, then adds vectors for similarity search.
        Uses L2 (Euclidean) distance metric for similarity computation.
        
        Args:
            embeddings (np.ndarray): 2D array of shape (n_samples, embedding_dim).
            metadata (List[dict]): List of metadata dictionaries, one per embedding.
            
        Raises:
            ValueError: If embeddings are empty or have incorrect dimensions.
        """
        print("Adding embeddings to Faiss index...")
        
        # Validate embeddings
        if embeddings is None or embeddings.size == 0:
            raise ValueError("Embeddings are empty. Ensure documents were loaded and embeddings were generated.")
        if embeddings.ndim != 2:
            raise ValueError(f"Expected embeddings to be 2D (n_samples, dim), got shape {embeddings.shape}.")
        
        # Get embedding dimension (e.g., 384 for all-MiniLM-L6-v2)
        dimension = embeddings.shape[1]
        
        # Initialize FAISS index on first addition (IndexFlatL2 = exact L2 distance search)
        if self.index is None:
            self.index = faiss.IndexFlatL2(dimension)
        
        # Add embeddings to the index
        self.index.add(embeddings)
        
        # Store metadata alongside embeddings
        if metadata:
            self.metadata.extend(metadata)
        print(f"Added {embeddings.shape[0]} embeddings to Faiss index.")
        
    def save_index(self):
        """
        Save the FAISS index and metadata to disk.
        
        Persists two files:
        - faiss_index.bin: Binary file containing the FAISS index
        - metadata.pkl: Pickle file containing metadata for each vector
        """
        # Create directory if it doesn't exist
        os.makedirs(self.persist_dir, exist_ok=True)
        
        # Define file paths
        faiss_path = os.path.join(self.persist_dir, "faiss_index.bin")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        
        # Save FAISS index to binary file
        faiss.write_index(self.index, faiss_path)
        
        # Save metadata using pickle
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"Saved Faiss index to {faiss_path} and metadata to {meta_path}.")
        
    def load_index(self):
        """
        Load a previously saved FAISS index and metadata from disk.
        
        Restores the vector store to its saved state, allowing immediate querying
        without rebuilding the index.
        """
        # Define file paths
        faiss_path = os.path.join(self.persist_dir, "faiss_index.bin")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        
        # Load FAISS index from binary file
        self.index = faiss.read_index(faiss_path)
        
        # Load metadata from pickle file
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)
        print(f"Loaded Faiss index from {faiss_path} and metadata from {meta_path}.")
        
    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        """
        Search for the most similar vectors in the index.
        
        Performs k-nearest neighbors search using L2 distance.
        
        Args:
            query_embedding (np.ndarray): Query vector of shape (1, embedding_dim).
            top_k (int): Number of nearest neighbors to retrieve.
            
        Returns:
            List[dict]: List of results, each containing index, distance, and metadata.
            
        Raises:
            ValueError: If the index hasn't been loaded or built.
        """
        if self.index is None:
            raise ValueError("Faiss index is not loaded. Please load or build the index first.")
        
        # Perform k-NN search (returns distances and indices of top_k results)
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Format results with metadata
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            # Retrieve metadata for this result (if available)
            meta = self.metadata[idx] if idx < len(self.metadata) else None
            results.append({"index": idx, "distance": dist, "metadata": meta})
        return results
    
    def query(self, query_text: str, top_k: int = 5):
        """
        Query the vector store using natural language text.
        
        Converts the query text to an embedding and performs similarity search.
        
        Args:
            query_text (str): Natural language query string.
            top_k (int): Number of most similar documents to retrieve.
            
        Returns:
            List[dict]: Search results with document content and similarity scores.
        """
        # Convert query text to embedding vector
        query_embedding = self.embedding_generator.generate_embeddings([query_text])
        
        # Perform similarity search
        return self.search(query_embedding, top_k)
    
    
    


# Example usage: Run this script directly to test the vector store
if __name__ == "__main__":
    from data_loader import load_all_documents
    
    # Step 1: Load documents from data directory
    docs = load_all_documents("data")
    
    # Step 2: Initialize vector store
    store = FaiassVectorStore("faiss_store")
    
    # Step 3: Build FAISS index from documents
    store.build_from_documents(docs)
    
    # Step 4: Load the saved index (to demonstrate persistence)
    store.load_index()
    
    # Step 5: Perform a sample query
    print(store.query("What is attention mechanism?", top_k=3))
           