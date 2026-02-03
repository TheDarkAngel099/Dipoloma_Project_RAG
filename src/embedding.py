"""
Embedding Generator Module

This module provides functionality for splitting documents into chunks and generating
vector embeddings using Sentence Transformers models.
"""

from typing import List, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer  # Hugging Face sentence embeddings
import numpy as np
from .data_loader import load_all_documents


class EmbeddingGenerator:
    """
    Handles document chunking and embedding generation for RAG applications.
    
    This class uses RecursiveCharacterTextSplitter to break documents into manageable chunks
    and SentenceTransformer to convert text into dense vector embeddings.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the EmbeddingGenerator with a specific model and chunking parameters.
        
        Args:
            model_name (str): Name of the SentenceTransformer model to use.
                            Default is 'all-MiniLM-L6-v2' (384-dimensional embeddings).
            chunk_size (int): Maximum size of each text chunk in characters.
            chunk_overlap (int): Number of characters to overlap between consecutive chunks
                               to maintain context at boundaries.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Load the pre-trained sentence transformer model
        self.model = SentenceTransformer(model_name)
        print(f"Loaded embedding model: {model_name}")

    def split_documents(self, documents: List[Any]) -> List[str]:
        """
        Split documents into smaller chunks for efficient embedding generation.
        
        Uses a recursive strategy that tries to split on paragraph boundaries first,
        then sentences, then words, to maintain semantic coherence within chunks.
        
        Args:
            documents (List[Any]): List of LangChain Document objects to split.
            
        Returns:
            List[str]: List of text chunks ready for embedding.
        """
        # Create a text splitter with specified chunk size and overlap
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,  # Use character count as length metric
            # Try splitting on these separators in order: paragraphs -> lines -> spaces -> characters
            separators=["\n\n", "\n", " ", ""]
        ).split_documents(documents)
        print(f"Split documents into {len(chunks)} chunks.")
        return chunks
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate dense vector embeddings for a list of text chunks.
        
        Converts text into numerical representations (vectors) that can be used
        for semantic similarity search in a vector database.
        
        Args:
            texts (List[str]): List of text strings or Document objects to embed.
            
        Returns:
            np.ndarray: 2D array of embeddings with shape (n_texts, embedding_dimension).
                       Each row represents the embedding vector for one text chunk.
        """
        # Ensure all inputs are strings (handles both str and Document objects)
        texts = [str(text) for text in texts]
        print(f"Generating embeddings for {len(texts)} texts.")
        
        # Encode texts into embeddings using the sentence transformer model
        # show_progress_bar=True displays a progress indicator during encoding
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings
    
# Example usage: Run this script directly to test embedding generation pipeline
if __name__ == "__main__":
    # Specify the directory containing your documents
    data_directory = "/mnt/d/RAG/data/pdfs"
    
    # Step 1: Load all documents from the specified directory
    documents = load_all_documents(data_directory)
    
    # Step 2: Initialize the embedding generator with default settings
    embedding_generator = EmbeddingGenerator()
    
    # Step 3: Split documents into smaller, overlapping chunks
    text_chunks = embedding_generator.split_documents(documents)
    
    # Step 4: Generate vector embeddings for all text chunks
    embeddings = embedding_generator.generate_embeddings(text_chunks)
    
    # Display summary of the embedding generation process
    print(f"Total embeddings generated: {embeddings.shape[0]}")         