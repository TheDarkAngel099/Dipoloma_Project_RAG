"""RAG (Retrieval-Augmented Generation) package for document search and summarization."""

__version__ = "0.1.0"

from .data_loader import *
from .embedding import EmbeddingGenerator
from .vectorstore import FaiassVectorStore
from .search import RAGSearch

__all__ = [
    "EmbeddingGenerator",
    "FaiassVectorStore",
    "RAGSearch",
]
