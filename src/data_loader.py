from pathlib import Path
from typing import List, Any
# Import document loaders from LangChain community package for various file formats
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader, Docx2txtLoader, JSONLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader

def load_all_documents(data_dir: str) -> List[Any]:
    """
    Load all documents from the specified directory using appropriate loaders based on file extensions.
    
    This function recursively scans the data directory and automatically selects the appropriate
    loader based on file extension. Supports PDF, TXT, CSV, DOCX, JSON, and Excel files.

    Args:
        data_dir (str): The directory containing the documents to load.
    Returns:
        List[Any]: A list of loaded LangChain Document objects.
    """
    # Map file extensions to their corresponding LangChain loader classes
    loaders = {
        '.pdf': PyPDFLoader,           # For PDF documents
        '.txt': TextLoader,            # For plain text files
        '.csv': CSVLoader,             # For CSV spreadsheets
        '.docx': Docx2txtLoader,       # For Word documents
        '.json': JSONLoader,           # For JSON files
        '.xlsx': UnstructuredExcelLoader,  # For modern Excel files
        '.xls': UnstructuredExcelLoader,   # For legacy Excel files
    }

    documents = []  # Container for all loaded documents
    data_path = Path(data_dir)  # Convert string path to Path object

    # Recursively iterate through all files in the directory and subdirectories
    for file_path in data_path.rglob('*'):
        if file_path.is_file():  # Skip directories
            ext = file_path.suffix.lower()  # Get file extension in lowercase
            loader_class = loaders.get(ext)  # Get corresponding loader class
            
            if loader_class:  # Only process supported file types
                # Instantiate the loader with the file path
                loader = loader_class(str(file_path))
                # Load the document(s) from the file
                loaded_docs = loader.load()
                # Add all loaded documents to the main list
                documents.extend(loaded_docs)

    return documents

# Example usage: Run this script directly to test document loading
if __name__ == "__main__":
    # Specify the directory containing your documents
    data_directory = "/mnt/d/RAG/data/pdfs"
    # Load all supported documents from the directory
    all_documents = load_all_documents(data_directory)
    # Display the count of loaded documents
    print(f"Loaded {len(all_documents)} documents.")