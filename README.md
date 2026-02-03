# 🎓 Diploma Project Q&A System

A Retrieval-Augmented Generation (RAG) application that enables natural language question-answering over diploma project documents. Built with FAISS for vector search, Sentence Transformers for embeddings, and Groq's LLaMA for AI-powered responses.

> **Note:** This is a friendly tinkering project while I explore and learn about RAGs.

![Python](https://img.shields.io/badge/python-3.12+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.53+-red.svg)

## ✨ Features

- 📄 **Multi-format Document Support** - Load PDFs, DOCX, TXT, CSV, JSON, and Excel files
- 🔍 **Semantic Search** - FAISS-powered vector similarity search
- 🤖 **AI-Powered Answers** - Context-aware responses using Groq's LLaMA 3.1 8B
- 💬 **Interactive Chat Interface** - Beautiful Streamlit web UI with chat history
- ⚡ **Fast & Efficient** - Persistent vector store for instant retrieval
- 🎛️ **Configurable** - Adjust chunk size, retrieval count, and model settings

## 🛠️ Tech Stack

- **Vector Store:** FAISS (Facebook AI Similarity Search)
- **Embeddings:** Sentence Transformers (all-MiniLM-L6-v2)
- **LLM:** Groq LLaMA 3.1 8B Instant
- **Document Processing:** LangChain
- **Frontend:** Streamlit
- **Python:** 3.12+

## 📦 Installation

### Prerequisites

- Python 3.12 or higher
- Groq API key ([Get one here](https://console.groq.com))

### Setup

1. **Clone the repository:**
```bash
git clone <your-repo-url>
cd RAG
```

2. **Create and activate virtual environment:**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirement.txt
```

4. **Configure environment variables:**
Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_groq_api_key_here
```

5. **Add your documents:**
Place your PDF, DOCX, or other supported documents in the `data/pdfs/` directory.

## 🚀 Usage

### Running the Web Application

Start the Streamlit app (from the venv):
```bash
python -m streamlit run app.py
```

The app will be available at a local URL printed in the terminal (usually `http://localhost:8501` or `http://localhost:8502`).

**Important:** The app now lazy-loads the RAG system for faster startup. Use the sidebar buttons:
- **Load RAG (fast):** Loads existing FAISS index (no rebuild)
- **Build/Refresh Index (slow):** Builds index from documents if missing

### Using the Python API

```python
from src.search import RAGSearch

# Initialize the RAG system
rag = RAGSearch(
    persist_dir="faiss_store",
    embedding_model="all-MiniLM-L6-v2",
    llm_model="llama-3.1-8b-instant"
)

# Ask a question
query = "What are the main contributions of this research?"
answer = rag.search_and_sumarize(query, top_k=3)
print(answer)
```

### Building the Vector Store

If you want to rebuild the vector store from scratch:
```bash
cd src
python vectorstore.py
```

## 📁 Project Structure

```
RAG/
├── app.py                  # Streamlit web application
├── src/
│   ├── __init__.py        # Package initialization
│   ├── data_loader.py     # Document loading utilities
│   ├── embedding.py       # Text chunking and embedding generation
│   ├── vectorstore.py     # FAISS vector store management
│   └── search.py          # RAG search orchestration
├── data/
│   └── pdfs/              # Place your documents here
├── faiss_store/           # Persisted vector index (auto-generated)
├── .env                   # Environment variables (create this)
├── .gitignore            # Git ignore rules
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## ⚙️ Configuration

### Embedding Settings

Edit parameters in `src/embedding.py`:
- `chunk_size`: Maximum characters per chunk (default: 1000)
- `chunk_overlap`: Overlap between chunks (default: 200)
- `model_name`: Sentence transformer model (default: "all-MiniLM-L6-v2")

### LLM Settings

Modify in `src/search.py`:
- `llm_model`: Groq model name (default: "llama-3.1-8b-instant")
- `top_k`: Number of context chunks to retrieve (default: 5)

### Supported Document Formats

- PDF (`.pdf`)
- Microsoft Word (`.docx`)
- Plain Text (`.txt`)
- CSV (`.csv`)
- JSON (`.json`)
- Excel (`.xlsx`, `.xls`)

## 💡 Example Queries

- "What are the main contributions of this project?"
- "Explain the methodology used in the research"
- "What were the experimental results?"
- "Which model performed the best?"
- "Summarize the conclusion section"

## 🔧 Troubleshooting

### Vector Store Not Found
If you see "FAISS index not found", use **Build/Refresh Index (slow)** in the sidebar to build it from documents in `data/pdfs/`.

### Out of Memory
Reduce `chunk_size` or process documents in smaller batches if you encounter memory issues.

### Groq API Errors
Ensure your `GROQ_API_KEY` is valid and you have API credits. Check the [Groq Console](https://console.groq.com) for rate limits.


## 🙏 Acknowledgments

- [FAISS](https://github.com/facebookresearch/faiss) by Facebook Research
- [Sentence Transformers](https://www.sbert.net/) by UKPLab
- [LangChain](https://www.langchain.com/) for document processing
- [Groq](https://groq.com/) for fast LLM inference
- [Streamlit](https://streamlit.io/) for the web interface


---

Made with ❤️ for diploma project documentation
