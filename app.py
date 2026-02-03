"""
Streamlit RAG Q&A Application

This is a web-based chat interface for querying diploma project documents using
Retrieval-Augmented Generation (RAG). Users can ask questions and receive AI-generated
answers based on retrieved document context.
"""

import streamlit as st
import os
from src.search import RAGSearch

# ==================== PAGE CONFIGURATION ====================
# Configure the Streamlit page with title, icon, and layout
st.set_page_config(
    page_title="Diploma Project Q&A",
    page_icon="🎓",
    layout="wide"  # Use full width of the browser
)

# ==================== CUSTOM CSS STYLING ====================
# Apply custom CSS for improved UI appearance
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTextInput > div > div > input {
        font-size: 16px;
    }
    .context-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# ==================== RAG SYSTEM INITIALIZATION ====================
# Cache the RAG system to avoid reloading on every interaction
@st.cache_resource
def load_rag_system():
    """
    Load and cache the RAG system.
    
    Uses Streamlit's cache_resource to load the system once and reuse it
    across user sessions, improving performance.
    
    Returns:
        RAGSearch: Initialized RAG search system with loaded vector store and LLM.
    """
    with st.spinner("🔄 Loading RAG system..."):
        rag = RAGSearch(
            persist_dir="faiss_store",           # FAISS index directory
            embedding_model="all-MiniLM-L6-v2",  # Sentence transformer model
            llm_model="llama-3.1-8b-instant"     # Groq LLM model
        )
    return rag

# ==================== PAGE HEADER ====================
st.title("🎓 Diploma Project Q&A System")
st.markdown("Ask questions about your diploma project and get AI-powered answers from our report.")
st.markdown("---")

# ==================== SESSION STATE INITIALIZATION ====================
# Initialize chat history in session state (persists across reruns)
if "messages" not in st.session_state:
    st.session_state.messages = []

# ==================== SIDEBAR CONFIGURATION ====================
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Slider to control number of document chunks to retrieve
    top_k = st.slider(
        "Number of context chunks",
        min_value=1,
        max_value=10,
        value=3,
        help="How many relevant document chunks to retrieve"
    )
    
    # Checkbox to toggle context visibility
    show_context = st.checkbox(
        "Show retrieved context",
        value=True,
        help="Display the document chunks used to generate the answer"
    )
    
    st.markdown("---")
    
    # Display system information
    st.header("📊 System Info")
    st.info("""
    **Model:** LLaMA 3.1 8B
    
    **Embedding:** all-MiniLM-L6-v2
    
    **Vector Store:** FAISS
    """)
    
    # Button to clear chat history
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# ==================== LOAD RAG SYSTEM ====================
# Load the RAG system with error handling
try:
    rag_system = load_rag_system()
    st.success("✅ RAG system loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading RAG system: {e}")
    st.stop()  # Stop execution if system fails to load

# ==================== DISPLAY CHAT HISTORY ====================
# Render all previous messages in the chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Show context in expandable section if available and enabled
        if "context" in message and show_context:
            with st.expander("📄 Retrieved Context"):
                st.markdown(message["context"])

# ==================== CHAT INPUT AND RESPONSE GENERATION ====================
# Handle new user input
if prompt := st.chat_input("Ask a question about your diploma project..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate AI response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            try:
                # Step 1: Retrieve relevant document chunks from vector store
                results = rag_system.vectorstore.query(prompt, top_k=top_k)
                
                # Step 2: Extract and format context from retrieved chunks
                texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
                context = "\n\n---\n\n".join([f"**Chunk {i+1}:**\n{text}" for i, text in enumerate(texts)])
                
                # Step 3: Generate AI summary using LLM with retrieved context
                summary = rag_system.search_and_sumarize(prompt, top_k=top_k)
                
                # Step 4: Display the generated response
                st.markdown(summary)
                
                # Step 5: Store assistant message and context in chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": summary,
                    "context": context
                })
                
                # Step 6: Show retrieved context in expandable section if enabled
                if show_context and context:
                    with st.expander("📄 Retrieved Context"):
                        st.markdown(context)
                        
            except Exception as e:
                # Handle errors gracefully and display to user
                error_msg = f"❌ Error generating response: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>💡 <strong>Tip:</strong> Ask specific questions about your diploma project for better results.</p>
    <p>Example: "What are the main contributions?" or "Explain the methodology used"</p>
</div>
""", unsafe_allow_html=True)
