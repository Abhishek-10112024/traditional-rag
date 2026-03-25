"""Streamlit UI for RAG application."""

import streamlit as st
from streamlit_chat import message
from typing import List, Dict
import sys
from pathlib import Path
from loguru import logger
import time

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.rag_pipeline import RAGPipeline
from config import settings, LOGS_DIR

# Configure logger
logger.add(f"{LOGS_DIR}/streamlit.log", rotation="500 MB")

# Page configuration
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        color: #1976d2;
    }
    .assistant-message {
        background-color: #f3e5f5;
        color: #7b1fa2;
    }
</style>
""", unsafe_allow_html=True)


class RAGApp:
    """Streamlit RAG Application."""
    
    def __init__(self):
        """Initialize the app."""
        self.rag_pipeline = None
        self.init_session_state()
    
    def init_session_state(self):
        """Initialize session state."""
        if "rag_pipeline" not in st.session_state:
            st.session_state.rag_pipeline = None
        
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        if "documents_loaded" not in st.session_state:
            st.session_state.documents_loaded = False
        
        if "doc_count" not in st.session_state:
            st.session_state.doc_count = 0
    
    def initialize_pipeline(self):
        """Initialize RAG pipeline."""
        try:
            with st.spinner("Initializing RAG pipeline..."):
                self.rag_pipeline = RAGPipeline(
                    use_reranking=st.session_state.get("use_reranking", True),
                    use_hybrid_search=st.session_state.get("use_hybrid_search", True)
                )
                st.session_state.rag_pipeline = self.rag_pipeline
                
                # Check if documents already exist in the persisted vector store
                try:
                    if self.rag_pipeline.hybrid_search:
                        doc_count = self.rag_pipeline.hybrid_search.vector_store.get_document_count()
                        if doc_count > 0:
                            st.session_state.documents_loaded = True
                            st.session_state.doc_count = doc_count
                            st.success(f"✅ RAG pipeline initialized with {doc_count} documents!")
                            logger.info(f"RAG pipeline initialized with {doc_count} persisted documents")
                        else:
                            st.success("✅ RAG pipeline initialized successfully! (No documents)")
                            logger.info("RAG pipeline initialized via Streamlit UI")
                    else:
                        st.success("✅ RAG pipeline initialized successfully!")
                except Exception as e:
                    logger.warning(f"Could not detect document count: {e}")
                    st.success("✅ RAG pipeline initialized successfully!")
        except Exception as e:
            st.error(f"❌ Error initializing RAG pipeline: {e}")
            logger.error(f"Error initializing RAG pipeline: {e}")
    
    def handle_document_upload(self):
        """Handle document upload."""
        st.subheader("📄 Document Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Upload Documents**")
            uploaded_files = st.file_uploader(
                "Upload text files (.txt, .md)",
                type=["txt", "md"],
                accept_multiple_files=True
            )
            
            if uploaded_files:
                if st.button("📤 Process and Index Documents", use_container_width=True):
                    with st.spinner("Processing documents..."):
                        try:
                            documents = []
                            for file in uploaded_files:
                                # Try multiple encodings for robust file reading
                                content = None
                                for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
                                    try:
                                        content = file.read().decode(encoding)
                                        break
                                    except (UnicodeDecodeError, AttributeError):
                                        file.seek(0)  # Reset file pointer
                                        continue
                                
                                if content is None:
                                    st.warning(f"⚠️ Could not decode {file.name} - skipping")
                                    logger.warning(f"Skipped {file.name} - unsupported encoding")
                                    continue
                                
                                # Split by paragraphs or chunks
                                chunk_size = 800
                                overlap = 100

                                words = content.split()
                                chunks = []

                                for i in range(0, len(words), chunk_size - overlap):
                                    chunk = " ".join(words[i:i + chunk_size])
                                    chunks.append(chunk)
                                documents.extend(chunks)
                            
                            if not documents:
                                st.error("❌ No documents were successfully processed")
                                return
                            
                            if self.rag_pipeline is None:
                                self.initialize_pipeline()
                            
                            self.rag_pipeline.add_documents(documents)
                            st.session_state.documents_loaded = True
                            st.session_state.doc_count = len(documents)
                            
                            st.success(f"✅ Successfully indexed {len(documents)} document chunks!")
                            logger.info(f"Indexed {len(documents)} document chunks via Streamlit")
                        except Exception as e:
                            st.error(f"❌ Error processing documents: {e}")
                            logger.error(f"Error processing documents: {e}")
        
        with col2:
            st.write("**Dataset Information**")
            if st.session_state.documents_loaded:
                st.metric("Documents Indexed", st.session_state.doc_count)
                if st.button("🗑️ Clear All Documents", use_container_width=True):
                    try:
                        if self.rag_pipeline:
                            self.rag_pipeline.hybrid_search.clear()
                            st.session_state.documents_loaded = False
                            st.session_state.doc_count = 0
                            st.success("✅ Documents cleared!")
                    except Exception as e:
                        st.error(f"Error clearing documents: {e}")
            else:
                st.info("No documents indexed yet. Upload files to get started.")
    
    def handle_settings(self):
        """Handle settings panel."""
        st.sidebar.header("⚙️ Settings")
        
        with st.sidebar.expander("Search Configuration"):
            st.session_state.use_hybrid_search = st.checkbox(
                "Enable Hybrid Search (BM25 + Vector)",
                value=True,
                help="Use both sparse and dense retrieval"
            )
            
            st.session_state.vector_weight = st.slider(
                "Vector Search Weight",
                0.0, 1.0, 0.7,
                help="Weight for vector search in hybrid search"
            )
            
            st.session_state.top_k_retrieval = st.slider(
                "Top-K Documents to Retrieve",
                1, 20, 10,
                help="Number of documents to retrieve before reranking"
            )
        
        with st.sidebar.expander("Reranking Configuration"):
            st.session_state.use_reranking = st.checkbox(
                "Enable Reranking",
                value=True,
                help="Use cross-encoder for reranking"
            )
            
            st.session_state.top_k_rerank = st.slider(
                "Top-K Documents After Reranking",
                1, 10, 4,
                help="Number of documents to pass to LLM"
            )
        
        with st.sidebar.expander("LLM Configuration"):
            st.session_state.temperature = st.slider(
                "Temperature",
                0.0, 2.0, 0.7,
                step=0.1,
                help="Higher = more creative, Lower = more focused"
            )
            
            st.session_state.top_p = st.slider(
                "Top-P (Nucleus Sampling)",
                0.0, 1.0, 0.9,
                step=0.05,
                help="Controls diversity of output"
            )
            
            st.session_state.max_tokens = st.slider(
                "Max Tokens",
                1, 1000, settings.MAX_NEW_TOKENS,
                step=50,
                help="Maximum length of generated response"
            )
        
        with st.sidebar.expander("Cache Configuration"):
            st.session_state.cache_enabled = st.checkbox(
                "Enable Response Caching",
                value=True,
                help="Cache responses to identical queries"
            )
            
            if st.button("Clear Cache", use_container_width=True):
                if self.rag_pipeline:
                    self.rag_pipeline.clear_cache()
                    st.success("✅ Cache cleared!")
    
    def handle_chat(self):
        """Handle chat interface."""
        st.subheader("💬 Chat with RAG Assistant")
        
        # Check if pipeline is initialized
        if not st.session_state.documents_loaded:
            st.warning("⚠️ Please upload and index documents first!")
            return
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for i, msg in enumerate(st.session_state.chat_history):
                if msg["role"] == "user":
                    with st.chat_message("user", avatar="👤"):
                        st.markdown(msg["content"])
                else:
                    with st.chat_message("assistant", avatar="🤖"):
                        st.markdown(msg["content"])
        
        # Input area
        st.divider()
        col1, col2 = st.columns([0.85, 0.15])
        
        with col1:
            user_input = st.text_input(
                "Ask a question:",
                key="user_input_",
                placeholder="What would you like to know?"
            )
        
        with col2:
            send_button = st.button("Send", use_container_width=True)
        
        # Process query
        if send_button and user_input:
            # Add user message to history
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input
            })
            
            # Generate response
            try:
                with st.spinner("Thinking..."):
                    # Initialize pipeline if not already done
                    if self.rag_pipeline is None:
                        self.initialize_pipeline()
                    
                    # Check if documents are loaded (either from UI or from script)
                    has_documents = st.session_state.get("documents_loaded", False)
                    
                    # Also check the actual vector store for documents
                    if not has_documents and self.rag_pipeline is not None:
                        try:
                            if self.rag_pipeline.hybrid_search:
                                doc_count = self.rag_pipeline.hybrid_search.vector_store.get_document_count()
                                has_documents = doc_count > 0
                                if has_documents:
                                    st.session_state.documents_loaded = True
                                    st.session_state.doc_count = doc_count
                        except:
                            pass
                    
                    if not has_documents:
                        st.warning("⚠️ Please upload and index documents first!")
                        st.info("💡 You can either:\n1. Upload files in the 'Document Management' tab, or\n2. Run: `python scripts/ingest_documents.py <files>`")
                        st.session_state.chat_history.pop()  # Remove the user message we just added
                    else:
                        result = self.rag_pipeline.query(
                            user_input,
                            top_k_retrieval=st.session_state.get("top_k_retrieval", 10),
                            top_k_rerank=st.session_state.get("top_k_rerank", 4)
                        )
                    
                        # Display retrieved documents
                        with st.expander("📚 Retrieved Documents"):
                            for i, doc in enumerate(result["retrieved_docs"], 1):
                                score = doc.get("score", "")

                                # Handle float vs dict safely
                                if isinstance(score, (int, float)):
                                    score_display = f"{score:.3f}"
                                else:
                                    score_display = str(score)

                                st.markdown(f"**Document {i}** (Score: {score_display})")
                                st.write(doc["text"][:200] + "...")
                    
                        # Add assistant response
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": result["answer"]
                        })
                
                # Display response
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(result["answer"])
                
                logger.info(f"Query processed: {user_input[:50]}...")
                st.rerun()
            
            except Exception as e:
                st.error(f"❌ Error processing query: {e}")
                logger.error(f"Error processing query: {e}")
    
    def run(self):
        """Run the Streamlit app."""
        # Header
        st.title("🤖 RAG Assistant")
        st.markdown("*A free, production-ready Retrieval-Augmented Generation system*")
        
        # Auto-initialize pipeline on page load
        if self.rag_pipeline is None and st.session_state.rag_pipeline is None:
            self.initialize_pipeline()
        
        # Main layout
        tab1, tab2, tab3 = st.tabs(["💬 Chat", "📄 Documents", "ℹ️ About"])
        
        # Load settings
        self.handle_settings()
        
        with tab1:
            self.handle_chat()
        
        with tab2:
            self.handle_document_upload()
        
        with tab3:
            st.markdown("""
            ## About RAG Assistant
            
            This is a production-ready Retrieval-Augmented Generation (RAG) system built with:
            
            ### Features
            - **Hybrid Search**: Combines BM25 (sparse) and vector (dense) retrieval
            - **Reranking**: Uses cross-encoder models to improve result relevance
            - **Free LLMs**: Supports Ollama for local inference
            - **Caching**: Response caching for improved performance
            - **Production Ready**: Error handling, logging, and performance optimization
            
            ### Tech Stack
            - **Embeddings**: Sentence-Transformers (all-MiniLM-L6-v2)
            - **Vector Store**: Chroma (free, open-source)
            - **Reranker**: Cross-Encoder (ms-marco-MiniLM-L-6-v2)
            - **LLM**: Ollama (mistral, neural-chat, orca-mini)
            - **Frontend**: Streamlit
            - **Cloud**: HuggingFace Spaces / Streamlit Cloud
            
            ### Getting Started
            1. Upload documents in the Documents tab
            2. Configure settings as needed
            3. Start chatting with the assistant
            
            ### Tips
            - Use more documents for better context
            - Adjust temperature for different response styles
            - Enable reranking for better relevance
            - Check retrieved documents to understand sources
            
            ---
            Built with ❤️ using free and open-source tools
            """)


def main():
    """Main entry point."""
    app = RAGApp()
    app.run()


if __name__ == "__main__":
    main()
