import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from huggingface_hub import login
import tempfile
import faiss
import pickle
from embeddings import (
    read_file, chunk_text, get_embedding, 
    create_faiss_index, save_index, load_index, 
    search_similar
)

load_dotenv()

# Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
DEFAULT_INDEX_PATH = "askgalore_index.index"
DEFAULT_DATA_PATH = "askgalore_data.pkl"

# Initialize Gemini API
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# Page configuration
st.set_page_config(
    page_title="DocBot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* Global Styles */
.main {
    font-family: 'Inter', sans-serif;
}

/* Logo and Header */
.logo-container {
    text-align: center;
    padding: 2rem 0 1rem 0;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 15px;
    margin-bottom: 2rem;
    box-shadow: 0 8px 16px rgba(0,0,0,0.1);
}

.logo-icon {
    font-size: 4rem;
    margin-bottom: 0.5rem;
}

.main-header {
    font-size: 2.8rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0;
    letter-spacing: -0.5px;
}

.sub-header {
    font-size: 1.1rem;
    color: #e0e7ff;
    margin-top: 0.5rem;
    font-weight: 400;
}

/* ===================== */
/* MODERN SIDEBAR STYLES */
/* ===================== */
[data-testid="stSidebar"] {
    background-color: #0f172a !important;
    background-image: linear-gradient(180deg, #0f172a 0%, #1e293b 100%) !important;
    border-right: 1px solid #334155;
}

[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}

/* Sidebar headings */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #f8fafc !important;
    font-weight: 600 !important;
    letter-spacing: 0.025em;
}

/* Sidebar inputs */
[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] .stSelectbox > div > div {
    background-color: #1e293b !important;
    color: #f8fafc !important;
    border: 1px solid #475569 !important;
    border-radius: 6px;
}

[data-testid="stSidebar"] .stTextInput input:focus,
[data-testid="stSidebar"] .stSelectbox > div > div:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 1px #667eea !important;
}

/* Sidebar buttons */
[data-testid="stSidebar"] button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: #ffffff !important;
    border: none !important;
    font-weight: 600 !important;
    padding: 0.5rem 1rem !important;
    transition: all 0.2s ease;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
}

[data-testid="stSidebar"] button:hover {
    transform: translateY(-1px);
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    filter: brightness(1.1);
}

/* File Uploader */
[data-testid="stFileUploader"] {
    background-color: #1e293b;
    border: 1px dashed #475569;
    border-radius: 8px;
    padding: 1rem;
}

[data-testid="stFileUploader"] section {
    background-color: transparent !important;
}

[data-testid="stFileUploader"] button {
    background: #334155 !important; /* Secondary button style for browse */
    box-shadow: none !important;
}

/* Mode Cards */
.mode-card {
    background: white;
    padding: 1.2rem;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    margin-bottom: 1rem;
    border-left: 4px solid #667eea;
}

[data-testid="stSidebar"] .mode-card * {
    color: #1f2937 !important;
}

.mode-card.success {
    border-left-color: #10b981;
    background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
    color: #000000;
}

.mode-card.warning {
    border-left-color: #f59e0b;
    background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
    color: #000000;
}

/* Chat Messages */
.chat-message {
    padding: 1.2rem;
    border-radius: 12px;
    margin-bottom: 1rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    transition: transform 0.2s;
    color: #1f2937;
}

.chat-message:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

.user-message {
    background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
    border-left: 4px solid #3b82f6;
}

.bot-message {
    background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
    border-left: 4px solid #10b981;
}

.message-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.5rem;
    font-weight: 600;
}

.message-icon {
    font-size: 1.3rem;
}

.message-content {
    line-height: 1.6;
    color: #374151;
}

/* Source Box */
.source-box {
    background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
    padding: 0.8rem;
    border-radius: 8px;
    margin-top: 0.5rem;
    font-size: 0.85rem;
    color: #78350f;
    border-left: 3px solid #f59e0b;
}

/* Metric Cards */
.metric-card {
    background: white;
    padding: 1rem;
    border-radius: 10px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    margin: 0.5rem 0;
}

/* Buttons */
.stButton > button {
    border-radius: 8px;
    font-weight: 500;
    transition: all 0.3s;
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}

/* Divider */
hr {
    margin: 1.5rem 0;
    border: none;
    border-top: 2px solid #e5e7eb;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "model" not in st.session_state:
    st.session_state.model = None
if "index" not in st.session_state:
    st.session_state.index = None
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "mode" not in st.session_state:
    st.session_state.mode = "default"
if "previous_mode" not in st.session_state:
    st.session_state.previous_mode = "default"
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False

@st.cache_resource
def load_embedding_model():
    """Load the embedding model (cached)."""
    try:
        login(ACCESS_TOKEN)
        model = SentenceTransformer("google/embeddinggemma-300M")
        return model
    except Exception as e:
        st.error(f"Error loading embedding model: {e}")
        return None

@st.cache_resource
def load_gemini_model():
    """Load Gemini model for chat."""
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        return model
    except Exception as e:
        st.error(f"Error loading Gemini model: {e}")
        return None

def load_default_index():
    """Load the default Ask Galore index."""
    try:
        if os.path.exists(DEFAULT_INDEX_PATH) and os.path.exists(DEFAULT_DATA_PATH):
            index, chunks, embeddings = load_index(DEFAULT_INDEX_PATH, DEFAULT_DATA_PATH)
            return index, chunks, embeddings
        else:
            st.warning("⚠️ Default index not found. Please create it by running embeddings.py first.")
            return None, None, None
    except Exception as e:
        st.error(f"Error loading default index: {e}")
        return None, None, None

def process_uploaded_files(uploaded_files, embedding_model):
    """Process multiple uploaded PDF or TXT files."""
    try:
        all_chunks = []
        all_embeddings = []
        
        progress_text = st.empty()
        overall_progress = st.progress(0)
        
        for idx, uploaded_file in enumerate(uploaded_files):
            progress_text.text(f"📄 Processing file {idx+1}/{len(uploaded_files)}: {uploaded_file.name}")
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            # Read and process file
            text = read_file(tmp_path)
            
            # Clean up temp file immediately after reading
            os.unlink(tmp_path)
            
            if text:
                chunks = chunk_text(text)
                
                # Generate embeddings for this file's chunks
                file_embeddings = []
                for chunk in chunks:
                    embedding = get_embedding(chunk.page_content, embedding_model)
                    file_embeddings.append(embedding)
                
                all_chunks.extend(chunks)
                all_embeddings.extend(file_embeddings)
            
            overall_progress.progress((idx + 1) / len(uploaded_files))
        
        if not all_chunks:
            st.error("❌ No valid text found in uploaded files.")
            return None, None, None

        with st.spinner("🔍 Creating combined FAISS index..."):
            index = create_faiss_index(all_embeddings)
        
        progress_text.empty()
        overall_progress.empty()
        
        st.success(f"✅ Successfully processed {len(uploaded_files)} files ({len(all_chunks)} total chunks)!")
        return index, all_chunks, all_embeddings
    
    except Exception as e:
        st.error(f"Error processing files: {e}")
        return None, None, None

def get_relevant_context(query, index, chunks, model, k=3):
    """Retrieve relevant context for the query."""
    try:
        results = search_similar(query, model, index, chunks, k=k)
        context_parts = []
        sources = []
        
        for result in results:
            context_parts.append(result['text'])
            sources.append({
                'rank': result['rank'],
                'distance': result['distance'],
                'preview': result['text'][:200]
            })
        
        context = "\n\n".join(context_parts)
        return context, sources
    except Exception as e:
        st.error(f"Error retrieving context: {e}")
        return "", []

def generate_response(query, context, gemini_model):
    """Generate response using Gemini with retrieved context."""
    try:
        prompt = f"""You are a helpful AI assistant. Use the following context to answer the user's question.
If the answer cannot be found in the context, say so politely and provide a general helpful response.
If the user is asking general questions or greating then you will not find relevent context so answer it politly and casualy.

Context:
{context}

Question: {query}

Answer:"""
        
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "I apologize, but I encountered an error generating a response. Please try again."

def main():
    # Auto-load default index and model on startup if in default mode
    if st.session_state.mode == "default" and st.session_state.index is None:
        if os.path.exists(DEFAULT_INDEX_PATH) and os.path.exists(DEFAULT_DATA_PATH):
            with st.spinner("🔄 Loading Ask Galore index and embeddings..."):
                try:
                    st.session_state.index, st.session_state.chunks, st.session_state.embeddings = load_default_index()
                    if st.session_state.model is None:
                        st.session_state.model = load_embedding_model()
                except Exception as e:
                    st.error(f"Error auto-loading index: {e}")
    
    # Header with Logo
    st.markdown('''
        <div class="logo-container">
            <div class="logo-icon">🤖</div>
            <h1 class="main-header">DocBot</h1>
            <p class="sub-header">✨ AI-Powered Document Intelligence at Your Fingertips</p>
        </div>
    ''', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings & Configuration")
        
        # Mode selection
        mode = st.radio(
            "📦 Select Knowledge Source",
            ["📚 Ask Galore Documentation", "📄 Upload Custom Document"],
            index=0 if st.session_state.mode == "default" else 1
        )
        
        # Detect mode change
        new_mode = "default" if mode.startswith("📚") else "upload"
        mode_changed = new_mode != st.session_state.previous_mode
        
        if mode_changed:
            st.session_state.previous_mode = new_mode
            st.session_state.mode = new_mode
            st.session_state.messages = []  # Clear chat history on mode change
            
            # Reload default index when switching back to default mode
            if new_mode == "default":
                if os.path.exists(DEFAULT_INDEX_PATH) and os.path.exists(DEFAULT_DATA_PATH):
                    with st.spinner("🔄 Loading default index..."):
                        try:
                            st.session_state.index, st.session_state.chunks, st.session_state.embeddings = load_default_index()
                            if st.session_state.model is None:
                                st.session_state.model = load_embedding_model()
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error loading default index: {e}")
        
        st.session_state.mode = new_mode
        
        st.divider()
        
        # File upload section
        if st.session_state.mode == "upload":
            st.markdown("#### 📄 Document Upload")
            uploaded_files = st.file_uploader(
                "Drop your files here (Max 4)",
                type=["pdf", "txt"],
                accept_multiple_files=True,
                help="📎 Supported formats: PDF, TXT. Max 4 files."
            )
            
            if uploaded_files:
                if len(uploaded_files) > 4:
                    st.error("⚠️ Maximum 4 files allowed. Please remove some files.")
                else:
                    st.info(f"📁 **Selected:** {len(uploaded_files)} file(s)")
                    if st.button("🚀 Process Documents", type="primary", use_container_width=True):
                        # Load embedding model
                        if st.session_state.model is None:
                            with st.spinner("🧠 Loading AI model..."):
                                st.session_state.model = load_embedding_model()
                        
                        if st.session_state.model:
                            index, chunks, embeddings = process_uploaded_files(
                                uploaded_files, 
                                st.session_state.model
                            )
                        
                        if index is not None:
                            st.session_state.index = index
                            st.session_state.chunks = chunks
                            st.session_state.embeddings = embeddings
                            st.session_state.messages = []  # Clear chat history
                            st.rerun()
        else:
            st.markdown("#### 📚 Default Knowledge Base")
            if st.session_state.index is not None:
                st.markdown('''
                    <div class="mode-card success">
                        <span class="status-icon">✅</span>
                        <strong>Active: Ask Galore Documentation</strong> 
                    </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown('''
                    <div class="mode-card warning">
                        <span class="status-icon">⚠️</span>
                        <strong>Not Found:</strong> Run embeddings.py first
                    </div>
                ''', unsafe_allow_html=True)
        
        st.divider()
        
        # Info section
        st.markdown("#### 📊 Analytics")
        if st.session_state.index:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📦 Chunks", st.session_state.index.ntotal)
            with col2:
                st.metric("🎯 Mode", st.session_state.mode.title())
        else:
            st.info("🔍 No index loaded")
        
        st.divider()
        
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    # Main chat interface
    st.markdown("### 💬 Conversation")
    
    # Check if index is loaded
    if st.session_state.index is None:
        if st.session_state.mode == "default":
            st.warning("⚠️ **Default index not found.** Please run `python embeddings.py` to create the Ask Galore index first.")
        else:
            st.info("👈 **Get Started:** Upload a document in the sidebar to begin chatting!")
        return
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.container():
            if message["role"] == "user":
                st.markdown(f'''
                    <div class="chat-message user-message">
                        <div class="message-header">
                            <span class="message-icon">👤</span>
                            <span>You</span>
                        </div>
                        <div class="message-content">{message["content"]}</div>
                    </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                    <div class="chat-message bot-message">
                        <div class="message-header">
                            <span class="message-icon">🤖</span>
                            <span>DocBot</span>
                        </div>
                        <div class="message-content">{message["content"]}</div>
                    </div>
                ''', unsafe_allow_html=True)
                
                if "sources" in message and message["sources"]:
                    with st.expander("📚 View Knowledge Sources", expanded=False):
                        for source in message["sources"]:
                            st.markdown(f"""
                                <div class="source-box">
                                    <strong>🔖 Source {source['rank']}</strong> (Relevance: {source['distance']:.4f})<br>
                                    <em>{source['preview']}...</em>
                                </div>
                            """, unsafe_allow_html=True)
    
    # Chat input
    user_query = st.chat_input("💬 Type your question here...")
    
    if user_query:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        # Load models if needed
        if st.session_state.model is None:
            st.session_state.model = load_embedding_model()
        
        gemini_model = load_gemini_model()
        
        if st.session_state.model and gemini_model and st.session_state.index:
            # Get relevant context
            with st.spinner("🔍 Searching for relevant information..."):
                context, sources = get_relevant_context(
                    user_query,
                    st.session_state.index,
                    st.session_state.chunks,
                    st.session_state.model,
                    k=3
                )
            
            # Generate response
            with st.spinner("💭 Generating response..."):
                response = generate_response(user_query, context, gemini_model)
            
            # Add bot message
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "sources": sources
            })
            
            st.rerun()
        else:
            st.error("Models not loaded properly. Please check your API keys.")

if __name__ == "__main__":
    main()