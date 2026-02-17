"""
Healthcare RAG Chatbot - Beautiful Streamlit Frontend
Features: Medical-themed UI, Chat history with avatars, Real-time responses
"""

import streamlit as st
from rag_agent import create_rag_graph, KB
from datetime import datetime

# Page Configuration
st.set_page_config(
    page_title="AskGalore - Healthcare Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/askgalore',
        'Report a bug': None,
        'About': "AskGalore Healthcare Assistant - AI-Powered Health Information"
    }
)

# Custom CSS Styling
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
        padding: 2.5rem 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .logo-container:before {
        content: "";
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: pulse 3s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 0.5; }
        50% { opacity: 0.8; }
    }
    
    .logo-icon {
        font-size: 4.5rem;
        margin-bottom: 0.5rem;
        filter: drop-shadow(0 4px 8px rgba(0,0,0,0.2));
    }
    
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0;
        letter-spacing: -1px;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        font-size: 1.2rem;
        color: #e0e7ff;
        margin-top: 0.5rem;
        font-weight: 400;
    }
    
    /* ===================== */
    /* MODERN SIDEBAR STYLES */
    /* ===================== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%) !important;
        border-right: 1px solid #334155;
    }
    
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #f8fafc !important;
        font-weight: 600 !important;
    }
    
    [data-testid="stSidebar"] button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: #ffffff !important;
        border: none !important;
        font-weight: 600 !important;
        transition: all 0.2s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    [data-testid="stSidebar"] button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
        filter: brightness(1.1);
    }
    
    /* Chat Messages Container */
    .message-container {
        display: flex;
        align-items: flex-start;
        gap: 1rem;
        margin-bottom: 1.5rem;
        animation: slideIn 0.4s ease;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Avatar Styles */
    .avatar {
        width: 50px;
        height: 50px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.8rem;
        flex-shrink: 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        position: relative;
        transition: all 0.3s ease;
    }
    
    .avatar:hover {
        transform: scale(1.1) rotate(5deg);
        box-shadow: 0 6px 16px rgba(0,0,0,0.2);
    }
    
    .user-avatar {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: 3px solid #ffffff;
    }
    
    .bot-avatar {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        border: 3px solid #ffffff;
    }
    
    /* Chat Bubble */
    .chat-message {
        flex: 1;
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
        position: relative;
    }
    
    .chat-message:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.12);
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-left: 4px solid #4c63d2;
    }
    
    /* Speech bubble arrow for user */
    .user-message:before {
        content: '';
        position: absolute;
        left: -10px;
        top: 20px;
        width: 0;
        height: 0;
        border-style: solid;
        border-width: 10px 10px 10px 0;
        border-color: transparent #667eea transparent transparent;
    }
    
    .bot-message {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        color: #1f2937;
        border-left: 4px solid #10b981;
    }
    
    /* Speech bubble arrow for bot */
    .bot-message:before {
        content: '';
        position: absolute;
        left: -10px;
        top: 20px;
        width: 0;
        height: 0;
        border-style: solid;
        border-width: 10px 10px 10px 0;
        border-color: transparent #f0fdf4 transparent transparent;
    }
    
    .message-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.75rem;
        font-weight: 600;
        font-size: 1.05rem;
    }
    
    .status-indicator {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #10b981;
        animation: pulse-dot 2s infinite;
    }
    
    @keyframes pulse-dot {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* Typing Indicator */
    .typing-indicator {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 1.5rem;
        animation: slideIn 0.3s ease;
    }
    
    .typing-dots {
        display: flex;
        gap: 0.4rem;
        padding: 0.8rem 1.2rem;
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border-radius: 12px;
        border-left: 4px solid #10b981;
    }
    
    .typing-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: #10b981;
        animation: typing 1.4s infinite;
    }
    
    .typing-dot:nth-child(2) {
        animation-delay: 0.2s;
    }
    
    .typing-dot:nth-child(3) {
        animation-delay: 0.4s;
    }
    
    @keyframes typing {
        0%, 60%, 100% {
            transform: translateY(0);
            opacity: 0.7;
        }
        30% {
            transform: translateY(-10px);
            opacity: 1;
        }
    }
    
    .message-content {
        line-height: 1.7;
        font-size: 1rem;
    }
    
    .user-message .message-content {
        color: #ffffff;
    }
    
    .bot-message .message-content {
        color: #374151;
    }
    
    /* Badges */
    .intent-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.3rem 0.3rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: all 0.2s ease;
    }
    
    .intent-badge:hover {
        transform: translateY(-1px);
        box-shadow: 0 3px 6px rgba(0,0,0,0.15);
    }
    
    .badge-high { 
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        color: #065f46;
        border: 1px solid #6ee7b7;
    }
    .badge-medium { 
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        color: #92400e;
        border: 1px solid #fbbf24;
    }
    .badge-low { 
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        color: #991b1b;
        border: 1px solid #f87171;
    }
    
    /* Source Citation */
    .source-citation {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-left: 4px solid #f59e0b;
        padding: 1rem 1.25rem;
        margin-top: 1rem;
        border-radius: 12px;
        font-size: 0.9rem;
        color: #78350f;
        box-shadow: 0 2px 8px rgba(245, 158, 11, 0.2);
    }
    
    .source-citation strong {
        color: #92400e;
    }
    
    /* Disclaimer Box */
    .disclaimer {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border: 2px solid #ef4444;
        border-radius: 12px;
        padding: 1.25rem;
        margin: 1rem 0;
        font-size: 0.9rem;
        color: #991b1b;
        box-shadow: 0 2px 8px rgba(239, 68, 68, 0.2);
    }
    
    /* Metric Cards */
    .metric-card {
        background: white;
        padding: 1.25rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin: 0.75rem 0;
        border-left: 4px solid #667eea;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);
    }
    
    /* Welcome Info Box */
    .welcome-box {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        padding: 2rem;
        border-radius: 16px;
        border-left: 4px solid #3b82f6;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.2);
        margin-bottom: 2rem;
    }
    
    .welcome-box h3 {
        color: #1e40af;
        margin-top: 0;
    }
    
    .welcome-box ul {
        color: #1e3a8a;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #764ba2 0%, #667eea 100%);
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'rag_app' not in st.session_state:
    with st.spinner("🏥 Initializing Healthcare RAG System..."):
        st.session_state.rag_app = create_rag_graph()

# Helper Functions
def get_confidence_badge(confidence):
    badges = {
        'high': ('✅ High', 'badge-high'),
        'medium': ('⚠️ Medium', 'badge-medium'),
        'low': ('❌ Low', 'badge-low')
    }
    label, cls = badges.get(confidence, ('Unknown', 'badge-medium'))
    return f'<span class="intent-badge {cls}">{label}</span>'

def format_sources(sources):
    if not sources:
        return ""
    unique = list(set(sources))
    return f'<div class="source-citation">📚 <strong>Sources:</strong> {", ".join(unique)}</div>'

# Header with Logo
st.markdown("""
<div class="logo-container">
    <div class="logo-icon">🏥</div>
    <h1 class="main-header">AskGalore Healthcare</h1>
    <p class="sub-header">✨ AI-Powered Healthcare Information at Your Fingertips</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 📊 System Status")
    
    if KB and KB.loaded:
        st.success("✅ Knowledge Base Loaded")
        st.metric("Documents", f"{KB.index.ntotal}")
        st.metric("Model", "EmbeddingGemma")
    else:
        st.error("❌ KB Not Loaded")
    
    st.markdown("---")
    st.markdown("### 💬 Session Stats")
    st.metric("Messages", len(st.session_state.chat_history))
    
    st.markdown("---")
    st.markdown("### 💡 Example Queries")
    
    examples = [
        "How much water should I drink daily?",
        "What is diabetes?",
        "Side effects of aspirin",
        "Benefits of regular exercise"
    ]
    
    for ex in examples:
        if st.button(ex, key=f"ex_{ex[:20]}"):
            st.session_state.example_query = ex
    
    st.markdown("---")
    
    if st.button("🗑️ Clear Chat", type="secondary"):
        st.session_state.chat_history = []
        st.rerun()
    
    st.markdown("""
    <div class="disclaimer">
        <strong>⚠️ Medical Disclaimer</strong><br>
        This AI provides general information only. 
        Always consult healthcare professionals.
    </div>
    """, unsafe_allow_html=True)

# Display Chat History
chat_container = st.container()

with chat_container:
    if len(st.session_state.chat_history) == 0:
        st.markdown("""
        <div class="welcome-box">
            <h3>👋 Welcome to AskGalore Healthcare Assistant!</h3>
            <p>I'm here to help you with healthcare information. You can ask me about:</p>
            <ul>
                <li>💡 <strong>General Health:</strong> Wellness tips, nutrition, exercise</li>
                <li>📚 <strong>Medical Facts:</strong> Conditions, symptoms, terminology</li>
                <li>💊 <strong>Medications:</strong> Drug information, side effects</li>
                <li>🏃 <strong>Lifestyle:</strong> Sleep, stress management, preventive care</li>
            </ul>
            <p><strong>Start by typing your question below!</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    for chat in st.session_state.chat_history:
        if chat['role'] == 'user':
            st.markdown(f"""
            <div class="message-container">
                <div class="avatar user-avatar">👤</div>
                <div class="chat-message user-message">
                    <div class="message-header">
                        <span>You</span>
                    </div>
                    <div class="message-content">{chat['content']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            confidence_badge = get_confidence_badge(chat.get('confidence', 'low'))
            sources_html = format_sources(chat.get('sources', []))
            timestamp = chat.get('timestamp', '')
            
            st.markdown(f"""
            <div class="message-container">
                <div class="avatar bot-avatar">🏥</div>
                <div class="chat-message bot-message">
                    <div class="message-header">
                        <span>AskGalore Assistant</span>
                        <span class="status-indicator"></span>
                    </div>
                    <div class="message-content">
                        {chat['content']}
                        <br><br>
                        {confidence_badge}
                    </div>
                    {sources_html}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Auto-scroll to bottom after rendering messages
    if len(st.session_state.chat_history) > 0:
        st.markdown('<div id="scroll-anchor"></div>', unsafe_allow_html=True)
        st.markdown("""
        <script>
            window.parent.document.querySelector('#scroll-anchor').scrollIntoView({behavior: 'smooth', block: 'end'});
        </script>
        """, unsafe_allow_html=True)

# Chat Input with Enter to Send
if 'example_query' in st.session_state:
    user_query = st.session_state.example_query
    del st.session_state.example_query
else:
    user_query = None

# Initialize clear_input flag
if 'clear_input' not in st.session_state:
    st.session_state.clear_input = False

# Form for Enter key submission
with st.form(key='chat_form', clear_on_submit=True):
    col1, col2 = st.columns([6, 1])
    
    with col1:
        query_input = st.text_input(
            "💬 Ask your question:",
            value=user_query if user_query else "",
            placeholder="e.g., What are the symptoms of diabetes? (Press Enter to send)",
            key="query_input",
            label_visibility="collapsed"
        )
    
    with col2:
        send_button = st.form_submit_button("🚀 Send", type="primary", use_container_width=True)

# Auto-focus on input field
st.markdown("""
<script>
    // Auto-focus the input field
    const interval = setInterval(() => {
        const input = window.parent.document.querySelector('input[aria-label="💬 Ask your question:"]');
        if (input) {
            input.focus();
            clearInterval(interval);
        }
    }, 100);
    setTimeout(() => clearInterval(interval), 1000);
</script>
""", unsafe_allow_html=True)

# Process Query
if send_button and query_input:
    st.session_state.chat_history.append({
        'role': 'user',
        'content': query_input,
        'timestamp': datetime.now().strftime("%H:%M:%S")
    })
    
    # Show typing indicator
    typing_placeholder = st.empty()
    typing_placeholder.markdown("""
    <div class="message-container">
        <div class="avatar bot-avatar">🏥</div>
        <div class="typing-dots">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    initial_state = {
        "query": query_input,
        "chat_history": [],
        "cleaned_query": None,
        "intent": None,
        "retrieved_docs": None,
        "retrieval_score": None,
        "context_valid": None,
        "confidence_level": None,
        "response": None,
        "sources": None,
        "needs_clarification": None,
        "error": None
    }
    
    try:
        result = st.session_state.rag_app.invoke(initial_state)
        
        st.session_state.chat_history.append({
            'role': 'assistant',
            'content': result.get('response', 'Sorry, I could not generate a response.'),
            'intent': result.get('intent', 'unknown'),
            'confidence': result.get('confidence_level', 'low'),
            'sources': result.get('sources', []),
            'timestamp': datetime.now().strftime("%H:%M:%S")
        })
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
    finally:
        typing_placeholder.empty()
    
    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🏥 <strong>AskGalore Healthcare Assistant</strong> | Powered by AI</p>
    <p style="font-size: 0.8rem;">Educational tool only • Consult healthcare professionals for medical advice</p>
</div>
""", unsafe_allow_html=True)