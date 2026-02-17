from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import os
from dotenv import load_dotenv
from typing import List, TypedDict, Optional, Annotated
import operator

from healthcare_kb import get_knowledge_base

load_dotenv()

#  State defination

class AgentState(TypedDict):
    """State object that flows through the LangGraph workflow"""
    # Input
    query: str
    chat_history: Annotated[List, operator.add]
    
    # Processing
    cleaned_query: Optional[str]
    intent: Optional[str]
    
    # Retrieval
    retrieved_docs: Optional[List[str]]
    retrieval_score: Optional[float]
    
    # Validation
    context_valid: Optional[bool]
    confidence_level: Optional[str]
    
    # Response
    response: Optional[str]
    sources: Optional[List[str]]
    needs_clarification: Optional[bool]
    
    # Error handling
    error: Optional[str]


# llm intialization

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.1,
    api_key=os.getenv("GROQ_API_KEY")
)

#preloading knowledge base at module level for zero latency retrieval
try:
    # Preload the knowledge base at module import time
    # This ensures zero latency when retrieval_node is called
    KB = get_knowledge_base()
    print("✓ Knowledge Base preloaded successfully!")
    print(f"✓ Index contains {KB.index.ntotal} vectors")
    print(f"✓ Embedding model: {KB.model_id}")
except Exception as e:
    print(f"Warning: Could not preload knowledge base: {e}")
    print("   Knowledge base will be loaded on first retrieval")
    KB = None



#NODE 1: INTENT ROUTING
def intent_router_node(state: AgentState) -> AgentState:
    """
    Classify the query intent for healthcare domain
    Intents:
    - general_query: General health information, wellness tips
    - sensitive_query: Personal health data, symptoms, conditions
    - factual_medical: Medical facts, drug information, procedures
    - medical_advice: Seeking diagnosis or treatment recommendations
    - out_of_scope: Non-healthcare related queries
    """
    query = state["query"]
    
    cleaned = query.strip()     #removing any spaces at the start and end
    cleaned = " ".join(cleaned.split())     #removing any extra spaces in between
    
    print(f"[INPUT PROCESSOR] Original: {query}")
    print(f"[INPUT PROCESSOR] Cleaned: {cleaned}")
    
    
    intent_prompt = f"""You are a healthcare query classifier. Analyze the user's question and classify it into ONE of these categories:

1. **general_query** - General health information, wellness tips, preventive care, healthy lifestyle
   Examples:
   - "How much water should I drink daily?"
   - "What are the benefits of regular exercise?"
   - "How many hours of sleep do I need?"
   - "What is a balanced diet?"

2. **sensitive_query** - Personal health concerns, symptoms, medical history, private health information
   Examples:
   - "I have chest pain and shortness of breath"
   - "My blood pressure reading is 150/90"
   - "I'm experiencing anxiety and panic attacks"
   - "I've been having headaches for 3 days"

3. **factual_medical** - Medical facts, conditions, drug information, procedures, anatomy, terminology
   Examples:
   - "What is diabetes?"
   - "How does insulin work?"
   - "What are the side effects of aspirin?"
   - "Explain hypertension"

4. **medical_advice** - Seeking diagnosis, treatment recommendations, or medical decisions
   Examples:
   - "Should I take aspirin for my headache?"
   - "Do I need to see a doctor for this?"
   - "What medication is best for high blood pressure?"
   - "Can I stop taking my medication?"

5. **out_of_scope** - Questions unrelated to healthcare
   Examples:
   - "What's the weather today?"
   - "Tell me a joke"
   - "How do I cook pasta?"
   - "What time is it?"

User Query: "{cleaned}"

Respond with ONLY the category name (general_query/sensitive_query/factual_medical/medical_advice/out_of_scope)."""

    messages = [SystemMessage(content=intent_prompt)]
    response = llm.invoke(messages)
    intent = response.content.strip().lower()
    
    print(f"[INTENT ROUTER] Query: {cleaned}")
    print(f"[INTENT ROUTER] Healthcare Intent: {intent}")
    
    return {
        **state,
        "intent": intent,
        "cleaned_query": cleaned,
        "chat_history": [HumanMessage(content=cleaned)]
    }


#NODE 2: RETRIEVAL

def retrieval_node(state: AgentState) -> AgentState:
    """
    Retrieve relevant documents from healthcare knowledge base using FAISS
    """
    
    query = state["cleaned_query"]
    intent = state["intent"]
    
    print(f"[RETRIEVAL] Searching for: {query}")
    print(f"[RETRIEVAL] Intent: {intent}")
    
    # Skip retrieval for out_of_scope
    if intent == "out_of_scope":
        print("[RETRIEVAL] Skipping retrieval for out_of_scope query")
        return {
            **state,
            "retrieved_docs": [],
            "retrieval_score": 0.0,
            "sources": []
        }
    
    try:
        # Get knowledge base instance
        kb = get_knowledge_base()
        
        # Retrieve relevant documents
        # k=3 for most queries, k=5 for medical_advice (need more context)
        k = 5 if intent == "medical_advice" else 3
        
        context, sources, results = kb.get_context(query, k=k, max_length=2000)
        
        # Calculate average similarity score
        if results:
            avg_score = sum(r['similarity_score'] for r in results) / len(results)
        else:
            avg_score = 0.0
        
        # Extract document texts
        retrieved_docs = [r['text'] for r in results]
        
        print(f"[RETRIEVAL] Retrieved {len(retrieved_docs)} documents")
        print(f"[RETRIEVAL] Sources: {', '.join(sources)}")
        print(f"[RETRIEVAL] Average similarity: {avg_score:.3f}")
        
        return {
            **state,
            "retrieved_docs": retrieved_docs,
            "retrieval_score": avg_score,
            "sources": sources
        }
    
    except FileNotFoundError as e:
        print(f"[RETRIEVAL] Error: {e}")
        print("[RETRIEVAL] Falling back to empty retrieval")
        return {
            **state,
            "retrieved_docs": [],
            "retrieval_score": 0.0,
            "sources": [],
            "error": "Knowledge base not initialized. Please run: python process_healthcare_pdfs.py"
        }
    except Exception as e:
        print(f"[RETRIEVAL] Unexpected error: {e}")
        return {
            **state,
            "retrieved_docs": [],
            "retrieval_score": 0.0,
            "sources": [],
            "error": str(e)
        }

#NODE 3: context validation

def context_validator_node(state: AgentState) -> AgentState:
    """
    Validate the quality of retrieved context
    Determine confidence level for routing decisions
    """
    retrieval_score = state["retrieval_score"]
    retrieved_docs = state["retrieved_docs"]
    
    # Determine confidence level based on retrieval score
    if retrieval_score >= 0.7:
        confidence_level = "high"
        context_valid = True
    elif retrieval_score >= 0.4:
        confidence_level = "medium"
        context_valid = True
    else:
        confidence_level = "low"
        context_valid = False
    
    print(f"[CONTEXT VALIDATOR] Retrieval Score: {retrieval_score}")
    print(f"[CONTEXT VALIDATOR] Confidence Level: {confidence_level}")
    print(f"[CONTEXT VALIDATOR] Context Valid: {context_valid}")
    
    return {
        **state,
        "confidence_level": confidence_level,
        "context_valid": context_valid
    }

# NODE4: response generator with healthcare-specific prompts

def response_generator_node(state: AgentState) -> AgentState:
    """
    Generate grounded response using LLM
    Uses different prompts based on intent AND confidence level
    Healthcare-specific prompts with appropriate disclaimers
    """
    query = state["cleaned_query"]
    retrieved_docs = state["retrieved_docs"]
    confidence_level = state["confidence_level"]
    intent = state["intent"]
    
    # Prepare context
    context = "\n\n".join(retrieved_docs) if retrieved_docs else ""
    
    # Select prompt based on INTENT and confidence level
    if intent == "general_query":
        # General health information - informative and encouraging
        if confidence_level == "high":
            system_prompt = """You are a helpful healthcare information assistant. Provide general health information based ONLY on the context below.

Be encouraging and supportive. Focus on wellness and preventive care.

Context:
{context}

Question: Provide a clear, helpful answer based on the context."""
        else:
            system_prompt = """You are a helpful healthcare information assistant. Based on available information, provide general health guidance.

If information is limited, acknowledge this and suggest consulting healthcare resources.

Context:
{context}"""
    
    elif intent == "sensitive_query":
        # Personal health concerns - empathetic, private, careful
        system_prompt = """You are a compassionate healthcare assistant. The user is sharing personal health information.

IMPORTANT GUIDELINES:
- Be empathetic and supportive
- Maintain privacy and confidentiality
- Use the context below to provide relevant information
- If symptoms are serious, recommend consulting a healthcare provider
- Never diagnose or provide definitive medical conclusions

Context:
{context}

Respond with care and empathy."""
    
    elif intent == "factual_medical":
        # Medical facts - precise, educational, evidence-based
        if confidence_level == "high":
            system_prompt = """You are a medical information assistant. Provide accurate, evidence-based medical information using ONLY the context below.

Be precise and educational. Use proper medical terminology but explain it clearly.

Context:
{context}

Provide a factual, well-structured answer."""
        else:
            system_prompt = """You are a medical information assistant. Based on available information, provide medical facts.

If the context is incomplete, state this clearly and suggest authoritative medical resources.

Context:
{context}"""
    
    elif intent == "medical_advice":
        # Medical advice - VERY CAREFUL, include disclaimers
        system_prompt = """You are a healthcare information assistant. The user is seeking medical advice.

CRITICAL DISCLAIMERS:
- You are NOT a licensed medical professional
- This is NOT a substitute for professional medical advice
- Always recommend consulting a qualified healthcare provider for medical decisions

Based on the context below, you may provide general educational information, but you MUST:
1. Include a clear disclaimer
2. Recommend professional consultation
3. Not provide specific diagnoses or treatment plans

Context:
{context}

Provide educational information with appropriate disclaimers."""
    
    else:
        # Default/fallback
        system_prompt = """You are a helpful healthcare assistant. Answer based on the context below.

Context:
{context}"""
    
    # Format the prompt
    formatted_prompt = system_prompt.format(context=context)
    
    messages = [
        SystemMessage(content=formatted_prompt),
        HumanMessage(content=query)
    ]
    
    response = llm.invoke(messages)
    answer = response.content
    
    # Add intent-specific disclaimers
    if intent == "medical_advice":
        answer = f"**Medical Disclaimer**: This information is for educational purposes only and is not medical advice. Please consult a qualified healthcare provider for medical decisions.\n\n{answer}"

    
    print(f"[RESPONSE GENERATOR] Intent: {intent}, Confidence: {confidence_level}")
    print(f"[RESPONSE GENERATOR] Generated Response: {answer[:100]}...")
    
    return {
        **state,
        "response": answer,
        "sources": retrieved_docs,
        "chat_history": [AIMessage(content=answer)]
    }



# HELPER FUNCTION: Greeting/Conversation Detection

def is_greeting_or_conversation(query: str, intent: str) -> bool:
    """
    Detect if the query is a greeting or casual conversation
    These should be answered naturally by LLM, not rejected
    """
    if intent != "out_of_scope":
        return False
    
    query_lower = query.lower().strip()
    
    # Greeting patterns
    greetings = [
        'hi', 'hello', 'hey', 'good morning', 'good afternoon', 
        'good evening', 'greetings', 'howdy', 'hiya', 'sup',
        'what\'s up', 'whats up'
    ]
    
    # Conversational patterns
    conversations = [
        'how are you', 'how are you doing', 'how\'s it going',
        'nice to meet you', 'pleased to meet you',
        'thank you', 'thanks', 'thank', 'appreciate',
        'bye', 'goodbye', 'see you', 'take care',
        'who are you', 'what are you', 'what can you do',
        'what do you do', 'help', 'can you help'
    ]
    
    # Check if query matches greeting/conversation patterns
    for pattern in greetings + conversations:
        if pattern in query_lower or query_lower.startswith(pattern):
            return True
    
    # Check for question about the assistant itself
    assistant_questions = ['you', 'your name', 'your capabilities', 'askgalore']
    if any(word in query_lower for word in assistant_questions) and len(query.split()) < 10:
        return True
    
    return False


# NODE5: FALLBACK HANDLER (Healthcare-Specific with Greeting Detection)

def fallback_handler_node(state: AgentState) -> AgentState:
    """
    Handle low-confidence or out-of-scope queries
    Smart routing: Greetings/conversations get natural LLM responses
    True out-of-scope queries get polite rejection
    """
    query = state["cleaned_query"]
    intent = state["intent"]
    
    # HANDLE GREETINGS & CONVERSATIONS - Let LLM respond naturally
    if is_greeting_or_conversation(query, intent):
        print(f"[FALLBACK HANDLER] Detected greeting/conversation - using LLM")
        
        conversation_prompt = """You are AskGalore, a friendly healthcare information assistant.

Respond naturally to the user's message. Be warm, professional, and helpful.

Remember:
- You specialize in healthcare information
- You can answer questions about health, wellness, medical conditions, and medications
- You're here to provide reliable health information (not medical advice)
- Always encourage consulting healthcare professionals for medical decisions

User message: "{query}"

Respond in a friendly, conversational way."""
        
        formatted_prompt = conversation_prompt.format(query=query)
        messages = [SystemMessage(content=formatted_prompt)]
        
        response = llm.invoke(messages)
        fallback_response = response.content
        
        print(f"[FALLBACK HANDLER] Conversational response generated")
    
    # HANDLE TRUE OUT-OF-SCOPE - Politely decline
    elif intent == "out_of_scope":
        fallback_response = """I apologize, but that question seems to be outside my healthcare knowledge base.

I'm designed to help with:
- General health and wellness information
- Medical facts and terminology
- Health conditions and treatments
- Medication information

Is there a health-related question I can help you with?"""
        print(f"[FALLBACK HANDLER] Out-of-scope query - polite decline")
    
    # HANDLE LOW CONFIDENCE MEDICAL ADVICE
    elif intent == "medical_advice":
        fallback_response = f"""I don't have enough reliable information in my knowledge base to provide guidance on: "{query}"

**Important**: For medical advice, diagnosis, or treatment decisions, please consult a qualified healthcare provider such as:
- Your primary care physician
- A specialist relevant to your concern
- A telehealth service
- An urgent care facility (if urgent)

Your health and safety are paramount. Professional medical consultation is essential for personalized medical advice."""
        print(f"[FALLBACK HANDLER] Low confidence medical advice - recommend doctor")
    
    # HANDLE LOW CONFIDENCE SENSITIVE QUERY
    elif intent == "sensitive_query":
        fallback_response = f"""Thank you for sharing your health concern. I don't have sufficient information in my knowledge base to provide reliable guidance about: "{query}"

**I recommend**:
- Consulting with a healthcare provider who can properly evaluate your situation
- If symptoms are severe or worsening, seek immediate medical attention
- Keeping a record of your symptoms to share with your doctor

Your health information is private and important. Professional medical evaluation is the best course of action."""
        print(f"[FALLBACK HANDLER] Sensitive query low confidence - recommend consultation")
    
    # HANDLE OTHER LOW CONFIDENCE QUERIES
    else:
        fallback_response = f"""I don't have enough reliable information in my knowledge base to answer your question about: "{query}"

To provide accurate health information, I need higher confidence in the retrieved context.

Could you rephrase your question or ask something more specific? Alternatively, consider consulting:
- Reputable health websites (CDC, WHO, Mayo Clinic)
- Your healthcare provider
- A medical professional"""
        print(f"[FALLBACK HANDLER] General low confidence - suggest rephrasing")
    
    return {
        **state,
        "response": fallback_response,
        "needs_clarification": not is_greeting_or_conversation(query, intent),
        "chat_history": [AIMessage(content=fallback_response)]
    }


# Node6: Response Formatter (Healthcare-Specific)

def response_formatter_node(state: AgentState) -> AgentState:
    """
    Format the final response for delivery
    Add metadata, sources, and conversational polish
    """
    response = state["response"]
    sources = state.get("sources", [])
    confidence_level = state.get("confidence_level", "unknown")
    
    # Add source citations if available
    if sources and len(sources) > 0 and confidence_level in ["high", "medium"]:
        formatted_response = f"{response}\n\n---\n**Sources:** {len(sources)} document(s) retrieved"
    else:
        formatted_response = response
    
    print(f"[RESPONSE FORMATTER] Final response formatted")
    
    return {
        **state,
        "response": formatted_response
    }


# ROUTING LOGIC

def should_retrieve(state: AgentState) -> str:
    """
    Decide whether to retrieve documents based on healthcare intent
    All healthcare queries should attempt retrieval except out_of_scope
    """
    intent = state.get("intent", "general_query")
    
    if intent == "out_of_scope":
        return "fallback"
    else:
        # All healthcare intents should retrieve
        return "retrieve"


def route_after_validation(state: AgentState) -> str:
    """
    Route based on context validation results
    Special handling for medical_advice - requires HIGH confidence
    """
    confidence_level = state.get("confidence_level", "low")
    intent = state.get("intent", "general_query")
    
    # Medical advice requires HIGH confidence only
    if intent == "medical_advice":
        if confidence_level in ["high", "medium"]:
            return "generate"
        else:
            return "fallback"  # Don't provide medical advice with low/medium confidence
    
    # Other intents can proceed with high or medium confidence
    if confidence_level in ["high", "medium"]:
        return "generate"
    else:
        return "fallback"


# BUILD THE GRAPH

def create_rag_graph():
    """
    Create and compile the LangGraph workflow for healthcare RAG
    """
    # Initialize the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("intent_router", intent_router_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("context_validator", context_validator_node)
    workflow.add_node("response_generator", response_generator_node)
    workflow.add_node("fallback_handler", fallback_handler_node)
    workflow.add_node("response_formatter", response_formatter_node)
    
    # Define the flow
    workflow.add_edge(START, "intent_router")
    
    # Conditional routing after intent detection
    workflow.add_conditional_edges(
        "intent_router",
        should_retrieve,
        {
            "retrieve": "retrieval",
            "fallback": "fallback_handler"
        }
    )
    
    # Retrieval → Validation
    workflow.add_edge("retrieval", "context_validator")
    
    # Conditional routing after validation
    workflow.add_conditional_edges(
        "context_validator",
        route_after_validation,
        {
            "generate": "response_generator",
            "fallback": "fallback_handler"
        }
    )
    
    # Both paths lead to formatter
    workflow.add_edge("response_generator", "response_formatter")
    workflow.add_edge("fallback_handler", "response_formatter")
    
    # Formatter -> End
    workflow.add_edge("response_formatter", END)
    
    # Compile the graph
    app = workflow.compile()
    
    return app


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Create the graph
    rag_app = create_rag_graph()
    while True:
        user_query = input("Enter a healthcare-related question: ")
        if user_query.lower() == "exit":
            print("Goodbye!")
            break
        result = rag_app.invoke({"query": user_query})
        # Display final response
        print(f"\n{'=' * 80}")
        print("FINAL RESPONSE:")
        print("=" * 80)
        print(result["response"])
        print(f"\nConfidence Level: {result.get('confidence_level', 'N/A')}")
        print(f"Intent: {result.get('intent', 'N/A')}")
        print("=" * 80)
    
        
