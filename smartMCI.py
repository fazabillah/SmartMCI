import streamlit as st
import os
import re
import hashlib
import json
from datetime import datetime, timedelta
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from typing import Dict, List, Tuple, Optional
import requests

# Load environment variables
load_dotenv(override=True)

# Page configuration
st.set_page_config(
    page_title="SmartMCI", 
    layout="wide",
    page_icon="🛡️"
)

# API Documents configuration
API_INDEXES = {
    "api571": "api571-damage-mechanisms",
    "api970": "api970-corrosion-control", 
    "api584": "api584-integrity-windows"
}

# Configuration data for analysis page
EQUIPMENT_TYPES = [
    "Pressure Vessels", "Piping Systems", "Heat Exchangers", 
    "Storage Tanks", "Reactors", "Columns/Towers"
]

MATERIALS = [
    "Carbon Steel", "Stainless Steel 304/316", "Duplex Stainless Steel", 
    "Super Duplex (2507)", "Inconel 625", "Hastelloy C-276"
]

DAMAGE_MECHANISMS = [
    "Pitting Corrosion", "Crevice Corrosion", "Stress Corrosion Cracking",
    "General Corrosion", "Hydrogen Embrittlement", "Fatigue Cracking",
    "High Temperature Corrosion", "Erosion-Corrosion"
]

ENVIRONMENTS = [
    "Marine/Offshore", "Sour Service (H2S)", "High Temperature", 
    "Chloride Environment", "Caustic Service", "Atmospheric"
]

# Simple caching system
class SimpleCache:
    def __init__(self, ttl_hours=24):
        self.ttl_hours = ttl_hours
        if "chat_cache" not in st.session_state:
            st.session_state.chat_cache = {}
    
    def _create_key(self, query: str, context: dict = None) -> str:
        """Create cache key from query and context"""
        normalized = re.sub(r'[^\w\s]', '', query.lower())
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        if context:
            context_str = "|".join([f"{k}:{v}" for k, v in context.items() if v and v != "Not Specified"])
            cache_string = f"{normalized}|{context_str}"
        else:
            cache_string = normalized
            
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def get(self, query: str, context: dict = None):
        """Get cached response if valid"""
        key = self._create_key(query, context)
        if key in st.session_state.chat_cache:
            cache_data = st.session_state.chat_cache[key]
            cache_time = datetime.fromisoformat(cache_data["timestamp"])
            if datetime.now() - cache_time < timedelta(hours=self.ttl_hours):
                return cache_data["response"]
            else:
                del st.session_state.chat_cache[key]
        return None
    
    def set(self, query: str, response: str, context: dict = None):
        """Cache a response"""
        key = self._create_key(query, context)
        st.session_state.chat_cache[key] = {
            "response": response,
            "timestamp": datetime.now().isoformat()
        }

# Initialize components
@st.cache_resource
def setup_embeddings():
    """Setup embeddings model"""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

@st.cache_resource  
def setup_vectorstores():
    """Setup vector stores for API documents"""
    if not os.environ.get("PINECONE_API_KEY"):
        st.error("❌ PINECONE_API_KEY not found. Please check your .env file.")
        return {}, False
    
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    embeddings = setup_embeddings()
    vectorstores = {}
    
    for api_name, index_name in API_INDEXES.items():
        try:
            existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
            if index_name in existing_indexes:
                index = pc.Index(index_name)
                stats = index.describe_index_stats()
                vector_count = stats.get('total_vector_count', 0)
                
                if vector_count > 0:
                    vectorstore = PineconeVectorStore(index=index, embedding=embeddings)
                    vectorstores[api_name] = vectorstore
                else:
                    st.warning(f"⚠️ {api_name.upper()} index is empty")
            else:
                st.warning(f"⚠️ {api_name.upper()} index not found")
                
        except Exception as e:
            st.error(f"❌ Error connecting to {api_name.upper()}: {e}")
    
    if vectorstores:
        st.success(f"✅ Connected to: {', '.join([f'{k.upper()}' for k in vectorstores.keys()])}")
    else:
        st.error("❌ No API documents found. Please run ingestion script first.")
    
    return vectorstores, len(vectorstores) > 0

@st.cache_resource
def setup_llm():
    """Setup LLM"""
    if not os.environ.get("GROQ_API_KEY"):
        st.error("❌ GROQ_API_KEY not found. Please check your .env file.")
        return None
    
    return ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.1)

def is_mci_related(query: str) -> bool:
    """Check if query is related to materials, corrosion, or integrity"""
    mci_keywords = [
        # Damage mechanisms
        'corrosion', 'cracking', 'damage', 'degradation', 'failure', 'pitting', 
        'crevice', 'stress', 'fatigue', 'erosion', 'embrittlement', 'oxidation',
        # Materials
        'steel', 'stainless', 'carbon', 'alloy', 'metal', 'material', 'duplex',
        'inconel', 'hastelloy', 'aluminum', 'copper', 'titanium',
        # Equipment
        'pipe', 'vessel', 'tank', 'exchanger', 'reactor', 'equipment', 'piping',
        'pressure', 'storage', 'column', 'tower', 'pipeline',
        # Environment/conditions
        'temperature', 'chloride', 'sour', 'h2s', 'caustic', 'marine', 'offshore',
        'high temp', 'environment', 'service', 'operating', 'process',
        # API standards
        'api 571', 'api 970', 'api 584', 'api571', 'api970', 'api584',
        # Prevention/control
        'mitigation', 'prevention', 'protection', 'inhibitor', 'coating',
        'cathodic', 'inspection', 'monitoring', 'integrity', 'limits'
    ]
    
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in mci_keywords)

def retrieve_documents(query: str, vectorstores: dict) -> List:
    """Retrieve relevant documents from all API sources"""
    all_docs = []
    
    for api_name, vectorstore in vectorstores.items():
        try:
            retriever = vectorstore.as_retriever(k=3)
            docs = retriever.invoke(query)
            all_docs.extend(docs)
        except Exception as e:
            st.warning(f"⚠️ Error retrieving from {api_name.upper()}: {e}")
    
    return all_docs

def format_documents(docs) -> str:
    """Format retrieved documents for the prompt"""
    if not docs:
        return "No relevant API documentation found."
    
    formatted_docs = []
    for i, doc in enumerate(docs):
        api_standard = doc.metadata.get('api_standard', 'Unknown API')
        page_num = doc.metadata.get('page', 'Unknown Page')
        
        formatted_docs.append(
            f"[Source {i+1}] {api_standard} - Page {page_num}\n"
            f"Content: {doc.page_content}\n"
        )
    
    return "\n".join(formatted_docs)

def get_conversation_context(chat_history: List[dict], max_messages: int = 4) -> str:
    """Get recent conversation context"""
    if len(chat_history) <= 1:
        return ""
    
    recent_messages = chat_history[-(max_messages*2):]  # Get recent Q&A pairs
    context_parts = []
    
    for msg in recent_messages:
        if msg["role"] == "user":
            context_parts.append(f"User: {msg['content']}")
        elif msg["role"] == "assistant":
            # Include only first 150 characters of response for context
            summary = msg['content'][:150] + "..." if len(msg['content']) > 150 else msg['content']
            context_parts.append(f"Assistant: {summary}")
    
    return "\n".join(context_parts) if context_parts else ""

def search_web_tavily(query: str, max_results: int = 5) -> str:
    """Search web using Tavily API as fallback when API docs are insufficient"""
    
    tavily_api_key = os.environ.get("TAVILY_API_KEY")
    if not tavily_api_key:
        return "Web search unavailable - TAVILY_API_KEY not configured."
    
    try:
        # Focus search on MCI and engineering topics
        focused_query = f"materials corrosion integrity engineering {query}"
        
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": tavily_api_key,
            "query": focused_query,
            "search_depth": "basic",
            "include_answer": True,
            "include_raw_content": False,
            "max_results": max_results,
            "include_domains": [
                "asme.org", "api.org", "nace.org", "astm.org", 
                "engineeringtoolbox.com", "corrosionpedia.com",
                "sciencedirect.com", "springer.com", "onepetro.org"
            ]
        }
        
        response = requests.post(url, json=payload, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            web_content = []
            
            # Add Tavily's answer if available
            if data.get("answer"):
                web_content.append(f"Web Search Summary: {data['answer']}")
            
            # Add search results
            results = data.get("results", [])
            for i, result in enumerate(results[:max_results]):
                title = result.get("title", "No title")
                content = result.get("content", "No content")
                url = result.get("url", "No URL")
                
                web_content.append(
                    f"[Web Source {i+1}] {title}\n"
                    f"URL: {url}\n"
                    f"Content: {content[:500]}...\n"
                )
            
            return "\n".join(web_content) if web_content else "No relevant web results found."
            
        else:
            return f"Web search error: HTTP {response.status_code}"
            
    except Exception as e:
        return f"Web search failed: {str(e)}"

def assess_content_sufficiency(docs: List, query: str) -> Tuple[bool, str]:
    """Assess if retrieved API documents are sufficient to answer the query"""
    
    if not docs:
        return False, "No relevant API documentation found."
    
    # Simple heuristics to determine if content is sufficient
    total_content_length = sum(len(doc.page_content) for doc in docs)
    
    # Check for key query terms in retrieved content
    query_terms = set(re.findall(r'\b\w+\b', query.lower()))
    doc_content = " ".join([doc.page_content.lower() for doc in docs])
    doc_terms = set(re.findall(r'\b\w+\b', doc_content))
    
    term_overlap = len(query_terms.intersection(doc_terms)) / len(query_terms) if query_terms else 0
    
    # Consider content sufficient if:
    # 1. We have substantial content AND good term overlap
    # 2. OR we have very relevant but shorter content
    is_sufficient = (
        (total_content_length > 500 and term_overlap > 0.3) or
        (total_content_length > 200 and term_overlap > 0.5)
    )
    
    if is_sufficient:
        return True, "API documentation appears sufficient."
    else:
        return False, f"API documentation limited (content: {total_content_length} chars, term overlap: {term_overlap:.2f})"

def create_chat_prompt():
    """Create conversational prompt template (backwards compatibility)"""
    return create_hybrid_chat_prompt()

def create_hybrid_chat_prompt():
    """Create prompt template that can handle both API docs and web content"""
    return PromptTemplate(
        input_variables=["api_context", "web_context", "conversation_history", "query", "search_used"],
        template="""You are a specialized MCI (Materials, Corrosion, and Integrity) engineering assistant with expertise in API 571, API 970, and API 584 standards.

IMPORTANT UNIT REQUIREMENTS:
- ALWAYS use metric SI units
- Convert imperial units if needed: °C = (°F - 32) × 5/9, bar = psi × 0.0689476

API Documentation:
{api_context}

{search_used}Web Search Results:
{web_context}

Conversation History:
{conversation_history}

Current Question: {query}

RESPONSE GUIDELINES:
1. Provide DIRECT, CONCISE answers without preambles
2. Do NOT mention "based on API documentation" or "web search results" 
3. Start immediately with the technical information
4. Use metric SI units consistently
5. Reference sources naturally: "API 571 states..." or "Per API 970..."
6. Maintain conversational flow and acknowledge previous context
7. If you have information from any source, present it confidently
8. Be technically accurate and specific

Give a direct technical response:"""
    )

def create_analysis_prompt():
    """Create structured analysis prompt template"""
    return PromptTemplate(
        input_variables=["context", "equipment_context", "query"],
        template="""You are an expert MCI (Materials, Corrosion, and Integrity) engineering consultant providing structured analysis.

Equipment Context:
{equipment_context}

IMPORTANT UNIT REQUIREMENTS:
- ALWAYS use metric SI units in your responses
- Temperature: Use degrees Celsius (°C) ONLY
- Pressure: Use bar ONLY (not psi, kPa, or MPa)
- If source documents mention Fahrenheit (°F) or psi, convert to °C and bar
- Common conversions: °C = (°F - 32) × 5/9, bar = psi × 0.0689476

API Documentation Available:
{context}

Analysis Request: {query}

Provide a comprehensive structured analysis covering:

## 1. DAMAGE MECHANISMS
- Specific conditions that cause damage
- Environmental factors and thresholds
- Material susceptibility factors
- Critical parameters (temperature in °C, pressure in bar)

## 2. MITIGATION STRATEGIES
- Material selection recommendations
- Environmental control measures
- Protective systems and coatings
- Design modifications
- Process optimization strategies

## 3. OPERATING LIMITS
- Safe operating windows (temperature in °C, pressure in bar)
- Critical control points and alarm settings
- Monitoring requirements
- Inspection frequencies
- Deviation consequences

## 4. SPECIFIC RECOMMENDATIONS
- Context-specific guidance based on equipment and environment
- Risk assessment considerations
- Implementation priorities

Express all temperatures in °C and pressures in bar. Be thorough but concise in each section.

Analysis:"""
    )

def generate_response(query: str, vectorstores: dict, llm, chat_history: List[dict] = None, equipment_context: dict = None) -> str:
    """Generate response using RAG with web search fallback"""
    try:
        # Retrieve relevant documents from API sources
        docs = retrieve_documents(query, vectorstores)
        api_context = format_documents(docs)
        
        # Assess if API documentation is sufficient
        is_sufficient, assessment_msg = assess_content_sufficiency(docs, query)
        
        web_context = ""
        search_used = ""
        
        # Use web search as fallback if API docs are insufficient
        if not is_sufficient and is_mci_related(query):
            st.info(f"🔍 API documentation limited. Searching web for additional information... ({assessment_msg})")
            web_context = search_web_tavily(query)
            search_used = "✅ "
        
        if equipment_context:
            # Structured analysis mode (keep original logic)
            context_parts = []
            for key, value in equipment_context.items():
                if value and value != "Not Specified":
                    context_parts.append(f"{key.replace('_', ' ').title()}: {value}")
            
            context_string = " | ".join(context_parts) if context_parts else "General analysis"
            
            prompt_template = create_analysis_prompt()
            formatted_prompt = prompt_template.format(
                context=api_context,
                equipment_context=context_string,
                query=query
            )
        else:
            # Chat mode with hybrid sources
            conversation_context = get_conversation_context(chat_history or [])
            
            prompt_template = create_hybrid_chat_prompt()
            formatted_prompt = prompt_template.format(
                api_context=api_context,
                web_context=web_context,
                conversation_history=conversation_context,
                query=query,
                search_used=search_used
            )
        
        # Generate response
        response = llm.invoke(formatted_prompt)
        
        if hasattr(response, 'content'):
            result = response.content
        else:
            result = str(response)
        
        # Add search indicator to response if web search was used
        if web_context and search_used:
            result += "\n\n*📡 Response includes web search results to supplement API documentation*"
        
        return result
            
    except Exception as e:
        return f"I encountered an error while processing your request: {str(e)}. Please try again."

def chatbot_page():
    """Chatbot page implementation"""
    st.title("🛡️ SmartMCI ChatBot")
    st.markdown("**Conversational AI for Materials, Corrosion & Integrity**")
    
    # Initialize session state for chatbot
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    
    # Initialize cache
    cache = SimpleCache()
    
    # Setup components
    with st.spinner("Initializing SmartMCI..."):
        vectorstores, available = setup_vectorstores()
        llm = setup_llm()
        
        if not available or not llm:
            st.error("❌ System initialization failed. Please check configuration.")
            st.stop()
        
        # Check Tavily API key
        tavily_available = bool(os.environ.get("TAVILY_API_KEY"))
        if tavily_available:
            st.success("🌐 Web search enabled via Tavily")
        else:
            st.warning("⚠️ Web search disabled - TAVILY_API_KEY not found")
    
    # Sidebar with quick actions
    with st.sidebar:
        st.header("Quick Actions")
        
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_messages = []
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 💡 Example Questions")
        
        examples = [
            "What causes stress corrosion cracking?",
            "How to prevent pitting corrosion?",
            "Operating limits for sour service?",
            "Damage mechanisms in stainless steel",
            "Corrosion control methods for offshore"
        ]
        
        for example in examples:
            if st.button(example, key=f"chat_ex_{hash(example)}", use_container_width=True):
                # Add user message
                st.session_state.chat_messages.append({"role": "user", "content": example})
                
                # Process the example question through AI
                if is_mci_related(example):
                    # Check cache first
                    cached_response = cache.get(example)
                    if cached_response:
                        response = cached_response + "\n\n*[Cached response]*"
                    else:
                        # Generate new response
                        response = generate_response(example, vectorstores, llm, st.session_state.chat_messages)
                        cache.set(example, response)
                else:
                    response = """I'm specialized in Materials, Corrosion, and Integrity (MCI) engineering based on API 571, 970, and 584 standards. 

Your question doesn't appear to be related to:
- Damage mechanisms and failure analysis
- Corrosion control and prevention
- Integrity operating windows
- Materials selection and behavior

Could you please ask about topics related to materials, corrosion, or equipment integrity?

"""
                
                # Add assistant response
                st.session_state.chat_messages.append({"role": "assistant", "content": response})
                st.rerun()
        
        # Show cache stats if available
        if st.session_state.get("chat_cache"):
            st.markdown("---")
            st.caption(f"💾 Cached responses: {len(st.session_state.chat_cache)}")
    
    # Display chat history
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about materials, corrosion, or integrity..."):
        # Add user message
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        
        # Check if MCI-related
        if not is_mci_related(prompt):
            response = """I'm specialized in Materials, Corrosion, and Integrity (MCI) engineering based on API 571, 970, and 584 standards. 

Your question doesn't appear to be related to:
- Damage mechanisms and failure analysis
- Corrosion control and prevention
- Integrity operating windows
- Materials selection and behavior

Could you please ask about topics related to materials, corrosion, or equipment integrity? For example:
- "What causes pitting corrosion?"
- "How to prevent stress corrosion cracking?"
- "Operating limits for high temperature service?"

"""
        else:
            # Check cache first
            cached_response = cache.get(prompt)
            if cached_response:
                response = cached_response + "\n\n*[Cached response]*"
            else:
                # Generate new response
                with st.spinner("Analyzing API standards..."):
                    response = generate_response(prompt, vectorstores, llm, st.session_state.chat_messages)
                    cache.set(prompt, response)
        
        # Add assistant response
        st.session_state.chat_messages.append({"role": "assistant", "content": response})
        st.rerun()
    
    # Welcome message for new users
    if not st.session_state.chat_messages:
        st.markdown("""
        ## 👋 Welcome to SmartMCI ChatBot!
        
        I'm your conversational consultant for **Materials, Corrosion & Integrity** based on:
        - **API 571** - Damage Mechanisms & Failure Analysis
        - **API 970** - Corrosion Control & Prevention  
        - **API 584** - Integrity Operating Windows
        
        **Enhanced Capabilities:**
        - 📚 **Primary Source**: API standards documentation
        - 🌐 **Web Search**: Additional information when API docs are limited
        - 💬 **Conversational**: Natural chat with context awareness
        
        **Ask me about:**
        - Damage mechanisms and their causes
        - Corrosion prevention strategies
        - Material selection guidance
        - Operating limits and safe parameters
        - Equipment integrity challenges
        
        **Units:** All responses use metric SI units
        
        **Try asking:** *"What causes stress corrosion cracking in chloride environments?"*
        """)

def analysis_page():
    """Structured analysis page implementation"""
    st.title("🔬 SmartMCI Analysis")
    st.markdown("**Structured Analysis with Equipment Parameters**")
    
    # Initialize cache
    cache = SimpleCache()
    
    # Setup components
    with st.spinner("Initializing analysis tools..."):
        vectorstores, available = setup_vectorstores()
        llm = setup_llm()
        
        if not available or not llm:
            st.error("❌ System initialization failed. Please check configuration.")
            st.stop()
    
    # Input parameters in sidebar
    with st.sidebar:
        st.header("Equipment Parameters")
        
        # Equipment context inputs
        equipment_type = st.selectbox(
            "Equipment Type:", 
            ["Not Specified"] + EQUIPMENT_TYPES,
            help="Select equipment for specific guidance"
        )
        
        material = st.selectbox(
            "Material:", 
            ["Not Specified"] + MATERIALS,
            help="Select material for specific recommendations"
        )  
        
        environment = st.selectbox(
            "Service Environment:", 
            ["Not Specified"] + ENVIRONMENTS,
            help="Select environment for specific analysis"
        )
        
        damage_mechanism = st.selectbox(
            "Damage Type:", 
            ["Not Specified"] + DAMAGE_MECHANISMS,
            help="Select damage type for specific information"
        )
        
        # Operating conditions
        st.subheader("Operating Conditions")
        
        col1, col2 = st.columns(2)
        with col1:
            temperature = st.number_input(
                "Temperature (°C)", 
                value=None, 
                min_value=-50, 
                max_value=1000, 
                placeholder="Optional"
            )
        with col2:
            pressure = st.number_input(
                "Pressure (bar)", 
                value=None, 
                min_value=0, 
                max_value=500, 
                placeholder="Optional"
            )
        
        # Analysis type
        st.subheader("Analysis Focus")
        analysis_focus = st.selectbox(
            "Focus Area:",
            [
                "Comprehensive Analysis",
                "Damage Mechanisms Only",
                "Mitigation Strategies Only", 
                "Operating Limits Only"
            ]
        )
        
        st.markdown("---")
        
        # Quick analysis buttons
        st.subheader("Quick Analysis")
        
        if st.button("🔍 Run Analysis", type="primary", use_container_width=True):
            # Create equipment context
            equipment_context = {
                'equipment_type': equipment_type,
                'material': material,
                'environment': environment,
                'damage_mechanism': damage_mechanism,
                'temperature': f"{temperature}°C" if temperature is not None else "Not Specified",
                'pressure': f"{pressure} bar" if pressure is not None else "Not Specified"
            }
            
            # Create analysis query based on context
            context_parts = []
            if damage_mechanism != "Not Specified":
                context_parts.append(damage_mechanism)
            if equipment_type != "Not Specified":
                context_parts.append(f"in {equipment_type}")
            if material != "Not Specified":
                context_parts.append(f"({material})")
            if environment != "Not Specified":
                context_parts.append(f"for {environment}")
            
            if context_parts:
                query = f"{analysis_focus} for {' '.join(context_parts)}"
            else:
                query = f"{analysis_focus} - general MCI engineering guidance"
            
            # Store in session state for processing
            st.session_state.analysis_query = query
            st.session_state.analysis_context = equipment_context
            st.session_state.run_analysis = True
            st.rerun()
        
        # Clear results
        if st.button("🗑️ Clear Results", use_container_width=True):
            if "analysis_result" in st.session_state:
                del st.session_state.analysis_result
            if "analysis_query" in st.session_state:
                del st.session_state.analysis_query
            st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Process analysis if requested
        if st.session_state.get("run_analysis", False):
            query = st.session_state.get("analysis_query", "")
            context = st.session_state.get("analysis_context", {})
            
            # Check cache
            cached_response = cache.get(query, context)
            
            if cached_response:
                st.session_state.analysis_result = cached_response + "\n\n*[Cached response]*"
            else:
                with st.spinner("Generating comprehensive analysis..."):
                    response = generate_response(query, vectorstores, llm, equipment_context=context)
                    cache.set(query, response, context)
                    st.session_state.analysis_result = response
            
            # Clear the run flag
            st.session_state.run_analysis = False
        
        # Display results
        if "analysis_result" in st.session_state:
            st.markdown("## 📊 Analysis Results")
            st.markdown(st.session_state.analysis_result)
            
            # Export option
            if st.button("📄 Export Analysis Report"):
                context = st.session_state.get("analysis_context", {})
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
                
                report = f"""MCI Engineering Analysis Report
Generated: {timestamp}

EQUIPMENT PARAMETERS:
"""
                for key, value in context.items():
                    if value != "Not Specified":
                        report += f"- {key.replace('_', ' ').title()}: {value}\n"
                
                report += f"\nANALYSIS RESULTS:\n{'-' * 50}\n"
                report += st.session_state.analysis_result
                report += f"\n\n{'-' * 50}\n⚠️ DISCLAIMER: Results must be verified by qualified engineers."
                
                st.download_button(
                    "📄 Download Report",
                    report,
                    file_name=f"mci_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                    mime="text/plain"
                )
        else:
            # Welcome message
            st.markdown("""
            ## 🔬 Welcome to SmartMCI Analysis!
            
            This page provides **structured analysis** based on your specific equipment parameters.
            
            **How to use:**
            1. Set your equipment parameters in the sidebar
            2. Choose your analysis focus
            3. Click "Run Analysis" for comprehensive results
            
            **Analysis covers:**
            - **Damage Mechanisms** (API 571) - Conditions and causes
            - **Mitigation Strategies** (API 970) - Prevention methods
            - **Operating Limits** (API 584) - Safe parameters
            - **Specific Recommendations** - Context-based guidance
            
            **Units:** All results use metric SI units
            """)
    
    with col2:
        # Context summary
        st.markdown("### 📋 Current Context")
        
        context_items = []
        if equipment_type != "Not Specified":
            context_items.append(f"**Equipment:** {equipment_type}")
        if material != "Not Specified":
            context_items.append(f"**Material:** {material}")
        if environment != "Not Specified":
            context_items.append(f"**Environment:** {environment}")
        if damage_mechanism != "Not Specified":
            context_items.append(f"**Damage Type:** {damage_mechanism}")
        if temperature is not None:
            context_items.append(f"**Temperature:** {temperature}°C")
        if pressure is not None:
            context_items.append(f"**Pressure:** {pressure} bar")
        
        if context_items:
            for item in context_items:
                st.markdown(item)
        else:
            st.markdown("*No specific context set*")
        
        st.markdown("---")
        
        # Cache info
        if st.session_state.get("chat_cache"):
            st.markdown("### 💾 Cache Status")
            st.caption(f"Stored analyses: {len(st.session_state.chat_cache)}")

def main():
    """Main application with page navigation"""
    
    # Initialize session state for page selection
    if "current_page" not in st.session_state:
        st.session_state.current_page = "ChatBot"
    
    # Page navigation in sidebar with buttons
    st.sidebar.title("🛡️ SmartMCI App")
    
    st.sidebar.markdown("### 📱 Navigation")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button(
            "💬 ChatBot", 
            use_container_width=True,
            type="primary" if st.session_state.current_page == "ChatBot" else "secondary"
        ):
            st.session_state.current_page = "ChatBot"
            st.rerun()
    
    with col2:
        if st.button(
            "🔬 Analysis", 
            use_container_width=True,
            type="primary" if st.session_state.current_page == "Analysis" else "secondary"
        ):
            st.session_state.current_page = "Analysis"
            st.rerun()
    
    st.sidebar.markdown("---")
    
    # Route to appropriate page
    if st.session_state.current_page == "ChatBot":
        chatbot_page()
    elif st.session_state.current_page == "Analysis":
        analysis_page()
    
    # Common footer
    st.markdown("---")
    st.caption("⚠️ **Engineering verification required** | Based on API 571/970/584 standards")

if __name__ == "__main__":
    main()