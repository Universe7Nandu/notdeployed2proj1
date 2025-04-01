import os
import sys
import tempfile
import streamlit as st
from dotenv import load_dotenv
import pandas as pd
from PyPDF2 import PdfReader
from docx import Document
import nest_asyncio
import time
from datetime import datetime
import traceback
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.card import card

# Force use of newer SQLite
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Apply nest_asyncio to resolve async issues
nest_asyncio.apply()

# Import LangChain modules after SQLite fix
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Page configuration and styling
st.set_page_config(
    page_title="RAG AI Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .chat-message.user {
        background-color: #e6f7ff;
        border-left: 5px solid #1890ff;
    }
    .chat-message.assistant {
        background-color: #f6ffed;
        border-left: 5px solid #52c41a;
    }
    .chat-avatar {
        width: 50px;
        height: 50px;
        border-radius: 50%;
        object-fit: cover;
        margin-right: 1rem;
    }
    .chat-content {
        flex-grow: 1;
        overflow-wrap: break-word;
    }
    .stButton button {
        background-color: #1890ff;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        border: none;
        transition: all 0.3s;
    }
    .stButton button:hover {
        background-color: #096dd9;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    .metrics-card {
        background-color: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .sidebar .stTextInput input {
        border-radius: 5px;
    }
    .stFileUploader {
        padding: 0.5rem;
        border-radius: 5px;
        border: 1px dashed #1890ff;
    }
    .status-chip {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 50px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .status-chip.active {
        background-color: #f6ffed;
        color: #52c41a;
        border: 1px solid #b7eb8f;
    }
    .status-chip.inactive {
        background-color: #fff7e6;
        color: #fa8c16;
        border: 1px solid #ffd591;
    }
    .error-box {
        background-color: #fff2f0;
        border: 1px solid #ffccc7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #fffbe6;
        border: 1px solid #ffe58f;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "document_mode" not in st.session_state:
    st.session_state.document_mode = False
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = None
if "knowledge_base" not in st.session_state:
    st.session_state.knowledge_base = None
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "metrics" not in st.session_state:
    st.session_state.metrics = {
        "total_queries": 0,
        "avg_response_time": 0,
        "total_response_time": 0,
        "chunks_retrieved": 0,
        "document_uploads": 0
    }
if "chunking_strategy" not in st.session_state:
    st.session_state.chunking_strategy = "recursive"
if "chunk_size" not in st.session_state:
    st.session_state.chunk_size = 1000
if "chunk_overlap" not in st.session_state:
    st.session_state.chunk_overlap = 200
if "api_key_entered" not in st.session_state:
    st.session_state.api_key_entered = False
if "embedding_model_loaded" not in st.session_state:
    st.session_state.embedding_model_loaded = False
if "error_log" not in st.session_state:
    st.session_state.error_log = []

# Cache the embedding model to avoid reloading
@st.cache_resource(show_spinner=False)
def get_embedding_model():
    try:
        model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        st.session_state.embedding_model_loaded = True
        return model
    except Exception as e:
        error_msg = f"Error loading embedding model: {str(e)}"
        st.session_state.error_log.append(error_msg)
        st.error(error_msg)
        return None

# Create or get the vector store
def create_document_vectorstore(file, text_chunks):
    try:
        # Get or create the embedding model
        if st.session_state.embedding_model is None:
            with st.spinner("Loading embedding model..."):
                st.session_state.embedding_model = get_embedding_model()
                
        if st.session_state.embedding_model is None:
            st.error("Failed to load embedding model. Please try again.")
            return None
        
        # Create FAISS vector store
        with st.spinner("Processing document..."):
            vectorstore = FAISS.from_texts(
                texts=text_chunks,
                embedding=st.session_state.embedding_model
            )
            # Update metrics
            st.session_state.metrics["document_uploads"] += 1
            st.session_state.metrics["chunks_retrieved"] = len(text_chunks)
            return vectorstore
    except Exception as e:
        error_msg = f"Error creating vector store: {str(e)}"
        st.session_state.error_log.append(error_msg)
        st.error(error_msg)
        st.error(traceback.format_exc())
        return None

# Initialize knowledge base with default information
def initialize_knowledge_base():
    chunks = [
        "Nandesh Babu is a skilled developer from Bengaluru, India.",
        "He specializes in Python, JavaScript, and various web frameworks.",
        "Nandesh has several years of experience in software development.",
        "He enjoys solving complex problems and building efficient applications.",
        "Nandesh has expertise in Generative AI and RAG implementations.",
        "He is knowledgeable about prompt engineering techniques.",
        "Nandesh works with LangChain framework for building AI applications.",
        "He has experience with vector database management for efficient retrieval.",
        "Nandesh has skills in selecting appropriate embedding models for different use cases.",
        "He implements various chunking strategies for optimal text processing.",
        "Nandesh evaluates chatbots using tools like Arize AI.",
        "He deploys applications using Streamlit for user-friendly interfaces.",
        "Nandesh has integrated Groq API for optimized model inference.",
        "He works with Llama 3 and other open-source LLMs."
    ]
    
    try:
        if st.session_state.embedding_model is None:
            st.session_state.embedding_model = get_embedding_model()
            
        if st.session_state.embedding_model is None:
            st.warning("Unable to load default knowledge base. Some features may be limited.")
            return None
            
        vectorstore = FAISS.from_texts(
            texts=chunks,
            embedding=st.session_state.embedding_model
        )
        return vectorstore
    except Exception as e:
        error_msg = f"Error initializing knowledge base: {str(e)}"
        st.session_state.error_log.append(error_msg)
        st.error(error_msg)
        st.error(traceback.format_exc())
        return None

# Process uploaded document
def get_document_text(uploaded_file):
    text = ""
    file_name = uploaded_file.name
    tmp_path = None
    
    try:
        if file_name.endswith(".pdf"):
            try:
                pdf_reader = PdfReader(uploaded_file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text() or ""
                    text += page_text + "\n\n"
            except Exception as e:
                error_msg = f"Error reading PDF file: {str(e)}"
                st.session_state.error_log.append(error_msg)
                st.error(error_msg)
                return None
                
        elif file_name.endswith(".docx"):
            try:
                # Create a temporary file path
                with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                
                # Read the document
                doc = Document(tmp_path)
                for para in doc.paragraphs:
                    text += para.text + "\n"
            except Exception as e:
                error_msg = f"Error reading DOCX file: {str(e)}"
                st.session_state.error_log.append(error_msg)
                st.error(error_msg)
                return None
            finally:
                # Clean up temporary file
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except Exception as e:
                        st.warning(f"Could not remove temporary file: {str(e)}")
                
        elif file_name.endswith(".txt"):
            try:
                text = uploaded_file.getvalue().decode("utf-8")
            except Exception as e:
                error_msg = f"Error reading TXT file: {str(e)}"
                st.session_state.error_log.append(error_msg)
                st.error(error_msg)
                return None
        else:
            st.error("Unsupported file format. Please upload a PDF, DOCX, or TXT file.")
            return None
    except Exception as e:
        error_msg = f"Error processing document: {str(e)}"
        st.session_state.error_log.append(error_msg)
        st.error(error_msg)
        st.error(traceback.format_exc())
        return None
    
    # Check if we got any text
    if not text.strip():
        st.warning("No text could be extracted from the document.")
        return None
        
    return text

# Split text into chunks
def split_text(text):
    try:
        if st.session_state.chunking_strategy == "recursive":
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=st.session_state.chunk_size,
                chunk_overlap=st.session_state.chunk_overlap,
                length_function=len
            )
        else:  # Default to recursive if something else is selected
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=st.session_state.chunk_size,
                chunk_overlap=st.session_state.chunk_overlap,
                length_function=len
            )
        chunks = text_splitter.split_text(text)
        
        if not chunks:
            st.warning("Text splitting resulted in 0 chunks. The document might be too short or empty.")
            
        return chunks
    except Exception as e:
        error_msg = f"Error splitting text: {str(e)}"
        st.session_state.error_log.append(error_msg)
        st.error(error_msg)
        st.error(traceback.format_exc())
        return []

# Generate response using LLM
def generate_response(user_question):
    start_time = time.time()
    
    try:
        if not GROQ_API_KEY and not st.session_state.api_key_entered:
            return "Please enter your Groq API key in the sidebar to continue."
            
        if st.session_state.document_mode and st.session_state.vector_store:
            # Use document-based knowledge
            vectorstore = st.session_state.vector_store
        else:
            # Use default knowledge base
            if st.session_state.knowledge_base is None:
                st.session_state.knowledge_base = initialize_knowledge_base()
            vectorstore = st.session_state.knowledge_base
        
        if vectorstore is None:
            return "I'm having trouble accessing the knowledge base. Please try again later."
        
        # Initialize memory and groq LLM
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Define prompt template that guides the model to cite sources
        prompt_template = """
        You are a helpful AI assistant designed to provide accurate information.
        
        If using document knowledge:
        - Base your answer only on the provided document context
        - If the answer is not in the document, say "The document doesn't provide information about this."
        - Use clear bullet points and formatting for complex answers
        - Cite relevant sections from the document
        
        If using general knowledge:
        - Provide information about Nandesh Babu and his skills
        - Stay within the scope of the information you have
        
        Question: {question}
        """
        
        # Use the API key from session state if available
        api_key = st.session_state.get("groq_api_key", GROQ_API_KEY)
        
        llm = ChatGroq(
            temperature=0.5,
            groq_api_key=api_key,
            model_name="mixtral-8x7b-32768"
        )
        
        # Create conversation chain
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
            memory=memory
        )
        
        # Get response from conversation chain
        response = conversation_chain.invoke({"question": user_question})
        answer = response["answer"]
        
        # Update metrics
        end_time = time.time()
        response_time = end_time - start_time
        st.session_state.metrics["total_queries"] += 1
        st.session_state.metrics["total_response_time"] += response_time
        st.session_state.metrics["avg_response_time"] = (
            st.session_state.metrics["total_response_time"] / 
            st.session_state.metrics["total_queries"]
        )
        
        return answer
    
    except Exception as e:
        error_msg = f"Error generating response: {str(e)}"
        st.session_state.error_log.append(error_msg)
        st.error(error_msg)
        st.error(traceback.format_exc())
        return f"I encountered an error while generating a response. Error details: {str(e)}"

# Custom message display
def display_message(message, is_user=False):
    avatar = "👤" if is_user else "🤖"
    role = "user" if is_user else "assistant"
    st.markdown(f"""
    <div class="chat-message {role}">
        <div class="chat-avatar">{avatar}</div>
        <div class="chat-content">{message}</div>
    </div>
    """, unsafe_allow_html=True)

# Main application
def main():
    # Sidebar for settings and document upload
    with st.sidebar:
        colored_header(
            label="RAG AI Assistant",
            description="Powered by Groq & LangChain",
            color_name="blue-70"
        )
        
        add_vertical_space(2)
        
        # API Key input
        st.subheader("🔑 API Key")
        groq_api_key = st.text_input(
            "Enter your Groq API Key",
            type="password",
            value=GROQ_API_KEY if GROQ_API_KEY else "",
            help="Required to use the Groq language model"
        )
        
        if groq_api_key:
            st.session_state.groq_api_key = groq_api_key
            st.session_state.api_key_entered = True
        
        add_vertical_space(1)
        
        # Document upload section
        st.subheader("📄 Document Upload")
        uploaded_file = st.file_uploader(
            "Upload a document (PDF, DOCX, or TXT)",
            type=["pdf", "docx", "txt"],
            help="The document will be processed and made available for questions"
        )
        
        if uploaded_file and (uploaded_file != st.session_state.uploaded_file or st.button("Process Document")):
            st.session_state.uploaded_file = uploaded_file
            with st.spinner("Processing document..."):
                document_text = get_document_text(uploaded_file)
                if document_text:
                    text_chunks = split_text(document_text)
                    if text_chunks:
                        vectorstore = create_document_vectorstore(
                            uploaded_file,
                            text_chunks
                        )
                        if vectorstore:
                            st.session_state.vector_store = vectorstore
                            st.session_state.document_mode = True
                            st.success(f"✅ Document processed: {len(text_chunks)} chunks created")
                        else:
                            st.error("Failed to create vector store from document")
                    else:
                        st.error("Failed to extract text chunks from the document.")
                else:
                    st.error("Failed to extract text from the document.")
        
        add_vertical_space(1)
        
        # Advanced settings collapsible section
        with st.expander("⚙️ Advanced Settings"):
            # Chunking strategy
            st.subheader("Chunking Strategy")
            chunking_strategy = st.selectbox(
                "Choose chunking method",
                options=["recursive"],
                index=0,
                help="The method used to split text into chunks"
            )
            
            # Update chunking strategy in session state
            if chunking_strategy != st.session_state.chunking_strategy:
                st.session_state.chunking_strategy = chunking_strategy
            
            # Chunk size and overlap
            chunk_size = st.slider(
                "Chunk Size",
                min_value=500,
                max_value=2000,
                value=st.session_state.chunk_size,
                step=100,
                help="The target size of each text chunk"
            )
            
            chunk_overlap = st.slider(
                "Chunk Overlap",
                min_value=0,
                max_value=500,
                value=st.session_state.chunk_overlap,
                step=50,
                help="The overlap between consecutive chunks"
            )
            
            # Update chunk settings in session state
            if chunk_size != st.session_state.chunk_size:
                st.session_state.chunk_size = chunk_size
            
            if chunk_overlap != st.session_state.chunk_overlap:
                st.session_state.chunk_overlap = chunk_overlap
            
            # Reprocess button
            if st.session_state.uploaded_file and st.button("Reprocess Document"):
                with st.spinner("Reprocessing document with new settings..."):
                    document_text = get_document_text(st.session_state.uploaded_file)
                    if document_text:
                        text_chunks = split_text(document_text)
                        if text_chunks:
                            vectorstore = create_document_vectorstore(
                                st.session_state.uploaded_file,
                                text_chunks
                            )
                            if vectorstore:
                                st.session_state.vector_store = vectorstore
                                st.success(f"✅ Document reprocessed: {len(text_chunks)} chunks created")
                            else:
                                st.error("Failed to create vector store during reprocessing")
                        else:
                            st.error("Failed to extract text chunks during reprocessing.")
                    else:
                        st.error("Failed to extract text during reprocessing.")
            
            # Error log viewer
            if st.session_state.error_log and st.button("View Error Log"):
                st.write("Error Log:")
                for i, error in enumerate(st.session_state.error_log):
                    st.markdown(f"**Error {i+1}:** {error}")
                    
            if st.session_state.error_log and st.button("Clear Error Log"):
                st.session_state.error_log = []
                st.success("Error log cleared")
        
        add_vertical_space(1)
        
        # Knowledge source toggle
        if st.session_state.vector_store is not None:
            st.subheader("🧠 Knowledge Source")
            document_mode = st.toggle(
                "Use Document Knowledge", 
                value=st.session_state.document_mode,
                help="Toggle between document knowledge and default information"
            )
            
            if document_mode != st.session_state.document_mode:
                st.session_state.document_mode = document_mode
                if document_mode:
                    st.success("Using knowledge from the uploaded document")
                else:
                    st.success("Using default knowledge base")
        
        add_vertical_space(1)
        
        # Clear conversation button
        if st.button("🗑️ Clear Conversation"):
            st.session_state.messages = []
            st.session_state.conversation = None
            st.success("Conversation cleared!")
    
    # Main content area
    colored_header(
        label="RAG-Powered AI Assistant",
        description="Chat with documents using Retrieval Augmented Generation",
        color_name="blue-70"
    )
    
    # Show current mode and embedding model status
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if st.session_state.document_mode and st.session_state.uploaded_file:
            st.markdown(f"""
            <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                <div class="status-chip active">Document Mode: ON</div>
                <span style="margin-left: 10px;">Using: {st.session_state.uploaded_file.name}</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                <div class="status-chip inactive">Document Mode: OFF</div>
                <span style="margin-left: 10px;">Using: Default Knowledge Base</span>
            </div>
            """, unsafe_allow_html=True)
    
    # Show embedding model status
    with col2:
        if st.session_state.embedding_model_loaded:
            st.markdown("""
            <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                <div class="status-chip active">Embedding Model: Loaded</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                <div class="status-chip inactive">Embedding Model: Not Loaded</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Initialize Embedding Model"):
                with st.spinner("Loading embedding model..."):
                    st.session_state.embedding_model = get_embedding_model()
                    if st.session_state.embedding_model:
                        st.success("Embedding model loaded successfully")
                    else:
                        st.error("Failed to load embedding model")
    
    # Metrics display
    with st.expander("📊 Performance Metrics", expanded=False):
        metrics_cols = st.columns(4)
        with metrics_cols[0]:
            st.metric("Total Queries", st.session_state.metrics["total_queries"])
        with metrics_cols[1]:
            st.metric("Avg Response Time", f"{st.session_state.metrics['avg_response_time']:.2f}s")
        with metrics_cols[2]:
            st.metric("Document Uploads", st.session_state.metrics["document_uploads"])
        with metrics_cols[3]:
            st.metric("Chunks Retrieved", st.session_state.metrics["chunks_retrieved"])
    
    # Display chat messages
    st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)
    
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            display_message(message["content"], message["role"] == "user")
    
    # Chat input
    st.markdown("<div style='height: 100px'></div>", unsafe_allow_html=True)
    
    user_input = st.chat_input("Type your question here...")
    
    if user_input:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        display_message(user_input, is_user=True)
        
        # Generate and display assistant response
        with st.spinner("Thinking..."):
            response = generate_response(user_input)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        display_message(response, is_user=False)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.error(traceback.format_exc()) 
