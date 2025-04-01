import os
import sys
import tempfile
import streamlit as st
from dotenv import load_dotenv
import pandas as pd
from PyPDF2 import PdfReader
from docx import Document
import nest_asyncio

# Force use of newer SQLite
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Import LangChain modules after SQLite fix
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

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
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

# Cache the embedding model to avoid reloading
@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create or get the vector store
def create_document_vectorstore(file, text_chunks):
    try:
        # Get or create the embedding model
        if st.session_state.embedding_model is None:
            with st.spinner("Loading embedding model..."):
                st.session_state.embedding_model = get_embedding_model()
        
        # Create FAISS vector store
        with st.spinner("Processing document..."):
            vectorstore = FAISS.from_texts(
                texts=text_chunks,
                embedding=st.session_state.embedding_model
            )
            return vectorstore
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None

# Initialize knowledge base with default information
def initialize_knowledge_base():
    chunks = [
        "Nandesh Babu is a skilled developer from Bengaluru, India.",
        "He specializes in Python, JavaScript, and various web frameworks.",
        "Nandesh has several years of experience in software development.",
        "He enjoys solving complex problems and building efficient applications.",
    ]
    
    try:
        if st.session_state.embedding_model is None:
            st.session_state.embedding_model = get_embedding_model()
            
        vectorstore = FAISS.from_texts(
            texts=chunks,
            embedding=st.session_state.embedding_model
        )
        return vectorstore
    except Exception as e:
        st.error(f"Error initializing knowledge base: {e}")
        return None

# Process uploaded document
def get_document_text(uploaded_file):
    text = ""
    file_name = uploaded_file.name
    try:
        if file_name.endswith(".pdf"):
            pdf_reader = PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        elif file_name.endswith(".docx"):
            doc = Document(uploaded_file)
            for para in doc.paragraphs:
                text += para.text + "\n"
        elif file_name.endswith(".txt"):
            text = uploaded_file.getvalue().decode("utf-8")
        else:
            st.error("Unsupported file format. Please upload a PDF, DOCX, or TXT file.")
            return None
    except Exception as e:
        st.error(f"Error processing document: {e}")
        return None
    
    return text if text.strip() else None

# Split text into chunks
def split_text(text):
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        return text_splitter.split_text(text)
    except Exception as e:
        st.error(f"Error splitting text: {e}")
        return []

# Generate response using LLM
def generate_response(user_question):
    try:
        if st.session_state.document_mode and st.session_state.vector_store:
            # Use document-based knowledge
            vectorstore = st.session_state.vector_store
        else:
            # Use default knowledge base
            if not hasattr(st.session_state, 'knowledge_base') or st.session_state.knowledge_base is None:
                st.session_state.knowledge_base = initialize_knowledge_base()
            vectorstore = st.session_state.knowledge_base
        
        if vectorstore is None:
            return "I'm having trouble accessing the knowledge base. Please try again later."
        
        # Initialize memory and groq LLM
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        llm = ChatGroq(
            temperature=0.7,
            groq_api_key=GROQ_API_KEY,
            model_name="mixtral-8x7b-32768"
        )
        
        # Create conversation chain
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory
        )
        
        # Get response from conversation chain
        response = conversation_chain.invoke({"question": user_question})
        return response["answer"]
    
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return f"I encountered an error while generating a response. Error details: {str(e)}"

# Main application
def main():
    # Apply nest_asyncio to resolve async issues
    nest_asyncio.apply()
    
    # Set page configuration
    st.set_page_config(
        page_title="AI Assistant",
        page_icon="🤖",
        layout="wide"
    )
    
    # App header
    st.header("💬 AI Chat Assistant")
    
    # Initialize knowledge base with default info if not already done
    if "knowledge_base_initialized" not in st.session_state:
        st.session_state.knowledge_base = initialize_knowledge_base()
        st.session_state.knowledge_base_initialized = True
    
    # Sidebar for document upload
    with st.sidebar:
        st.title("Settings")
        
        # Document upload section
        st.subheader("Upload Document")
        uploaded_file = st.file_uploader("Choose a file (PDF, DOCX, or TXT)", 
                                        type=["pdf", "docx", "txt"])
        
        if uploaded_file and uploaded_file != st.session_state.uploaded_file:
            st.session_state.uploaded_file = uploaded_file
            with st.spinner("Processing document..."):
                document_text = get_document_text(uploaded_file)
                if document_text:
                    text_chunks = split_text(document_text)
                    if text_chunks:
                        st.session_state.vector_store = create_document_vectorstore(
                            uploaded_file,
                            text_chunks
                        )
                        st.session_state.document_mode = True
                        st.success("Document processed successfully!")
                    else:
                        st.error("Failed to extract text chunks from the document.")
                else:
                    st.error("Failed to extract text from the document.")
        
        # Toggle for using document knowledge
        if st.session_state.vector_store is not None:
            document_mode = st.toggle("Use Document Knowledge", value=st.session_state.document_mode)
            if document_mode != st.session_state.document_mode:
                st.session_state.document_mode = document_mode
                if document_mode:
                    st.success("Now using knowledge from the uploaded document")
                else:
                    st.success("Now using default knowledge base")
        
        # Clear conversation button
        if st.button("Clear Conversation"):
            st.session_state.messages = []
            st.session_state.conversation = None
            st.success("Conversation cleared!")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if user_input := st.chat_input("Ask a question:"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_response(user_input)
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main() 
