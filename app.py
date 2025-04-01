#############################
# 1. FORCE NEWER SQLITE3 (Optional)
#############################
import pysqlite3 as sqlite3
import sys
sys.modules["sqlite3"] = sqlite3
# Optionally, if you know the bundled SQLite is new enough but misreported:
# sqlite3.sqlite_version = "3.35.5"

#############################
# 2. IMPORTS
#############################
import asyncio
import nest_asyncio
import streamlit as st
import pandas as pd
import os
from PyPDF2 import PdfReader
import docx
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
import faiss
import tempfile
import json
import uuid
from datetime import datetime
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma

# ChromaDB
import chromadb

# Initialize environment
load_dotenv()

#############################
# 3. CONFIGURATION
#############################
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found in environment variables. Please set it in the .env file.")
    st.stop()

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

#############################
# 4. DEFAULT INFORMATION
#############################
NANDESH_INFO = """
#### **Personal Details**
- **Name**: Nandesh Kalashetti
- **Role**: Full-Stack Web/Gen-AI Developer
- **Email**: nandeshkalshetti1@gmail.com
- **Phone**: 9420732657
- **Location**: Samarth Nagar, Akkalkot
- **Portfolio**: [Portfolio Link](https://nandesh-kalashettiportfilio2386.netlify.app/)
- **GitHub**: [GitHub Profile](https://github.com/Universe7Nandu)
- **LeetCode**: [LeetCode Profile](https://leetcode.com/u/Nandesh2386/)

---

### **Education**
- **Bachelor of Information Technology** - Walchand Institute of Technology, Solapur (CGPA: 8.8/10)
- **HSC (12th Grade)** - Walchand College of Arts and Science, Solapur (89%)
- **SSC (10th Grade)** - Mangrule High School, Akkalkot (81.67%)

---

### **Experience**
- **Full-Stack Developer** at Katare Informatics (May 2023 - Oct 2023)  
  - Web development, error handling (Apache_Foundation), advanced PHP
  - Worked on front-end, back-end, and database management

---

### **Skills**
#### **Programming Languages**
- Java, JavaScript, TypeScript, Python

#### **Web Development**
- HTML, CSS, React.js, Node.js, Express.js, MongoDB

#### **Frameworks & Libraries**
- React.js, Redux, TypeScript, Laravel

#### **Tools & Platforms**
- Git, Jenkins, Docker, Tomcat, Maven

#### **Cloud & DevOps**
- AWS Cloud Foundations, CI/CD pipelines

#### **Database**
- MySQL, MongoDB

#### **Chatbot & AI Development**
- **Frameworks & Libraries**: LangChain, OpenAI API, Hugging Face, TensorFlow, PyTorch
- **NLP**: Chatbot Development, Sentiment Analysis, Named Entity Recognition (NER), Tokenization

---

### **Projects**
1. **ActivityHub**  
   - Social learning platform (React.js, Advanced PHP, MySQL)
   - Secure authentication, interactive learning modules, notification systems

2. **Advanced Counter Application**  
   - Mathematical utility counter using React.js, JavaScript ES6+, modular CSS

3. **E-Cart**  
   - Online shopping website with search, UI customization, light/dark mode

4. **Generative AI Chatbot**  
   - Uses Retrieval-Augmented Generation (RAG), semantic embeddings, ChromaDB for knowledge base

5. **Online Course Catalog Web Application**  
   - Tools: Jenkins, Maven, Tomcat, GitHub
   - Features: Interlinked course pages, instructor details, automated deployment

---

### **Certifications**
- AWS Cloud Foundations - AWS Academy
- DevOps Workshop
- Infosys Courses

---

### **Achievements**
- 4/5 rating in AICTE Assessment Test (Institute Level)
- Improved organizational efficiency by 30%
- Completed 10+ projects showcasing innovation

---

### **Additional Training**
- Worked on DevOps pipelines using Jenkins & Docker
- Hands-on experience in cybersecurity via TryHackMe Challenges
"""

#############################
# 5. SYSTEM PROMPTS
#############################
DEFAULT_SYSTEM_PROMPT = f"""## Friendly AI Assistant
- For Nandesh info queries, provide detailed knowledge about Nandesh Kalashetti.
- For shorter queries, provide concise but complete responses.
- For longer queries, provide detailed, structured explanations.
- Be responsive in English, Hindi, or Marathi as preferred by the user.
- Always maintain a professional, helpful tone.

### Nandesh Info
{NANDESH_INFO}
"""

DOCUMENT_SYSTEM_PROMPT = """## Document-based Assistant
- Use ONLY the information from the uploaded document to respond.
- DO NOT reference Nandesh's information when in document mode.
- For shorter queries, provide concise but complete responses.
- For longer queries, provide detailed, structured explanations.
- Be responsive in English, Hindi, or Marathi as preferred by the user.
- Always maintain a professional, helpful tone.
"""

# Allow nested asyncio loops (needed for async functions in Streamlit)
nest_asyncio.apply()

#############################
# 6. HELPER FUNCTIONS
#############################
def get_chroma_db(collection_name="nandesh_knowledge", persist_directory="./data/chroma_db_5"):
    """Initialize or get ChromaDB vector store for Nandesh's knowledge"""
    try:
        # Create client
        client = chromadb.Client()
        
        # Get or create collection
        return Chroma(
            collection_name=collection_name,
            embedding_function=embedding_model,
            persist_directory=persist_directory
        )
    except Exception as e:
        print(f"Error initializing ChromaDB: {e}")
        return None

def create_document_db(collection_name="document_knowledge", persist_directory="./data/document_db"):
    """Create a separate database for document knowledge"""
    # Ensure directory exists
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)
        
    # Create and return the vector store
    try:
        # Create client
        client = chromadb.Client()
        
        # Get or create collection
        return Chroma(
            collection_name=collection_name,
            embedding_function=embedding_model,
            persist_directory=persist_directory
        )
    except Exception as e:
        print(f"Error initializing document ChromaDB: {e}")
        return None

def initialize_knowledge_base():
    """Initialize knowledge base with Nandesh info"""
    # Ensure directories exist
    if not os.path.exists("data"):
        os.makedirs("data")
        
    if not os.path.exists("data/chroma_db_5"):
        os.makedirs("data/chroma_db_5")
        
    if not os.path.exists("data/conversations"):
        os.makedirs("data/conversations")
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_text(NANDESH_INFO)
    
    # Create vector store
    db = get_chroma_db()
    
    # If database is empty, add Nandesh's information
    try:
        if db._collection.count() == 0:
            db.add_texts(chunks)
    except:
        db.add_texts(chunks)
    
    return db

def process_document(file):
    """Process uploaded document using PyPDF or other methods"""
    try:
        ext = os.path.splitext(file.name)[1].lower()
        
        if ext == ".pdf":
            # Read PDF file
            pdf = PdfReader(file)
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
            return text
        
        elif ext == ".txt":
            return file.getvalue().decode("utf-8")
        
        elif ext == ".docx":
            # Use python-docx to read DOCX files
            doc = docx.Document(file)
            return "\n".join([para.text for para in doc.paragraphs])
        
        elif ext == ".csv":
            df = pd.read_csv(file)
            return df.to_string()
        
        else:
            return "Unsupported file format"
    
    except Exception as e:
        return f"Error processing file: {str(e)}"

def add_document_to_db(text):
    """Add document text to vector database"""
    # Create directories if needed
    if not os.path.exists("data/document_db"):
        os.makedirs("data/document_db")
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_text(text)
    
    # Create or get document database
    db = create_document_db()
    
    # Clear existing data
    try:
        db._collection.delete(where={})
    except:
        pass
    
    # Add chunks to vector store
    db.add_texts(chunks)
    
    return db, len(chunks)

def get_llm():
    """Initialize and return Groq LLM"""
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY environment variable not set")
    
    # Initialize Groq LLM
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama2-70b-chat",  # Using llama2 instead of llama3 for compatibility
        temperature=0.7,
        max_tokens=2000
    )
    
    return llm

def get_retrieval_chain(document_mode=False):
    """Create a conversational retrieval chain with RAG"""
    # Determine which database to use
    if document_mode and "document_db" in st.session_state:
        db = st.session_state.document_db
        system_prompt = DOCUMENT_SYSTEM_PROMPT
    else:
        db = get_chroma_db()
        system_prompt = DEFAULT_SYSTEM_PROMPT
    
    # Create retriever
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    
    # Create prompt template
    template = f"""
    {system_prompt}
    
    CONTEXT:
    {{context}}
    
    CHAT HISTORY:
    {{chat_history}}
    
    USER QUESTION: {{question}}
    
    YOUR ANSWER (respond in a professional, friendly tone and in the language the user uses):
    """
    
    # Create QA prompt
    qa_prompt = PromptTemplate(
        template=template,
        input_variables=["context", "chat_history", "question"]
    )
    
    # Get LLM
    llm = get_llm()
    
    # Create chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": qa_prompt}
    )
    
    return chain

def generate_response(query, chat_history=[], document_mode=False):
    """Generate a response using the LLM and RAG"""
    try:
        # Get conversational chain
        chain = get_retrieval_chain(document_mode=document_mode)
        
        # Format chat history for the chain
        formatted_history = []
        for q, a in chat_history:
            formatted_history.append((q, a))
        
        # Generate response
        response = chain({"question": query, "chat_history": formatted_history})
        
        return {
            "answer": response["answer"],
            "source_documents": response.get("source_documents", [])
        }
    except Exception as e:
        return {
            "answer": f"I encountered an error: {str(e)}. Please try again or contact support.",
            "source_documents": []
        }

def save_conversation():
    """Save current conversation to file"""
    if not os.path.exists("data/conversations"):
        os.makedirs("data/conversations")
    
    # Save only if there are messages
    if st.session_state.messages:
        conversation = {
            "id": st.session_state.current_conversation_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "messages": st.session_state.messages,
            "chat_history": st.session_state.chat_history
        }
        
        # Save to file
        with open(f"data/conversations/{st.session_state.current_conversation_id}.json", "w") as f:
            json.dump(conversation, f)
        
        # Add to conversations list if not already there
        if st.session_state.current_conversation_id not in [conv["id"] for conv in st.session_state.conversations]:
            st.session_state.conversations.append({
                "id": st.session_state.current_conversation_id,
                "timestamp": conversation["timestamp"],
                "preview": st.session_state.messages[0]["content"] if st.session_state.messages else "New Conversation"
            })

def load_conversation(conversation_id):
    """Load a conversation from file"""
    try:
        with open(f"data/conversations/{conversation_id}.json", "r") as f:
            conversation = json.load(f)
            
        st.session_state.current_conversation_id = conversation_id
        st.session_state.messages = conversation["messages"]
        st.session_state.chat_history = conversation.get("chat_history", [])
    except Exception as e:
        st.error(f"Failed to load conversation: {e}")

def new_conversation():
    """Start a new conversation"""
    save_conversation()  # Save current conversation first
    st.session_state.current_conversation_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.chat_history = []

#############################
# 7. MAIN APPLICATION
#############################
def main():
    # Page configuration
    st.set_page_config(
        page_title="NandeshBot - GenAI Chatbot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # ---- CUSTOM CSS for beautiful UI ----
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600&display=swap');
    
    /* Main styles */
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    /* App container */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Header styles */
    .main-header {
        background: linear-gradient(90deg, #4776E6 0%, #8E54E9 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        background: linear-gradient(90deg, #FFFFFF 0%, #EFEFEF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .header-subtitle {
        font-size: 1.2rem;
        font-weight: 400;
        opacity: 0.9;
    }
    
    /* Chat container */
    .chat-container {
        background-color: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
        max-height: 600px;
        overflow-y: auto;
    }
    
    /* Message bubbles */
    .user-bubble {
        background: linear-gradient(90deg, #4776E6 0%, #8E54E9 100%);
        color: white;
        border-radius: 18px 18px 0 18px;
        padding: 12px 18px;
        margin: 15px 0;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    }
    
    .bot-bubble {
        background: white;
        border-radius: 18px 18px 18px 0;
        padding: 12px 18px;
        margin: 15px 0;
        max-width: 80%;
        border-left: 5px solid #4776E6;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    }
    
    /* Input area */
    .input-area {
        background-color: white;
        border-radius: 50px;
        padding: 10px 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        margin-top: 1rem;
        display: flex;
        align-items: center;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2b5876 0%, #4e4376 100%);
        color: white;
    }
    
    /* Upload button */
    .upload-btn {
        background: linear-gradient(90deg, #4776E6 0%, #8E54E9 100%);
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 50px;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .upload-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    /* New Chat button */
    .new-chat-btn {
        background: linear-gradient(90deg, #FF416C 0%, #FF4B2B 100%);
        color: white;
        border: none;
        padding: 12px;
        border-radius: 8px;
        font-weight: 600;
        width: 100%;
        cursor: pointer;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .new-chat-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    /* History buttons */
    .history-btn {
        background: rgba(255, 255, 255, 0.1);
        border: none;
        padding: 10px;
        border-radius: 8px;
        color: white;
        margin-bottom: 0.5rem;
        text-align: left;
        transition: all 0.3s ease;
    }
    
    .history-btn:hover {
        background: rgba(255, 255, 255, 0.2);
    }
    
    /* Expander */
    .css-1fcdlhc {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    
    /* Remove padding */
    div.block-container {padding-top: 1rem;}
    
    /* Media queries for responsive design */
    @media (max-width: 768px) {
        .header-title {
            font-size: 2rem;
        }
        
        .header-subtitle {
            font-size: 1rem;
        }
        
        .user-bubble, .bot-bubble {
            max-width: 90%;
        }
    }
    </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    if "conversations" not in st.session_state:
        st.session_state.conversations = []

    if "current_conversation_id" not in st.session_state:
        st.session_state.current_conversation_id = str(uuid.uuid4())

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    if "document_mode" not in st.session_state:
        st.session_state.document_mode = False

    if "knowledge_base_initialized" not in st.session_state:
        # Initialize knowledge base with Nandesh info
        initialize_knowledge_base()
        st.session_state.knowledge_base_initialized = True

    # Load existing conversations
    if os.path.exists("data/conversations"):
        for filename in os.listdir("data/conversations"):
            if filename.endswith(".json"):
                try:
                    with open(f"data/conversations/{filename}", "r") as f:
                        conversation = json.load(f)
                        
                    # Add to conversations list if not already there
                    if conversation["id"] not in [conv["id"] for conv in st.session_state.conversations]:
                        st.session_state.conversations.append({
                            "id": conversation["id"],
                            "timestamp": conversation["timestamp"],
                            "preview": conversation["messages"][0]["content"] if conversation["messages"] else "New Conversation"
                        })
                except Exception as e:
                    st.error(f"Failed to load conversation {filename}: {e}")

    # Sort conversations by timestamp (newest first)
    if st.session_state.conversations:
        st.session_state.conversations.sort(key=lambda x: x["timestamp"], reverse=True)

    # ---- SIDEBAR CONTENT ----
    with st.sidebar:
        st.markdown("<h2 style='text-align: center; color: white;'>NandeshBot</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #ccc;'>Powered by Groq & RAG</p>", unsafe_allow_html=True)
        
        # New chat button
        st.markdown(
            """
            <button class="new-chat-btn" id="new-chat-btn">
                🌟 New Chat
            </button>
            """, 
            unsafe_allow_html=True
        )
        if st.button("🌟 New Chat", key="new_chat_btn", use_container_width=True):
            new_conversation()
            # Reset document mode
            st.session_state.document_mode = False
            if "document_db" in st.session_state:
                del st.session_state.document_db
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # Upload document section
        st.markdown("### Upload Document")
        uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt", "docx", "csv"])
        
        if uploaded_file is not None:
            if st.button("Process Document", key="process_doc"):
                with st.spinner("Processing document..."):
                    # Extract text
                    text = process_document(uploaded_file)
                    
                    # Add to vector DB
                    db, num_chunks = add_document_to_db(text)
                    
                    # Store DB in session state
                    st.session_state.document_db = db
                    
                    # Enable document mode
                    st.session_state.document_mode = True
                    
                    # Clear current conversation for new document context
                    new_conversation()
                    
                    st.success(f"Document processed into {num_chunks} chunks! The chatbot will now answer based on this document.")
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # Chat history
        st.markdown("### Chat History")
        
        if not st.session_state.conversations:
            st.markdown("<p style='color: #ccc;'>No previous conversations</p>", unsafe_allow_html=True)
        else:
            for conv in st.session_state.conversations:
                # Show only the first 30 characters of the preview
                preview = conv["preview"][:30] + "..." if len(conv["preview"]) > 30 else conv["preview"]
                if st.button(f"🗨️ {preview}", key=f"conv_{conv['id']}", use_container_width=True):
                    load_conversation(conv["id"])

    # ---- MAIN CONTENT ----
    # Header
    st.markdown(
        """
        <div class="main-header">
            <h1 class="header-title">NandeshBot - GenAI Chatbot</h1>
            <p class="header-subtitle">Powered by Retrieval Augmented Generation & Groq</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Info about current mode
    if st.session_state.document_mode:
        st.info("📄 Document Mode: Responses will be based on the uploaded document only.")
    else:
        st.info("👤 Profile Mode: Responses will be based on Nandesh's information.")

    # Chat interface
    st.markdown("<div class='chat-container' id='chat-container'>", unsafe_allow_html=True)
    
    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"<div class='user-bubble'>{message['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='bot-bubble'>{message['content']}</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

    # Input area
    col1, col2 = st.columns([6, 1])
    
    with col1:
        user_input = st.text_input("Your message", key="user_input", label_visibility="collapsed", placeholder="Type your message here...")
    
    with col2:
        send_button = st.button("Send", key="send_button")

    # Handle input
    if (send_button or user_input) and user_input:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Generate response
        with st.spinner("Thinking..."):
            # Update chat history
            if st.session_state.chat_history:
                history = st.session_state.chat_history.copy()
            else:
                history = []
            
            # Generate response based on mode
            response = generate_response(
                user_input, 
                history, 
                document_mode=st.session_state.document_mode
            )
            
            # Add bot response to chat
            st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
            
            # Update chat history
            st.session_state.chat_history.append((user_input, response["answer"]))
        
        # Save conversation
        save_conversation()
        
        # Clear input and rerun to show new messages
        st.session_state.user_input = ""
        st.experimental_rerun()

    # Project Description Expander
    with st.expander("Project Description"):
        st.markdown("""
        ## Generative AI Chatbot with RAG
        
        This project demonstrates a functional Generative AI chatbot leveraging advanced techniques including:
        
        - **Retrieval Augmented Generation (RAG)** for accurate and contextual responses
        - **Effective prompt engineering** tailored to different types of queries
        - **ChromaDB vector database** for efficient semantic search
        - **Sentence transformers embeddings** for high-quality text representation
        - **Groq API integration** with Llama 2 for optimized inference
        
        ### Key Features
        - Toggle between Nandesh's information and document-based responses
        - Multilingual support (English, Hindi, Marathi)
        - Conversation history tracking
        - Beautiful, responsive UI
        - PDF document processing and knowledge extraction
        
        ### Skills Demonstrated
        Generative AI, RAG implementation, prompt engineering, LangChain framework, vector database management, embedding model selection, chunking strategies, chatbot evaluation, Streamlit deployment, Groq API integration, Llama 2 usage.
        """)

# Run the app
if __name__ == "__main__":
    main()
