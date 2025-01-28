import os
import streamlit as st
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    DirectoryLoader,
    TextLoader,
    UnstructuredFileLoader
)
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
import logging
from typing import List, Dict, Optional
from tqdm import tqdm
import shutil
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ayurveda_training.log'),
        logging.StreamHandler()
    ]
)


GOOGLE_API_KEY = "AIzaSyAeW-u0NYxcdvC5kOSH94svVsMfqVM3UKg"
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")
genai.configure(api_key=GOOGLE_API_KEY)

# Constants
DATA_DIR = Path("ayurveda_data")
PROCESSED_DIR = DATA_DIR / "processed"
KNOWLEDGE_BASE_DIR = DATA_DIR / "knowledge_base"
METADATA_FILE = KNOWLEDGE_BASE_DIR / "metadata.json"

# Create necessary directories
for directory in [DATA_DIR, PROCESSED_DIR, KNOWLEDGE_BASE_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

class AyurvedaKnowledgeBase:
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        """Load or create metadata tracking file"""
        if METADATA_FILE.exists():
            return json.loads(METADATA_FILE.read_text())
        return {"processed_files": [], "last_updated": None}

    def _save_metadata(self):
        """Save current metadata state"""
        METADATA_FILE.write_text(json.dumps(self.metadata, indent=2))

    def process_documents(self, input_directory: str):
        """Process all supported documents in the input directory"""
        input_path = Path(input_directory)
        
        # Configure loaders for different file types
        loaders = {
            ".pdf": DirectoryLoader(input_directory, glob="**/*.pdf", loader_cls=PyPDFLoader),
            ".txt": DirectoryLoader(input_directory, glob="**/*.txt", loader_cls=TextLoader)
        }

        all_docs = []
        for file_type, loader in loaders.items():
            try:
                docs = loader.load()
                all_docs.extend(docs)
                logging.info(f"Processed {len(docs)} {file_type} files")
            except Exception as e:
                logging.error(f"Error processing {file_type} files: {e}")

        return all_docs

    def create_knowledge_base(self, documents: List) -> Optional[FAISS]:
        """Create or update the knowledge base from processed documents"""
        try:
            # Split documents into chunks
            texts = self.text_splitter.split_documents(documents)
            logging.info(f"Split documents into {len(texts)} chunks")

            # Create vector store
            vector_store = FAISS.from_documents(texts, self.embeddings)
            
            # Save the vector store
            vector_store.save_local(str(KNOWLEDGE_BASE_DIR / "vector_store"))
            logging.info("Knowledge base created and saved successfully")
            
            return vector_store
        except Exception as e:
            logging.error(f"Error creating knowledge base: {e}")
            return None

class AyurvedaBot:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-latest",
            temperature=0.3,
            top_p=0.85
        )
        self.prompt = PromptTemplate.from_template("""
            You are an expert Ayurvedic practitioner and scholar with deep knowledge of traditional texts and practices.
            Use the following context from authentic Ayurvedic texts to answer the question.
            If the answer isn't directly in the context, use the principles from the texts to provide guidance.
            
            Context: {context}
            Question: {question}
            
            Please structure your response as follows:
            1. Main Answer: Direct response to the question
            2. Textual Reference: Relevant references from Ayurvedic texts (if available)
            3. Practical Application: How this knowledge can be applied
            4. Additional Context: Related concepts or principles
            
            Remember to:
            - Use proper Sanskrit terms with their meanings
            - Cite specific texts when possible
            - Explain concepts in both traditional and modern contexts
            - Consider the holistic nature of Ayurveda
            
            Answer:
        """)

    def setup_chain(self, retriever):
        return (
            {"context": retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

def main():
    st.set_page_config(page_title="Ayurveda Knowledge Base Training", layout="wide")
    
    st.title("🌿 Ayurveda Knowledge Base Training System")
    
    # Initialize components
    knowledge_base = AyurvedaKnowledgeBase()
    ayurveda_bot = AyurvedaBot()
    
    # Sidebar for training
    with st.sidebar:
        st.header("📚 Knowledge Base Management")
        
        # Upload multiple files
        uploaded_files = st.file_uploader(
            "Upload Ayurvedic Texts (PDF/TXT)", 
            accept_multiple_files=True,
            type=['pdf', 'txt']
        )
        
        if uploaded_files:
            if st.button("Process & Train"):
                with st.spinner("Processing documents and updating knowledge base..."):
                    # Save uploaded files
                    for file in uploaded_files:
                        file_path = DATA_DIR / file.name
                        with open(file_path, "wb") as f:
                            f.write(file.getvalue())
                    
                    # Process documents
                    docs = knowledge_base.process_documents(str(DATA_DIR))
                    if docs:
                        # Create/update knowledge base
                        vector_store = knowledge_base.create_knowledge_base(docs)
                        if vector_store:
                            st.success("Knowledge base updated successfully!")
                        else:
                            st.error("Error updating knowledge base")
                    
                    # Cleanup temporary files
                    for file in DATA_DIR.glob("*.*"):
                        if file.is_file() and file.suffix in [".pdf", ".txt"]:
                            file.unlink()

    # Main chat interface
    st.header("🔍 Test Your Knowledge Base")
    
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    for q, a in st.session_state.chat_history:
        st.text_area("Question:", value=q, height=100, disabled=True)
        st.markdown("**Answer:**")
        st.markdown(a)
        st.markdown("---")
    
    # Question input
    question = st.text_input("Ask about Ayurveda:")
    
    if st.button("Ask"):
        if question:
            try:
                # Load the vector store
                vector_store = FAISS.load_local(
                    str(KNOWLEDGE_BASE_DIR / "vector_store"),
                    knowledge_base.embeddings
                )
                retriever = vector_store.as_retriever(search_kwargs={'k': 5})
                
                # Get response
                chain = ayurveda_bot.setup_chain(retriever)
                with st.spinner("Consulting the texts..."):
                    response = chain.invoke(question)
                
                # Update chat history
                st.session_state.chat_history.append((question, response))
                st.experimental_rerun()
                
            except Exception as e:
                if "vector_store" not in os.listdir(KNOWLEDGE_BASE_DIR):
                    st.warning("Please upload and process some Ayurvedic texts first.")
                else:
                    st.error(f"Error: {str(e)}")
    
    if st.button("Clear History"):
        st.session_state.chat_history = []
        st.experimental_rerun()

if __name__ == "__main__":
    main()
