import os
import json
import logging
from pathlib import Path
from multiprocessing import Pool
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Google API key setup
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyAeW-u0NYxcdvC5kOSH94svVsMfqVM3UKg")

# Directory and file constants
DATA_DIR = Path("ayurveda_data")
PROCESSED_DIR = DATA_DIR / "processed"
KNOWLEDGE_BASE_DIR = DATA_DIR / "knowledge_base"
METADATA_FILE = KNOWLEDGE_BASE_DIR / "metadata.json"

# Ensure necessary directories exist
DATA_DIR.mkdir(exist_ok=True, parents=True)
PROCESSED_DIR.mkdir(exist_ok=True, parents=True)
KNOWLEDGE_BASE_DIR.mkdir(exist_ok=True, parents=True)

# Knowledge Base Manager
class AyurvedaKnowledgeBase:
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        self.metadata = self._load_metadata()

    def _load_metadata(self):
        if METADATA_FILE.exists():
            return json.loads(METADATA_FILE.read_text())
        return {"processed_files": []}

    def _save_metadata(self):
        METADATA_FILE.write_text(json.dumps(self.metadata, indent=2))

    def process_document(self, file_path: Path):
        try:
            if file_path.suffix == ".pdf":
                loader = PyPDFLoader(file_path)
            elif file_path.suffix == ".txt":
                loader = TextLoader(file_path)
            else:
                return []
            return loader.load()
        except Exception as e:
            logging.error(f"Error processing {file_path.name}: {e}")
            return []

    def process_documents(self, input_files):
        with Pool() as pool:
            results = pool.map(self.process_document, input_files)
        return [doc for result in results for doc in result]

    def update_knowledge_base(self, documents):
        try:
            texts = self.text_splitter.split_documents(documents)
            vector_store = FAISS.from_documents(texts, self.embeddings)
            vector_store.save_local(str(KNOWLEDGE_BASE_DIR / "vector_store"))
            logging.info("Knowledge base updated successfully.")
        except Exception as e:
            logging.error(f"Error updating knowledge base: {e}")

# Chatbot Manager
class AyurvedaBot:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-latest",
            temperature=0.3,
            top_p=0.85
        )
        self.prompt = PromptTemplate.from_template(
            """
            You are an expert Ayurvedic practitioner.
            Use the context to answer the question with depth and accuracy.

            Context: {context}
            Question: {question}

            Answer:
            """
        )

    def setup_chain(self, retriever):
        return (
            {"context": retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

# Streamlit App
@st.cache_resource
def load_knowledge_base(embeddings):
    try:
        return FAISS.load_local(str(KNOWLEDGE_BASE_DIR / "vector_store"), embeddings)
    except Exception as e:
        logging.error(f"Error loading vector store: {e}")
        return None

def main():
    st.set_page_config(page_title="Ayurveda Knowledge Base", layout="wide")
    st.title("\U0001F33F Ayurveda Knowledge Base")

    knowledge_base = AyurvedaKnowledgeBase()
    ayurveda_bot = AyurvedaBot()

    with st.sidebar:
        st.header("\U0001F4DA Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload Ayurvedic texts (PDF/TXT)", 
            accept_multiple_files=True,
            type=["pdf", "txt"]
        )
        if uploaded_files and st.button("Process & Train"):
            with st.spinner("Processing files..."):
                input_files = []
                for file in uploaded_files:
                    file_path = DATA_DIR / file.name
                    with open(file_path, "wb") as f:
                        f.write(file.getvalue())
                    input_files.append(file_path)

                docs = knowledge_base.process_documents(input_files)
                if docs:
                    knowledge_base.update_knowledge_base(docs)
                    st.success("Knowledge base updated successfully!")

    st.header("\U0001F50D Query Ayurveda Knowledge Base")
    question = st.text_input("Ask a question about Ayurveda:")
    if st.button("Ask"):
        vector_store = load_knowledge_base(knowledge_base.embeddings)
        if vector_store:
            retriever = vector_store.as_retriever(search_kwargs={"k": 5})
            chain = ayurveda_bot.setup_chain(retriever)
            with st.spinner("Consulting the knowledge base..."):
                try:
                    response = chain.invoke(question)
                    st.markdown(f"**Answer:** {response}")
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.error("Knowledge base not found. Please upload and process documents.")

if __name__ == "__main__":
    main()
