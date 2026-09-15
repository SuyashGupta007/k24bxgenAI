#load pdf
# split into chuncks
# create the embeddings
# store into chroma db

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
load_dotenv()


loader = PyPDFLoader("D:/bx24genai/documentloader/genAI pdf.pdf")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
chuncks = splitter.split_documents(docs)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma.from_documents(chuncks,embeddings,persist_directory="D:/bx24genai/vector_store/chroma_db")
print("vector store created")