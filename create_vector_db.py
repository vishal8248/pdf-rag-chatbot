from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("pdf/ai_notes.pdf")
documents = loader.load()

from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200
    )
chunks = text_splitter.split_documents(documents)

from langchain_huggingface import HuggingFaceEmbeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

from langchain_community.vectorstores import FAISS
vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model
    )

vectorstore.save_local("faiss_index")