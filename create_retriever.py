import dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader

REVIEWS_CSV_PATH = "sodapdf-converted-1.pdf"
REVIEWS_CHROMA_PATH = "chroma_data"

dotenv.load_dotenv()

loader = PyPDFLoader(file_path=REVIEWS_CSV_PATH)
reviews = loader.load()

# split the doc into smaller chunks i.e. chunk_size=500
text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=3000,
        chunk_overlap=500,
        length_function=len)
chunks = text_splitter.split_documents(reviews)

reviews_vector_db = Chroma.from_documents(
    chunks, OpenAIEmbeddings(), persist_directory=REVIEWS_CHROMA_PATH
)