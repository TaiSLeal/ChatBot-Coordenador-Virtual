from langchain.embeddings import OpenAIEmbeddings
from qdrant_client import QdrantClient, models
import os
from langchain.vectorstores import Qdrant
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, PyPDFDirectoryLoader

load_dotenv()
embeddings = OpenAIEmbeddings()

client = QdrantClient(
    url = os.environ['QDRANT_HOST'], 
    api_key = os.environ['QDRANT_API_KEY']
)

client.recreate_collection(
    collection_name = os.environ['QDRANT_COLLECTION_NAME2'],
    vectors_config = models.VectorParams(
        size = 1536, # Vector size is defined by used model
        distance = models.Distance.COSINE
    )
)

vectorstore = Qdrant(
    client = client,
    collection_name = os.getenv("QDRANT_COLLECTION_NAME2"),
    embeddings = embeddings
)

def get_chunks(text):
    text_splitter = CharacterTextSplitter(
        chunk_size=800,
        chunk_overlap = 100,
    )
    chunks = text_splitter.split_documents(text)
    return chunks

root_dir = 'C:\\Users\\taina\\OneDrive\\Documentos\\UFBA\\ENGQUÍMICA\\TCC\\versão5\\automacao'
loader = PyPDFDirectoryLoader(root_dir)
raw_text = loader.load()
texts = get_chunks(raw_text)

vectorstore.add_documents(texts)
