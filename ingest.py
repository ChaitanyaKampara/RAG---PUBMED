# if we have a pdf
# if u want to build Q&A or RAG
# you need TO create EMBEDDING
# $$$ => HOW TO STORE ?


from langchain.vectorstores import Qdrant  # STORING
# EMBEDDING MODEL
from langchain.embeddings import HuggingFaceBgeEmbeddings
# PDF LOADER
from langchain.document_loaders import PyPDFLoader
# TOKENIZATION
from langchain.text_splitter import RecursiveCharacterTextSplitter

# loader = PyPDFLoader("data.pdf")  # HOLDING DATA
# documents = loader.load()  # LOADING


from langchain.document_loaders import CsvLoader  # Importing CsvLoader

# Assuming you have already imported other necessary modules and defined variables

csv_loader = CsvLoader("medicine_data.csv")  # Define CsvLoader with the path to your CSV file

documents = csv_loader.load()  # Load documents from the CSV file

# Rest of your code remains unchanged











text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # first 500 chars becomes 1 chunk or documents
    chunk_overlap=50
)

texts = text_splitter.split_documents(documents)

# LOAD EMBEDDING MODEL :

model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}

embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

print("Embedding model loaded.............")

url = "http://localhost:6333"
collection_name = "gpt_db"
# these will go into the collection folder
qdrant = Qdrant.from_documents(
    texts,
    embeddings,
    url=url,
    collection_name=collection_name,
    prefer_grpc=False
)

print("Qdrant vector database created............")
