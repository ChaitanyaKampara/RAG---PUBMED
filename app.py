from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceBgeEmbeddings
from qdrant_client import QdrantClient

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

client = QdrantClient(
    url=url,

    prefer_grpc=False
)

print(client)
print('######################')  # if there is any wrong then print this

db = Qdrant(
    client=client,
    embeddings=embeddings,
    collection_name=collection_name
)

print(db)
print("#######################")

query = "What are the Common side effects of systemic therapeutic agents ?"

docs = db.similarity_search_with_score(query=query, k=5)

for i in docs:
    doc, score = i
    print({"score": score, "content": doc.page_content, "metadata": doc.metadata})
