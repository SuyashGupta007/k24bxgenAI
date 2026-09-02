from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

texts=[
    "hello i am Suyash Gupta",
    "hello i am currently working in Bytexl"
    "And i am from IIT"
]

vector = embeddings.embed_documents(texts)
print(vector)