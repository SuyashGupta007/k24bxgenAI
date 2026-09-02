from langchain_community.document_loaders import WebBaseLoader

url = "https://www.samsung.com/in/smartphones/galaxy-s26-ultra/" 
loader = WebBaseLoader(url)
docs = loader.load()
print(docs[0].page_content)