from langchain_community.document_loaders import PyPDFLoader

data = PyPDFLoader("D:/bx24genai/documentloader/genAI pdf.pdf")
docs = data.load()
print(docs[0].page_content)
