from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

data = TextLoader("D:/bx24genai/documentloader/startingcommands.txt")

splitter = RecursiveCharacterTextSplitter(separators=["\n\n","\n","\t","."," "],chunk_size=10,chunk_overlap=2)
docs = data.load()
chunk = splitter.split_documents(docs)
print(len(chunk))
for i in chunk:
    print(i.page_content)
    print()