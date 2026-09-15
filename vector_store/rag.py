from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# 1. Create embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# 2. Create Chat LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)


# 3. Connect with vector store
vectorstore = Chroma(
    persist_directory="D:/bx24genai/vector_store/chroma_db",
    embedding_function=embedding_model
)


# 4. Create retriever
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.7
    }
)


# 5. Create prompt template
prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant.

Answer the question using only the given context.

If the answer is not available in the context, say:
"I don't know."

Context:
{context}

Question:
{question}
""")


print("RAG System is ready")
print("Type 0 to exit")


# 6. Chat loop
while True:

    query = input("\nYOU: ")

    if query == "0":
        break

    # 7. Retrieve relevant documents
    docs = retriever.invoke(query)

    # 8. Combine retrieved documents
    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    # Debugging: see retrieved context
    print("\nRetrieved Context:\n", context)

    # 9. Create final prompt
    final_prompt = prompt.invoke({
        "context": context,
        "question": query
    })

    # 10. Generate answer
    response = llm.invoke(final_prompt)

    # 11. Print answer
    print("AI:", response.content)