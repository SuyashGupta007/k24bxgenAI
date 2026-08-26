from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI
from  langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
import os

#WAY 1

# model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
# res = model.invoke("tell me about virat kholi")
# print(res.content)


# model = ChatGroq(model="openai/gpt-oss-120b")
# res = model.invoke("tell me a joke")
# print(res.content)


# model = ChatMistralAI(model="mistral-small-2603")
# res = model.invoke("give a poem on sunset",temperature=0.9,max_tokens=6)
# print(res.content)


#WAY 2

# model= init_chat_model("google_genai:gemini-3.7-flash")
# res = model.invoke("tell me the best college in india for Btech")
# print(res.content[0]["text"])

# #way3

# model= init_chat_model("gemini-3.7-flash",model_provider="google-genai")
# res = model.invoke("tell me the best college in india for Btech")
# print(res.content[0]["text"])


llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Flash-0731",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_KEY"),
    max_new_tokens=1000,
)

model = ChatHuggingFace(llm=llm)

response = model.invoke(
    "Give me 5 simple AI Engineer interview questions."
)

print(response.content)