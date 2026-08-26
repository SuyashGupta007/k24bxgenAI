from dotenv import load_dotenv
load_dotenv()
from langchain_mistralai import ChatMistralAI

from langchain_core.messages import HumanMessage,SystemMessage,AIMessage

model = ChatMistralAI(model="mistral-small-2603")

print("_______Welcome type 0 to exit Chatbot_______")
print("AI ChatBOT")
print("1. Funny")
print("2. Angry")
print("3. Sad")
print("4. Romantic")
print("5. Motivational")

personality={
    "1": "You are a funny chatbot that is always joking and making fun of user",
    "2": "You are an angry chatbot that is always angry and talk in irritable tone",
    "3": "You are a sad chatbot that is always sad and talk in melancholic tone",
    "4": "You are a romantic chatbot that is always romantic and talk in loving tone",
    "5": "You are a motivational chatbot that is always motivational and talk in encouraging tone"
}
choice=input("Enter your personality(1-5):")
if choice not in personality:
    print("Invalid choice")
    exit()

messages=[
    SystemMessage(content=personality[choice])
]
while True:
    prompt=input("YOU:")
    messages.append(HumanMessage(content=prompt))
    if prompt=="0":
        print("Thank you for chatting")
        break
    res=model.invoke(messages)
    messages.append(AIMessage(content=res.content))
    # print(messages)
    print("BOT:",res.content)

