from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()
chat_model = ChatGoogleGenerativeAI(model="gemini/gemini-1.5-pro")
result =  chat_model.invoke("Write a poem about a lonely computer.")
print(result.content)