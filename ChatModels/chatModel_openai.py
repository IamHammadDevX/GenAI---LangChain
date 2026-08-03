from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()
chat_model = ChatOpenAI(model="gpt-4", temperature=0.7, max_completion_tokens=15)

result =  chat_model.invoke("Write a poem about a lonely computer.")
print(result.content)