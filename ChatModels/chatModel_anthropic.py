from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import os

load_dotenv()
chat_model = ChatAnthropic(model="claude-3.5-sonnet-20241022", temperature=0.7, max_completion_tokens=15)
result =  chat_model.invoke("Write a poem about a lonely computer.")
print(result.content)