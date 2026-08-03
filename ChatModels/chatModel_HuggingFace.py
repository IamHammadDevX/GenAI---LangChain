from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not token:
    raise ValueError(
        "HUGGINGFACEHUB_API_TOKEN is missing from the .env file."
    )

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation",
    provider="auto",
    max_new_tokens=10,
    temperature=1.8,
    huggingfacehub_api_token=token,
)

chat_model = ChatHuggingFace(llm=llm)

result = chat_model.invoke(
    "What is the name of capital of France?"
)

print(result.content)