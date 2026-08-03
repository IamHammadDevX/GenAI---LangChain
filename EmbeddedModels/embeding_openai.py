from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimension=32)

document = [
    "In the quiet hum of circuits, a lonely computer sits,",
    "Its screen aglow, yet no one to share its wits.",
    "In the digital dark, it waits for a friend to come."
]

result = embeddings.embed_documents(document)
# result = embeddings.embed_query("Write a poem about a lonely computer.")
print(str(result))