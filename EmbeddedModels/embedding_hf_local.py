from langchain_huggingface import HuggingFaceEmbeddings
import os

os.environ["HF_HOME"] = "D:/LangChain/huggingface_cache"


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
texts = ["Hello world", "Goodbye world"]
embeddings_list = embeddings.embed_documents(texts)
print(str(embeddings_list))