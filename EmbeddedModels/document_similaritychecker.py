import os

# ==========================================================
# Hugging Face Configuration
# ==========================================================
os.environ["HF_HOME"] = "D:/LangChain/huggingface_cache"
os.environ["HF_HUB_OFFLINE"] = "1"                  # Use only local files
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1" # Hide Windows symlink warning

from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ==========================================================
# Load Embedding Model
# ==========================================================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={
        "device": "cpu",
        "local_files_only": True
    },
    encode_kwargs={
        "normalize_embeddings": True
    }
)

# ==========================================================
# Documents
# ==========================================================
texts = [
    "Virat Kohli is an Indian cricket legend who shattered the record for the most centuries in One Day Internationals.",
    "He successfully captained India across formats and led them to major ICC white-ball championship victories.",

    "Sachin Tendulkar is an iconic Indian batsman widely revered as the God of Cricket for his flawless technique.",
    "He remains the leading run scorer in Test and ODI history and is the only player to score 100 international centuries.",

    "Sir Don Bradman is a legendary Australian batsman universally acknowledged as the greatest run scorer in cricket history.",
    "His Test batting average of 99.94 remains one of the greatest records in all of sports.",

    "Sir Garfield Sobers is regarded as the greatest all-rounder in cricket history.",
    "He famously hit six sixes in one over and held the world record for the highest individual Test score for 36 years.",

    "Shane Warne was one of the greatest leg-spin bowlers in cricket history.",
    "He became the first player to take 700 Test wickets and bowled the famous Ball of the Century."
]

# ==========================================================
# Query
# ==========================================================
query = "Tell me about Smith"

# ==========================================================
# Generate Embeddings
# ==========================================================
document_embeddings = embeddings.embed_documents(texts)
query_embedding = embeddings.embed_query(query)

# ==========================================================
# Calculate Cosine Similarity
# ==========================================================
similarities = cosine_similarity(
    [query_embedding],
    document_embeddings
)[0]

# ==========================================================
# Output
# ==========================================================
print("=" * 70)
print("Embedding Model :", "sentence-transformers/all-MiniLM-L6-v2")
print("Embedding Dimension :", len(query_embedding))
print("=" * 70)

# Sort from highest similarity to lowest
results = sorted(
    zip(texts, similarities),
    key=lambda x: x[1],
    reverse=True
)

print("\nSimilarity Ranking\n")

for i, (text, score) in enumerate(results, start=1):
    print(f"{i}. Score: {score:.4f}")
    print(text)
    print("-" * 70)


best_index = np.argmax(similarities)

print("\nBEST MATCH")
print("=" * 70)
print(texts[best_index])
print(f"\nSimilarity Score: {similarities[best_index]:.4f}")