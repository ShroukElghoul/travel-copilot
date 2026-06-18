# scripts/peek.py
# Quick look at what's actually stored in ChromaDB after indexing.
# Run with: poetry run python -m scripts.peek

import chromadb
from src.travel_copilot import config

# Connect to the same on-disk database we wrote during indexing.
client = chromadb.PersistentClient(path=str(config.CHROMA_DB_PATH))
collection = client.get_or_create_collection(name="wikivoyage")

# How many records are stored in total?
print(f"Total chunks stored: {collection.count()}\n")
print("-" * 50)

# Pull the first 3 records to inspect. include=[...] says which parts we want
# back (the text and metadata). We don't ask for embeddings here because a
# 768-number vector is noise to read — but we'll peek at one separately below.
sample = collection.get(limit=3, include=["documents", "metadatas"])

for i in range(len(sample["ids"])):
    print(f"ID:       {sample['ids'][i]}")
    print(f"Title:    {sample['metadatas'][i]['title']}")
    print(f"Section:  {sample['metadatas'][i]['section']}")
    print(f"Text:     {sample['documents'][i][:200]}...")
    print("-" * 50)

# Now peek at ONE actual embedding vector, just to see what a chunk looks like
# as numbers. We only print its length and the first 8 values.
one = collection.get(limit=1, include=["embeddings"])
vec = one["embeddings"][0]
print(f"\nOne chunk's vector → length: {len(vec)}")
print(f"First 8 numbers: {[round(x, 4) for x in vec[:8]]}")