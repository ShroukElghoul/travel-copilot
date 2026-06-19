# retriever.py
# ---------------------------------------------------------------------------
# PURPOSE: The first step of the QUERY flow. Given a user's question, find the
# most relevant chunks from the vector store.
#
#   question -> embed -> search ChromaDB -> return nearest chunks
#
# This is the counterpart to ingestion: ingestion PUT vectors in; retrieval
# PULLS the closest ones out. Note we embed the question with the SAME embed()
# (same model) used for the chunks — required for the vectors to be comparable.
# ---------------------------------------------------------------------------

import chromadb

from .. import config
from .. import llm

COLLECTION_NAME = "wikivoyage"


def retrieve(query: str, top_k: int = 5) -> list[dict]:
    """Find the top_k chunks most relevant to `query`.

    Args:
        query: the user's question / search text.
        top_k: how many chunks to return.

    Returns:
        A list of dicts, each: {"text", "title", "section", "distance"}.
        `distance` = how far the chunk is from the query (smaller = closer =
        more relevant, since we configured cosine distance).
    """
    # Connect to the SAME on-disk database we built during indexing.
    client = chromadb.PersistentClient(path=str(config.CHROMA_DB_PATH))
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    # 1. Turn the question into a vector (same model as the chunks).
    query_vector = llm.embed(query)

    # 2. Ask ChromaDB for the nearest chunks to that vector.
    #    include=[...] picks what we want back: the chunk text (documents),
    #    the metadata (title/section), and the distances (how close each is).
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    # 3. ChromaDB returns "list of lists" (one inner list per query). We sent
    #    a single query, so we read index [0] of each. Then we zip the parallel
    #    lists together into clean per-chunk dicts.
    chunks = []
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for text, meta, dist in zip(documents, metadatas, distances):
        chunks.append({
            "text": text,
            "title": meta["title"],
            "section": meta["section"],
            "distance": dist,
        })

    return chunks


if __name__ == "__main__":
    # Run with: poetry run python -m src.travel_copilot.retrieval.retriever
    sample_query = "medieval city center to explore on foot"

    print(f"Query: {sample_query}\n")
    print("-" * 60)

    for i, chunk in enumerate(retrieve(sample_query, top_k=5), start=1):
        # Lower distance = more relevant. Round it for readability.
        print(f"Result {i} | {chunk['title']} > {chunk['section']} "
              f"| distance={round(chunk['distance'], 4)}")
        print(f"  {chunk['text'][:150]}...")
        print("-" * 60)