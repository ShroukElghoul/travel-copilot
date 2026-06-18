# index.py
# ---------------------------------------------------------------------------
# PURPOSE: The final ingestion step. Take the chunk stream (parse -> chunk),
# turn each chunk's text into a vector (embed), and store it in ChromaDB so it
# can later be searched by MEANING.
#
# This is the end of the OFFLINE "ingestion flow":
#   dump -> parse -> chunk -> EMBED + STORE  (you are here)
# After this runs once, the data is searchable and we never re-run it unless
# the source data changes.
# ---------------------------------------------------------------------------

import chromadb

from .. import config
from .. import llm
from .parse import parse_articles
from .chunk import chunk_articles


# Name of the collection (think: a "table") inside ChromaDB that holds our vectors.
COLLECTION_NAME = "wikivoyage"

# How many chunks to accumulate before writing them to the DB in one go.
# Batching is much faster than adding one chunk at a time.
BATCH_SIZE = 100


def get_collection():
    """Create (or fetch, if it already exists) the ChromaDB collection.

    - PersistentClient writes to disk at CHROMA_DB_PATH, so the index survives
      between runs (you embed once, reuse forever).
    - metadata={"hnsw:space": "cosine"} tells Chroma to measure similarity by
      COSINE distance, which is the right choice for text embeddings.
    """
    client = chromadb.PersistentClient(path=str(config.CHROMA_DB_PATH))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def index_dump(dump_path: str, limit: int = None) -> int:
    """Embed all chunks from the dump and store them in ChromaDB.

    Returns the number of chunks stored.
    """
    config.ensure_dirs()           # make sure the data/db folders exist
    collection = get_collection()

    # Build the lazy chunk stream: parse articles -> chunk them.
    chunk_stream = chunk_articles(parse_articles(dump_path, limit=limit))

    # ChromaDB's add() wants parallel lists (ids, embeddings, documents, metadatas).
    # We fill these per-batch, flush, then clear them and continue.
    ids, embeddings, documents, metadatas = [], [], [], []
    total = 0

    def flush():
        """Write the current batch to ChromaDB, then empty the buffers."""
        if not ids:
            return
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        # Clear in place so the same lists keep being reused.
        ids.clear()
        embeddings.clear()
        documents.clear()
        metadatas.clear()

    for chunk in chunk_stream:
        # 1. Turn this chunk's text into a vector (the actual "AI" step).
        vector = llm.embed(chunk["text"])

        # 2. Stage it into the current batch buffers.
        ids.append(chunk["chunk_id"])
        embeddings.append(vector)
        documents.append(chunk["text"])
        metadatas.append({"title": chunk["title"], "section": chunk["section"]})

        total += 1

        # 3. When the batch is full, write it and reset.
        if len(ids) >= BATCH_SIZE:
            flush()
            print(f"  ...stored {total} chunks so far")

    # Write whatever is left in the final (partial) batch.
    flush()
    return total


if __name__ == "__main__":
    # Run with: poetry run python -m src.travel_copilot.ingestion.index
    dump_file = config.DATA_PATH / "enwikivoyage-latest-pages-articles.xml.bz2"

    print(f"Indexing from: {dump_file.name} (limit=20 articles)")
    print("This embeds every chunk via Ollama, so it takes a little while...\n")

    stored = index_dump(str(dump_file), limit=20)

    # Confirm by asking the collection how many records it now holds.
    collection = get_collection()
    print(f"\nDone. Chunks stored this run: {stored}")
    print(f"Total chunks in collection '{COLLECTION_NAME}': {collection.count()}")
