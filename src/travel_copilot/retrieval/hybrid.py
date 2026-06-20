# retrieval/hybrid.py
# ---------------------------------------------------------------------------
# PURPOSE: Hybrid retrieval = DENSE (vector/cosine) + SPARSE (BM25 keyword),
# fused with Reciprocal Rank Fusion (RRF).
#
# WHY: Vector search is great at MEANING but weak at exact names ("Aachen").
# BM25 is great at exact WORDS but blind to paraphrase. Running both and fusing
# them covers each other's weaknesses — better recall than either alone.
#
# FLOW:  query ─┬─ vector search → ranked list A
#               └─ BM25 search   → ranked list B
#                        └─ RRF fuse(A, B) → final ranked chunks
# ---------------------------------------------------------------------------

import chromadb
from rank_bm25 import BM25Okapi

from .. import config
from .. import llm

COLLECTION_NAME = "wikivoyage"


def _tokenize(text: str) -> list[str]:
    """Very simple tokenizer for BM25: lowercase and split on whitespace.

    BM25 matches on tokens (words). This keeps it simple; a fancier tokenizer
    could strip punctuation, but lowercase+split is enough to catch "aachen".
    """
    return text.lower().split()


class HybridRetriever:
    """Holds the vector collection AND an in-memory BM25 index over the same
    chunks, so we can search both ways and fuse the results.

    We build this as a class because the BM25 index must be built ONCE from the
    whole corpus and reused for every query (rebuilding it per query would be
    wasteful). The class loads everything once in __init__.
    """

    def __init__(self):
        # 1. Connect to the same persistent vector store we built at index time.
        client = chromadb.PersistentClient(path=str(config.CHROMA_DB_PATH))
        self.collection = client.get_or_create_collection(name=COLLECTION_NAME)

        # 2. Pull ALL chunks out of Chroma (text + metadata). Chroma is our
        #    single source of truth for the documents; BM25 indexes the same
        #    text. get() with no limit returns everything.
        stored = self.collection.get(include=["documents", "metadatas"])
        self.ids = stored["ids"]
        self.documents = stored["documents"]
        self.metadatas = stored["metadatas"]

        # 3. Build a quick lookup: chunk_id -> its data. RRF works with ids, and
        #    at the end we use this to turn winning ids back into full chunks.
        self.by_id = {
            cid: {"text": doc, "title": meta["title"], "section": meta["section"]}
            for cid, doc, meta in zip(self.ids, self.documents, self.metadatas)
        }

        # 4. Build the BM25 index over the tokenized corpus (built once, here).
        tokenized_corpus = [_tokenize(doc) for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def _vector_search(self, query: str, k: int) -> list[str]:
        """DENSE search: embed the query, ask Chroma for nearest chunks.
        Returns a list of chunk_ids, best first."""
        query_vector = llm.embed(query)
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=k,
            include=[],  # we only need the ids, which Chroma always returns
        )
        return results["ids"][0]

    def _bm25_search(self, query: str, k: int) -> list[str]:
        """SPARSE search: score every chunk by BM25 keyword relevance, take
        the top k. Returns a list of chunk_ids, best first."""
        scores = self.bm25.get_scores(_tokenize(query))
        # Pair each chunk_id with its score, sort high→low, take top k.
        ranked = sorted(zip(self.ids, scores), key=lambda pair: pair[1], reverse=True)
        return [cid for cid, _score in ranked[:k]]

    def retrieve(self, query: str, top_k: int = 5, candidate_k: int = 20) -> list[dict]:
        """Run both searches over a wider candidate set, fuse with RRF, and
        return the top_k fused chunks.

        candidate_k: how many to pull from EACH retriever before fusing (a wide
                     net). top_k: how many to return after fusion.
        """
        vector_ids = self._vector_search(query, candidate_k)
        bm25_ids = self._bm25_search(query, candidate_k)

        # --- Reciprocal Rank Fusion (RRF) ---------------------------------
        # Each retriever gives a ranked list. For a chunk at rank r
        # in a list, it earns 1 / (rrf_k + r) points from that list. We sum a
        # chunk's points across both lists. Chunks ranked highly in BOTH lists
        # bubble to the top. rrf_k=60 is the standard constant; it softens the
        # difference between top ranks so a strong #1 doesn't dominate entirely.
        # RRF uses RANK position, not raw scores — which is why it can combine
        # cosine distances and BM25 scores even though they're on different scales.
        rrf_k = 60
        fused_scores: dict[str, float] = {}

        for ranked_list in (vector_ids, bm25_ids):
            for rank, cid in enumerate(ranked_list, start=1):
                fused_scores[cid] = fused_scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)

        # Sort all seen chunk_ids by fused score, high→low, take top_k.
        top_ids = sorted(fused_scores, key=lambda c: fused_scores[c], reverse=True)[:top_k]

        # Turn the winning ids back into full chunk dicts (with the fused score).
        results = []
        for cid in top_ids:
            chunk = dict(self.by_id[cid])          # copy text/title/section
            chunk["score"] = fused_scores[cid]
            chunk["chunk_id"] = cid
            results.append(chunk)
        return results


if __name__ == "__main__":
    import sys
    retriever = HybridRetriever()
    # Run with: poetry run python -m src.travel_copilot.retrieval.hybrid
    # Build the retriever once (loads corpus + builds BM25), then test queries.

    # Use a question from the command line, or a default if none given.
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "What is there to see in Aachen?" # proper noun — BM25 should rescue this
        #query = "walkable historic medieval city centre",   # conceptual — vector handles this

    print(f"\nQuery: {query}")
    print("-" * 60)
    for i, chunk in enumerate(retriever.retrieve(query, top_k=5), start=1):
        print(f"{i}. {chunk['title']} > {chunk['section']} (rrf={round(chunk['score'], 4)})")
        print(f"   {chunk['text'][:120]}...")