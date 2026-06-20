# retrieval/rerank.py
# ---------------------------------------------------------------------------
# PURPOSE: Stage 2 of retrieval — re-rank the candidates from hybrid search.
#
# Hybrid search (BM25 + vector) gives a good SHORTLIST but imperfect ORDER.
# A cross-encoder re-ranker reads the (question, chunk) pair TOGETHER and scores
# how well each chunk answers the question — far more accurate than the
# bi-encoder similarity used for retrieval, but too slow to run over the whole
# corpus. So we only run it over the ~20 candidates hybrid already narrowed to.
#
# FLOW:  query + ~20 candidate chunks ─→ FlashRank cross-encoder ─→ reordered top-k
# ---------------------------------------------------------------------------

from flashrank import Ranker, RerankRequest


# Create the re-ranker ONCE at module load. FlashRank downloads a small model
# the first time (needs internet once), then caches it. Loading it once and
# reusing it avoids re-loading the model on every query.
_ranker = Ranker()


def rerank(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """Re-order candidate chunks by true relevance to the query.

    Args:
        query:  the user's question.
        chunks: candidate chunks from hybrid retrieval (each has text/title/
                section/chunk_id).
        top_k:  how many to return after re-ranking.

    Returns:
        The top_k chunks, reordered best-first, each with a new "rerank_score".
    """
    # FlashRank wants "passages": dicts with an id, the text to score, and
    # optional meta we can carry through. We use the list index as the id so we
    # can map FlashRank's results back to our original chunk dicts afterwards.
    passages = [
        {"id": i, "text": chunk["text"], "meta": chunk}
        for i, chunk in enumerate(chunks)
    ]

    # Ask FlashRank to score every passage against the query. The cross-encoder
    # reads each (query, passage) pair jointly and returns a relevance score.
    request = RerankRequest(query=query, passages=passages)
    ranked = _ranker.rerank(request)  # returns passages sorted best-first, each with "score"

    # Rebuild our chunk dicts in the new order, attaching the rerank score.
    # We pull the original chunk back out of the "meta" we passed in.
    results = []
    for item in ranked[:top_k]:
        chunk = dict(item["meta"])          # copy the original chunk
        chunk["rerank_score"] = item["score"]
        results.append(chunk)
    return results

if __name__ == "__main__":
    import sys
    # Compare hybrid order vs. re-ranked order on a query that hybrid got wrong.
    # Run with: poetry run python -m src.travel_copilot.retrieval.rerank
    from .hybrid import HybridRetriever

    # Question from command line, or a default if none given.
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "how do I get around in Aarhus?"

    retriever = HybridRetriever()

    # Pull a WIDE candidate set from hybrid (re-rank needs candidates to reorder;
    # it can only reorder what it's given, so we give it ~20).
    candidates = retriever.retrieve(query, top_k=20, candidate_k=20)

    print(f"Query: {query}\n")
    print("HYBRID order (before re-ranking):")
    for i, c in enumerate(candidates[:5], 1):
        print(f"  {i}. {c['title']} > {c['section']}")

    reranked = rerank(query, candidates, top_k=5)

    print("\nRE-RANKED order (after FlashRank):")
    for i, c in enumerate(reranked, 1):
        print(f"  {i}. {c['title']} > {c['section']} (score={round(c['rerank_score'], 4)})")
        print(f"     {c['text'][:150]}...")