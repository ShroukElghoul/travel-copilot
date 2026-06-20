# tools.py
# ---------------------------------------------------------------------------
# PURPOSE: Define the TOOLS the agent can choose to use. A tool = a function +
# a description the LLM reads to decide WHEN to call it.
#
# Right now: one tool — search_travel_guides — which wraps the FULL retrieval
# stack (hybrid search -> FlashRank re-rank) and returns readable text the agent
# can use to answer. (The sandboxed calculator tool will be added next, here.)
#
# Key idea: the LLM decides whether to call a tool based on its NAME + DOCSTRING.
# So the docstring is not just documentation — it's the agent's instruction
# manual. Write it to clearly say WHEN to use the tool.
# ---------------------------------------------------------------------------

from langchain_core.tools import tool

from .retrieval.hybrid import HybridRetriever
from .retrieval.rerank import rerank


# Build the retriever ONCE at import time. Creating a HybridRetriever loads the
# whole corpus from ChromaDB and builds the BM25 index — expensive — so we do it
# a single time and reuse it on every tool call, rather than rebuilding per call.
_retriever = HybridRetriever()


@tool
def search_travel_guides(query: str) -> str:
    """Search travel guides for information about destinations.

    Use this whenever you need facts about a place — what to see, where to eat,
    how to get around, history, neighborhoods, etc. Input should be a short
    search query describing what travel information you need.
    """
    # 1. Hybrid retrieval: pull a WIDE candidate set (BM25 + vector via RRF).
    candidates = _retriever.retrieve(query, top_k=20, candidate_k=20)

    # 2. Re-rank those candidates with the cross-encoder, keep the best few.
    top_chunks = rerank(query, candidates, top_k=5)

    # 3. Format the chunks into one readable string the LLM can reason over.
    #    We include the source (title/section) on each so the agent can cite it.
    if not top_chunks:
        return "No relevant travel information found."

    blocks = []
    for chunk in top_chunks:
        source = f"{chunk['title']} ({chunk['section']})"
        blocks.append(f"[{source}]\n{chunk['text']}")

    return "\n\n".join(blocks)


if __name__ == "__main__":
    # Quick test of the tool on its own.
    # Run with: poetry run python -m src.travel_copilot.tools
    #
    # Note: a @tool-decorated function is called via .invoke({...}) rather than
    # like a plain function, because LangChain wraps it as a Tool object.
    result = search_travel_guides.invoke({"query": "what to see in Aachen"})
    print(result)