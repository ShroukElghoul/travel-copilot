# tools.py
# ---------------------------------------------------------------------------
# PURPOSE: Define the TOOLS the agent can choose to use. A tool = a function +
# a description the LLM reads to decide WHEN to call it.
#
# Tools so far:
#   1. search_travel_guides — wraps the full retrieval stack (hybrid + rerank).
#   2. calculate           — safely evaluates a math expression (budgets, etc.).
#
# Key idea: the LLM decides whether to call a tool based on its NAME + DOCSTRING.
# So the docstring is the agent's instruction manual — write it to say clearly
# WHEN to use the tool.
# ---------------------------------------------------------------------------

from langchain_core.tools import tool
from simpleeval import simple_eval

from .retrieval.hybrid import HybridRetriever
from .retrieval.rerank import rerank


# Build the retriever ONCE at import time (loading the corpus + BM25 is
# expensive), then reuse it on every call.
_retriever = HybridRetriever()


@tool
def search_travel_guides(query: str) -> str:
    """Search travel guides for information about destinations.

    Use this whenever you need facts about a place — what to see, where to eat,
    how to get around, history, neighborhoods, etc. Input should be a short
    search query describing what travel information you need.
    """
    candidates = _retriever.retrieve(query, top_k=20, candidate_k=20)
    top_chunks = rerank(query, candidates, top_k=5)

    if not top_chunks:
        return "No relevant travel information found."

    blocks = []
    for chunk in top_chunks:
        source = f"{chunk['title']} ({chunk['section']})"
        blocks.append(f"[{source}]\n{chunk['text']}")
    return "\n\n".join(blocks)


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression and return the result.

    ALWAYS use this for any arithmetic — budgets, totals, distances, durations,
    currency sums — instead of doing the math yourself. Input must be a plain
    math expression, e.g. "5*80 + 200" or "(3+4)*12". Do not include words,
    currency symbols, or units — only numbers and the operators + - * / ( ).
    """
    try:
        # simple_eval safely evaluates the expression. Unlike Python's eval(),
        # it does NOT run arbitrary code — no imports, no function calls, no file
        # or system access — so it's safe to run on LLM-generated input. It only
        # permits arithmetic, which is exactly what we want here.
        result = simple_eval(expression)
        return str(result)
    except Exception as exc:
        # If the expression is malformed (or tries something not allowed),
        # return a clear message instead of crashing the agent.
        return f"Could not evaluate '{expression}': {exc}"


if __name__ == "__main__":
    # Quick test of both tools.
    # Run with: poetry run python -m src.travel_copilot.tools
    # (@tool functions are called via .invoke({...}), not like plain functions.)
    print("calculate test:")
    print("  5 nights x 80 + 200 flights =",
          calculate.invoke({"expression": "5*80 + 200"}))
    print("  bad input =",
          calculate.invoke({"expression": "import os"}))  # should be refused safely

    print("\nsearch test:")
    print(search_travel_guides.invoke({"query": "what to see in Aachen"})[:300], "...")
