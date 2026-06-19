# scripts/ask.py
# ---------------------------------------------------------------------------
# PURPOSE: The full RAG loop end-to-end. Take a user's travel question, retrieve
# relevant Wikivoyage chunks, hand them to the LLM as context, and print a
# grounded answer that cites its sources.
#
#   question -> retrieve chunks -> build grounded prompt -> generate -> answer
#
# This is the payoff: instead of asking the model from its own memory, we feed
# it OUR data and tell it to answer only from that. That's what makes RAG
# answers grounded, current, and citable.
#
# Run with:  poetry run python -m scripts.ask "what is there to see in Aachen?"
# (or with no argument to use the default demo question)
# ---------------------------------------------------------------------------

import sys

from src.travel_copilot import llm
from src.travel_copilot.retrieval.retriever import retrieve


# The system prompt sets the rules for HOW the model should answer. The key
# instruction is "use ONLY the provided context" — this is what keeps the model
# grounded and stops it inventing facts (a core RAG safety habit).
SYSTEM_PROMPT = (
    "You are a helpful travel assistant. Answer the user's question using ONLY "
    "the provided context from travel guides. If the answer is not in the "
    "context, say you don't have enough information rather than guessing. "
    "Be concise and practical."
)


def build_prompt(question: str, chunks: list[dict]) -> str:
    """Stitch the retrieved chunks + the question into one grounded prompt.

    We label each chunk with its source (title/section) so the model can see
    where each piece of context came from — and so it can reference them.
    """
    # Turn each chunk into a labeled block of context text.
    context_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        source = f"{chunk['title']} ({chunk['section']})"
        context_blocks.append(f"[Source {i}: {source}]\n{chunk['text']}")

    # Join all context blocks, then append the actual question at the end.
    context = "\n\n".join(context_blocks)
    return (
        f"Context from travel guides:\n\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer using only the context above."
    )


def ask(question: str, top_k: int = 5) -> None:
    """Run the full RAG loop and stream the grounded answer to the screen."""
    # 1. Retrieve the most relevant chunks for this question (existing code).
    chunks = retrieve(question, top_k=top_k)

    # 2. Build the grounded prompt from those chunks + the question.
    prompt = build_prompt(question, chunks)

    # 3. Generate the answer, streaming tokens as they arrive (existing code).
    print(f"\nQuestion: {question}\n")
    print("Answer:")
    for token in llm.generate(prompt, system_prompt=SYSTEM_PROMPT):
        print(token, end="", flush=True)

    # 4. Show the sources, so the answer is traceable/citable. We de-duplicate
    #    because several chunks can come from the same article.
    print("\n\nSources:")
    seen = set()
    for chunk in chunks:
        source = f"{chunk['title']} ({chunk['section']})"
        if source not in seen:
            print(f"  - {source}")
            seen.add(source)


if __name__ == "__main__":
    # Use a question from the command line if given, else a default demo one.
    if len(sys.argv) > 1:
        user_question = " ".join(sys.argv[1:])
    else:
        user_question = "What is there to see in Aachen?"

    ask(user_question)