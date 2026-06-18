# llm.py
# ---------------------------------------------------------------------------
# THE single module that talks to an LLM provider. Everything else in the
# project calls generate() (for chat) or embed() (for embeddings) and never
# touches the provider directly. Swapping providers later = changes here only.
# ---------------------------------------------------------------------------

import ollama
from typing import Generator, Optional

from . import config


# ===========================================================================
# CHAT (text generation)
# ===========================================================================

def _call_ollama(messages: list, json_mode: bool) -> Generator[str, None, None]:
    """Provider-specific call for Ollama. Keeps Ollama's specifics isolated."""
    kwargs = {
        "model": config.CHAT_MODEL,
        "messages": messages,
        "stream": True,
    }
    if json_mode:
        kwargs["format"] = "json"

    try:
        stream = ollama.chat(**kwargs)
        for chunk in stream:
            yield chunk["message"]["content"]
    except ConnectionError as exc:
        raise RuntimeError(
            "Could not reach Ollama. Is it running? "
            "Start the Ollama app, or run `ollama serve` in a terminal."
        ) from exc


def _call_azure(messages: list, json_mode: bool) -> Generator[str, None, None]:
    """Placeholder for Azure OpenAI. Implemented later (different SDK/syntax)."""
    raise NotImplementedError(
        "Azure provider not implemented yet. Implement _call_azure() before "
        "setting PROVIDER='azure' in config."
    )


def generate(
    prompt: str,
    system_prompt: Optional[str] = None,
    json_mode: bool = False,
) -> Generator[str, None, None]:
    """Send a prompt to the active LLM provider and stream the response.

    This is the ONLY module that talks to an LLM provider directly. The rest of
    the project calls generate() and never knows which provider is behind it, so
    switching providers is a config change + one isolated helper here.

    Args:
        prompt: The user prompt to send.
        system_prompt: Optional system prompt to set context/behavior.
        json_mode: If True, ask the model to return valid JSON.

    Yields:
        Response text, token by token, as it streams in. (yield = produce values
        one at a time instead of returning them all at once, so the caller can
        print tokens as they arrive.)
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    # Pick the provider-specific implementation based on config.
    # Fail fast, fail loud if the configured provider has no handler.
    if config.PROVIDER == "ollama":
        yield from _call_ollama(messages, json_mode)
    elif config.PROVIDER == "azure":
        yield from _call_azure(messages, json_mode)
    else:
        raise ValueError(f"Unknown PROVIDER in config: {config.PROVIDER!r}")


# ===========================================================================
# EMBEDDINGS (text -> vector)
# ===========================================================================

def _embed_ollama(text: str) -> list[float]:
    """Provider-specific embedding call for Ollama."""
    try:
        # ollama.embeddings returns a dict; the vector is under "embedding".
        response = ollama.embeddings(model=config.EMBED_MODEL, prompt=text)
        return response["embedding"]
    except ConnectionError as exc:
        raise RuntimeError(
            "Could not reach Ollama. Is it running? "
            "Start the Ollama app, or run `ollama serve` in a terminal."
        ) from exc


def _embed_azure(text: str) -> list[float]:
    """Placeholder for Azure embeddings. Implemented later."""
    raise NotImplementedError(
        "Azure embeddings not implemented yet. Implement _embed_azure() before "
        "setting PROVIDER='azure' in config."
    )


def embed(text: str) -> list[float]:
    """Turn a piece of text into an embedding vector (a list of floats).

    Uses the SAME provider switch as generate(), so chat and embeddings always
    come from the configured provider. We embed chunks at index time AND
    questions at query time through this one function — guaranteeing both use
    the same model (required for the vectors to be comparable).
    """
    if config.PROVIDER == "ollama":
        return _embed_ollama(text)
    elif config.PROVIDER == "azure":
        return _embed_azure(text)
    else:
        raise ValueError(f"Unknown PROVIDER in config: {config.PROVIDER!r}")


if __name__ == "__main__":
    # Quick smoke test of both functions.
    print(f"Provider: {config.PROVIDER} | Chat: {config.CHAT_MODEL} | Embed: {config.EMBED_MODEL}\n")

    print("Chat test:")
    for token in generate("Say hello in 3 words."):
        print(token, end="", flush=True)

    print("\n\nEmbed test:")
    vec = embed("A test sentence about travel.")
    print(f"Vector length: {len(vec)} | first 5 numbers: {vec[:5]}")
