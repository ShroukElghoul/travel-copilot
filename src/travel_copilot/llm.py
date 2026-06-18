import ollama
from typing import Generator, Optional

from . import config


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
        Response text, token by token, as it streams in.
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


if __name__ == "__main__":
    # Quick smoke test: run `poetry run python -m travel_copilot.llm`
    print(f"Provider: {config.PROVIDER} | Model: {config.CHAT_MODEL}\n")
    for token in generate("Say hello in 3 words."):
        print(token, end="", flush=True)
    print("\n\nTest complete.")
