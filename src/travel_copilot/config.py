from pathlib import Path

# Base directory = project root.
# config.py lives at: <root>/src/travel_copilot/config.py
# so three .parent hops climb: config.py -> travel_copilot -> src -> <root>
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# --- Active provider ---------------------------------------------------------
# Change this ONE line to switch the whole app to a different LLM provider.
PROVIDER = "ollama"   # "ollama" | "azure"

# --- Provider-specific settings (each block self-contained) ------------------
# Ollama (local, free — used for the sprint)
OLLAMA_CHAT_MODEL = "llama3.1"
OLLAMA_EMBED_MODEL = "nomic-embed-text"

# Azure OpenAI (placeholder for later; endpoint/key would come from .env)
AZURE_CHAT_MODEL = "gpt-4o"
AZURE_EMBED_MODEL = "text-embedding-3-small"

# --- Generic names the rest of the app uses ----------------------------------
# Everything outside config refers to CHAT_MODEL / EMBED_MODEL, never to the
# provider-specific names above. Switching providers happens here, once.
if PROVIDER == "ollama":
    CHAT_MODEL = OLLAMA_CHAT_MODEL
    EMBED_MODEL = OLLAMA_EMBED_MODEL
elif PROVIDER == "azure":
    CHAT_MODEL = AZURE_CHAT_MODEL
    EMBED_MODEL = AZURE_EMBED_MODEL
else:
    # Fail fast, fail loud: an unknown provider is caught here at import time
    # with a clear message, not deep inside an API call later.
    raise ValueError(f"Unknown PROVIDER in config: {PROVIDER!r}")

# --- Paths -------------------------------------------------------------------
DATA_PATH = BASE_DIR / "data"
CHROMA_DB_PATH = BASE_DIR / "chroma_db"


def ensure_dirs() -> None:
    """Create the project's data/output directories if they don't exist.

    Kept as an explicit function (not run at import time) so that importing
    this module has no surprising side effects on the filesystem. Call once
    at the start of scripts that read/write these paths.
    """
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)
