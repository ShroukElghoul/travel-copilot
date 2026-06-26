# httpx is the HTTP client library used internally by Ollama.
# We import it here so we can catch the specific error it raises
# when the Ollama server is not running (httpx.ConnectError).
import httpx

# FastAPI   — the web framework that maps URLs to Python functions.
# HTTPException — raised inside a route to send a clean HTTP error response
#                 (status code + message) instead of a raw Python traceback.
from fastapi import FastAPI, HTTPException

# BaseModel — Pydantic base class. Any class that inherits from it gets:
#   - automatic JSON parsing (request body → Python object)
#   - automatic validation (wrong type or missing field → 422 error)
#   - automatic schema generation (visible in /docs)
from pydantic import BaseModel

# ask_agent() lives in src/travel_copilot/agent.py.
# The ".." means "go up one level" from this file's package (api/) to the
# parent package (travel_copilot/), then import ask_agent from agent.py.
# ask_agent(question: str) -> str  — runs the full LangGraph RAG loop and
# returns the final answer as a plain string.
from ..agent import ask_agent

# The FastAPI application object. Uvicorn is pointed at this object when you
# run: poetry run uvicorn src.travel_copilot.api.main:app
# Everything — routes, middleware, docs — hangs off this one object.
app = FastAPI()


# --- REQUEST / RESPONSE MODELS -----------------------------------------------
# These Pydantic models define the exact shape of the JSON that goes IN and OUT
# of the /ask endpoint. FastAPI reads the type hints and does three things:
#   1. Parses incoming JSON into a Python object automatically.
#   2. Validates it — missing "question" or wrong type → 422 Unprocessable Entity.
#   3. Generates the schema you see in /docs (no manual documentation needed).

class AskRequest(BaseModel):
    question: str   # the travel question sent by the caller


class AskResponse(BaseModel):
    answer: str     # the RAG-generated answer returned to the caller


# --- ROUTES ------------------------------------------------------------------
# A route = URL + HTTP verb + Python function.
# The decorator (@app.get, @app.post) registers the function in FastAPI's
# routing table. When a matching request arrives, FastAPI calls that function.

@app.get("/health")
def health():
    # GET /health — no input, no logic.
    # Used by load balancers and container orchestrators (e.g. Kubernetes) to
    # check "is this service alive?". Returns 200 + JSON if the server is up.
    return {"status": "ok"}


@app.post("/ask")
def ask(body: AskRequest) -> AskResponse:
    # POST /ask — receives a question, runs the full RAG agent, returns the answer.
    #
    # FastAPI automatically:
    #   - reads the request body as JSON
    #   - validates it against AskRequest (must have a "question" string)
    #   - passes it here as `body`
    #
    # We then call ask_agent(), which runs: hybrid retrieval → reranking →
    # LangGraph agent → Ollama LLM → final answer string.
    try:
        answer = ask_agent(body.question)
        return AskResponse(answer=answer)

    except httpx.ConnectError:
        # Ollama is not running or unreachable.
        # httpx.ConnectError is the specific exception the Ollama HTTP client
        # raises on a refused connection — it is NOT Python's built-in
        # ConnectionError (different class, different library).
        # 503 = "Service Unavailable": our server is up, but a dependency is down.
        raise HTTPException(
            status_code=503,
            detail="AI model is unavailable. Is Ollama running?"
        )

    except Exception:
        # Catch-all for any other unexpected error (e.g. ChromaDB missing,
        # a bug in the agent). We return a generic 500 so internal details
        # (file paths, stack traces) are never exposed to the caller.
        # 500 = "Internal Server Error": something broke inside our code.
        raise HTTPException(
            status_code=500,
            detail="Something went wrong. Please try again."
        )
