from fastapi import FastAPI
from pydantic import BaseModel

from ..agent import ask_agent

app = FastAPI()


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(body: AskRequest) -> AskResponse:
    answer = ask_agent(body.question)
    return AskResponse(answer=answer)