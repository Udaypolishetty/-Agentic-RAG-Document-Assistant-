from fastapi import FastAPI
from pydantic import BaseModel

from app.agent.agent import run_agent
from fastapi.responses import FileResponse
from fastapi import HTTPException

app = FastAPI(
    title="Agentic RAG API",
    description="Agentic RAG system with memory, tools and MCP",
    version="1.0.0"
)


class Question(BaseModel):
    question: str


@app.get("/")
def home():
    return {
        "message": "Agentic RAG API is running"
    }


@app.post("/ask")
def ask_question(request: Question):

    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    try:
        answer = run_agent(request.question)

        return {
            "question": request.question,
            "answer": answer
        }

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Unable to process the question."
        )

@app.get("/ui")
def ui():
    return FileResponse("static/index.html")