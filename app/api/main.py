from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.agent.agent import run_agent


app = FastAPI(
    title="Agentic RAG API",
    description="Agentic RAG knowledge assistant with memory, RAG and MCP",
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

        result = run_agent(
            request.question
        )

        return {
            "question": request.question,
            "answer": result["answer"],
            "sources": result["sources"]
        }

    except Exception as e:

        print("Agent error:", e)

        raise HTTPException(
            status_code=500,
            detail="Unable to process the question."
        )


@app.get("/ui")
def ui():

    return FileResponse(
        "static/index.html"
    )