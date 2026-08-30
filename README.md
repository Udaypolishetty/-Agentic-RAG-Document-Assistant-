# Agentic RAG Document Assistant

> An AI-powered document assistant that combines Retrieval-Augmented Generation (RAG), agentic tool calling, conversation memory, MCP, and FastAPI.

---

## Overview

The goal is to let a user ask questions across a project or personal knowledge base and receive answers grounded in the information retrieved from the stored documents.**

This project is an **Agentic RAG (Retrieval-Augmented Generation) Document Assistant**** that answers questions using project-specific knowledge stored in a vector database.

Instead of relying only on the LLM's internal knowledge, the agent can retrieve relevant information from the project knowledge base and use that context to generate grounded answers.

### Key Features

\- Semantic document retrieval

\- Agentic LLM reasoning

\- Function/tool calling

\- Short-term conversation memory

\- Long-term semantic memory

\- Model Context Protocol (MCP)

\- FastAPI REST API

\- Automated API testing with Pytest

\- LangSmith observability

---

## Architecture

```text

User / Web UI

      |

      v

   FastAPI

      |

      v

 Agentic Layer

      |

      +-------------------+-------------------+

      |                   |                   |

      v                   v                   v

     RAG                 MCP               Memory

      |                   |                   |

      v                   v                   v

 ChromaDB            MCP Server          ChromaDB

      |                   |                   |

      +-------------------+-------------------+

                          |

                          v

                  Relevant Context

                          |

                          v

                         LLM

                          |

                          v

                    Final Answer

```

---

## How It Works

A typical request follows this flow:

```text

User Question

      |

      v

FastAPI / Agent

      |

      v

LLM decides whether information is needed

      |

      v

Tool Calling

      |

      v

RAG / MCP / Memory

      |

      v

Relevant Context

      |

      v

LLM generates response

      |

      v

Final Answer

```

The agent is instructed to use available project information and avoid inventing project-specific facts.

---

## Retrieval-Augmented Generation

Project information is stored in a vector database and retrieved using semantic similarity.

### RAG Pipeline

```text

Documents

    |

    v

Text Chunking

    |

    v

Embeddings

    |

    v

ChromaDB

    |

    v

Semantic Search

    |

    v

Relevant Context

    |

    v

LLM

    |

    v

Answer

```

### Why Semantic Search?

Traditional keyword search looks for exact words.

Semantic search instead compares the **meaning**** of the question with the meaning of stored document chunks.

For example:

```text

Question:

"What are the current risks in Project Phoenix?"

          |

          v

Semantic Retrieval

          |

          v

Relevant Project Phoenix information

          |

          v

LLM

          |

          v

Grounded Answer

```

---

## Agentic Tool Calling

The LLM can decide when it needs additional project information and call a tool to retrieve it.

### Current Tool

**`search_documents`***

Searches the project knowledge base and returns relevant information to the agent.

### Tool Calling Flow

```text

User Question

      |

      v

     LLM

      |

      v

Need project information?

      |

     Yes

      |

      v

Tool Call

      |

      v

search_documents()

      |

      v

Retrieved Information

      |

      v

     LLM

      |

      v

Final Answer

```

This makes the application **agentic**** because the LLM can decide when to use an external tool instead of following only a fixed retrieval sequence.

---
## Memory

The application uses two levels of memory.

### Short-Term Memory

Short-term memory stores the current conversation so the assistant can maintain context between messages.

```text

User Message

      |

      v

Conversation Memory

      |

      v

Agent

      |

      v

Assistant Response

```

### Long-Term Memory

Long-term memory stores useful previous information and retrieves relevant memories using semantic similarity.

```text

Previous Interaction

        |

        v

    Embedding

        |

        v

     ChromaDB

        |

        v

 Semantic Search

        |

        v

 Relevant Memory

        |

        v

   Agent Context

```

---

## Model Context Protocol (MCP)

The project includes an MCP server/client architecture for standardized tool integration.

```text

Agent

  |

  v

MCP Client

  |

  v

MCP Server

  |

  v

Available Tools

  |

  v

Tool Result

  |

  v

Agent

```

MCP separates tool implementations from the main agent logic and provides a standardized approach for exposing tools.

---

## FastAPI Backend

The agent is exposed through a REST API using FastAPI.

### Health Check

```http

GET /

```

Response:

```json

{

  "message": "Agentic RAG API is running"

}

```
### Ask a Question

```http

POST /ask

```

Request:

```json

{

  "question": "What are the current risks in Project Phoenix?"

}

```

Response:

```json

{

  "question": "What are the current risks in Project Phoenix?",

  "answer": "The current risks in Project Phoenix are..."

}

```

### Web Interface

```http

GET /ui

```

A simple browser-based interface is available for interacting with the assistant.

---

## Observability

The project integrates **LangSmith**** for LLM application tracing and debugging.

A typical trace can show:

```text

User Query

    |

    v

Agent

    |

    v

Tool Selection

    |

    v

Retrieval / MCP

    |

    v

Tool Result

    |

    v

LLM Response

```

This makes it easier to understand agent behavior and troubleshoot retrieval or tool-calling problems.

---

## Testing

The project includes automated API tests using **Pytest****.

Run:

```bash

python -m pytest

```

Example result:

```text

collected 2 items

Tests/test_api.py ..    [100%]

2 passed

```

The current tests cover:

\- API health check

\- Empty-question validation

---

## Project Structure

```text

Agentic-RAG/

|

+-- app/

\|   |

\|   +-- agent/

\|   |   +-- agent.py

\|   |

\|   +-- api/

\|   |   +-- main.py

\|   |

\|   +-- rag/

\|   |   +-- ingest.py

\|   |   +-- retrieve.py

\|   |   +-- rag.py

\|   |

\|   +-- memory/

\|   |   +-- memory.py

\|   |   +-- long_term_memory.py

\|   |

\|   +-- tools/

\|   |   +-- document_search.py

\|   |

\|   +-- mcp_server/

\|   |   +-- server.py

\|   |

\|   +-- mcp_client/

\|       +-- client.py

|

+-- static/

\|   +-- index.html

|

+-- Tests/

\|   +-- test_api.py

|

+-- data/

|

+-- .env.example

+-- .gitignore

+-- requirements.txt

+-- README.md

```

---

## Technology Stack

\| Technology | Purpose |

\|---|---|

\| Python 3.12 | Core application |

\| Gemini 2.5 Flash | Large Language Model |

\| OpenRouter | LLM API access |

\| ChromaDB | Vector database and semantic memory |

\| Hugging Face Embeddings | Text embeddings |

\| FastAPI | REST API |

\| MCP | Tool integration |

\| Pytest | Automated testing |

\| LangSmith | Observability and tracing |

\| HTML / CSS / JavaScript | Web interface |

\| Git / GitHub | Version control |

---

## Installation

### 1. Clone the Repository

```bash

git clone <your-repository-url>

cd Agentic-RAG

```

### 2. Create a Virtual Environment

```bash

python -m venv .venv

```

### 3. Activate the Environment

Windows PowerShell:

```powershell

.venv\Scripts\Activate.ps1

```

### 4. Install Dependencies

```bash

pip install -r requirements.txt

```

### 5. Configure Environment Variables

Create a `.env` file in the project root.

```text

OPENROUTER_API_KEY=your_openrouter_api_key_here

LANGCHAIN_TRACING_V2=true

LANGCHAIN_API_KEY=your_langsmith_api_key_here

LANGCHAIN_PROJECT=Agentic-RAG

LANGCHAIN_ENDPOINT=https://api.smith.langchain.com

```

> **Important:**** Never commit `.env` or real API keys to GitHub.

---

## Running the Application

### Run the Agent

```bash

python -m app.agent.agent

```

### Run the FastAPI Server

```bash

uvicorn app.api.main:app --reload

```

Then open:

**API documentation**

```text

http://127.0.0.1:8000/docs

```

**Web interface**

```text

http://127.0.0.1:8000/ui

```

**Health check**
```text

http://127.0.0.1:8000/

```

---

## Example Questions

Try questions such as:

```text

What are the current risks in Project Deloitte?

What is the migration deadline?

What technologies are used in Project Accenture?

What should the team prioritize?

What issues are currently high priority?

```

---

## Engineering Highlights

This project demonstrates practical experience with:

\- Agentic RAG architecture

\- Semantic vector search

\- LLM tool calling

\- Agent-based decision making

\- Short-term conversation memory

\- Long-term semantic memory

\- MCP tool integration

\- REST API development

\- Input validation

\- Exception handling

\- Automated backend testing

\- LLM observability

---

## Future Improvements

Potential improvements include:

\- Gmail integration through MCP

\- Jira integration

\- Notion integration

\- Additional project-management tools

\- RAG evaluation and retrieval metrics

\- Response streaming

\- Authentication and authorization

\- Response caching

\- Docker containerization

\- Cloud deployment

\- Production monitoring

---

## Key Learnings

Building this project provided hands-on experience with modern AI application development:

1. Designing a complete RAG pipeline

2. Working with vector databases and embeddings

3. Building agentic LLM workflows

4. Implementing tool calling

5. Managing short-term and long-term memory

6. Integrating MCP-based tools

7. Exposing AI functionality through REST APIs

8. Testing backend functionality with Pytest

9. Adding observability to LLM applications

---

## Author

### Uday Polishetty

**Software Developer | AI / LLM Projects****

Built as a practical project exploring **Agentic AI, RAG, vector databases, memory, tool calling, MCP, and backend engineering****.
