import os
import json
import asyncio

from dotenv import load_dotenv
from openai import OpenAI

from app.tools.document_search import search_documents
from app.memory.memory import ConversationMemory
from app.memory.long_term_memory import LongTermMemory

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langsmith import traceable


# ==================================================
# ENVIRONMENT
# ==================================================

load_dotenv()


# ==================================================
# OPENROUTER CLIENT
# ==================================================

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)


# ==================================================
# MEMORY
# ==================================================

memory = ConversationMemory()
long_term_memory = LongTermMemory()


# ==================================================
# MCP SERVER
# ==================================================

server_params = StdioServerParameters(
    command="python",
    args=["-m", "app.mcp_server.server"]
)


# ==================================================
# MCP TOOL RUNNER
# ==================================================

async def run_mcp_tool(tool_name, arguments):

    async with stdio_client(
        server_params
    ) as (read, write):

        async with ClientSession(
            read, write
        ) as session:

            await session.initialize()

            result = await session.call_tool(
                tool_name,
                arguments
            )

            return "\n".join(
                item.text
                for item in result.content
                if hasattr(item, "text")
            )


# ==================================================
# AGENT
# ==================================================

@traceable(name="Agentic RAG Agent")
def run_agent(question):

    # ------------------------------------------------
    # Conversation memory
    # ------------------------------------------------

    memory.add(
        "user",
        question
    )

    sources = []


    # ------------------------------------------------
    # Long-term memory
    # ------------------------------------------------

    past_memories = long_term_memory.search(
        question
    )

    memory_context = "\n".join(
        past_memories
    )


    # ------------------------------------------------
    # SYSTEM PROMPT
    # ------------------------------------------------

    system_prompt = f"""
You are an intelligent knowledge assistant.

Your job is to answer questions using the user's
knowledge base.

The knowledge base may contain:

- Companies
- Jobs and careers
- Projects
- Technology
- Personal notes
- Science
- Astronomy
- Other documents

IMPORTANT RULES:

1. Use the retrieved knowledge-base information
   whenever it is relevant to the user's question.

2. Do not invent information.

3. If retrieved information contains the answer,
   use it as the primary source.

4. Give a concise and natural answer.

5. You are NOT restricted to project questions.

6. For questions about Project Phoenix or other
   projects, relevant project information should
   be used.

7. If the retrieved information does not contain
   enough information, clearly say that the
   knowledge base does not contain enough
   information.

Relevant past memories:

{memory_context}
"""


    messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]


    # ------------------------------------------------
    # Conversation history
    # ------------------------------------------------

    messages.extend(
        memory.get_messages()
    )


    # ==================================================
    # RAG SEARCH FIRST
    # ==================================================

    rag_result = search_documents(
        question
    )


    # ------------------------------------------------
    # Extract retrieved context
    # ------------------------------------------------

    retrieved_context = rag_result.get(
        "context",
        ""
    )


    # ------------------------------------------------
    # Extract sources
    # ------------------------------------------------

    retrieved_sources = rag_result.get(
        "sources",
        []
    )


    for source in retrieved_sources:

        if source not in sources:

            sources.append(
                source
            )


    # ------------------------------------------------
    # Add retrieved information to conversation
    # ------------------------------------------------

    messages.append({

        "role": "user",

        "content": f"""
Use the following information retrieved from
the knowledge base to answer my question.

---------------- RETRIEVED INFORMATION ----------------

{retrieved_context}

---------------- END RETRIEVED INFORMATION ------------

Question:

{question}

Answer using the retrieved information when relevant.
"""
    })


    # ==================================================
    # TOOLS
    # ==================================================

    tools = [

        {
            "type": "function",

            "function": {

                "name": "get_project_info",

                "description": """
Get detailed project information using the MCP
project information server.

Use this tool when the user specifically asks
about a project.
""",

                "parameters": {

                    "type": "object",

                    "properties": {

                        "project_name": {

                            "type": "string",

                            "description":
                                "Name of the project."
                        }

                    },

                    "required": [
                        "project_name"
                    ]
                }
            }
        }

    ]


    # ==================================================
    # AGENT LOOP
    # ==================================================

    while True:

        response = client.chat.completions.create(

            model="google/gemini-2.5-flash",

            messages=messages,

            tools=tools,

            tool_choice="auto",

            max_tokens=500
        )


        message = response.choices[0].message


        # ------------------------------------------------
        # No more tool calls
        # ------------------------------------------------

        if not message.tool_calls:

            answer = message.content


            memory.add(
                "assistant",
                answer
            )


            long_term_memory.save(
                answer
            )


            return {

                "answer": answer,

                "sources": sources
            }


        # ------------------------------------------------
        # Add assistant tool request
        # ------------------------------------------------

        messages.append(
            message
        )


        # ------------------------------------------------
        # Execute MCP tools
        # ------------------------------------------------

        for tool_call in message.tool_calls:

            arguments = json.loads(
                tool_call.function.arguments
            )


            if (
                tool_call.function.name
                == "get_project_info"
            ):

                tool_result = asyncio.run(

                    run_mcp_tool(

                        "get_project_info",

                        arguments
                    )
                )


            else:

                tool_result = (
                    "Unknown tool."
                )


            messages.append({

                "role": "tool",

                "tool_call_id":
                    tool_call.id,

                "content":
                    tool_result
            })


# ==================================================
# COMMAND LINE TEST
# ==================================================

if __name__ == "__main__":

    question = input(
        "What do you want to know?: "
    )


    result = run_agent(
        question
    )


    print(
        "\nAnswer:\n"
    )

    print(
        result["answer"]
    )


    print(
        "\nSources:\n"
    )


    if result["sources"]:

        for source in result["sources"]:

            print(

                f"- {source['source']} "
                f"({source['category']})"

            )

    else:

        print(
            "No document sources used."
        )