import os
import json
import asyncio

from dotenv import load_dotenv
from openai import OpenAI

from app.tools.document_search import search_documents
from app.tools.gmail_search import search_gmail

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

async def run_mcp_tool(
    tool_name,
    arguments
):

    async with stdio_client(
        server_params
    ) as (read, write):

        async with ClientSession(
            read,
            write
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

    # ==================================================
    # CONVERSATION MEMORY
    # ==================================================

    memory.add(
        "user",
        question
    )

    sources = []


    # ==================================================
    # LONG-TERM MEMORY
    # ==================================================

    past_memories = long_term_memory.search(
        question
    )

    memory_context = "\n".join(
        past_memories
    )


    # ==================================================
    # SYSTEM PROMPT
    # ==================================================

    system_prompt = f"""
You are an intelligent knowledge assistant.

You can use three information sources:

1. The user's uploaded knowledge base
2. The user's Gmail
3. Project information through an MCP tool

IMPORTANT RULES:

- Do not invent information.
- Use retrieved information when relevant.
- For document questions, use the knowledge base.
- For Gmail questions, use Gmail.
- For project questions, use the project tool.
- You may combine information from multiple sources
  when necessary.
- Give concise and natural answers.
- If the available sources do not contain enough
  information, clearly say so.

GMAIL:

Use search_gmail when the user asks about:

- Emails
- Gmail
- Recruiters
- Job applications
- Interview emails
- Company communication
- Verification codes
- Application updates
- Recent email information

Do not use Gmail for questions that can be answered
from uploaded documents.

PROJECTS:

Use get_project_info when the user specifically
asks about a project.

Relevant past memories:

{memory_context}
"""


    # ==================================================
    # MESSAGES
    # ==================================================

    messages = [

        {
            "role": "system",
            "content": system_prompt
        }

    ]


    # ==================================================
    # CONVERSATION HISTORY
    # ==================================================

    messages.extend(
        memory.get_messages()
    )


    # ==================================================
    # RAG SEARCH
    # ==================================================

    rag_result = search_documents(
        question
    )


    retrieved_context = rag_result.get(
        "context",
        ""
    )


    retrieved_sources = rag_result.get(
        "sources",
        []
    )


    # --------------------------------------------------
    # Add document sources only if RAG returned context
    # --------------------------------------------------

    if retrieved_context.strip():

        for source in retrieved_sources:

            if source not in sources:

                sources.append(
                    source
                )


    # ==================================================
    # ADD RAG CONTEXT
    # ==================================================

    messages.append({

        "role": "user",

        "content": f"""
Here is information retrieved from the user's
knowledge base.

---------------- KNOWLEDGE BASE ----------------

{retrieved_context}

-------------- END KNOWLEDGE BASE --------------

Question:

{question}

Use this information if it is relevant.
"""
    })


    # ==================================================
    # TOOLS
    # ==================================================

    tools = [

        # ------------------------------------------------
        # PROJECT MCP TOOL
        # ------------------------------------------------

        {
            "type": "function",

            "function": {

                "name": "get_project_info",

                "description": """
Get detailed information about a specific project.

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
        },


        # ------------------------------------------------
        # GMAIL TOOL
        # ------------------------------------------------

        {
            "type": "function",

            "function": {

                "name": "search_gmail",

                "description": """
Search the user's Gmail when the question requires
information from emails.

Use this for:

- Recruiter emails
- Job applications
- Interview communication
- Company communication
- Verification emails
- Application updates
- Recent email information
""",

                "parameters": {

                    "type": "object",

                    "properties": {

                        "query": {

                            "type": "string",

                            "description": """
Gmail search query.

Examples:

Accenture
from:careers.accenture.com
subject:verification
newer_than:7d
"""
                        }

                    },

                    "required": [
                        "query"
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


        # ==================================================
        # NO TOOL CALL
        # ==================================================

        if not message.tool_calls:

            answer = (
                message.content
                or "I could not generate an answer."
            )


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


        # ==================================================
        # ADD ASSISTANT TOOL REQUEST
        # ==================================================

        messages.append(
            message
        )


        # ==================================================
        # EXECUTE TOOLS
        # ==================================================

        for tool_call in message.tool_calls:

            tool_name = (
                tool_call.function.name
            )


            arguments = json.loads(
                tool_call.function.arguments
            )


            # ==============================================
            # PROJECT MCP TOOL
            # ==============================================

            if tool_name == "get_project_info":

                tool_result = asyncio.run(

                    run_mcp_tool(

                        "get_project_info",

                        arguments

                    )
                )


            # ==============================================
            # GMAIL TOOL
            # ==============================================

            elif tool_name == "search_gmail":

                tool_result = search_gmail(

                    arguments["query"]

                )


            # ==============================================
            # UNKNOWN TOOL
            # ==============================================

            else:

                tool_result = (
                    "Unknown tool."
                )


            # ==================================================
            # ADD TOOL RESULT
            # ==================================================

            messages.append({

                "role": "tool",

                "tool_call_id":
                    tool_call.id,

                "content":
                    str(tool_result)

            })


    # ==================================================
    # END AGENT LOOP
    # ==================================================


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