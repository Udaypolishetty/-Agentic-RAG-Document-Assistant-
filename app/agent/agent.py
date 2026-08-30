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


load_dotenv()
from langsmith import traceable

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

memory = ConversationMemory()
long_term_memory = LongTermMemory()


server_params = StdioServerParameters(
    command="python",
    args=["-m", "app.mcp_server.server"]
)


async def run_mcp_tool(tool_name, arguments):

    async with stdio_client(server_params) as (read, write):

        async with ClientSession(read, write) as session:

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
  
@traceable(name="Agentic RAG Agent")
def run_agent(question):

    memory.add("user", question)

    past_memories = long_term_memory.search(question)
    memory_context = "\n".join(past_memories)

    messages = [
        {
            "role": "system",
           "content": f"""
You are an intelligent project assistant.

IMPORTANT:
When the user asks about a project, its risks,
issues, tasks, deadlines, architecture, team,
or other project-specific information,
ALWAYS use the search_documents tool first.

Use get_project_info only when appropriate.

Use relevant past memories when helpful.

Do not invent facts.
"""
        }
    ]

    messages.extend(memory.get_messages())

    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_documents",
                "description": "Search the internal project knowledge base.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string"
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_project_info",
                "description": "Get project information using an MCP server.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "project_name": {
                            "type": "string"
                        }
                    },
                    "required": ["project_name"]
                }
            }
        }
    ]

    while True:

        response = client.chat.completions.create(
            model="google/gemini-2.5-flash",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            max_tokens=500
        )

        message = response.choices[0].message

        if not message.tool_calls:

            memory.add("assistant", message.content)
            long_term_memory.save(message.content)

            return message.content

        messages.append(message)

        for tool_call in message.tool_calls:

            arguments = json.loads(
                tool_call.function.arguments
            )

            if tool_call.function.name == "search_documents":

                results = search_documents(
                    arguments["query"]
                )

                tool_result = "\n\n".join(results)

            elif tool_call.function.name == "get_project_info":

                tool_result = asyncio.run(
                    run_mcp_tool(
                        "get_project_info",
                        arguments
                    )
                )

            else:

                tool_result = "Unknown tool."

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result
            })


if __name__ == "__main__":

    question = input("What do you want to know?: ")

    answer = run_agent(question)

    print("\nAnswer:\n")
    print(answer)