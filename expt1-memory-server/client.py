import asyncio
import os
import sys
from contextlib import AsyncExitStack

from google import genai
from google.genai import types as genai_types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import load_dotenv

# override=True so the .env file wins over any old key set in the terminal
load_dotenv(override=True)

MODEL = "gemini-3.1-flash-lite"


def mcp_tools_to_gemini(tools_result):
    """Convert MCP tool definitions into Gemini function declarations."""
    declarations = []
    for t in tools_result.tools:
        schema = dict(t.inputSchema) if t.inputSchema else {"type": "object", "properties": {}}
        declarations.append(
            genai_types.FunctionDeclaration(
                name=t.name,
                description=t.description or "",
                parameters=schema,
            )
        )
    return genai_types.Tool(function_declarations=declarations)


async def generate_with_retry(client, max_attempts=5, **kwargs):
    """Call Gemini, retrying on temporary errors (503 overloaded, 429 rate limit)."""
    for attempt in range(max_attempts):
        try:
            # generate_content is blocking, so run it in a thread to keep asyncio responsive
            return await asyncio.to_thread(client.models.generate_content, **kwargs)
        except Exception as e:
            msg = str(e)
            temporary = any(code in msg for code in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED"))
            if temporary and attempt < max_attempts - 1:
                wait = 2 ** attempt
                print(f"[Model busy, retrying in {wait}s...]")
                await asyncio.sleep(wait)
            else:
                raise


class MCPClient:
    def __init__(self):
        self.session: ClientSession | None = None
        self.exit_stack = AsyncExitStack()
        self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    async def connect(self, server_script: str):
        server_params = StdioServerParameters(
            command=sys.executable,
            args=[server_script],
        )
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )
        await self.session.initialize()

        tools_result = await self.session.list_tools()
        print("Connected. Available tools:", [t.name for t in tools_result.tools])

    async def process_query(self, query: str) -> str:
        tools_result = await self.session.list_tools()
        gemini_tool = mcp_tools_to_gemini(tools_result)

        contents = [
            genai_types.Content(role="user", parts=[genai_types.Part(text=query)])
        ]
        config = genai_types.GenerateContentConfig(tools=[gemini_tool])

        response = await generate_with_retry(
            self.client, model=MODEL, contents=contents, config=config
        )

        # Loop until the model stops asking for tools and gives a final answer
        while True:
            candidate = response.candidates[0]
            parts = candidate.content.parts or []
            function_calls = [p.function_call for p in parts if getattr(p, "function_call", None)]

            if not function_calls:
                break

            # Add the model's tool-call message once, then answer every call in one message
            contents.append(candidate.content)
            response_parts = []

            for fc in function_calls:
                args = dict(fc.args) if fc.args else {}
                print(f"\n[Calling tool: {fc.name} with args {args}]")

                result = await self.session.call_tool(fc.name, args)
                result_text = "".join(
                    c.text for c in result.content if hasattr(c, "text")
                )
                print(f"[Tool result: {result_text}]")

                response_parts.append(
                    genai_types.Part.from_function_response(
                        name=fc.name,
                        response={"result": result_text},
                    )
                )

            contents.append(genai_types.Content(role="user", parts=response_parts))

            response = await generate_with_retry(
                self.client, model=MODEL, contents=contents, config=config
            )

        final_parts = response.candidates[0].content.parts or []
        return "\n".join(p.text for p in final_parts if getattr(p, "text", None))

    async def chat_loop(self):
        print("Personal Assistant Memory Client. Type 'quit' to exit.")
        while True:
            query = input("\nYou: ").strip()
            if query.lower() == "quit":
                break
            if not query:
                continue
            try:
                answer = await self.process_query(query)
                print(f"\nAssistant: {answer}")
            except Exception as e:
                print(f"Error: {e}")

    async def cleanup(self):
        await self.exit_stack.aclose()


async def main():
    client = MCPClient()
    try:
        await client.connect("server.py")
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())