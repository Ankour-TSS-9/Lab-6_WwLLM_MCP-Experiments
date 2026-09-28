import asyncio
import os
import sys

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv(override=True)

MODEL = "gemini-3.1-flash-lite"
SERVER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "server.py")

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))


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


async def generate_with_retry(log, max_attempts=5, **kwargs):
    """Call Gemini, retrying on temporary errors (503 overloaded, 429 rate limit)."""
    for attempt in range(max_attempts):
        try:
            return await asyncio.to_thread(client.models.generate_content, **kwargs)
        except Exception as e:
            msg = str(e)
            temporary = any(c in msg for c in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED"))
            if temporary and attempt < max_attempts - 1:
                wait = 2 ** attempt
                log(f"⏳ Model busy, retrying in {wait}s...")
                await asyncio.sleep(wait)
            else:
                raise


async def run_query(query, log):
    """Send the query to Gemini, let it call MCP tools, and return the final answer."""
    params = StdioServerParameters(command=sys.executable, args=[SERVER_PATH])

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools_result = await session.list_tools()
            tool_names = [t.name for t in tools_result.tools]

            config = genai_types.GenerateContentConfig(
                tools=[mcp_tools_to_gemini(tools_result)]
            )
            contents = [
                genai_types.Content(role="user", parts=[genai_types.Part(text=query)])
            ]

            log(f"**Step 1: Question sent to Gemini**\n\nAvailable tools: `{tool_names}`")
            response = await generate_with_retry(
                log, model=MODEL, contents=contents, config=config
            )

            step = 2
            while True:
                candidate = response.candidates[0]
                parts = candidate.content.parts or []
                calls = [p.function_call for p in parts if getattr(p, "function_call", None)]

                if not calls:
                    break

                # Any text the model wrote alongside its tool call
                for p in parts:
                    if getattr(p, "text", None):
                        log(f"**Step {step}: Model reasoning**\n\n{p.text}")
                        step += 1

                contents.append(candidate.content)
                response_parts = []

                for fc in calls:
                    args = dict(fc.args) if fc.args else {}
                    log(
                        f"**Step {step}: Model decides to call a tool**\n\n"
                        f"Tool: `{fc.name}`\n\nArguments: `{args}`"
                    )
                    step += 1

                    result = await session.call_tool(fc.name, args)
                    result_text = "".join(
                        c.text for c in result.content if hasattr(c, "text")
                    )
                    log(f"**Step {step}: Tool result (from wttr.in)**\n\n```\n{result_text}\n```")
                    step += 1

                    response_parts.append(
                        genai_types.Part.from_function_response(
                            name=fc.name, response={"result": result_text}
                        )
                    )

                contents.append(genai_types.Content(role="user", parts=response_parts))
                log(f"**Step {step}: Tool result sent back to Gemini**")
                step += 1
                response = await generate_with_retry(
                    log, model=MODEL, contents=contents, config=config
                )

            final_parts = response.candidates[0].content.parts or []
            return "\n".join(p.text for p in final_parts if getattr(p, "text", None))


# ---------------- Streamlit UI ----------------

st.set_page_config(page_title="Weather Dashboard", page_icon="🌦️")
st.title("🌦️ Weather Dashboard")
st.caption("Gemini + MCP server + wttr.in. Ask about the weather anywhere.")

if "history" not in st.session_state:
    st.session_state.history = []

# Show earlier conversation
for item in st.session_state.history:
    with st.chat_message("user"):
        st.write(item["query"])
    with st.chat_message("assistant"):
        with st.expander("🧠 Thought process"):
            for entry in item["trace"]:
                st.markdown(entry)
        st.write(item["answer"])

query = st.chat_input("e.g. What's the weather in Tokyo?")

if query:
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        trace = []
        with st.status("🧠 Thinking...", expanded=True) as status:

            def log(entry):
                trace.append(entry)
                status.markdown(entry)

            try:
                answer = asyncio.run(run_query(query, log))
                status.update(label="🧠 Thought process", state="complete", expanded=True)
            except Exception as e:
                answer = f"Error: {e}"
                status.update(label="Something went wrong", state="error")

        st.write(answer)

    st.session_state.history.append({"query": query, "trace": trace, "answer": answer})