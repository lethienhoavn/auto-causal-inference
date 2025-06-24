from langgraph.graph import StateGraph, MessagesState, END, START
from langchain_core.messages import HumanMessage
from openai import OpenAI
import asyncio
from typing import Dict, List, TypedDict

import json
import re
from dotenv import load_dotenv
from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters
from openai import OpenAI

load_dotenv()

server_params = StdioServerParameters(
    command="python",
    args=["server.py"],
)

class CustomState(TypedDict):
    messages: List[HumanMessage]
    treatment: str
    outcome: str

async def run_auto_causal_inference(treatment: str, outcome: str):
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                tools = await session.list_tools()
                target_tool = next((tool for tool in tools.tools if tool.name == "auto_causal_inference"), None)

                if not target_tool:
                    raise ValueError("Tool 'auto_causal_inference' not found in MCP server")

                # Call tool with structured arguments
                result = await session.call_tool(
                    name="auto_causal_inference",
                    arguments={
                        "treatment": treatment,
                        "outcome": outcome,
                    }
                )

                # Parse and return
                return json.loads(result.content[0].text)

    except Exception as e:
        print("❌ Error during tool execution:")
        import traceback
        traceback.print_exc()
        return None
    
def clean_json_response(s: str) -> str:
    # Loại bỏ code block markdown ```json ... ```
    s = s.strip()
    s = re.sub(r"^```json\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()

# 1. tool_calling node: dùng LLM để xác định biến
def tool_calling(state: CustomState) -> CustomState:
    user_msg = next((m for m in state['messages'] if isinstance(m, HumanMessage)), None)
    if not user_msg:
        raise ValueError("No user input message")

    variables = [
        "age", "income", "education", "branch_visits",
        "channel_preference", "customer_engagement",
        "region_code", "promotion_offer", "activated_ib"
    ]

    prompt = f"""
        You are a causal inference assistant.
        Variables: {', '.join(variables)}

        User question:
        "{user_msg.content.strip()}"

        Extract the treatment and outcome variables from the question.
        Return JSON like: {{ "treatment": "...", "outcome": "..." }}
    """

    llm = OpenAI()
    response = llm.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    parsed = response.choices[0].message.content
    parsed = clean_json_response(parsed)
    import json
    try:
        extracted = json.loads(parsed)
        print("extracted: ", extracted)
        return {"treatment": extracted["treatment"], "outcome": extracted["outcome"], "messages": state["messages"]}
    except:
        raise ValueError(f"Failed to parse tool_calling output: {parsed}")

# 2. auto_causal_inference node: gọi MCP tool server
async def auto_node(state: CustomState) -> CustomState:
    print("state: ", state)
    treatment = state["treatment"]
    outcome = state["outcome"]

    result = await run_auto_causal_inference(treatment, outcome)
    return {"messages": state["messages"] + [HumanMessage(content=json.dumps(result, indent=2))]}


# 3. Build LangGraph async
def build_graph():
    builder = StateGraph(CustomState)
    builder.add_node("tool_calling", tool_calling)
    builder.add_node("auto_causal_inference", auto_node)
    
    builder.add_edge(START, "tool_calling")
    builder.add_edge("tool_calling", "auto_causal_inference")
    builder.add_edge("auto_causal_inference", END)

    return builder.compile()

# 4. Run
async def main():
    graph = build_graph()

    user_query = "Does offering a promotion increase activation of Internet Banking?"
    input_state = {"messages": [HumanMessage(content=user_query)]}
    result = await graph.ainvoke(input_state)

    for m in result["messages"]:
        print(m.content)

if __name__ == "__main__":
    asyncio.run(main())
