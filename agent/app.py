from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from tools import auto_causal_inference, custom_tools_condition, VARIABLE_INFO_DICT

# Create LLM instance
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Bind tool to LLM
llm_with_tools = llm.bind_tools([auto_causal_inference])

# Node for calling LLM
def tool_calling(state: MessagesState):
    VARIABLE_DESC = "The available variables are:\n" + "\n".join(
        [f"- {var}: {desc}" for var, desc in VARIABLE_INFO_DICT.items()]
    )
    
    # Extract original user question
    user_msg = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
    if not user_msg:
        raise ValueError("Missing user question.")

    full_prompt = f"""
                    You are a causal inference assistant.
                    {VARIABLE_DESC}

                    User question:
                    "{user_msg.content.strip()}"

                    Please identify the treatment and outcome variables from the question, 
                    and use the `auto_causal_inference` tool to perform causal inference.
                  """
    
    return {"messages": [llm_with_tools.invoke([HumanMessage(content=full_prompt)])]}

# Define graph
builder = StateGraph(MessagesState)
builder.add_node("tool_calling", tool_calling)
builder.add_node("auto_causal_inference", ToolNode([auto_causal_inference]))

builder.add_edge(START, "tool_calling")
builder.add_conditional_edges("tool_calling", custom_tools_condition)
builder.add_edge("auto_causal_inference", END)

graph = builder.compile()

# View
display(Image(graph.get_graph().draw_mermaid_png()))









# from langchain_core.messages import HumanMessage
# from langchain_core.messages import ToolMessage
# import json

# # human_prompt = "Does offering a promotion increase digital product activation ?"
# human_prompt = "Do branch visits increase engagement?"

# messages = [HumanMessage(content=human_prompt)]
# messages = graph.invoke({"messages": messages})

# for m in messages['messages']:
#     m.pretty_print()

# #
# print("\n\n===========================================================================")
# print("Detailed result of Causal Inference:")
# tool_msg = next((m for m in messages["messages"] if isinstance(m, ToolMessage)), None)

# if tool_msg:
#     try:
#         result = json.loads(tool_msg.content)
#         for key, value in result.items():
#             print(f"\n🔑 {key}:\n")
#             if isinstance(value, list):
#                 for item in value:
#                     print(f"  - {item}")
#             else:
#                 if key == "causal_graph":
#                     # Extract edges from DOT string
#                     lines = value.strip().replace("digraph {", "").replace("}", "").strip().split(";")
#                     for line in lines:
#                         if "->" in line:
#                             src, dst = map(str.strip, line.strip().split("->"))
#                             print(f"  {src} → {dst}")
#                 else:
#                     print(value)
#     except json.JSONDecodeError as e:
#         print("❌ Failed to parse tool content as JSON.")
#         print("Raw content:", tool_msg.content)
# else:
#     print("❌ No ToolMessage found in output.")