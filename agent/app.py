from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from tools import (
    tool_calling,
    custom_tools_condition,
    identify_causal_relationships,
    identify_causal_variables,
    compute_ate,
    tune_model,
    refutation_test,
    summarize_result,
    CausalState
)
from IPython.display import Image, display
import json
from typing import Dict, Any, List


# Build graph with nodes
builder = StateGraph(CausalState)
builder.add_node("tool_calling", tool_calling)
builder.add_node("identify_causal_relationships", ToolNode([identify_causal_relationships]))
builder.add_node("identify_causal_variables", identify_causal_variables)
builder.add_node("compute_ate", compute_ate)
builder.add_node("tune_model", tune_model)
builder.add_node("refutation_test", refutation_test)
builder.add_node("summarize_result", summarize_result)

# Define edges to pass data from one node to next
builder.add_edge(START, "tool_calling")
builder.add_conditional_edges("tool_calling", 
                              path=custom_tools_condition,
                              path_map={
                                    "tool": "identify_causal_relationships",
                                    "__end__": END,
                                }
                                )
builder.add_edge("identify_causal_relationships", "identify_causal_variables")
builder.add_edge("identify_causal_variables", "compute_ate")
builder.add_edge("compute_ate", "tune_model")
builder.add_edge("tune_model", "refutation_test")
builder.add_edge("refutation_test", "summarize_result")
builder.add_edge("summarize_result", END)

graph = builder.compile()

# Display graph diagram
# display(Image(graph.get_graph().draw_mermaid_png()))


# Test run
human_prompt = "Does offering a promotion increase digital product activation?"
messages = [HumanMessage(content=human_prompt)]
messages = graph.invoke({"messages": messages})

for m in messages['messages']:
    m.pretty_print()
