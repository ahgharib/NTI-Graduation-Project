from langgraph.graph import StateGraph, END
from state import PlanState
from tools import search_node_func
from agents import generator_node, discriminator_node, editor_node, validator_node

def should_continue(state):
    if state.get("error"):
        return "end" # Or handle error specific logic
    if state.get("approved"):
        return "end"
    if state["attempt_count"] > 3:
        return "end"
    return "regenerate"

workflow = StateGraph(PlanState)

# Planner Nodes
workflow.add_node("search", search_node_func)
workflow.add_node("generator", generator_node)
workflow.add_node("validator", validator_node) # <--- New Node
workflow.add_node("discriminator", discriminator_node)

# Edges
workflow.set_entry_point("search")
workflow.add_edge("search", "generator")
workflow.add_edge("generator", "validator") # Check Generator output
workflow.add_edge("validator", "discriminator")

workflow.add_conditional_edges(
    "discriminator",
    should_continue,
    {
        "end": END,
        "regenerate": "generator"
    }
)

app_graph = workflow.compile()

# --- EDITOR GRAPH ---
# Create a small subgraph for the editor to ensure validation runs there too
edit_workflow = StateGraph(PlanState)
edit_workflow.add_node("editor", editor_node)
edit_workflow.add_node("validator", validator_node)

edit_workflow.set_entry_point("editor")
edit_workflow.add_edge("editor", "validator")
edit_workflow.add_edge("validator", END)

editor_graph = edit_workflow.compile()