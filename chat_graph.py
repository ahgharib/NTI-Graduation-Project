# chat_graph.py
from langgraph.graph import StateGraph, END
from chat_state import AgentState
from orchestrator import Orchestrator
from quiz_agent import quiz_node
from explainer_agent import explainer_node
from summarizer_agent import summarizer_node # <--- IMPORTED
from user_summary_node import user_summary_node # <--- IMPORTED
from langgraph.checkpoint.memory import MemorySaver


def scheduler(state: AgentState):
    """Schedule the next node to execute based on the plan."""
    
    queue_actions = state.get("plan_actions", [])
    queue_instructions = state.get("plan_instructions", [])
    
    if not queue_actions:
        return {"next": "END"}
    
    next_node = queue_actions[0]
    next_instr = queue_instructions[0]
    
    return {
        "next": next_node,
        "current_instruction": next_instr,
        "plan_actions": queue_actions[1:],
        "plan_instructions": queue_instructions[1:]
    }

def create_study_buddy_graph():
    workflow = StateGraph(AgentState)
    orchestrator = Orchestrator()
    
    # Add Nodes
    workflow.add_node("orchestrator", orchestrator.build_plan_node)
    workflow.add_node("scheduler", scheduler)
    workflow.add_node("quiz_generator", quiz_node)
    workflow.add_node("explain_node", explainer_node)
    workflow.add_node("summarizer", summarizer_node) # <--- ADDED NODE
    workflow.add_node("user_summary_node", user_summary_node) # <--- ADDED NODE
    
    # Entry Point
    workflow.set_entry_point("orchestrator")
    
    # Edges to Scheduler
    workflow.add_edge("orchestrator", "scheduler")
    workflow.add_edge("quiz_generator", "user_summary_node") # <--- MODIFIED EDGE
    workflow.add_edge("explain_node", "scheduler")
    workflow.add_edge("summarizer", "scheduler") # <--- ADDED EDGE

    workflow.add_edge("user_summary_node", "scheduler") # <--- ADDED EDGE

    # Conditional Edges from Scheduler
    workflow.add_conditional_edges(
        "scheduler",
        lambda state: state["next"] if "next" in state else "END",
        {
            "quiz_generator": "quiz_generator", 
            "explain_node": "explain_node",
            "summarizer": "summarizer", # <--- ADDED MAPPING
            "END": END
        }
    )
    # Memory Checkpointing
    memory_saver = MemorySaver()

    return workflow.compile(interrupt_before=["user_summary_node"], checkpointer=memory_saver)

study_buddy_graph = create_study_buddy_graph()