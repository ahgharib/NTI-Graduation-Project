# chat_graph.py
from langgraph.graph import StateGraph, END
from chat_state import AgentState
from orchestrator import Orchestrator
from quiz_agent import quiz_node
from explainer_agent import explainer_node

def scheduler(state: AgentState):
    """Schedule the next node to execute based on the plan."""
    # CHANGED: Removed load_plan_to_state() because we inject plan_data in app.py
    
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
    
    workflow.add_node("orchestrator", orchestrator.build_plan_node)
    workflow.add_node("scheduler", scheduler)
    workflow.add_node("quiz_generator", quiz_node)
    workflow.add_node("explain_node", explainer_node)
    
    workflow.set_entry_point("orchestrator")
    workflow.add_edge("orchestrator", "scheduler")
    workflow.add_edge("quiz_generator", "scheduler")
    workflow.add_edge("explain_node", "scheduler")
    
    workflow.add_conditional_edges(
        "scheduler",
        lambda state: state["next"] if "next" in state else "END",
        {
            "quiz_generator": "quiz_generator", 
            "explain_node": "explain_node",
            "END": END
        }
    )
    
    return workflow.compile()

study_buddy_graph = create_study_buddy_graph()