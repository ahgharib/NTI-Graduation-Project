from langgraph.graph import StateGraph, END
from state import AgentState
from orchestrator import Orchestrator
from planner_agent import PlanningAgent
from quiz_agent import quiz_node
from explainer_agent import explainer_node
from log import log_and_print

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
    """Create and configure the Study Buddy graph."""
    workflow = StateGraph(AgentState)
    orchestrator = Orchestrator()
    
    # Add nodes
    workflow.add_node("orchestrator", orchestrator.build_plan_node)
    workflow.add_node("scheduler", scheduler)
    workflow.add_node("planner", PlanningAgent.plan_node)
    workflow.add_node("quiz_generator", quiz_node)
    workflow.add_node("explain_node", explainer_node)
    
    # Configure edges
    workflow.set_entry_point("orchestrator")
    workflow.add_edge("orchestrator", "scheduler")
    workflow.add_edge("planner", "scheduler")
    workflow.add_edge("quiz_generator", "scheduler")
    workflow.add_edge("explain_node", "scheduler")
    
    # Conditional routing from scheduler
    workflow.add_conditional_edges(
        "scheduler",
        lambda state: state["next"] if "next" in state else "END",
        {
            "planner": "planner",
            "quiz_generator": "quiz_generator", 
            "explain_node": "explain_node",
            "END": END
        }
    )
    
    return workflow.compile()

# Export the graph creation function
def get_study_buddy_app():
    """Get the compiled Study Buddy application."""
    return create_study_buddy_graph()