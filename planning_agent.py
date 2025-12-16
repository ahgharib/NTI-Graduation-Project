from state import AgentState, ProjectPlan
from agents import PlanAgents
from tools import PlanTools # Use PlanTools for parsing/logging/formatting
from langchain_core.messages import HumanMessage
import json
from typing import Dict, Any

class PlanningAgent:
    """Main planning agent that generates and refines project plans."""
    
    @staticmethod
    def plan_node(state: AgentState) -> dict:
        """Generate or refine a project plan based on the current state."""
        print(f"\n[PLANNER NODE RUNNING]")
        
        # 1. Determine if this is an initial plan or a refinement
        errors = state.get("validation_errors", [])
        plan_in_state: ProjectPlan = state.get("parsed_output")
        is_refinement = bool(errors) and plan_in_state is not None
        
        user_prompt = state.get("user_prompt", PlanTools.extract_user_prompt(state.get("messages", [])))
        
        # 2. Select appropriate agent (Initial or Refinement)
        if is_refinement:
            attempts = state.get("refinement_attempts", 1)
            print(f"  Mode: Refinement (Errors: {len(errors)}, Attempt: {attempts})")
            agent = PlanAgents.create_refinement_agent()
            invoke_args = {
                "original_plan": json.dumps(plan_in_state.model_dump(), indent=2),
                "validation_errors": "\n".join([f"- {e}" for e in errors])
            }
            log_action = "refine_plan"
            log_details = {"error_count": len(errors)}
        else:
            print("  Mode: Initial Planning")
            agent = PlanAgents.create_planning_agent()
            invoke_args = {"user_prompt": user_prompt}
            log_action = "generate_plan"
            log_details = {"user_prompt": user_prompt[:100]}
            
        # 3. Create log entry
        log_entry = PlanTools.create_execution_log_entry("planner", log_action, log_details)
        if "execution_log" not in state: state["execution_log"] = []
        state["execution_log"].append(log_entry)
        
        # 4. Invoke LLM
        try:
            llm_response = agent.invoke(invoke_args)
            
            # 5. Parse the response
            plan = PlanTools.parse_llm_output(llm_response.content)
            
            # 6. Update log and state
            plan_summary = PlanTools.create_summary(plan)
            formatted_plan = PlanTools.format_plan_for_display(plan)
            
            log_entry["details"]["status"] = "success"
            
            # NOTE: We do NOT set the "next" field. The orchestrator determines the next step.
            return {
                "messages": [HumanMessage(content=formatted_plan, name="Planner")],
                "parsed_output": plan,
                "plan_summary": plan_summary,
                "execution_log": state["execution_log"],
                "validation_errors": [], # Clear errors for next validation check
                "last_node": "planner" # Signal to orchestrator that plan was generated
            }
            
        except Exception as e:
            log_entry["details"]["status"] = "error"
            log_entry["details"]["error"] = str(e)
            state["execution_log"].append(log_entry)
            
            error_msg = f"❌ Plan generation/refinement failed: {str(e)[:200]}. Passing error to Orchestrator."
            print(f"  Error: {str(e)[:100]}")
            
            # Set a distinct state to inform orchestrator of failure
            return {
                "messages": [HumanMessage(content=error_msg, name="Planner")],
                "execution_log": state["execution_log"],
                "parsed_output": None,
                "last_node": "planner_error" 
            }