from state import AgentState, ProjectPlan
from agents import PlanAgents
from tools import PlanTools
from langchain_core.messages import HumanMessage
import json
from datetime import datetime
class PlanningAgent:
    """Main planning agent that generates and refines project plans."""
    
    @staticmethod
    def plan_node(state: AgentState) -> dict:
        print(f"\n[PLANNER NODE RUNNING]")
        
        # --- KEY CHANGE: Use specific instructions if available ---
        # The Orchestrator might say: "Create a plan for Python learning using these search results: ..."
        instruction = state.get("task_instructions")
        if not instruction:
            instruction = state.get("user_prompt")
            
        print(f"  👉 processing instructions: {instruction[:100]}...")
        
        # 1. Determine if this is an initial plan or a refinement
        errors = state.get("validation_errors", [])
        plan_in_state: ProjectPlan = state.get("parsed_output")
        is_refinement = bool(errors) and plan_in_state is not None
        
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
            # Pass the DYNAMIC instruction, not just user_prompt
            invoke_args = {"user_prompt": instruction} 
            log_action = "generate_plan"
            log_details = {"instructions": instruction[:100]}
            
        # 3. Create log entry
        log_entry = PlanTools.create_execution_log_entry("planner", log_action, log_details)
        if "execution_log" not in state: state["execution_log"] = []
        state["execution_log"].append(log_entry)
        
        try:
            llm_response = agent.invoke(invoke_args)
            plan = PlanTools.parse_llm_output(llm_response.content)
            plan_summary = PlanTools.create_summary(plan)
            formatted_plan = PlanTools.format_plan_for_display(plan)
            
            # SAVE THE PLAN AS JSON
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plan_filename = f"roadmap_{timestamp}.json"
            with open(plan_filename, 'w', encoding='utf-8') as f:
                json.dump(plan.model_dump(), f, indent=2, ensure_ascii=False)
            
            log_entry["details"]["status"] = "success"
            log_entry["details"]["saved_file"] = plan_filename
            
            return {
                "messages": [HumanMessage(content=formatted_plan, name="Planner")],
                "parsed_output": plan,
                "plan_summary": plan_summary,
                "execution_log": state["execution_log"],
                "validation_errors": [],
                "last_node": "planner",
                "saved_plan_file": plan_filename,  # NEW: Save the filename
                "plan_data": plan.model_dump()     # NEW: Keep the plan data
            }
            
        except Exception as e:
            log_entry["details"]["status"] = "error"
            log_entry["details"]["error"] = str(e)
            state["execution_log"].append(log_entry)
            
            error_msg = f"❌ Plan generation/refinement failed: {str(e)[:200]}. Passing error to Orchestrator."
            print(f"  Error: {str(e)[:100]}")
            
            # Set a distinct state to inform orchestrator of failure
            return {
                "messages": [HumanMessage(content=str(e), name="Planner")],
                "execution_log": state["execution_log"],
                "parsed_output": None,
                "last_node": "planner_error" 
            }