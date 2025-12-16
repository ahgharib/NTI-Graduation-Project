import os
import json
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from state import AgentState, ProjectPlan
from langchain_core.messages import HumanMessage
# Assuming tools.py contains PlanTools, ValidationTools, and OrchestrationTools
from tools import PlanTools, ValidationTools, OrchestrationTools 
from config import Config
from langchain_core.prompts import ChatPromptTemplate

class RoutingDecision(BaseModel):
    """The orchestrator's decision on the next high-level action based on prompt analysis."""
    action: str = Field(
        ..., 
        description="The next action to take. Must be one of: 'plan_project', 'terminate_impossible'."
    )
    thought: str = Field(
        ..., 
        description="A brief explanation of why this action was chosen."
    )

class Orchestrator:
    """Main orchestrator that routes between agents and manages the planning loop."""
    MAX_ATTEMPTS = 3
    
    def __init__(self):
        # Using LLama for Orchestration (as requested)
        self.llm = Config.get_ollama_llm()
        # Use structured output for initial prompt analysis only
        self.router_llm = self.llm.with_structured_output(RoutingDecision)
        
    def supervisor_node(self, state: AgentState) -> dict:
        """The central agent that manages the workflow."""
        print(f"\n--- ORCHESTRATOR RUNNING ---")
        
        last_node = state.get("last_node", "Start")
        
        # Ensure refinement attempts counter is initialized
        if "refinement_attempts" not in state:
            state["refinement_attempts"] = 0
        
        # Add log entry
        log_entry = PlanTools.create_execution_log_entry("orchestrator", "supervisor_run", {"from_node": last_node, "attempts": state["refinement_attempts"]})
        if "execution_log" not in state: state["execution_log"] = []
        state["execution_log"].append(log_entry)
        
        # --- PHASE 1: Initial Call (Start) ---
        if last_node == "Start" or last_node.startswith("orchestrator_end_"):
            return self._handle_initial_call(state)
        
        # --- PHASE 2: Return from Planner ---
        elif last_node in ("planner", "planner_error"):
            return self._handle_planner_return(state)
            
        # --- PHASE 3: Fallback ---
        else:
            message = f"Orchestrator received state from unknown node: {last_node}. Terminating gracefully."
            return OrchestrationTools.send_user_message(message)

    def _handle_initial_call(self, state: AgentState) -> dict:
        """Analyzes the user prompt and decides the first step."""
        user_prompt = state.get("user_prompt", PlanTools.extract_user_prompt(state.get("messages", [])))
        state["user_prompt"] = user_prompt # Ensure prompt is set
        state["refinement_attempts"] = 0 # Reset attempts for a new task
        
        print(f"  Initial analysis for prompt: {user_prompt[:50]}...")
        
        # Analyze prompt using the LLM for high-level decision
        try:
            # IMPROVED PROMPT for stricter routing
            analysis_prompt = (
                f"Analyze the user's request: '{user_prompt}'. "
                f"The 'planner' node creates detailed project plans with milestones and tasks. "
                f"If the request is for a **roadmap, study plan, course structure, or a detailed project plan**, you MUST choose 'plan_project'. "
                f"If the request is a non-planning command (like 'close chat', 'search the internet', 'summarize this'), you MUST choose 'terminate_impossible'. "
                f"Output one of: 'plan_project' or 'terminate_impossible'."
            )
            
            prompt_analysis: RoutingDecision = self.router_llm.invoke(analysis_prompt)
            
            # Sanitize output against invalid actions the LLM might hallucinate
            valid_actions = {"plan_project", "terminate_impossible"}
            action = prompt_analysis.action
            
            if action not in valid_actions:
                # Default to planning on invalid/hallucinated action
                print(f"  LLM Decision: {action} (INVALID). Thought: {prompt_analysis.thought}. Defaulting to Planner.")
                action = "plan_project"
            else:
                 print(f"  LLM Decision: {action} | Thought: {prompt_analysis.thought}")
            
            if action == "plan_project":
                print("  Routing to Planner.")
                return {"next": "planner", "last_node": "orchestrator"}
            
            elif action == "terminate_impossible":
                message = "The orchestrator has determined that the requested task is not a project planning request or is impossible with the current available nodes (Planner). Task aborted. Please provide a clear project prompt."
                return OrchestrationTools.send_user_message(message)

        except Exception as e:
            message = f"Orchestrator failed LLM analysis: {str(e)[:100]}. Routing to Planner by default."
            print(message)
            return {"next": "planner", "last_node": "orchestrator"}

    def _handle_planner_return(self, state: AgentState) -> dict:
        """Handles the state after the Planner has run (generation or refinement)."""
        
        plan: ProjectPlan = state.get("parsed_output")
        attempts = state.get("refinement_attempts", 0)
        
        # Case A: Planner reported an internal error (e.g., failed JSON parsing, API Error)
        if state.get("last_node") == "planner_error":
            attempts += 1 # Increment attempt counter
            
            if attempts >= self.MAX_ATTEMPTS:
                message = f"❌ Planner failed to generate valid structured output after {self.MAX_ATTEMPTS} attempts due to internal errors (e.g., API issues). Please modify your prompt or check the API status."
                return OrchestrationTools.send_user_message(message)
            else:
                print(f"  Planner failed (Attempt {attempts}/{self.MAX_ATTEMPTS}). Rerouting to Planner.")
                # FIX: Return updated attempts counter
                return {"next": "planner", "last_node": "orchestrator", "refinement_attempts": attempts}


        # Case B: Planner returned a plan (success) -> Run Validation
        if not plan:
            # Should only happen if 'planner' returned successfully but output was None (a safeguard)
            message = "❌ Planner finished but returned no parsed plan. Terminating."
            return OrchestrationTools.send_user_message(message)

        # 1. Validation
        is_valid, errors = ValidationTools.validate_plan(plan)
        
        log_entry = PlanTools.create_execution_log_entry("orchestrator", "validation_check", {"is_valid": is_valid, "error_count": len(errors), "attempt": attempts + 1})
        state["execution_log"].append(log_entry)
        
        if is_valid:
            # 2. Validation Success -> Save JSON and End
            print(f"  ✅ Validation SUCCESS. Saving and Ending.")
            # Tool handles state update and routing to END
            return OrchestrationTools.save_plan_and_end(plan)
            
        else:
            # 3. Validation FAILURE -> Try Refinement Loop
            attempts += 1 # Increment attempt counter
            
            print(f"  ❌ Validation FAILED ({len(errors)} errors). Attempt {attempts}/{self.MAX_ATTEMPTS}.")

            if attempts >= self.MAX_ATTEMPTS:
                print("  Max attempts reached. Sending message to user and ending.")
                # Max attempts reached -> Send message to user and route to END
                error_list = "\n".join([f"- {e}" for e in errors])
                message = (
                    f"⚠️ **PLANNING FAILED**\n\n"
                    f"The project planner could not generate a valid plan after {self.MAX_ATTEMPTS} attempts.\n"
                    f"**Remaining Errors:**\n{error_list}\n\n"
                    f"Please refine your original prompt to be more specific or change the constraints."
                )
                return OrchestrationTools.send_user_message(message)
            else:
                # FIX: Return all updated state fields to enable the loop
                print(f"  Routing back to Planner for refinement (Attempt {attempts}/{self.MAX_ATTEMPTS}).")
                return {
                    "next": "planner", 
                    "last_node": "orchestrator", 
                    "refinement_attempts": attempts,
                    "validation_errors": errors # Pass errors so the Planner knows to refine
                }