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
    """The orchestrator's decision on the next high-level action."""
    action: str = Field(
        ..., 
        description="Next action. 'plan_project' for roadmaps, 'generate_quiz' for tests, 'search_videos' for YouTube, 'terminate_impossible' otherwise."
    )
    thought: str = Field(..., description="Reasoning for choosing this action.")

class Orchestrator:
    """Main orchestrator that routes between agents and manages the planning/quiz workflow."""
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
        
        # --- PHASE 3: Return from Quiz Generator ---
        elif last_node == "quiz_generator":
            # The orchestrator sees the new output from quiz
            print("  ✅ Quiz received by Orchestrator. Finishing workflow.")
            message = state.get("final_output", "Quiz generation completed successfully.")
            return OrchestrationTools.send_user_message(message)
        
        # --- PHASE 4: Return from Youtube Generator ---
        elif last_node == "youtube_search":
            print("  ✅ Video links received by Orchestrator.")
            message = state.get("final_output")
            return OrchestrationTools.send_user_message(message)
            
        # --- PHASE 4: Fallback ---
        else:
            message = f"Orchestrator received state from unknown node: {last_node}. Terminating gracefully."
            return OrchestrationTools.send_user_message(message)

    def _handle_initial_call(self, state: AgentState) -> dict:
        """Analyzes the user prompt and decides the first step."""
        user_prompt = state.get("user_prompt", PlanTools.extract_user_prompt(state.get("messages", [])))
        state["user_prompt"] = user_prompt 
        state["refinement_attempts"] = 0 
        
        print(f"  Initial analysis for prompt: {user_prompt[:50]}...")
        
        try:
            # FIX 1: Update instructions so the LLM knows when to search
            analysis_prompt = (
                f"Analyze the user request: '{user_prompt}'\n"
                "Decision Criteria:\n"
                "- Choose 'plan_project' if they want a roadmap, study plan, or project structure.\n"
                "- Choose 'generate_quiz' if they want questions, a test, or MCQs.\n"
                "- Choose 'search_videos' if they specifically ask for YouTube videos, tutorials, or visual content.\n" # NEW
                "- Choose 'terminate_impossible' if the request is vague or harmful."
            )
            
            prompt_analysis: RoutingDecision = self.router_llm.invoke(analysis_prompt)
            
            # FIX 2: Add 'search_videos' to the validation set
            valid_actions = {"plan_project", "generate_quiz", "search_videos", "terminate_impossible"}
            action = prompt_analysis.action
            
            if action not in valid_actions:
                print(f"  LLM Decision: {action} (INVALID). Defaulting to Planner.")
                action = "plan_project"
            else:
                 print(f"  LLM Decision: {action} | Thought: {prompt_analysis.thought}")
            
            # ROUTING LOGIC
            if action == "plan_project":
                return {"next": "planner", "last_node": "orchestrator"}
            
            elif action == "generate_quiz":
                return {"next": "quiz_generator", "last_node": "orchestrator"}
            
            elif action == "search_videos": # This will now be reached correctly
                print("  Routing to YouTube Search.")
                return {"next": "youtube_search", "last_node": "orchestrator"}
            
            elif action == "terminate_impossible":
                message = "The orchestrator determined the task is not supported. Please provide a clearer prompt."
                return OrchestrationTools.send_user_message(message)

        except Exception as e:
            # Fallback
            return {"next": "planner", "last_node": "orchestrator"}

    def _handle_planner_return(self, state: AgentState) -> dict:
        """Handles the state after the Planner has run (generation or refinement)."""
        plan: ProjectPlan = state.get("parsed_output")
        attempts = state.get("refinement_attempts", 0)
        
        if state.get("last_node") == "planner_error":
            attempts += 1
            if attempts >= self.MAX_ATTEMPTS:
                message = f"❌ Planner failed after {self.MAX_ATTEMPTS} attempts."
                return OrchestrationTools.send_user_message(message)
            else:
                return {"next": "planner", "last_node": "orchestrator", "refinement_attempts": attempts}

        if not plan:
            return OrchestrationTools.send_user_message("❌ Planner finished but returned no plan.")

        # Validation logic remains the same
        is_valid, errors = ValidationTools.validate_plan(plan)
        if is_valid:
            print(f"  ✅ Validation SUCCESS. Saving and Ending.")
            return OrchestrationTools.save_plan_and_end(plan)
        else:
            attempts += 1
            if attempts >= self.MAX_ATTEMPTS:
                error_list = "\n".join([f"- {e}" for e in errors])
                return OrchestrationTools.send_user_message(f"⚠️ **PLANNING FAILED** after max attempts.\n{error_list}")
            else:
                return {
                    "next": "planner", 
                    "last_node": "orchestrator", 
                    "refinement_attempts": attempts,
                    "validation_errors": errors 
                }