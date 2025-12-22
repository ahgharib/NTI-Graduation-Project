import json
import re
from datetime import datetime
from typing import Dict, Any

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from state import AgentState, ProjectPlan
from config import Config
from tools import PlanTools, web_search_tool, youtube_search_tool
from log import universal_debug_log, prepare_context

class PlanningAgent:
    """Main planning agent node for the LangGraph workflow."""
    
    @staticmethod
    def plan_node(state: AgentState) -> dict:
        node_name = "PLANNER_NODE"
        instruction = state.get("current_instruction", state.get("user_prompt", ""))
        
        universal_debug_log(node_name, "START", {"instruction": instruction})
        
        # Check for refinement needs
        errors = state.get("validation_errors", [])
        plan_in_state = state.get("parsed_output")
        is_refinement = bool(errors) and plan_in_state is not None
        
        # Create appropriate prompt
        if is_refinement:
            attempts = state.get("refinement_attempts", 1)
            universal_debug_log(node_name, "REFINEMENT", {"errors": errors, "attempt": attempts})
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a project plan refiner. Fix the validation errors.
Original plan: {original_plan}
Validation errors: {validation_errors}
Output ONLY the corrected JSON in the ProjectPlan format."""),
                ("human", "Please fix the project plan and output valid JSON.")
            ])
            
            llm = Config.get_gemini_llm()
            chain = prompt | llm
            invoke_args = {
                "original_plan": json.dumps(plan_in_state.model_dump(), indent=2),
                "validation_errors": "\n".join([f"- {e}" for e in errors])
            }
        else:
            universal_debug_log(node_name, "INITIAL_PLANNING", {"instruction": instruction})
            
            # Use tools for research if needed
            context = prepare_context(state)
            research = ""
            if any(keyword in instruction.lower() for keyword in ["search", "research", "find", "look up"]):
                research = web_search_tool.invoke(instruction)
            
            # FIXED: Use double curly braces {{ }} to escape JSON in the template
            system_prompt = """You are a professional project planner. Your task is to create a detailed project plan.
            
            Output format must be valid JSON:
{{
    "goal": "Overall project goal",
    "duration": "Estimated total duration",
    "milestones": [
        {{
            "title": "Milestone title",
            "description": "What this milestone achieves",
            "difficulty": "Easy|Medium|Hard|Expert",
            "duration": "Duration",
            "tasks": [
                {{
                    "description": "Task description",
                    "difficulty": "Easy|Medium|Hard|Expert",
                    "duration": "Duration"
                }}
            ]
        }}
    ]
}}"""
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "Create a project plan for: {user_prompt}\n\nResearch Context: {research}\n\nPrevious Context: {context}")
            ])
            
            llm = Config.get_ollama_llm()
            chain = prompt | llm
            invoke_args = {
                "user_prompt": instruction,
                "research": research,
                "context": context
            }
        
        # Execute the chain
        try:
            llm_response = chain.invoke(invoke_args)
            plan = PlanTools.parse_llm_output(llm_response.content)
            
            # Create formatted output
            formatted_plan = PlanTools.format_plan_for_display(plan)
            plan_summary = PlanTools.create_summary(plan)
            
            # Save plan to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plan_filename = f"roadmap_{timestamp}.json"
            with open(plan_filename, 'w', encoding='utf-8') as f:
                json.dump(plan.model_dump(), f, indent=2, ensure_ascii=False)
            
            universal_debug_log(node_name, "SUCCESS", {"filename": plan_filename, "summary": plan_summary})
            
            return {
                "messages": [HumanMessage(content=formatted_plan, name="Planner")],
                "parsed_output": plan,
                "plan_summary": plan_summary,
                "validation_errors": [],
                "saved_plan_file": plan_filename,
                "plan_data": plan.model_dump(),
                "research_memory": [f"Plan created: {plan_summary}"],
                "execution_log": state.get("execution_log", []) + [
                    PlanTools.create_execution_log_entry(
                        "planner", 
                        "refine_plan" if is_refinement else "generate_plan", 
                        {"status": "success", "filename": plan_filename, "instruction": instruction[:100]}
                    )
                ]
            }
            
        except Exception as e:
            error_msg = f"Planning error: {str(e)[:200]}"
            universal_debug_log(node_name, "ERROR", {"error": error_msg})
            
            return {
                "messages": [HumanMessage(content=error_msg, name="Planner")],
                "validation_errors": [error_msg],
                "parsed_output": None,
                "execution_log": state.get("execution_log", []) + [
                    PlanTools.create_execution_log_entry(
                        "planner",
                        "refine_plan" if is_refinement else "generate_plan",
                        {"status": "error", "error": str(e)[:200], "instruction": instruction[:100]}
                    )
                ]
            }