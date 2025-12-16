from langchain_core.prompts import ChatPromptTemplate
from config import Config
import json

class PlanAgents:
    """Agents for different stages of planning using Gemini"""
    
    # System prompt for planning - FIXED: escaped curly braces
    SYSTEM_PROMPT = """You are a professional project planner. Your task is to create a detailed project plan based on the user's request.

You MUST output a valid JSON object with the following structure:
{{
    "goal": "Overall project goal",
    "duration": "Estimated total duration (e.g., '2 months', '6 weeks')",
    "milestones": [
        {{
            "title": "Milestone title",
            "description": "What this milestone achieves",
            "difficulty": "Easy|Medium|Hard|Expert",
            "duration": "Estimated duration (e.g., '2 days', '1 week')",
            "tasks": [
                {{
                    "description": "Detailed task description",
                    "difficulty": "Easy|Medium|Hard|Expert",
                    "duration": "Estimated duration (e.g., '2 hours', '1 day')"
                }}
            ]
        }}
    ]
}}

Guidelines:
1. Create 3-5 meaningful milestones
2. Each milestone should have 2-4 specific tasks
3. Tasks should be actionable and measurable
4. Difficulty levels: Easy (simple, quick tasks), Medium (moderate complexity), Hard (requires expertise), Expert (complex, may need research)
5. Duration should be realistic estimates
6. Ensure the plan is comprehensive and achievable
7. Include duration at both project and milestone levels"""

    @staticmethod
    def create_planning_agent():
        """Agent that creates the initial project plan"""
        llm = Config.get_ollama_llm()
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", PlanAgents.SYSTEM_PROMPT),
            ("human", "Create a project plan for: {user_prompt}")
        ])
        
        return prompt | llm
    
    @staticmethod
    def create_refinement_agent():
        """Agent that refines the plan based on validation errors"""
        llm = Config.get_gemini_llm()
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a project plan refiner. Fix the following validation errors in the project plan.
Original plan: {original_plan}
Validation errors: {validation_errors}

Output ONLY the corrected JSON with the same structure."""),
            ("human", "Please fix the project plan and output valid JSON.")
        ])
        
        return prompt | llm