from typing import List, Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from config import Config
from state import AgentState
from log import universal_debug_log # Import the new logger


NodeName = Literal["planner", "quiz_generator", "explain_node"]

class OrchestratorPlan(BaseModel):
    """The plan containing the sequence of nodes and their instructions."""
    actions: List[NodeName] = Field(..., description="The sequence of worker nodes to call.")
    instructions: List[str] = Field(..., description="Specific instructions for each node.")

class Orchestrator:
    def __init__(self):
        self.llm = Config.get_ollama_llm() 

    def build_plan_node(self, state: AgentState) -> dict:
        node_name = "ORCHESTRATOR"
        user_input = state["user_prompt"]
        
        # LOG FULL INPUT
        universal_debug_log(node_name, "FULL_INPUT", {"user_prompt": user_input, "state": state})
        print(f"\n🧠 [ORCHESTRATOR] Planning for: {user_input}")

        system_prompt = """You are the Orchestrator for a Study Buddy AI.
        
        AVAILABLE WORKERS:
        1. planner: Creates comprehensive study roadmaps/project plans. (Use for: "Make a plan", "roadmap", "curriculum")
        2. quiz_generator: Creates quizzes (MCQ/Coding). (Use for: "Test me", "Quiz me")
        3. explainer: Explain complex topics (use for: explain, search on , give me an explaination)

        CAPABILITIES:
        - All workers have access to Web Search and YouTube Search tools. 
        - DO NOT create a separate "search" node. Assign the research task to the relevant worker in your instructions for that worker.
        
        YOUR TASK:
        Break the user input into a linear sequence of steps.
        
        SCENARIOS:
        Input: "Make a study plan for Python then give me a quiz."
        Output:
          actions: ["planner", "quiz_generator"]
          instructions: ["Create a beginner Python study roadmap.", "Generate a Python quiz based on the roadmap."]

        Input: "Find videos about React hooks and explain them."
        Output:
          actions: ["explain_node"]
          instructions: ["Search for videos on React hooks and explain the core concepts found."]

        Input: "Find NLP interview questions and make a quiz about them."
        Output:
          actions: ["quiz"]
          instructions: ["Search for NLP interview questions and make me a quiz from the gained information."]

        
        Input: "Research Quantum Physics and make a quiz about it."
        Output:
          actions: ["explain_node", "quiz_generator"]
          instructions: ["Research the basics of Quantum Physics using web search.", "Create a quiz based on the Quantum Physics research."]
        
        Input: "explain recursion, explain dynamic programming, exlpain basic data structure concepts."
        Output:
          actions: ["explain_node", "explain_node", "explain_node"]
          instructions: ["Explain  recursion", "Explain dynamic programming", "Explain data structure concepts"]

        Input: "make a 3 quizez one on Agentic AI, another on Computer Vision, another on NLP. Note Use the Web (Internet) search to find information on each task before making the quiz"
        Output:
          actions: ["quiz_generator", "quiz_generator", "quiz_generator"]
          instructions: ["Search and generate a quiz on Agentic AI", "Search and generate a quiz on Computer Vision", "Search and generate a quiz on NLP"]


        Notes:
          A task can be called multiple times If the Topics are different
          Know that each WORKERS has the ability to search the internet or video search
        """

        structured_llm = self.llm.with_structured_output(OrchestratorPlan)
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])
        
        chain = prompt | structured_llm
        plan: OrchestratorPlan = chain.invoke({"input": user_input})
        
        universal_debug_log(node_name, "PLAN_GENERATED", plan.dict())
        
        print(f"📋 PLAN: {list(zip(plan.actions, plan.instructions))}")
        
        return {
            "plan_actions": plan.actions,
            "plan_instructions": plan.instructions,
            "next": "scheduler" # Hand off to the scheduler
        }