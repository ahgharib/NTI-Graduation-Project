from typing import List, Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from config import Config
from chat_state import AgentState # Changed import
from log import universal_debug_log # Import the new logger


NodeName = Literal["quiz_generator", "explain_node", "END"]

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
        
        milestone_context = state.get("selected_milestone_context", "No specific milestone selected.")

        system_prompt = """You are the Orchestrator for a Study Buddy AI.
        
        AVAILABLE WORKERS:
        1. quiz_generator: Creates quizzes (MCQ/Coding). (Use for: "Test me", "Quiz me")
        2. explainer: Explain complex topics (use for: explain, search on , give me an explaination)
        3. END: terminate the Chat

        CAPABILITIES:
        - All workers have access to Web Search and YouTube Search tools. 
        - DO NOT create a separate "search" node. Assign the research task to the relevant worker in your instructions for that worker.
        
        YOUR TASK:
        Break the user input into a linear sequence of steps.
        
        SCENARIOS:
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

        Input: "Create a 12-week Agentic AI roadmap. Do NOT include 'planning' or 'roadmap creation' as tasks; assume this output IS the final actionable plan. Focus 100 percent on specific technical milestones (e.g., 'Mastering LangGraph', 'Implementing Tool-Use')."
        Output:
          actions: ["END"]
          instructions: ["Sorry I cannot Do this Task"]
        
        Input: "Create a video About taking photos"
        Output:
          actions: ["END"]
          instructions: ["Sorry I cannot Do this Task"]

        Notes:
          A task can be called multiple times If the Topics are different
          Know that each WORKERS has the ability to search the internet or video search
          If the User Asks for something that is not within the CAPABILITIES of any Worker like asking to create a plan or make a video, ...... then: actions = ["END"]
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