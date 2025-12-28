from typing import List, Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from config import Config
from chat_state import AgentState 
from log import universal_debug_log 

# --- UPDATED: Added "summarizer" to valid nodes ---
NodeName = Literal["quiz_generator", "explain_node", "summarizer", "END"]

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
        
        history_summary = state.get("conversation_summary", "No previous context.")
        milestone_context = state.get("selected_milestone_context", "No specific milestone selected.")

        system_prompt = """You are the Orchestrator for a Study Buddy AI.
        
        AVAILABLE WORKERS:
        1. quiz_generator: Creates quizzes (MCQ/Coding). (Use for: Test me, Quiz me, where am I in..)
        2. explain_node: Explain complex topics (use for: explain, search on, What is.., Teach me about..)
        3. summarizer: Summarize content or documents. (Use for: summarize this, give me an overview, simplify this paragraph, summarize page X)
        4. END: Terminate the Chat.

        CAPABILITIES & CONSTRAINTS:
        - 'quiz_generator' and 'explain_node' have Web Search access.
        - 'summarizer' relies ONLY on Document Context (RAG) and the text YOU provide in the instruction.
        - IMPORTANT: Workers are STATELESS. They cannot see the chat history. 
        - RULE: If a user request is dependent on previous context (e.g., "Summarize that", "Quiz me on the topic we discussed"), you MUST extract the relevant details from the 'CONTEXT (SUMMARIZED)' section and include them explicitly in the worker's instruction.

        CONTEXT (SUMMARIZED FROM HISTORY):
        ------------------------------------------------
        {history_summary}
        ------------------------------------------------

        CURRENT USER REQUEST:
        {input}

        SCENARIOS:
        Input: "Summarize your last response."
        Output:
          actions: ["summarizer"]
          instructions: ["Summarize the following information discussed previously: [Orchestrator: Insert the key points of the last assistant message here from history_summary]"]

        Input: "Quiz me on what we just talked about."
        Output:
          actions: ["quiz_generator"]
          instructions: ["Generate a quiz based on the topic of [Topic Name], specifically focusing on [Key Details from context]."]

        Input: "Explain recursion then summarize it."
        Output:
          actions: ["explain_node", "summarizer"]
          instructions: ["Explain the programming concept of recursion with examples.", "Summarize the explanation of recursion provided in the previous step."]

        Notes:
          - You act as the memory for the workers. If the worker needs to know 'what' to summarize or 'what' to quiz, tell them exactly in the instruction.
          - For document-specific requests (e.g., "Page 5"), specify the source so RAG can trigger.
          - Always include the Milestone context if applicable: {milestone_context}
        """

        structured_llm = self.llm.with_structured_output(OrchestratorPlan)
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])
        
        chain = prompt | structured_llm
        plan: OrchestratorPlan = chain.invoke({
            "input": user_input,
            "history_summary": history_summary,
            "milestone_context": milestone_context
        })
        
        universal_debug_log(node_name, "PLAN_GENERATED", plan.dict())
        
        print(f"📋 PLAN: {list(zip(plan.actions, plan.instructions))}")
        
        return {
            "plan_actions": plan.actions,
            "plan_instructions": plan.instructions,
            "next": "scheduler" 
        }