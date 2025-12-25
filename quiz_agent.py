from config import Config
from chat_state import AgentState, Quiz
from chat_tools import web_search_tool
from log import prepare_context, universal_debug_log
from langchain_core.messages import HumanMessage

def quiz_node(state: AgentState):
    node_name = "QUIZ_AGENT"
    instruction = state["current_instruction"]
    context = prepare_context(state)
    milestone_context = state.get("selected_milestone_context", "")
    
    # Uses the structured output logic from your Real Node code
    # We use Gemini for structured quiz generation as in your config
    llm = Config.get_gemini_llm().with_structured_output(Quiz)
    
    # 1. Gather fresh data if necessary
    research = web_search_tool.invoke(f"Questions and answers for {instruction}")
    
    prompt = f"""
    Generate a quiz based on: {instruction}
    Context: {context}
    New Research: {research}
    Entire Roadmap: {state.get("plan_data")}
    if the User selected a Specific Milestone then the User want you to focus on this milestone or a task in this milestone then you Should search for the Milestone in the "Entire Roadmap" above based on the ID and Tailor your Answer to this milestone
    and here is the Milestone ID: {milestone_context}
    
    Follow the Quiz schema (MCQs, Articles, Coding).
    """
    
    quiz_output = llm.invoke(prompt)
    # universal_debug_log(node_name, "QUIZ_GENERATED", quiz_output.dict())
    
    return {
        "quiz_output": quiz_output,
        "messages": [HumanMessage(content=f"Generated quiz for {quiz_output.topic}", name="QuizAgent")],
        "research_memory": [f"Quiz generated for {instruction}"]
    }