from config import Config
from state import AgentState, Quiz
from tools import web_search_tool
from log import prepare_context, universal_debug_log
from langchain_core.messages import HumanMessage

def quiz_node(state: AgentState):
    node_name = "QUIZ_AGENT"
    instruction = state["current_instruction"]
    context = prepare_context(state)
    
    # Uses the structured output logic from your Real Node code
    # We use Gemini for structured quiz generation as in your config
    llm = Config.get_gemini_llm().with_structured_output(Quiz)
    
    # 1. Gather fresh data if necessary
    research = web_search_tool.invoke(f"Questions and answers for {instruction}")
    
    prompt = f"""
    Generate a quiz based on: {instruction}
    Context: {context}
    New Research: {research}
    
    Follow the Quiz schema (MCQs, Articles, Coding).
    """
    
    quiz_output = llm.invoke(prompt)
    universal_debug_log(node_name, "QUIZ_GENERATED", quiz_output.dict())
    
    return {
        "quiz_output": quiz_output,
        "messages": [HumanMessage(content=f"Generated quiz for {quiz_output.topic}", name="QuizAgent")],
        "research_memory": [f"Quiz generated for {instruction}"]
    }