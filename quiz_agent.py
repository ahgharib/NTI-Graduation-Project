from config import Config
from chat_state import AgentState, Quiz
from chat_tools import web_search_tool
from log import prepare_context, universal_debug_log
from langchain_core.messages import HumanMessage
from RAG.rag import get_context_chunks

def quiz_node(state: AgentState):
    node_name = "QUIZ_AGENT"
    instruction = state["current_instruction"]
    context = prepare_context(state)
    milestone_context = state.get("selected_milestone_context", "")

    document_context = get_context_chunks(instruction)

    if document_context:
        research = ""
    else:
        research = web_search_tool.invoke(
            f"quiz questions and answers about {instruction}"
        )
    
    # Uses the structured output logic from your Real Node code
    # We use Gemini for structured quiz generation as in your config
    llm = Config.get_gemini_llm().with_structured_output(Quiz)
    
    
    prompt = f"""
    Generate a quiz based on this instruction: {instruction}
    Context: {context}
    New Research: {research}
    Entire Roadmap: {state.get("plan_data")}
    if the User selected a Specific Milestone then the User want you to focus on this milestone or a task in this milestone then you Should search for the Milestone in the "Entire Roadmap" above based on the ID and Tailor your Answer to this milestone
    and here is the Milestone ID: {milestone_context}
    
    Follow the Quiz schema (MCQs, Articles, Coding).
    """
    prompt = f"""
    You are an expert educator and assessment designer.

    TASK:
    Generate a high-quality quiz based on the user's instruction.

    USER INSTRUCTION:
    {instruction}

    DOCUMENT CONTEXT (highest priority):
    {document_context if document_context else "No relevant document content found."}

    WEB SEARCH CONTEXT (use only if document context is insufficient):
    {research if research else "Web search not required."}

    PRIOR CONVERSATION CONTEXT:
    {context}

    ROADMAP CONTEXT:
    {state.get("plan_data")}

    SELECTED MILESTONE CONTEXT:
    {milestone_context}

    --- RULES ---
    1. Base the quiz primarily on the DOCUMENT CONTEXT when available.
    2. Use WEB SEARCH data only if the document does not sufficiently cover the topic.
    3. Use internal knowledge ONLY if explicitly required or if both document and web data are insufficient.
    4. Do NOT invent facts or questions not supported by the provided sources.
    5. Ensure all questions are accurate, unambiguous, and educational.
    6. Follow the Quiz schema exactly:
    - MCQs
    - Article questions
    - Coding questions
    """    
    
    quiz_output = llm.invoke(prompt)
    # universal_debug_log(node_name, "QUIZ_GENERATED", quiz_output.dict())
    
    return {
        "quiz_output": quiz_output,
        "messages": [HumanMessage(content=f"Generated quiz for {quiz_output.topic}", name="QuizAgent")],
        "research_memory": [f"Quiz generated for {instruction}"]
    }