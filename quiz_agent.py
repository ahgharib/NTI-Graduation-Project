from config import Config
from chat_state import AgentState, Quiz
from chat_tools import web_search_tool
from log import prepare_context, universal_debug_log
from langchain_core.messages import HumanMessage
from RAG.rag import get_context_chunks
import json
from pathlib import Path


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

    
    # llm = Config.get_gemini_llm().with_structured_output(Quiz)
    # llm = Config.get_ollama_llm().with_structured_output(Quiz)
    llm = Config.get_groq_llm().with_structured_output(Quiz)
    
    # print("########### DOCUMENT CONTEXT #############\n", document_context)

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
    - MCQs 4 options with one correct option
    - The correct answer must be one of the provided options
    - Article questions
    """    
    
    quiz_output = llm.invoke(prompt)
    # universal_debug_log(node_name, "QUIZ_GENERATED", quiz_output.dict())
    SUBMISSION_DIR = Path(
    "/teamspace/studios/this_studio/NTI-Graduation-Project/quiz_submissions"
    )
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    safe_topic = quiz_output.topic.replace(" ", "_").lower()
    file_name = f"quiz_{safe_topic}.json"

    file_path = SUBMISSION_DIR / file_name

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(
            quiz_output.dict(),
            f,
            ensure_ascii=False,
            indent=2
        )


    return {
        "quiz_output": quiz_output,
        "messages": [HumanMessage(content=f"Generated quiz for {quiz_output.topic}", name="QuizAgent")],
        "research_memory": [f"Quiz generated for {instruction}"]
    }