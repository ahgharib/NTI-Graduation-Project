from config import Config
from chat_state import AgentState
from chat_tools import web_search_tool, youtube_search_tool
from log import prepare_context, universal_debug_log
from langchain_core.messages import HumanMessage
from RAG.rag import get_context_chunks

def explainer_node(state: AgentState):
    node_name = "EXPLAINER_AGENT"
    instruction = state["current_instruction"]
    print("########### EXPLAINER AGENT INSTRUCTION #############\n", instruction)
    doc_context = get_context_chunks(instruction)
    print("########### EXPLAINER AGENT DOC CONTEXT #############\n", doc_context)
    context = prepare_context(state)
    milestone_context = state.get("selected_milestone_context", "")
    
    # Logic from your provided Explainer: Search + Explain
    # We use Groq/Gemini as per your architecture in Config
    llm = Config.get_gemini_llm()
    
    # Perform internal research if needed
    research = web_search_tool.invoke(instruction)
    
    prompt = f"""
    You are an expert technical explainer and research assistant.

    USER QUESTION:
    {instruction}

    DOCUMENT CONTEXT (if available):
    {doc_context if doc_context else "No relevant document context found."}

    WEB SEARCH CONTEXT (if used):
    {research if research else "No web search was required."}

    PRIOR CONVERSATION:
    {context}

    ROADMAP CONTEXT:
    {state.get("plan_data")}

    SELECTED MILESTONE CONTEXT:
    {milestone_context}

    --- INSTRUCTIONS ---
    1. Prefer explaining using the DOCUMENT CONTEXT when available.
    2. If the user asked for information from a specific document/page, prioritize that.
    3. If information comes from a document:
    - Explicitly cite: File name + page number.
    4. If information comes from web search:
    - Explicitly state that it was web-sourced.
    5. If knowledge is general:
    - Say it is general domain knowledge.
    6. Do NOT hallucinate references.
    7. Explain clearly, step-by-step, and concisely.
    """   
    response = llm.invoke(prompt)
    # universal_debug_log(node_name, "OUTPUT", response.content)
    
    return {
        "messages": [HumanMessage(content=response.content, name="Explainer")],
        "research_memory": [f"Explanation of {instruction}: {response.content}"]
    }