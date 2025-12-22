from config import Config
from state import AgentState
from tools import web_search_tool, youtube_search_tool
from log import prepare_context, universal_debug_log
from langchain_core.messages import HumanMessage

def explainer_node(state: AgentState):
    node_name = "EXPLAINER_AGENT"
    instruction = state["current_instruction"]
    context = prepare_context(state)
    
    # Logic from your provided Explainer: Search + Explain
    # We use Groq/Gemini as per your architecture in Config
    llm = Config.get_groq_llm()
    
    # Perform internal research if needed
    research = web_search_tool.invoke(instruction)
    
    prompt = f"""
    You are a helpful research assistant.
    Topic: {instruction}
    Web Context: {research}
    Previous History: {context}
    
    Explain the topic clearly and simply.
    """
    
    response = llm.invoke(prompt)
    universal_debug_log(node_name, "OUTPUT", response.content)
    
    return {
        "messages": [HumanMessage(content=response.content, name="Explainer")],
        "research_memory": [f"Explanation of {instruction}: {response.content}"]
    }