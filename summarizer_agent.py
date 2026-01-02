from config import Config
from chat_state import AgentState
from log import prepare_context, universal_debug_log
from langchain_core.messages import HumanMessage
from RAG.rag import get_context_chunks

def summarizer_node(state: AgentState):
    node_name = "SUMMARIZER_AGENT"
    instruction = state["current_instruction"]
    
    # 1. Get RAG Context (crucial for "summarize page 5" requests)
    doc_context = get_context_chunks(instruction)
    
    # 2. Get Conversation History
    context = prepare_context(state)
    
    # 3. Initialize the LLM
    llm = Config.get_gemini_llm()
    
    # 4. Construct Prompt
    prompt = f"""
    You are an expert Summarizer and Simplifier.
    
    YOUR TASK:
    - Provide a clear, high-quality summary based on the user's request.
    - If the user asks to summarize a specific document/page, prioritize the 'DOCUMENT CONTEXT'.
    - If the user asks for a summary of a specific page number, use the chuncks from that page in the 'DOCUMENT CONTEXT'.
    - If the user provides raw text in the instruction, summarize that.
    - If the user asks for an overview of a complex topic, simplify it.
    - Do NOT add external information unless necessary for clarity.
    - Keep the tone helpful and objective.
    
    USER INSTRUCTION:
    {instruction}
    
    DOCUMENT CONTEXT (RAG):
    {doc_context if doc_context else "No specific document context found."}
    
    CONVERSATION HISTORY:
    {context}
    
    """
    
    # 5. Invoke LLM
    response = llm.invoke(prompt)
    
    # 6. Return Output
    return {
        "messages": [HumanMessage(content=response.content, name="Summarizer")],
        "research_memory": [f"Summary generated for: {instruction}"]
    }