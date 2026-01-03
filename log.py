# log.py
import json
from chat_state import AgentState
from config import Config
from langsmith import traceable # Use LangSmith

@traceable(name="Node Debug Log")
def universal_debug_log(node_name: str, stage: str, data):
    """
    Logs data to LangSmith. The @traceable decorator automatically 
    captures arguments and results.
    """
    # We still print to console for real-time visibility
    print(f"\n--- [LOG: {node_name} | {stage}] ---")
    if isinstance(data, dict):
        print(json.dumps(data, indent=2, default=lambda o: str(o)))
    else:
        print(data)
    
    # Return data so LangSmith captures the output in the trace
    return {"node": node_name, "stage": stage, "payload": data}

def log_and_print(title: str, content):
    """Simple wrapper for session starts/ends."""
    print(f"\n{'='*20} {title} {'='*20}")
    print(content)

@traceable(name="Context Preparation")
def prepare_context(state: AgentState) -> str:
    """Prepare context from research memory for LLM consumption."""
    raw_history = "\n".join(state.get("research_memory", []))
    word_count = len(raw_history.split())
    original_prompt = state.get("user_prompt", "")
    TOKEN_THRESHOLD = 150 

    if word_count < TOKEN_THRESHOLD:
        return f"ORIGINAL USER REQUEST: {original_prompt}\n\nRAW HISTORY:\n{raw_history}"

    llm = Config.get_ollama_llm()
    summary_res = llm.invoke(f"Provide a high-density summary of this research: {raw_history}")
    return f"ORIGINAL USER REQUEST: {original_prompt}\n\nCOMPRESSED RESEARCH CONTEXT: {summary_res.content}"