import json
import logging
from state import AgentState
from config import Config

# Enhanced Logger Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.FileHandler("study_buddy.log", mode='a', encoding="utf-8")]
)

def universal_debug_log(node_name: str, stage: str, data):
    """Unified logger for debugging every aspect of node execution."""
    sep = "█" * 80
    sub_sep = "-" * 40
    
    # Format data for JSON-like readability in logs
    if isinstance(data, dict):
        clean_data = data.copy()
        if "messages" in clean_data:
            clean_data["messages"] = [f"[{type(m).__name__}] {m.content[:100]}..." for m in clean_data["messages"]]
        formatted = json.dumps(clean_data, indent=2, default=lambda o: str(o))
    else:
        formatted = str(data)

    log_entry = (
        f"\n{sep}\n"
        f"DEBUG | NODE: {node_name} | STAGE: {stage}\n"
        f"{sub_sep}\n"
        f"{formatted}\n"
        f"{sep}\n"
    )
    print(log_entry)
    logging.info(log_entry)

def log_and_print(title: str, content):
    """Preserved original utility."""
    separator = "="*60
    msg = f"\n{separator}\n[{title}]\n{content}\n{separator}"
    print(msg)
    logging.info(f"{title}: {content}")

def prepare_context(state: AgentState) -> str:
    """Prepare context from research memory for LLM consumption."""
    raw_history = "\n".join(state.get("research_memory", []))
    word_count = len(raw_history.split())
    original_prompt = state.get("user_prompt", "")
    TOKEN_THRESHOLD = 1500 

    if word_count < TOKEN_THRESHOLD:
        return f"ORIGINAL USER REQUEST: {original_prompt}\n\nRAW HISTORY:\n{raw_history}"

    llm = Config.get_groq_llm()
    if word_count < TOKEN_THRESHOLD * 2:
        summary_res = llm.invoke(f"Provide a brief summary of this research, keeping key links and data: {raw_history[:4000]}")
        return f"ORIGINAL USER REQUEST: {original_prompt}\n\nSUMMARY: {summary_res.content}\n\nRAW DATA: {raw_history}"

    summary_res = llm.invoke(f"The research is too long. Provide a high-density summary: {raw_history}")
    return f"ORIGINAL USER REQUEST: {original_prompt}\n\nCOMPRESSED RESEARCH CONTEXT: {summary_res.content}"