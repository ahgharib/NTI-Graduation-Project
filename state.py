from typing import TypedDict, List, Dict, Optional, Any

class PlanState(TypedDict):
    user_request: str           # The user's initial prompt
    messages: List[Any]         # Chat history for the Editor
    current_plan: Dict          # The Structured JSON Roadmap
    feedback: Optional[str]     # Discriminator's feedback
    attempt_count: int          # Loop counter
    search_context: str         # Results from Search Tools
    ui_selected_node: Optional[str] # ID of the node clicked in UI (for Editor context)
    approved: Optional[bool]
    raw_output: Optional[str]   # Added: To hold unparsed LLM output
    error: Optional[str]        # Added: To track parsing errors