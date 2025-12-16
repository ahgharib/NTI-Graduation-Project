from typing import TypedDict, List, Optional, Dict, Any, Annotated
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages
import json

# Pydantic models for structured output
class Task(BaseModel):
    description: str = Field(..., description="Detailed description of the task")
    difficulty: str = Field(..., description="Difficulty level: Easy, Medium, Hard, Expert")
    duration: str = Field(..., description="Estimated duration (e.g., '2 hours', '1 day')")

class Milestone(BaseModel):
    title: str = Field(..., description="Title of the milestone")
    description: str = Field(..., description="Description of what this milestone achieves")
    tasks: List[Task] = Field(..., description="List of tasks in this milestone")
    difficulty: str = Field(..., description="Difficulty level: Easy, Medium, Hard, Expert")
    duration: str = Field(..., description="Estimated duration (e.g., '2 days', '1 week')")

class ProjectPlan(BaseModel):
    goal: str = Field(..., description="Overall goal of the project")
    milestones: List[Milestone] = Field(..., description="List of milestones to achieve the goal")
    duration: str = Field(..., description="Estimated duration (e.g., '2 months', '6 weeks')")

# Unified State for Multi-Agent System
class AgentState(TypedDict):
    """Represents the state of our multi-agent graph."""
    # Messages for conversation history
    messages: Annotated[List[BaseMessage], add_messages]
    
    # Next node for conditional routing - Kept for LangGraph internal use but controlled by Orchestrator logic
    next: str
    
    # Phase for termination logic - Removed as 'next' and 'final_output' handle this
    
    # Planning specific state
    user_prompt: Optional[str]
    parsed_output: Optional[ProjectPlan]
    validation_errors: List[str]
    is_valid: bool
    execution_log: List[Dict[str, Any]]
    final_output: Optional[str]
    saved_file: Optional[str]
    
    # Current plan summary for orchestrator
    plan_summary: Optional[str]
    
    # === NEW FIELDS FOR ORCHESTRATION CONTROL ===
    refinement_attempts: int
    last_node: str
    # ==========================================