import json
from typing import TypedDict, List, Optional, Dict, Any, Annotated
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# --- EXISTING PYDANTIC MODELS (Task, Milestone, ProjectPlan, Quiz, etc.) REMAIN THE SAME ---
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

class MCQ(BaseModel):
    question: str = Field(..., description="The multiple-choice question")
    options: List[str] = Field(..., description="List of answer options")
    correct_answer: str = Field(..., description="The correct answer from the options")

class ArticleQuestion(BaseModel):
    question: str = Field(..., description="The article question")
    answer: str = Field(..., description="The article answer")

class CodingQuestion(BaseModel):
    question: str = Field(..., description="The coding question")
    code_snippet: str = Field(..., description="The code snippet")
    explanation: str = Field(..., description="Explanation of the code snippet")

class Quiz(BaseModel):
    topic: str = Field(..., description="The topic of the quiz")
    proficiency_level: str = Field(..., description="The proficiency level")
    mcq_questions: List[MCQ] = Field(..., description="List of MCQs")
    article_questions: List[ArticleQuestion] = Field(..., description="List of article questions")
    coding_questions: List[CodingQuestion] = Field(..., description="List of coding questions")

# --- UPDATED AGENT STATE ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    next: str
    user_prompt: str             # The original high-level request
    task_instructions: str       # NEW: Specific instructions for the current node
    memory: List[str]            # NEW: Accumulates findings (Search results, previous plans, etc.)
    
    parsed_output: Optional[ProjectPlan]
    quiz_output: Optional[Quiz]
    videos: Optional[List[Dict[str, Any]]]
    
    validation_errors: List[str]
    is_valid: bool
    execution_log: List[Dict[str, Any]]
    final_output: Optional[str]
    saved_file: Optional[str]
    plan_summary: Optional[str]
    refinement_attempts: int
    last_node: str
    max_attempts: int  # Maximum attempts per node
    attempts_count: Dict[str, int]  # Count attempts per node
    visited_nodes: List[str]  # Track visited nodes
    is_stuck: bool  # Flag for stuck detection

    saved_plan_file: Optional[str]  # Path to saved plan JSON
    saved_videos_file: Optional[str]  # Path to saved videos JSON
    plan_data: Optional[Dict[str, Any]]  # Complete plan data
    videos_data: Optional[Dict[str, Any]]  # Complete videos data
    search_query: Optional[str]  # For video searches