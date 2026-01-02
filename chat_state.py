import json
import operator
from typing import TypedDict, List, Optional, Dict, Any, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

# --- EXISTING PYDANTIC MODELS REMAIN UNCHANGED ---
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
    skill: str = Field(..., description="The skill being tested")
    options: List[str] = Field(..., description="List of answer options")
    correct_answer: str = Field(..., description="The correct answer from the options")

class ArticleQuestion(BaseModel):
    question: str = Field(..., description="The article question")
    skill: str = Field(..., description="The skill being tested")
    model_answer: str = Field(..., description="The article question model answer")

# class CodingQuestion(BaseModel):
#     question: str = Field(..., description="The coding question")
#     code_snippet: str = Field(..., description="The code snippet")
#     explanation: str = Field(..., description="Explanation of the code snippet")

class Quiz(BaseModel):
    topic: str = Field(..., description="The topic of the quiz")
    proficiency_level: str = Field(..., description="The proficiency level")
    mcq_questions: List[MCQ] = Field(..., description="List of MCQs")
    article_questions: List[ArticleQuestion] = Field(..., description="List of article questions")
    # coding_questions: List[CodingQuestion] = Field(..., description="List of coding questions")


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    user_prompt: str             
    final_output: Optional[str]  
    plan_actions: List[str]      
    plan_instructions: List[str] 
    current_instruction: str     
    next: str                    
    parsed_output: Optional[Any] 
    quiz_output: Optional[Any]   
    validation_errors: List[str]
    execution_log: List[Dict[str, Any]]
    research_memory: Annotated[List[str], operator.add]
    raw_data_storage: Annotated[List[Dict[str, Any]], operator.add]
    task_instructions: Optional[str]
    refinement_attempts: int
    saved_plan_file: Optional[str]
    
    # --- Context Passing ---
    plan_data: Optional[Dict[str, Any]] 
    selected_milestone_context: Optional[str] 
    
    # --- CHANGED: Added Summary Field ---
    conversation_summary: Optional[str] # Holds the LLM-summarized history
    user_submission: Optional[Dict[str, Any]]
    grader_output: Optional[Dict[str, Any]]