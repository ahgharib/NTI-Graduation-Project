from pydantic import BaseModel, Field
from typing import List, Optional

class Task(BaseModel):
    name: str = Field(description="Name of the task")
    description: str = Field(description="Brief description of what to do")
    resources: Optional[str] = Field(description="URL or title of a reading resource", default="")
    youtube: Optional[str] = Field(description="URL or title of a video resource", default="")

class Milestone(BaseModel):
    id: str = Field(description="Unique identifier, e.g., m1, m2")
    title: str = Field(description="Title of the milestone")
    description: str = Field(description="Overview of this phase")
    status: str = Field(description="Status: 'todo', 'in progress', or 'done'", default="todo")
    tasks: List[Task] = Field(description="List of tasks for this milestone")

class Roadmap(BaseModel):
    goal: str = Field(description="The main learning goal")
    duration: str = Field(description="Estimated duration")
    milestones: List[Milestone] = Field(description="List of milestones")