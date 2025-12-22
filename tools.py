import json
import os
import re
import requests
from typing import Dict, Any, List, Tuple
from datetime import datetime
from state import ProjectPlan
from langchain_core.messages import HumanMessage
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field

# --- PLANNER TOOLS ---

class PlanTools:
    """Tools for handling project plans (parsing, logging, formatting)."""
    
    @staticmethod
    def save_to_json(plan: ProjectPlan, filename: str = None) -> str:
        """Save project plan to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"project_plan_{timestamp}.json"
        
        plan_dict = plan.model_dump()
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(plan_dict, f, indent=2, ensure_ascii=False)
        
        return filename
    
    @staticmethod
    def parse_llm_output(llm_output: str) -> ProjectPlan:
        """Parse LLM output into structured ProjectPlan"""
        try:
            # Try to extract JSON from markdown code blocks
            if "```json" in llm_output:
                json_str = llm_output.split("```json")[1].split("```")[0].strip()
            elif "```" in llm_output:
                json_str = llm_output.split("```")[1].split("```")[0].strip()
            else:
                json_str = llm_output
            
            # Parse JSON
            data = json.loads(json_str)
            
            # Convert to Pydantic model
            return ProjectPlan(**data)
            
        except json.JSONDecodeError as e:
            # If JSON parsing fails, try to find JSON object
            json_match = re.search(r'\{.*\}', llm_output, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return ProjectPlan(**data)
            else:
                raise ValueError(f"Could not parse JSON from LLM output: {e}")
    
    @staticmethod
    def create_execution_log_entry(node_name: str, action: str, details: Dict[str, Any]) -> Dict[str, Any]:
        """Create a log entry for execution tracking"""
        return {
            "timestamp": datetime.now().isoformat(),
            "node": node_name,
            "action": action,
            "details": details
        }
    
    @staticmethod
    def extract_user_prompt(messages: list) -> str:
        """Extract the original user prompt from messages"""
        for message in messages:
            if isinstance(message, HumanMessage):
                return message.content
        return ""
    
    @staticmethod
    def format_plan_for_display(plan: ProjectPlan) -> str:
        """Format the plan for a readable output message."""
        output = ["✨ **PROJECT PLAN CREATED** ✨", "=" * 60]
        output.append(f"🎯 Goal: {plan.goal}")
        output.append(f"⏱️ Total Duration: {plan.duration}")
        output.append(f"🚩 Total Milestones: {len(plan.milestones)}")
        
        total_tasks = sum(len(m.tasks) for m in plan.milestones)
        output.append(f"✅ Total Tasks: {total_tasks}")
        output.append("=" * 60)
        
        for i, milestone in enumerate(plan.milestones):
            output.append(f"\n🚩 MILESTONE {i+1}: {milestone.title}")
            output.append(f"   Description: {milestone.description}")
            output.append(f"   Difficulty: {milestone.difficulty} | Duration: {milestone.duration}")
            output.append(f"   Tasks: {len(milestone.tasks)}")
            
            for j, task in enumerate(milestone.tasks):
                output.append(f"   └─ Task {j+1}: {task.description}")
                output.append(f"      Difficulty: {task.difficulty} | Duration: {task.duration}")
        
        return "\n".join(output)
    
    @staticmethod
    def create_summary(plan: ProjectPlan) -> str:
        """Create a concise summary for the orchestrator"""
        total_tasks = sum(len(m.tasks) for m in plan.milestones)
        # Simplified avg difficulty calculation for summary
        difficulty_levels = {"Easy": 1, "Medium": 2, "Hard": 3, "Expert": 4}
        all_difficulties = [difficulty_levels.get(m.difficulty, 0) for m in plan.milestones]
        avg_difficulty = sum(all_difficulties) / len(plan.milestones) if plan.milestones else 0
        
        if avg_difficulty < 1.5: difficulty_level = "Easy"
        elif avg_difficulty < 2.5: difficulty_level = "Medium"
        elif avg_difficulty < 3.5: difficulty_level = "Hard"
        else: difficulty_level = "Expert"
        
        return (
            f"Goal: {plan.goal[:50]}... | Milestones: {len(plan.milestones)} | "
            f"Tasks: {total_tasks} | Duration: {plan.duration} | Difficulty: {difficulty_level}"
        )


# --- VALIDATION TOOL (Used by Orchestrator) ---
class ValidationTools:
    """Tools for plan validation and rule checking."""
    
    VALID_DIFFICULTIES = {"Easy", "Medium", "Hard", "Expert"}
    
    @staticmethod
    def validate_plan(plan: ProjectPlan) -> Tuple[bool, List[str]]:
        """Validate the generated project plan against rules."""
        errors = []
        
        # Validate goal and project duration
        if not plan.goal or len(plan.goal.strip()) < 10:
            errors.append("Goal is too short or empty (minimum 10 characters)")
        if not plan.duration:
            errors.append("Project duration is missing")
        
        # Validate milestones count
        if not plan.milestones:
            errors.append("No milestones defined")
        elif len(plan.milestones) > 10 or len(plan.milestones) < 2:
            errors.append(f"Milestones count ({len(plan.milestones)}) must be between 2 and 10.")
        
        # Validate each milestone and its tasks
        for i, milestone in enumerate(plan.milestones):
            m_prefix = f"Milestone {i+1} ('{milestone.title[:20]}...')"
            
            if not milestone.title or len(milestone.title.strip()) < 3:
                errors.append(f"{m_prefix}: title is too short or empty.")
            if milestone.difficulty not in ValidationTools.VALID_DIFFICULTIES:
                errors.append(f"{m_prefix}: has invalid difficulty: {milestone.difficulty}")
            if not milestone.tasks:
                errors.append(f"{m_prefix}: has no tasks defined.")
            
            for j, task in enumerate(milestone.tasks):
                t_prefix = f"{m_prefix}, Task {j+1}"
                if task.difficulty not in ValidationTools.VALID_DIFFICULTIES:
                    errors.append(f"{t_prefix}: has invalid difficulty: {task.difficulty}")
        
        return not errors, errors

# --- ORCHESTRATION TOOLS (Used by Orchestrator) ---

class OrchestrationTools:
    """Tools for the Orchestrator agent (Messaging and Saving)."""
    
    @staticmethod
    def send_user_message(message: str) -> dict:
        """Outputs a final message to the user and signals termination."""
        print(f"\n[ORCHESTRATOR MESSAGE TO USER]")
        print(f"Message: {message}")
        return {
            "messages": [HumanMessage(content=message, name="Orchestrator")],
            "final_output": message,
            "next": "END",
            "last_node": "orchestrator_end_message"
        }

    @staticmethod
    def save_plan_and_end(plan: ProjectPlan) -> dict:
        """Saves the valid plan to a JSON file and signals termination."""
        # Note: Orchestrator should only call this if plan is already validated.
        
        try:
            # 1. Save
            filename = PlanTools.save_to_json(plan)
            file_size = os.path.getsize(filename)
            
            final_message = f"✅ Plan is valid and has been successfully saved to **{filename}** ({file_size} bytes)."

            # 2. Update State and End
            return {
                "messages": [HumanMessage(content=final_message, name="Orchestrator/Saver")],
                "saved_file": filename,
                "is_valid": True,
                "next": "END",
                "last_node": "orchestrator_saver"
            }
        
        except Exception as e:
            error_msg = f"❌ Save failed during final step: {str(e)[:200]}"
            print(f"  Save error: {e}")
            return OrchestrationTools.send_user_message(error_msg)

# --- SEARCH TOOLS ---

from langchain_core.tools import tool
from config import Config
from langchain_groq import ChatGroq
from pydantic import SecretStr

@tool
def web_search_tool(query: str) -> str:
    """
    Performs a web search using Tavily and summarizes the results.
    Use this when you need current events, documentation, or study material text.
    """
    print(f"  🔎 [Tool] Web Searching for: {query}")
    
    try:
        # Perform search with Tavily
        tavily = TavilySearch(max_results=3)
        results = tavily.invoke(query)
        
        # Summarize results with Groq
        if Config.GROQ_API_KEY:
            llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                api_key=SecretStr(Config.GROQ_API_KEY)
            )
            summary = llm.invoke(f"Summarize these search results for the query '{query}': {results}")
            return f"Web Search Results for '{query}':\n{summary.content}"
        else:
            return f"Web Search Results for '{query}':\n{results}"
    except Exception as e:
        return f"Error performing web search: {str(e)}"

@tool
def youtube_search_tool(query: str) -> str:
    """
    Searches YouTube for videos. 
    Use this when the user asks for videos, visual tutorials, or lectures.
    Returns a list of video titles and links.
    """
    print(f"  🎥 [Tool] YouTube Searching for: {query}")
    
    youtube_api_key = os.getenv("YOUTUBE_API_KEY")
    if not youtube_api_key:
        return "Error: YOUTUBE_API_KEY not found."

    try:
        # Optimize query with Groq if available
        if Config.GROQ_API_KEY:
            llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                api_key=SecretStr(Config.GROQ_API_KEY)
            )
            refined = llm.invoke(f"Create an optimized YouTube search query for: {query}. Return ONLY the query.")
            search_query = refined.content.strip()
        else:
            search_query = query

        # Search YouTube API
        url = "https://www.googleapis.com/youtube/v3/search"
        params = {
            "part": "snippet",
            "q": search_query,
            "maxResults": 3,
            "type": "video",
            "key": youtube_api_key
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        items = response.json().get("items", [])
        
        results = []
        for item in items:
            title = item["snippet"]["title"]
            video_id = item["id"]["videoId"]
            link = f"https://www.youtube.com/watch?v={video_id}"
            results.append(f"- Title: {title}\n  Link: {link}")
            
        return "\n".join(results) if results else "No videos found."
        
    except Exception as e:
        return f"Error searching YouTube: {str(e)}"