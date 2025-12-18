import json
from typing import Dict, Any, List, Optional, Literal
from pydantic import BaseModel, Field
from state import AgentState, ProjectPlan
from langchain_core.messages import HumanMessage, SystemMessage
from tools import PlanTools, ValidationTools, OrchestrationTools 
from config import Config

class OrchestratorDecision(BaseModel):
    """The orchestrator's decision on the next step."""
    thought: str = Field(..., description="Reasoning for the decision.")
    next_node: Literal['planner', 'quiz_generator', 'youtube_search', 'web_search', 'FINISH'] = Field(...)
    instructions: str = Field(..., description="Instructions for the next node.")

class Orchestrator:
    def __init__(self):
        self.llm = Config.get_ollama_llm()
        self.decision_llm = self.llm.with_structured_output(OrchestratorDecision)
        self.completed_tasks = set()
        
    def supervisor_node(self, state: AgentState) -> dict:
        print(f"\n--- ORCHESTRATOR THINKING ---")
        
        last_node = state.get("last_node", "")
        user_prompt = state.get("user_prompt", "").lower()
        
        # Track completed tasks based on actual outputs
        if last_node == "planner" and state.get("parsed_output"):
            self.completed_tasks.add("roadmap")
            print(f"  ✅ Roadmap created: {state.get('parsed_output').goal[:50]}...")
        
        elif last_node == "quiz_generator" and state.get("quiz_output"):
            self.completed_tasks.add("quiz")
            print(f"  ✅ Quiz generated: {state.get('quiz_output').topic}")
        
        elif last_node == "youtube_search" and state.get("videos"):
            self.completed_tasks.add("video")
            print(f"  ✅ Videos found: {len(state.get('videos'))} videos")
        
        elif last_node == "web_search" and state.get("saved_file"):
            self.completed_tasks.add("search")
            print(f"  ✅ Web search completed")
        
        # Analyze what user actually requested
        requested_tasks = self._parse_user_request(user_prompt)
        print(f"  📋 Requested: {requested_tasks}")
        print(f"  ✅ Completed: {self.completed_tasks}")
        
        # Check what's still pending
        pending_tasks = [task for task in requested_tasks if task not in self.completed_tasks]
        
        # If nothing pending, we're done
        if not pending_tasks:
            final_message = self._generate_final_message(state, requested_tasks)
            return {
                "next": "END",
                "final_output": final_message,
                "messages": [HumanMessage(content=final_message, name="Orchestrator")],
                "last_node": "orchestrator"
            }
        
        # Determine next task and create SPECIFIC instructions
        next_task = pending_tasks[0]
        instructions = self._create_instructions(next_task, user_prompt, state)
        
        # Map task to node
        task_to_node = {
            "roadmap": "planner",
            "quiz": "quiz_generator", 
            "video": "youtube_search",
            "search": "web_search"
        }
        
        next_node = task_to_node.get(next_task, "planner")
        
        print(f"  👉 Next: {next_node}")
        print(f"  📋 Instructions: {instructions}")
        
        return {
            "next": next_node,
            "task_instructions": instructions,
            "last_node": "orchestrator",
            "memory": list(self.completed_tasks)
        }
    
    def _parse_user_request(self, user_prompt: str) -> List[str]:
        """Parse user request to determine what tasks are needed"""
        tasks = []
        
        # Check for roadmap/plan requests
        roadmap_keywords = ["roadmap", "plan", "project plan", "learning path", "curriculum"]
        if any(keyword in user_prompt for keyword in roadmap_keywords):
            tasks.append("roadmap")
        
        # Check for quiz requests
        quiz_keywords = ["quiz", "test", "questions", "assessment", "exam"]
        if any(keyword in user_prompt for keyword in quiz_keywords):
            tasks.append("quiz")
        
        # Check for video requests
        video_keywords = ["video", "youtube", "tutorial", "watch", "visual"]
        if any(keyword in user_prompt for keyword in video_keywords):
            tasks.append("video")
        
        # Check for search/research requests
        search_keywords = ["search", "research", "information", "find", "look up"]
        if any(keyword in user_prompt for keyword in search_keywords):
            tasks.append("search")
        
        # If no specific tasks but mentions learning, default to roadmap
        if not tasks and ("learn" in user_prompt or "study" in user_prompt):
            tasks.append("roadmap")
        
        return tasks
    
    def _create_instructions(self, task: str, user_prompt: str, state: AgentState) -> str:
        """Create specific instructions for each task type"""
        # Extract the main topic from user prompt
        topic = self._extract_topic(user_prompt)
        
        if task == "roadmap":
            return f"Create a comprehensive roadmap for learning: {topic}. Include milestones, tasks, durations, and difficulty levels."
        
        elif task == "quiz":
            return f"Generate a quiz about: {topic}. Include 5 MCQs, 2 article questions, and 2 coding questions. Make sure questions are relevant to the topic."
        
        elif task == "video":
            # If we have a roadmap, search for videos for each milestone
            if "roadmap" in self.completed_tasks and state.get("parsed_output"):
                plan = state.get("parsed_output")
                return f"Search for educational YouTube videos about: {topic}. Specifically look for tutorials covering: {plan.goal[:100]}..."
            return f"Search for tutorial videos about: {topic} on YouTube. Focus on educational content from reputable sources."
        
        elif task == "search":
            return f"Search the web for latest information and tutorials about: {topic}. Include both beginner and advanced resources."
        
        return f"Please work on: {user_prompt}"
    
    def _extract_topic(self, user_prompt: str) -> str:
        """Extract the main topic from user prompt"""
        # Remove common request phrases
        remove_phrases = [
            "make a", "create a", "generate a", "build a", "give me",
            "roadmap", "plan", "quiz", "test", "video", "youtube", 
            "search", "find", "for", "about", "on", "then", "and",
            "based on", "according to", "learning", "study"
        ]
        
        topic = user_prompt.lower()
        for phrase in remove_phrases:
            topic = topic.replace(phrase, "")
        
        # Clean up extra spaces and punctuation
        topic = topic.strip(" ,.!?;:-")
        
        # If topic is too short, return original prompt
        if len(topic) < 10:
            return user_prompt[:50] + "..."
        
        return topic.capitalize()
    
    def _generate_final_message(self, state: AgentState, requested_tasks: List[str]) -> str:
        """Generate final summary message"""
        parts = ["🎉 **All Tasks Completed Successfully!** 🎉", ""]
        
        if "roadmap" in requested_tasks and state.get("parsed_output"):
            plan = state.get("parsed_output")
            parts.append(f"📋 **Roadmap Created:**")
            parts.append(f"   • Goal: {plan.goal}")
            parts.append(f"   • Duration: {plan.duration}")
            parts.append(f"   • Milestones: {len(plan.milestones)}")
            parts.append("")
        
        if "quiz" in requested_tasks and state.get("quiz_output"):
            quiz = state.get("quiz_output")
            parts.append(f"📝 **Quiz Generated:**")
            parts.append(f"   • Topic: {quiz.topic}")
            parts.append(f"   • Questions: {len(quiz.mcq_questions)} MCQs, {len(quiz.article_questions)} articles, {len(quiz.coding_questions)} coding")
            parts.append("")
        
        if "video" in requested_tasks and state.get("videos"):
            videos = state.get("videos", [])
            parts.append(f"🎥 **Videos Found:**")
            for i, video in enumerate(videos[:3], 1):
                parts.append(f"   {i}. {video.get('title', 'No title')}")
            parts.append("")
        
        if "search" in requested_tasks and state.get("saved_file"):
            parts.append(f"🔍 **Search completed and saved to file**")
            parts.append("")
        
        parts.append("✅ All requested tasks have been completed!")
        return "\n".join(parts)