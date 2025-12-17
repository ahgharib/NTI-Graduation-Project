from langgraph.graph import StateGraph, END
from state import AgentState
from orchestrator import Orchestrator
from planning_agent import PlanningAgent
from quiz_agent import QuizAgent  # Ensure this file exists as created in previous step
from langchain_core.messages import HumanMessage
import sys
import json
import os
from state import ProjectPlan
from tools import PlanTools
from youtube_agent import YouTubeAgent

class MultiAgentPlanner:
    """Main multi-agent planner system with Orchestrator control."""
    
    def __init__(self):
        self.orchestrator = Orchestrator()
        self.final_state = {} 
        self._build_graph()  # CRITICAL: This must be called to initialize self.app

    def _build_graph(self):
        self.workflow = StateGraph(AgentState)
        
        # 1. Add nodes
        self.workflow.add_node("orchestrator", self.orchestrator.supervisor_node)
        self.workflow.add_node("planner", PlanningAgent.plan_node)
        self.workflow.add_node("quiz_generator", QuizAgent.quiz_node)
        self.workflow.add_node("youtube_search", YouTubeAgent.youtube_node) # NEW NODE
        
        # 2. Set entry point
        self.workflow.set_entry_point("orchestrator")
        
        # 3. Add conditional edges
        self.workflow.add_conditional_edges(
            "orchestrator",
            lambda state: state.get("next"),
            {
                "planner": "planner",
                "quiz_generator": "quiz_generator",
                "youtube_search": "youtube_search", # NEW ROUTE
                "END": END
            }
        )
        
        # 4. Add return edges
        self.workflow.add_edge("planner", "orchestrator")
        self.workflow.add_edge("quiz_generator", "orchestrator")
        self.workflow.add_edge("youtube_search", "orchestrator") # RETURN TO SUPERVISOR

        self.app = self.workflow.compile()
        print("✅ Graph compiled: Orchestrator <-> [Planner | Quiz | Youtube]")
        
    def run(self, user_prompt: str) -> dict:
        """Run the process with the given prompt."""
        
        initial_state = {
            "messages": [HumanMessage(content=user_prompt, name="User")],
            "user_prompt": user_prompt,
            "validation_errors": [],
            "is_valid": False,
            "execution_log": [],
            "refinement_attempts": 0,
            "last_node": "Start"
        }
        
        print(f"\n--- STARTING SYSTEM ---\nPrompt: {user_prompt[:80]}...")
        
        current_state = initial_state.copy()
        
        try:
            # Using stream to track node execution
            for event in self.app.stream(initial_state, {"recursion_limit": 20}):
                for node_name, output in event.items():
                    print(f"  > Executed node: {node_name}")
                    if node_name != '__end__':
                        # Merge the node's output into our tracking state
                        current_state.update(output)
            
            self.final_state = current_state
            print(f"\n--- PROCESS COMPLETE ---")
            return self.final_state
            
        except Exception as e:
            print(f"\n❌ A fatal error occurred during graph execution: {e}")
            # Ensure we return at least the current state or the error
            return current_state if current_state else {"error": str(e)}

def display_results(result: dict):
    """Cleanly display the final output based on which agent ran."""
    print("\n\n" + "=" * 60)
    print("           FINAL EXECUTION RESULTS")
    print("=" * 60)
    
    # 1. Final User Message
    final_message = result.get("final_output")
    if final_message:
        print(f"\n🤖 FINAL MESSAGE:\n{final_message}")
    
    # 2. Display Quiz Details (if applicable)
    quiz = result.get("quiz_output")
    if quiz:
        print(f"\n📝 QUIZ GENERATED: {quiz.topic}")
        print(f"   Level: {quiz.proficiency_level}")
        print(f"   Questions: {len(quiz.mcq_questions)} MCQs, {len(quiz.coding_questions)} Coding")

    # 3. Display Project Plan summary (if applicable)
    saved_file = result.get("saved_file")
    if saved_file:
        print(f"\n✅ PLAN SAVED: {saved_file}")
        plan_data = result.get("parsed_output")
        if isinstance(plan_data, ProjectPlan):
            print(f"📊 PLAN SUMMARY: {plan_data.goal[:50]}... ({plan_data.duration})")
        
    # 4. Show execution log summary
    if result.get("execution_log"):
        print(f"\n📊 EXECUTION STEPS: {len(result['execution_log'])}")
    
    print(f"\n🔍 FINAL STATE KEYS: {list(result.keys())}")

def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        user_prompt = " ".join(sys.argv[1:])
    else:
        print("🤖 Multi-Agent Project & Quiz System")
        print("Enter your request (e.g., 'Plan a Python app' or 'Make a quiz on Java'):")
        user_prompt = input("> ")
        
        if user_prompt.lower() in ['quit', 'exit', 'q']:
            return
    
    if not user_prompt.strip():
        print("❌ Error: Empty prompt provided")
        return
    
    planner = MultiAgentPlanner()
    result = planner.run(user_prompt)
    display_results(result)

if __name__ == "__main__":
    main()