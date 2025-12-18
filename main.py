from langgraph.graph import StateGraph, END
from state import AgentState
from orchestrator import Orchestrator
from planning_agent import PlanningAgent
from quiz_agent import QuizAgent
from youtube_agent import YouTubeAgent
from web_search_agent import WebSearchAgent
from langchain_core.messages import HumanMessage
import sys

class MultiAgentPlanner:
    """Main multi-agent planner system with Orchestrator control."""
    
    def __init__(self):
        self.orchestrator = Orchestrator()
        self._build_graph() 

    def _build_graph(self):
        self.workflow = StateGraph(AgentState)
        
        # 1. Add nodes
        self.workflow.add_node("orchestrator", self.orchestrator.supervisor_node)
        self.workflow.add_node("planner", PlanningAgent.plan_node)
        self.workflow.add_node("quiz_generator", QuizAgent.quiz_node)
        self.workflow.add_node("youtube_search", YouTubeAgent.youtube_node)
        self.workflow.add_node("web_search", WebSearchAgent.search_node)
        
        # 2. Set Entry point
        self.workflow.set_entry_point("orchestrator")

        # 3. Add conditional edges
        # The Orchestrator decides 'next'. We map 'FINISH' to END.
        self.workflow.add_conditional_edges(
            "orchestrator",
            lambda state: state.get("next"),
            {
                "planner": "planner",
                "quiz_generator": "quiz_generator",
                "youtube_search": "youtube_search",
                "web_search": "web_search",
                "END": END
            }
        )
        
        # 4. Add return edges (All agents return to Orchestrator)
        self.workflow.add_edge("planner", "orchestrator")
        self.workflow.add_edge("quiz_generator", "orchestrator")
        self.workflow.add_edge("youtube_search", "orchestrator")
        self.workflow.add_edge("web_search", "orchestrator")

        self.app = self.workflow.compile()
        print("✅ Graph compiled: Orchestrator <-> [Agents]")
        
    def run(self, user_prompt: str) -> dict:
        initial_state = {
            "messages": [HumanMessage(content=user_prompt, name="User")],
            "user_prompt": user_prompt,
            "memory": [],
            "execution_log": [],
        }
        
        print(f"\n--- STARTING SYSTEM ---\nPrompt: {user_prompt}...")
        
        try:
            final_state = {}
            for event in self.app.stream(initial_state, {"recursion_limit": 25}):
                # Reset orchestrator for new run
                for key, value in event.items():
                    if key == "orchestrator":
                        self.orchestrator = Orchestrator()  # Reset orchestrator
                    final_state = value
            
            print(f"\n--- PROCESS COMPLETE ---")
            
            # SHOW THE RESULTS TO THE USER
            if "final_output" in final_state:
                print(f"\n🤖 FINAL MESSAGE:\n{final_state['final_output']}")
            
            if "saved_file" in final_state:
                print(f"\n📄 File saved: {final_state['saved_file']}")
            
            if "videos" in final_state:
                print(f"\n🎥 Videos found: {len(final_state['videos'])}")
                for video in final_state.get('videos', [])[:3]:
                    print(f"   • {video.get('title', 'No title')}")
            
            return final_state
            
        except Exception as e:
            print(f"\n❌ Execution Error: {e}")
            return {}

def main():
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
    else:
        prompt = input("Enter Request: ")
    
    system = MultiAgentPlanner()
    system.run(prompt)

if __name__ == "__main__":
    main()