from langgraph.graph import StateGraph, END
from state import AgentState
from orchestrator import Orchestrator
from planning_agent import PlanningAgent
from langchain_core.messages import HumanMessage
import sys
import json
import os
from state import ProjectPlan
from tools import PlanTools

# Removed unnecessary imports: ValidationAgent, file_analyzer

class MultiAgentPlanner:
    """Main multi-agent planner system with Orchestrator control."""
    
    def __init__(self):
        self.orchestrator = Orchestrator()
        self._build_graph()
        self.final_state = None  # Track final state
    
    def _build_graph(self):
        """Build the complete LangGraph workflow with minimal nodes."""
        # Initialize the graph
        self.workflow = StateGraph(AgentState)
        
        # Add ONLY the required nodes
        self.workflow.add_node("orchestrator", self.orchestrator.supervisor_node)
        self.workflow.add_node("planner", PlanningAgent.plan_node)
        
        # Set entry point
        self.workflow.set_entry_point("orchestrator")
        
        # Add a single conditional edge from orchestrator to planner.
        # The orchestrator's logic determines the final 'next' value: 'planner' or 'END'
        self.workflow.add_conditional_edges(
            "orchestrator",
            lambda state: state.get("next", "planner"),
            {
                "planner": "planner",
                END: END
            }
        )
        
        # The planner always returns to the orchestrator for validation and routing
        self.workflow.add_edge("planner", "orchestrator")

        # Compile the graph
        self.app = self.workflow.compile()
        print("✅ Graph compiled with minimal nodes: Orchestrator <-> Planner")
        
    def run(self, user_prompt: str) -> dict:
        """Run the planning process with the given prompt."""
        
        initial_state = {
            "messages": [HumanMessage(content=user_prompt, name="User")],
            "user_prompt": user_prompt,
            "validation_errors": [],
            "is_valid": False,
            "execution_log": [],
            "refinement_attempts": 0,
            "last_node": "Start" # Initial entry point indicator
        }
        
        print(f"\n--- STARTING PLANNER ---\nPrompt: {user_prompt[:80]}...")
        
        # We need to track the final state outside the stream loop
        final_state = initial_state.copy()
        
        try:
            # Stream the result
            for s in self.app.stream(initial_state, {"recursion_limit": 15}):
                for key, value in s.items():
                    print(f"  > Executed node: {key}")
                    if key != '__end__':
                        # Use update to merge all changes into final_state
                        final_state.update(value)
            
            # If the loop finishes without StopIteration (rare, but possible):
            self.final_state = final_state
            print(f"\n--- PROCESS COMPLETE ---")
            return self.final_state
            
        except StopIteration:
            # ✅ FIX: This is the expected and desired exception when the graph reaches END in stream mode.
            self.final_state = final_state
            print(f"\n--- PROCESS COMPLETE (Graph END reached) ---")
            return self.final_state
            
        except Exception as e:
            # Handle unexpected graph execution errors
            print(f"\n❌ A fatal error occurred during graph execution: {e}")
            return self.final_state if self.final_state else {"error": str(e)}

def display_results(result: dict):
    """Cleanly display the final output."""
    print("\n\n" + "=" * 60)
    print("           FINAL EXECUTION RESULTS")
    print("=" * 60)
    
    # Final User Message
    final_message = result.get("final_output")
    if final_message:
        print(f"\n🤖 FINAL MESSAGE:\n{final_message}")
    
    # Display saved file info
    saved_file = result.get("saved_file")
    if saved_file:
        print(f"\n✅ PLAN SAVED: {saved_file}")
        
        # Try to read and display the plan from the parsed output if available
        plan_data = result.get("parsed_output")
        if isinstance(plan_data, ProjectPlan):
            plan: ProjectPlan = plan_data
            total_tasks = sum(len(m.tasks) for m in plan.milestones)
            print(f"\n📊 PLAN SUMMARY:")
            print(f"   Goal: {plan.goal[:80]}...")
            print(f"   Duration: {plan.duration}")
            print(f"   Total Milestones: {len(plan.milestones)}")
            print(f"   Total Tasks: {total_tasks}")
        
    # Show execution log summary
    if result.get("execution_log"):
        print(f"\n📊 EXECUTION LOG:")
        nodes_executed = {}
        for log in result["execution_log"]:
            node = log['node']
            nodes_executed[node] = nodes_executed.get(node, 0) + 1
        
        for node, count in sorted(nodes_executed.items()):
            print(f"   {node}: {count} time(s)")
        
        print(f"\n   Total Steps: {len(result['execution_log'])}")
    
    # Show all keys in final state for debugging
    print(f"\n🔍 FINAL STATE KEYS: {list(result.keys())}")


def main():
    """Main entry point"""
    
    if len(sys.argv) > 1:
        user_prompt = " ".join(sys.argv[1:])
    else:
        print("🤖 Multi-Agent Project Planner (Orchestrated)")
        print("Enter your project description (or 'quit' to exit):")
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