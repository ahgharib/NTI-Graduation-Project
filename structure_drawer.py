import requests
import sys
import os
from langgraph.graph import StateGraph, END
from state import AgentState
from orchestrator import Orchestrator
from planning_agent import PlanningAgent
from langchain_core.messages import HumanMessage
from tools import PlanTools, ValidationTools, OrchestrationTools
from config import Config

# --- Your MultiAgentPlanner Class (Copied from main.py for full context) ---

class MultiAgentPlanner:
    """Main multi-agent planner system with Orchestrator control."""
    
    def __init__(self):
        self.orchestrator = Orchestrator()
        self._build_graph()
        self.final_state = None
    
    def _build_graph(self):
        """Build the complete LangGraph workflow with minimal nodes."""
        self.workflow = StateGraph(AgentState)
        self.workflow.add_node("orchestrator", self.orchestrator.supervisor_node)
        self.workflow.add_node("planner", PlanningAgent.plan_node)
        self.workflow.set_entry_point("orchestrator")
        
        # Conditional Edge (Orchestrator -> Planner/END)
        self.workflow.add_conditional_edges(
            "orchestrator",
            lambda state: state.get("next", "planner"),
            {
                "planner": "planner",
                END: END
            }
        )
        
        # Fixed Edge (Planner -> Orchestrator)
        self.workflow.add_edge("planner", "orchestrator")
        self.app = self.workflow.compile()
        print("✅ Graph compiled with minimal nodes: Orchestrator <-> Planner")


# --- CORRECTED VISUALIZATION FUNCTION ---

def generate_graph_visualization():
    """Generates the graph.png and prints the corrected Mermaid code."""
    
    # 1. Initialize the planner to build the graph structure
    planner = MultiAgentPlanner()
    app = planner.app
    
    # Define detailed descriptions using <br> for newlines (Mermaid standard)
    node_descriptions = {
        "orchestrator": (
            "**Orchestrator** (Supervisor)<br>LLM: Llama3 (Ollama)<br>Roles:<br>1. Initial **Agentic Routing** (plan/terminate)<br>2. **Pythonic Routing** (Validation Loop)"
        ),
        "planner": (
            "**Planner Agent** (Generator & Refiner)<br>LLM: Gemini/Llama3<br>Roles:<br>* Generate Initial Plan (JSON)<br>* Refine Plan (based on errors)"
        ),
        # Use simple labels for start/end
        "__start__": "START: User Prompt",
        "__end__": "END: Valid Plan or Max Attempts Reached"
    }

    # 2. Get the basic Mermaid code
    mermaid_code = app.get_graph().draw_mermaid()
    
    # 3. Use string replacement to update the simple node labels with detailed ones
    # We replace 'node[node]' with 'node["Detailed Label"]' using <br> for newlines
    detailed_mermaid_code = mermaid_code.replace(
        "orchestrator[orchestrator]", 
        f'orchestrator("{node_descriptions["orchestrator"]}")'
    ).replace(
        "planner[planner]", 
        f'planner("{node_descriptions["planner"]}")'
    ).replace(
        "__start__[__start__]", 
        f'__start__[{node_descriptions["__start__"]}]'
    ).replace(
        "__end__[__end__]", 
        f'__end__[{node_descriptions["__end__"]}]'
    )

    # 4. Add custom edge labels and external tool visualization
    tools_section = (
        "\n%% --- Data Flow and External Tool Access ---\n"
        "orchestrator -->|**Condition: next = planner**| planner\n"
        "orchestrator -->|**Condition: next = END**| __end__\n"
        "planner -->|Plan JSON + Errors| orchestrator\n"
        "\nsubgraph Tools and Validation\n"
        "    V[ValidationTools.validate_plan]:::tool\n"
        "    O[OrchestrationTools.save_plan_and_end]:::tool\n"
        "    P[PlanTools.parse_llm_output]:::tool\n"
        "end\n"
        "%% Conceptual links for clarity\n"
        "orchestrator -.-> V(Rule-Based Validation)\n"
        "orchestrator -.-> O(File Saving/Termination)\n"
        "planner -.-> P(JSON Parsing)\n"
        
        "%% Style definitions\n"
        "classDef tool fill:#add8e6,stroke:#333,stroke-width:2px,color:#000;\n"
    )
    
    final_mermaid_code = detailed_mermaid_code + tools_section

    # 5. Output to Kroki.io for PNG generation
    print("\n--- Sending Corrected Mermaid Code to Kroki.io ---")
    print("\n" + final_mermaid_code) # Print final code for transparency
    
    resp = requests.post(
        "https://kroki.io/mermaid/png",
        data=final_mermaid_code.encode("utf-8"),
        headers={"Content-Type": "text/plain"}
    )
    
    if resp.status_code == 200:
        with open("graph.png", "wb") as f:
            f.write(resp.content)
        print("✅ Success! 'graph.png' file created.")
    else:
        print(f"❌ Error generating image: HTTP Status {resp.status_code}")
        print("Response Content:", resp.content.decode("utf-8")[:200]) # Print first 200 chars of error


if __name__ == "__main__":
    # Ensure the main logic runs only if executed directly
    if len(sys.argv) > 1:
        # If running with a prompt, use the main function
        from main import main as run_main
        run_main()
    else:
        # Otherwise, generate the visualization
        print("Starting Graph Visualization Generator...")
        generate_graph_visualization()