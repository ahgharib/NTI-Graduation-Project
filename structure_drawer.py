import requests
import sys
import os
import json
from langgraph.graph import StateGraph, END
from state import AgentState
from orchestrator import Orchestrator
from planning_agent import PlanningAgent
from quiz_agent import QuizAgent
from youtube_agent import YouTubeAgent  # Import the new agent
from langchain_core.messages import HumanMessage
from tools import PlanTools, ValidationTools, OrchestrationTools

class MultiAgentPlanner:
    """Main multi-agent planner system with Orchestrator control."""
    
    def __init__(self):
        self.orchestrator = Orchestrator()
        self._build_graph()
    
    def _build_graph(self):
        """Build the complete LangGraph workflow: Planner, Quiz, and YouTube Search."""
        self.workflow = StateGraph(AgentState)
        
        # 1. Add nodes
        self.workflow.add_node("orchestrator", self.orchestrator.supervisor_node)
        self.workflow.add_node("planner", PlanningAgent.plan_node)
        self.workflow.add_node("quiz_generator", QuizAgent.quiz_node)
        self.workflow.add_node("youtube_search", YouTubeAgent.youtube_node) # NEW NODE
        
        # 2. Set entry point
        self.workflow.set_entry_point("orchestrator")
        
        # 3. Add conditional edges (Orchestrator -> Branch)
        self.workflow.add_conditional_edges(
            "orchestrator",
            lambda state: state.get("next", "planner"),
            {
                "planner": "planner",
                "quiz_generator": "quiz_generator",
                "youtube_search": "youtube_search",  # NEW ROUTE
                "END": END
            }
        )
        
        # 4. Add return edges (Nodes -> Orchestrator)
        self.workflow.add_edge("planner", "orchestrator")
        self.workflow.add_edge("quiz_generator", "orchestrator")
        self.workflow.add_edge("youtube_search", "orchestrator") # NEW RETURN

        self.app = self.workflow.compile()
        print("✅ Graph compiled: Orchestrator <-> [Planner | Quiz | YouTube]")

def generate_graph_visualization():
    """Generates the graph.png and prints updated Mermaid code for the 4-node system."""
    
    planner_sys = MultiAgentPlanner()
    app = planner_sys.app
    
    node_descriptions = {
        "orchestrator": (
            "**Orchestrator** (Supervisor)<br>LLM: Llama3<br>Decision Logic:<br>1. Route to **Planner** (Roadmaps)"
            "<br>2. Route to **Quiz** (Testing)<br>3. Route to **YouTube** (Visual Tutorials)"
        ),
        "planner": (
            "**Planner Agent**<br>LLM: Gemini<br>Task: Generate/Refine Project Plans"
        ),
        "quiz_generator": (
            "**Quiz Agent**<br>LLM: Llama3/Gemini<br>Task: Generate MCQs & Coding Challenges"
        ),
        "youtube_search": (
            "**YouTube Search Agent**<br>Internal Pipeline:<br>1. **Groq Refine** (Query Optimization)"
            "<br>2. **YouTube API** (Search)<br>3. **File Output** (JSON Links)"
        ),
        "__start__": "START: User Input",
        "__end__": "END: Output Generated"
    }

    mermaid_code = app.get_graph().draw_mermaid()
    
    # Replacement logic for labels
    detailed_mermaid_code = mermaid_code.replace(
        "orchestrator[orchestrator]", f'orchestrator("{node_descriptions["orchestrator"]}")'
    ).replace(
        "planner[planner]", f'planner("{node_descriptions["planner"]}")'
    ).replace(
        "quiz_generator[quiz_generator]", f'quiz_generator("{node_descriptions["quiz_generator"]}")'
    ).replace(
        "youtube_search[youtube_search]", f'youtube_search("{node_descriptions["youtube_search"]}")'
    ).replace(
        "__start__[__start__]", f'__start__[{node_descriptions["__start__"]}]'
    ).replace(
        "__end__[__end__]", f'__end__[{node_descriptions["__end__"]}]'
    )

    # Adding Tooling Subgraph and Logic Labels
    tools_section = (
        "\n%% --- Conditional Routing and Tools ---\n"
        "orchestrator -->|**Condition: planner**| planner\n"
        "orchestrator -->|**Condition: quiz_generator**| quiz_generator\n"
        "orchestrator -->|**Condition: youtube_search**| youtube_search\n"
        "orchestrator -->|**Condition: END**| __end__\n"
        "\nsubgraph shared_tools [Shared Resources]\n"
        "     V[ValidationTools]:::tool\n"
        "     P[PlanTools/Parser]:::tool\n"
        "     Y[YouTube API / Groq]:::tool\n"
        "end\n"
        "planner -.-> P\n"
        "quiz_generator -.-> P\n"
        "youtube_search -.-> Y\n"
        "orchestrator -.-> V\n"
        "classDef tool fill:#e1f5fe,stroke:#01579b,stroke-width:2px,color:#000;\n"
    )
    
    final_mermaid_code = detailed_mermaid_code + tools_section

    print("\n--- Sending Updated Mermaid Code to Kroki.io ---")
    
    resp = requests.post(
        "https://kroki.io/mermaid/png",
        data=final_mermaid_code.encode("utf-8"),
        headers={"Content-Type": "text/plain"}
    )
    
    if resp.status_code == 200:
        with open("graph.png", "wb") as f:
            f.write(resp.content)
        print("✅ Success! 'graph.png' updated with YouTube Search Node.")
    else:
        print(f"❌ Error: {resp.status_code}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Import main logic if needed, otherwise just run visualizer
        from main import main as run_main
        run_main()
    else:
        generate_graph_visualization()