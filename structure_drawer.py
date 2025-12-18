import requests
import sys
import os
import json
from langgraph.graph import StateGraph, END
from state import AgentState
from orchestrator import Orchestrator
from planning_agent import PlanningAgent
from quiz_agent import QuizAgent
from youtube_agent import YouTubeAgent
from web_search_agent import WebSearchAgent  # Added WebSearchAgent
from langchain_core.messages import HumanMessage
from tools import PlanTools, ValidationTools, OrchestrationTools

class MultiAgentPlanner:
    """Main multi-agent planner system with Orchestrator control."""
    
    def __init__(self):
        self.orchestrator = Orchestrator()
        self._build_graph()
    
    def _build_graph(self):
        """Build the complete LangGraph workflow: Planner, Quiz, YouTube, and Web Search."""
        self.workflow = StateGraph(AgentState)
        
        # 1. Add all nodes
        self.workflow.add_node("orchestrator", self.orchestrator.supervisor_node)
        self.workflow.add_node("planner", PlanningAgent.plan_node)
        self.workflow.add_node("quiz_generator", QuizAgent.quiz_node)
        self.workflow.add_node("youtube_search", YouTubeAgent.youtube_node)
        self.workflow.add_node("web_search", WebSearchAgent.search_node)  # NEW NODE
        
        # 2. Set entry point
        self.workflow.set_entry_point("orchestrator")
        
        # 3. Add conditional edges (Orchestrator -> Branch)
        self.workflow.add_conditional_edges(
            "orchestrator",
            lambda state: state.get("next", "planner"),
            {
                "planner": "planner",
                "quiz_generator": "quiz_generator",
                "youtube_search": "youtube_search",
                "web_search": "web_search",  # NEW ROUTE
                "END": END
            }
        )
        
        # 4. Add return edges (All Nodes -> Orchestrator)
        self.workflow.add_edge("planner", "orchestrator")
        self.workflow.add_edge("quiz_generator", "orchestrator")
        self.workflow.add_edge("youtube_search", "orchestrator")
        self.workflow.add_edge("web_search", "orchestrator")  # NEW RETURN

        self.app = self.workflow.compile()
        print("✅ Graph compiled: Orchestrator <-> [Planner | Quiz | YouTube | WebSearch]")

def generate_graph_visualization():
    """Generates the graph.png with clean Mermaid syntax."""
    
    planner_sys = MultiAgentPlanner()
    app = planner_sys.app
    
    # Basic Mermaid code - start with just the graph structure
    mermaid_template = """graph TB
    %% --- Main Nodes ---
    start([User Request]):::start
    orchestrator{{Orchestrator<br/>LLM: Llama3}}:::orchestrator
    planner[Planning Agent<br/>LLM: Llama3<br/>Output: JSON roadmap]:::planner
    quiz[Quiz Agent<br/>LLM: Llama3<br/>Output: JSON quiz]:::quiz
    youtube[YouTube Agent<br/>APIs: Groq + YouTube]:::youtube
    websearch[Web Search Agent<br/>APIs: Tavily + Groq]:::websearch
    finish([Tasks Complete]):::finish
    
    %% --- Main Flow ---
    start --> orchestrator
    orchestrator -->|roadmap| planner
    orchestrator -->|quiz| quiz
    orchestrator -->|video| youtube
    orchestrator -->|search| websearch
    planner --> orchestrator
    quiz --> orchestrator
    youtube --> orchestrator
    websearch --> orchestrator
    orchestrator -->|done| finish
    
    %% --- Tools Subgraph ---
    subgraph tools[Internal Tools]
        plantools[PlanTools]:::tool
        validation[ValidationTools]:::tool
        orchestration[OrchestrationTools]:::tool
    end
    
    %% --- APIs Subgraph ---
    subgraph apis[External APIs]
        groq[Groq API]:::api
        youtube_api[YouTube API]:::api
        tavily[Tavily Search]:::api
    end
    
    %% --- Tool Connections ---
    planner -.-> plantools
    orchestrator -.-> validation
    orchestrator -.-> orchestration
    youtube -.-> groq
    youtube -.-> youtube_api
    websearch -.-> tavily
    websearch -.-> groq
    
    %% --- Styling ---
    classDef orchestrator fill:#FFF3E0,stroke:#FF9800,stroke-width:3px
    classDef planner fill:#E8F5E8,stroke:#4CAF50,stroke-width:2px
    classDef quiz fill:#F3E5F5,stroke:#9C27B0,stroke-width:2px
    classDef youtube fill:#E3F2FD,stroke:#2196F3,stroke-width:2px
    classDef websearch fill:#FFF8E1,stroke:#FFC107,stroke-width:2px
    classDef start fill:#E8EAF6,stroke:#3F51B5,stroke-width:2px
    classDef finish fill:#FCE4EC,stroke:#E91E63,stroke-width:2px
    classDef tool fill:#E0F2F1,stroke:#009688,stroke-width:2px
    classDef api fill:#FFEBEE,stroke:#F44336,stroke-width:2px
    """
    
    print("\n--- Generating Graph Visualization ---")
    
    try:
        # First, save the Mermaid code for debugging
        with open("graph_code.mmd", "w", encoding="utf-8") as f:
            f.write(mermaid_template)
        print("✅ Mermaid code saved to 'graph_code.mmd' for debugging")
        
        # Send to Kroki
        resp = requests.post(
            "https://kroki.io/mermaid/png",
            data=mermaid_template.encode("utf-8"),
            headers={"Content-Type": "text/plain"},
            timeout=30
        )
        
        if resp.status_code == 200:
            with open("graph.png", "wb") as f:
                f.write(resp.content)
            print(f"✅ Graph generated successfully! Saved as 'graph.png'")
            return True
        else:
            print(f"❌ Kroki API Error {resp.status_code}: {resp.text[:200]}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Import main logic if needed
        from main import main as run_main
        run_main()
    else:
        generate_graph_visualization()