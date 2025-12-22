import requests
import sys
import os
import json
from langgraph.graph import StateGraph, END
from state import AgentState
from orchestrator import Orchestrator
from planner_agent import PlanningAgent
from quiz_agent import quiz_node
from explainer_agent import explainer_node
from tools import PlanTools, ValidationTools, OrchestrationTools, web_search_tool, youtube_search_tool
from config import Config
from langchain_core.messages import HumanMessage

def scheduler(state: AgentState):
    """Schedule the next node to execute based on the plan."""
    queue_actions = state.get("plan_actions", [])
    queue_instructions = state.get("plan_instructions", [])
    
    if not queue_actions:
        return {"next": "END"}
    
    next_node = queue_actions[0]
    next_instr = queue_instructions[0]
    
    return {
        "next": next_node,
        "current_instruction": next_instr,
        "plan_actions": queue_actions[1:],
        "plan_instructions": queue_instructions[1:]
    }

def create_study_buddy_graph():
    """Create and configure the Study Buddy graph."""
    workflow = StateGraph(AgentState)
    orchestrator = Orchestrator()
    
    # Add nodes
    workflow.add_node("orchestrator", orchestrator.build_plan_node)
    workflow.add_node("scheduler", scheduler)
    workflow.add_node("planner", PlanningAgent.plan_node)
    workflow.add_node("quiz_generator", quiz_node)
    workflow.add_node("explain_node", explainer_node)
    
    # Configure edges
    workflow.set_entry_point("orchestrator")
    workflow.add_edge("orchestrator", "scheduler")
    workflow.add_edge("planner", "scheduler")
    workflow.add_edge("quiz_generator", "scheduler")
    workflow.add_edge("explain_node", "scheduler")
    
    # Conditional routing from scheduler
    workflow.add_conditional_edges(
        "scheduler",
        lambda state: state["next"] if "next" in state else "END",
        {
            "planner": "planner",
            "quiz_generator": "quiz_generator", 
            "explain_node": "explain_node",
            "END": END
        }
    )
    
    return workflow.compile()

def generate_graph_visualization():
    """Generates the graph.png with detailed Mermaid syntax for current architecture."""
    
    mermaid_template = """graph TB
    %% ===== STATE =====
    state["<b>AgentState</b><br/>messages, user_prompt, plan_actions,<br/>plan_instructions, current_instruction,<br/>parsed_output, quiz_output, validation_errors,<br/>execution_log, research_memory, raw_data_storage,<br/>task_instructions, refinement_attempts, saved_plan_file, plan_data"]:::state
    
    %% ===== ENTRY POINT =====
    start(["User Input"]):::start
    orchestrator["<b>ORCHESTRATOR</b><br/>LLM: Ollama/Llama3<br/>Pipeline: Structured Output → OrchestratorPlan<br/>Tools: None"]:::orchestrator
    scheduler["<b>SCHEDULER</b><br/>Function: schedule<br/>Logic: Routes tasks from plan_actions queue"]:::scheduler
    
    %% ===== AGENT NODES =====
    planner["<b>PLANNER AGENT</b><br/>LLM: Ollama/Llama3 + Gemini<br/>Pipeline: Research → PromptTemplate → Parse JSON<br/>Output: ProjectPlan"]:::planner
    
    quiz["<b>QUIZ AGENT</b><br/>LLM: Gemini with Structured Output<br/>Pipeline: Research → Structured Quiz Schema<br/>Output: Quiz object"]:::quiz
    
    explainer["<b>EXPLAINER AGENT</b><br/>LLM: Groq/Llama-3.3-70b<br/>Pipeline: Research → Explanation Generation<br/>Output: Text explanation"]:::explainer
    
    %% ===== TOOLS =====
    plantools["<b>PlanTools</b><br/>parse_llm_output, format_plan_for_display,<br/>create_summary, save_to_json, create_execution_log_entry"]:::tool
    
    validation["<b>ValidationTools</b><br/>validate_plan, VALID_DIFFICULTIES"]:::tool
    
    orchestration["<b>OrchestrationTools</b><br/>send_user_message, save_plan_and_end"]:::tool
    
    websearch["<b>web_search_tool</b><br/>Tavily Search → Groq Summarization<br/>API: Tavily + Groq"]:::tool
    
    youtubesearch["<b>youtube_search_tool</b><br/>Groq Optimization → YouTube API<br/>API: Groq + YouTube Data API"]:::tool
    
    %% ===== HELPER NODES =====
    context_prep["<b>prepare_context</b><br/>Function: Tiered memory system<br/>Logic: Raw → Summary → Compressed based on length"]:::helper
    
    debug_log["<b>universal_debug_log</b><br/>Function: Unified logging<br/>Output: Structured debug logs to file"]:::helper
    
    %% ===== DATA MODEL RELATIONSHIPS =====
    projectplan["<b>ProjectPlan Model</b><br/>goal, duration, milestones[]"]:::model
    
    quizmodel["<b>Quiz Model</b><br/>topic, proficiency_level,<br/>mcq_questions[], article_questions[], coding_questions[]"]:::model
    
    milestone["<b>Milestone Model</b><br/>title, description, difficulty,<br/>duration, tasks[]"]:::model
    
    task["<b>Task Model</b><br/>description, difficulty, duration"]:::model
    
    %% ===== END NODE =====
    finish(["END"]):::end
    
    %% ===== MAIN FLOW =====
    start -->|user_prompt| orchestrator
    orchestrator -->|plan_actions, plan_instructions| scheduler
    scheduler -->|current_instruction, next=planner| planner
    scheduler -->|current_instruction, next=quiz| quiz
    scheduler -->|current_instruction, next=explainer| explainer
    
    %% ===== RETURN PATHS =====
    planner -->|parsed_output, messages| scheduler
    quiz -->|quiz_output, messages| scheduler
    explainer -->|messages, research_memory| scheduler
    scheduler -->|plan_actions empty| finish
    
    %% ===== TOOL CONNECTIONS =====
    planner -.-> plantools
    planner -.-> websearch
    planner -.-> youtubesearch
    
    quiz -.-> websearch
    explainer -.-> websearch
    explainer -.-> youtubesearch
    
    orchestrator -.-> validation
    orchestrator -.-> orchestration
    
    planner -.-> context_prep
    quiz -.-> context_prep
    explainer -.-> context_prep
    
    orchestrator -.-> debug_log
    planner -.-> debug_log
    quiz -.-> debug_log
    explainer -.-> debug_log
    
    %% ===== DATA MODEL CONNECTIONS =====
    plantools --> projectplan
    projectplan --> milestone
    milestone --> task
    
    quiz --> quizmodel
    
    %% ===== STYLING =====
    classDef start fill:#E8EAF6,stroke:#3F51B5,stroke-width:3px
    classDef end fill:#FCE4EC,stroke:#E91E63,stroke-width:3px
    classDef orchestrator fill:#FFF3E0,stroke:#FF9800,stroke-width:3px
    classDef scheduler fill:#F3E5F5,stroke:#9C27B0,stroke-width:2px
    classDef planner fill:#E8F5E8,stroke:#4CAF50,stroke-width:2px
    classDef quiz fill:#E3F2FD,stroke:#2196F3,stroke-width:2px
    classDef explainer fill:#FFF8E1,stroke:#FFC107,stroke-width:2px
    classDef tool fill:#E0F2F1,stroke:#009688,stroke-width:2px
    classDef helper fill:#F5F5F5,stroke:#757575,stroke-width:2px
    classDef model fill:#FFEBEE,stroke:#F44336,stroke-width:2px
    classDef state fill:#E1F5FE,stroke:#03A9F4,stroke-width:4px
    """
    
    print("\n" + "="*80)
    print("GENERATING GRAPH VISUALIZATION")
    print("="*80)
    
    try:
        # First, save the Mermaid code
        with open("graph.mmd", "w", encoding="utf-8") as f:
            f.write(mermaid_template)
        print("✅ Mermaid code saved to 'graph.mmd'")
        
        # Also save as Markdown for easy viewing
        with open("graph_architecture.md", "w", encoding="utf-8") as f:
            f.write("# Study Buddy AI - System Architecture\n\n")
            f.write("## Graph Visualization\n\n")
            f.write("```mermaid\n")
            f.write(mermaid_template)
            f.write("\n```\n\n")
            f.write("## Architecture Summary\n\n")
            f.write("### Core Components:\n")
            f.write("1. **Orchestrator** (Ollama/Llama3) - Routes user requests to appropriate agents\n")
            f.write("2. **Planner Agent** (Ollama + Gemini) - Creates structured study roadmaps\n")
            f.write("3. **Quiz Agent** (Gemini) - Generates quizzes with structured output\n")
            f.write("4. **Explainer Agent** (Groq) - Provides explanations with research\n")
            f.write("5. **Scheduler** - Manages sequential task execution\n\n")
            f.write("### Key Tools:\n")
            f.write("- **PlanTools** - Plan parsing, formatting, and saving\n")
            f.write("- **ValidationTools** - Plan validation and rule checking\n")
            f.write("- **OrchestrationTools** - User messaging and plan saving\n")
            f.write("- **web_search_tool** - Tavily search with Groq summarization\n")
            f.write("- **youtube_search_tool** - YouTube search with Groq optimization\n\n")
            f.write("### Helper Functions:\n")
            f.write("- **prepare_context** - Tiered memory system for LLM context\n")
            f.write("- **universal_debug_log** - Unified logging across all nodes\n\n")
            f.write("### Data Models:\n")
            f.write("- **ProjectPlan** - Goal, duration, milestones\n")
            f.write("- **Quiz** - Topic, proficiency level, questions\n")
            f.write("- **Milestone** - Title, description, difficulty, tasks\n")
            f.write("- **Task** - Description, difficulty, duration\n\n")
            f.write("### State Management:\n")
            f.write("AgentState contains all workflow data including messages, plans, instructions, memory, and execution logs.")
        
        print("✅ Architecture documentation saved to 'graph_architecture.md'")
        
        # Generate PNG via Kroki
        print("📡 Generating PNG via Kroki API...")
        resp = requests.post(
            "https://kroki.io/mermaid/png",
            data=mermaid_template.encode("utf-8"),
            headers={"Content-Type": "text/plain"},
            timeout=30
        )
        
        if resp.status_code == 200:
            with open("graph.png", "wb") as f:
                f.write(resp.content)
            
            # Try to get dimensions
            try:
                from PIL import Image
                img = Image.open("graph.png")
                width, height = img.size
                print(f"✅ Graph generated successfully!")
                print(f"   📏 Dimensions: {width}x{height} pixels")
                print(f"   💾 Saved as: 'graph.png'")
                print(f"   📄 Mermaid source: 'graph.mmd'")
                print(f"   📋 Documentation: 'graph_architecture.md'")
            except ImportError:
                print(f"✅ Graph generated successfully! Saved as 'graph.png'")
            except Exception as e:
                print(f"✅ Graph generated! Saved as 'graph.png' (Could not read dimensions: {e})")
            
            return True
        else:
            print(f"❌ Kroki API Error {resp.status_code}: {resp.text[:200]}")
            
            # Try alternative endpoint
            print("🔄 Trying alternative Mermaid renderer...")
            try:
                # Try mermaid.ink as alternative
                import urllib.parse
                encoded = urllib.parse.quote(mermaid_template)
                resp2 = requests.get(f"https://mermaid.ink/img/{encoded}", timeout=30)
                
                if resp2.status_code == 200:
                    with open("graph.png", "wb") as f:
                        f.write(resp2.content)
                    print("✅ Graph generated successfully via alternative renderer!")
                    return True
            except Exception as e2:
                print(f"❌ Alternative renderer also failed: {e2}")
            
            return False
            
    except Exception as e:
        print(f"❌ Error generating graph: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        # Run the actual application
        print("Running Study Buddy application...")
        from main import main
        main()
    else:
        # Generate visualization by default
        success = generate_graph_visualization()
        
        if success:
            print("\n" + "="*80)
            print("GRAPH GENERATION COMPLETE")
            print("="*80)
            print("\nFiles created:")
            print("1. graph.png - Visual graph diagram")
            print("2. graph.mmd - Mermaid source code")
            print("3. graph_architecture.md - Detailed documentation")
            print("\nTo run the application: python structure_drawer.py run")
        else:
            print("\n❌ Graph generation failed. Check the error messages above.")
            print("\n⚠️  You can still view the Mermaid code in 'graph.mmd'")
            print("   Try opening it in a Mermaid editor like:")
            print("   - https://mermaid.live")
            print("   - https://mermaid-js.github.io/mermaid-live-editor/")
        
        sys.exit(0 if success else 1)