import json
from langsmith import traceable
from graph import get_study_buddy_app
from log import log_and_print

@traceable
def main():
    # Get the compiled application
    app = get_study_buddy_app()
    
    # Example prompts to test
    # prompt = "Search for a video on recursion, explain it, then make a quiz."
    # prompt = "Search on each topic and make a quiz on each topic of those: python, C++, Java"
    # prompt = "Make a quiz on mohamed salah and make a search on salah last goal"
    # prompt = "Search for the latest trends in Agentic AI market and make RoadMap from zero to hero"
    
    prompt = f"Create a 12-week Agentic AI roadmap. Do NOT include 'planning' or 'roadmap creation' as tasks; assume this output IS the final actionable plan. Focus 100% on specific technical milestones (e.g., 'Mastering LangGraph', 'Implementing Tool-Use')."
    
    log_and_print("SESSION START", prompt)
    
    # Initial state
    initial_state = {
        "user_prompt": prompt, 
        "messages": [], 
        "plan_actions": [],
        "plan_instructions": [],
        "research_memory": [],
        "raw_data_storage": [],
        "execution_log": [],
        "validation_errors": [],
        "refinement_attempts": 0
    }
    
    # Stream the execution
    for state_snapshot in app.stream(initial_state, stream_mode="values"):
        from log import universal_debug_log
        universal_debug_log("GRAPH_MANAGER", "POST_NODE_STATE_UPDATE", state_snapshot)
        
        print("\n" + "💠" * 30)
        print("📜 FULL STATE UPDATE:")
        
        # Clean up message display for CMD
        display_state = state_snapshot.copy()
        if "messages" in display_state:
            display_state["messages"] = [f"{m.name}: {m.content[:50]}..." for m in display_state["messages"]]
        
        print(json.dumps(display_state, indent=2, default=lambda o: "<Structured Object>"))
        print("💠" * 30 + "\n")

if __name__ == "__main__":
    main()