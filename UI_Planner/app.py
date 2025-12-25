# app.py
import streamlit as st
import json
from streamlit_agraph import agraph, Node, Edge, Config as AgConfig
from graph import app_graph, editor_graph # Import the new editor_graph
from state import PlanState

st.set_page_config(layout="wide", page_title="AI Planner Agent")

# --- SESSION STATE INITIALIZATION ---
if "plan_json" not in st.session_state:
    st.session_state.plan_json = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "clicked_node" not in st.session_state:
    st.session_state.clicked_node = None

# Sidebar
with st.sidebar:
    st.header("Planner Settings")
    user_goal = st.text_input("What do you want to learn?", "Python Backend Development")
    
    # --- FEATURE 4: SAVE JSON ---
    if st.session_state.get("plan_json"):
        st.divider()
        json_str = json.dumps(st.session_state.plan_json, indent=2)
        st.download_button(
            label="Download Roadmap (JSON)",
            data=json_str,
            file_name="roadmap.json",
            mime="application/json"
        )
    
    if st.button("Generate Roadmap"):
        with st.spinner("AI is thinking..."):
            initial_state = PlanState(
                user_request=user_goal, attempt_count=0, messages=[], 
                current_plan={}, feedback=None, search_context="", 
                ui_selected_node=None, raw_output="", error=None
            )
            # Invoke Graph
            result = app_graph.invoke(initial_state)
            
            # --- FEATURE 2: BLUE ERROR MESSAGE ---
            if result.get("error"):
                st.markdown(
                    f"""
                    <div style='background-color: #00008B; color: white; padding: 10px; border-radius: 5px;'>
                        <strong>System Error:</strong> {result['error']}<br>
                        Please try adding more details to your request.
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            else:
                st.session_state.plan_json = result["current_plan"]
                st.session_state.chat_history.append({"role": "ai", "content": "Roadmap Generated!"})

# --- MAIN UI: GRAPH VISUALIZATION ---
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Interactive Roadmap")
    if st.session_state.plan_json:
        # Convert JSON to Nodes/Edges for AGraph
        nodes = []
        edges = []
        
        milestones = st.session_state.plan_json.get("milestones", [])
        
        for i, ms in enumerate(milestones):
            # Status Color Logic
            color = "#grey"
            if ms.get("status") == "in progress": color = "#3498db" # Blue
            elif ms.get("status") == "done": color = "#2ecc71" # Green
            
            # Milestone Node
            nodes.append(Node(id=ms.get("id", f"m{i}"), 
                              label=ms["title"], 
                              size=25, 
                              color=color))
            
            # Connect Milestones sequentially
            if i > 0:
                prev_id = milestones[i-1].get("id", f"m{i-1}")
                curr_id = ms.get("id", f"m{i}")
                edges.append(Edge(source=prev_id, target=curr_id, type="CURVE_SMOOTH"))

        # Interactive Graph Config
        config = AgConfig(width=700, height=500, directed=True, nodeHighlightBehavior=True)
        
        # RENDER GRAPH
        # return_value gets the ID of the clicked node
        selected_node_id = agraph(nodes=nodes, edges=edges, config=config)
        
        if selected_node_id:
            st.session_state.clicked_node = selected_node_id

# --- DETAIL PANEL & CHAT ---
with col2:
    st.subheader("Details & Edit")
    
    # --- FEATURE 5: FULL DETAILS ---
    if st.session_state.clicked_node and st.session_state.plan_json:
        # Safely find node
        milestones = st.session_state.plan_json.get("milestones", [])
        ms_data = next((m for m in milestones if m.get("id") == st.session_state.clicked_node), None)
        
        if ms_data:
            st.markdown(f"### {ms_data['title']}")
            st.caption(f"Status: {ms_data['status'].upper()}")
            st.write(ms_data['description'])
            
            st.markdown("#### Tasks")
            for task in ms_data.get("tasks", []):
                icon = "✅" if ms_data['status'] == 'done' else "⬜"
                st.markdown(f"**{icon} {task['name']}**")
                st.write(f"_{task['description']}_")
                
                # Show Resources if they exist
                if task.get('resources'):
                    st.markdown(f"📚 [Read Resource]({task['resources']})")
                if task.get('youtube'):
                    st.markdown(f"📺 [Watch Video]({task['youtube']})")
                st.divider()

    # --- CHAT INTERFACE ---
    # st.divider()
    st.subheader("Chat Assistant")
    
    # Display History
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            
    # Input
    user_input = st.chat_input("Ask to change plan or update status...")
    if user_input := st.chat_input("Update plan..."):
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        with st.spinner("Updating Plan..."):
            state_update = PlanState(
                current_plan=st.session_state.plan_json,
                messages=st.session_state.chat_history,
                user_request=user_goal,
                ui_selected_node=st.session_state.clicked_node,
                attempt_count=0, feedback=None, search_context="", raw_output="", error=None
            )
            
            # Use the new EDITOR GRAPH which includes validation
            result = editor_graph.invoke(state_update)
            
            if result.get("error"):
                 st.markdown(
                    f"""
                    <div style='background-color: #00008B; color: white; padding: 10px; border-radius: 5px;'>
                        <strong>Update Failed:</strong> {result['error']}<br>
                        I couldn't process that update. Please be more specific.
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            else:
                st.session_state.plan_json = result["current_plan"]
                st.session_state.chat_history.append({"role": "ai", "content": "Plan updated!"})
                st.rerun()