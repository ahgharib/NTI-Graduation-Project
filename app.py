import streamlit as st
import json
import uuid
from streamlit_agraph import agraph, Node, Edge, Config as AgConfig
from graph import app_graph, editor_graph 
from state import PlanState
from chat_graph import study_buddy_graph 
import os
import tempfile
from RAG.ingest import ingest_pdf
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from chat_tools import summarize_history
from search_agent import search_with_agent  # Import the search agent

UPLOAD_DIR = "RAG/data"
os.makedirs(UPLOAD_DIR, exist_ok=True)
st.set_page_config(layout="wide", page_title="AI Planner Agent")

# --- SESSION STATE INITIALIZATION ---
if "plan_json" not in st.session_state:
    st.session_state.plan_json = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "editor_chat_history" not in st.session_state:
    st.session_state.editor_chat_history = []
if "clicked_node" not in st.session_state:
    st.session_state.clicked_node = None
if "new_chat_input" not in st.session_state:
    st.session_state.new_chat_input = ""
if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = {}
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None  # chunk summaries
if "file_vectorstore" not in st.session_state:
    st.session_state.file_vectorstore = None # file summaries
    st.session_state.vectorstore = None
if "search_query" not in st.session_state:
    st.session_state.search_query = ""
if "is_searching" not in st.session_state:
    st.session_state.is_searching = False

def select_milestone(node_id):
    st.session_state.clicked_node = node_id

def perform_search():
    """Execute search and update chat history."""
    if st.session_state.search_query.strip():
        # Get the context from selected milestone
        context = ""
        if st.session_state.clicked_node and st.session_state.plan_json:
            ms_data = next((m for m in st.session_state.plan_json.get("milestones", []) 
                          if m.get("id") == st.session_state.clicked_node), None)
            if ms_data:
                context = ms_data['title']
        
        # Add user message to chat history
        user_message = f"🔍 Search: {st.session_state.search_query}"
        st.session_state.chat_history.append({"role": "user", "content": user_message})
        
        # Set searching flag
        st.session_state.is_searching = True
        
        # Perform the search
        try:
            with st.spinner("Searching web and YouTube..."):
                search_result = search_with_agent(
                    query=st.session_state.search_query,
                    context=context
                )
            
            # Add search result to chat history
            st.session_state.chat_history.append({
                "role": "ai", 
                "content": search_result
            })
            
        except Exception as e:
            error_msg = f"Search failed: {str(e)}"
            st.session_state.chat_history.append({"role": "ai", "content": error_msg})
        
        # Reset search state
        st.session_state.is_searching = False
        st.session_state.search_query = ""
        
        # Rerun to update UI
        st.rerun()

# --- SIDEBAR ---
with st.sidebar:
    st.header("Planner Settings")
    user_goal = st.text_input("What do you want to learn?", "Python Backend Development")
    
    # Custom CSS to hide the box and style the button
    st.markdown("""
        <style>
            [data-testid="stFileUploader"] section {
                padding: 0 !important;
            }
            [data-testid="stFileUploaderDropzoneInstructions"],
            [data-testid="stFileUploaderDropzone"] > div:first-child,
            [data-testid="stFileUploaderDeleteBtn"] {
                display: none !important;
            }
            [data-testid="stFileUploaderDropzone"] {
                border: none !important;
                background: transparent !important;
                padding: 0 !important;
                margin: 0 !important;
                min-height: 0px !important;
            }
            [data-testid="stFileUploaderDropzone"] button {
                width: 45px !important;
                height: 45px !important;
                border-radius: 8px !important;
                color: transparent !important; 
                overflow: hidden !important;
                position: relative !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                margin-top: 5px !important;
            }
            [data-testid="stFileUploaderDropzone"] button * {
                display: none !important;
            }
            [data-testid="stFileUploaderDropzone"] button::after {
                content: "📂";
                display: block !important;
                color: white !important; 
                font-size: 20px !important;
                position: absolute !important;
                left: 50% !important;
                top: 50% !important;
                transform: translate(-50%, -50%) !important;
                visibility: visible !important;
            }
        </style>
    """, unsafe_allow_html=True)

    col_gen, col_up = st.columns([4, 1])
    with col_gen:
        generate_clicked = st.button("Generate Roadmap", use_container_width=True)
    with col_up:
        uploaded_file = st.file_uploader("", type=["json"], label_visibility="collapsed", key="file_up_icon")

    if uploaded_file is not None:
        if "last_uploaded_file" not in st.session_state or st.session_state.last_uploaded_file != uploaded_file.name:
            try:
                uploaded_data = json.load(uploaded_file)
                st.session_state.plan_json = uploaded_data
                st.session_state.chat_history.append({"role": "ai", "content": f"Loaded: {uploaded_file.name}"})
                st.session_state.last_uploaded_file = uploaded_file.name
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

    if generate_clicked:
        with st.spinner("AI is thinking..."):
            initial_state = PlanState(
                user_request=user_goal, attempt_count=0, messages=[], 
                current_plan={}, feedback=None, search_context="", 
                ui_selected_node=None, raw_output="", error=None
            )
            result = app_graph.invoke(initial_state)
            if result.get("error"):
                st.error(f"System Error: {result['error']}")
            else:
                st.session_state.plan_json = result["current_plan"]
                st.session_state.chat_history.append({"role": "ai", "content": "Roadmap Generated!"})
                st.rerun()

    if st.session_state.get("plan_json"):
        st.divider()
        json_str = json.dumps(st.session_state.plan_json, indent=2)
        st.download_button(
            label="Download Roadmap (JSON)",
            data=json_str,
            file_name="roadmap.json",
            mime="application/json",
            use_container_width=True
        )

    if st.session_state.plan_json:
        st.divider()
        st.subheader("Milestones List")
        for ms in st.session_state.plan_json.get("milestones", []):
            ms_id = ms.get("id")
            is_selected = st.session_state.clicked_node == ms_id
            button_label = f"📍 {ms['title']}" if is_selected else ms['title']
            if st.button(button_label, key=f"btn_{ms_id}", use_container_width=True):
                select_milestone(ms_id)
                st.rerun()

# --- MAIN UI LAYOUT ---
col1, col2 = st.columns([3, 1.2])

with col1:
    st.subheader("Interactive Roadmap")
    if st.session_state.plan_json:
        nodes = []
        edges = []
        milestones = st.session_state.plan_json.get("milestones", [])
        for i, ms in enumerate(milestones):
            ms_id = ms.get("id", f"m{i}")
            
            # --- COLOR LOGIC EDITED HERE ---
            # Default node color is Red. If selected, it becomes Blue.
            is_selected = st.session_state.clicked_node == ms_id
            node_color = "blue" if is_selected else "red"
            node_size = 35 if is_selected else 25
            
            nodes.append(Node(
                id=ms_id, 
                label=ms["title"], 
                size=node_size, 
                color=node_color,
                font={'color': 'green'}  # LABEL COLOR: Green
            ))
            
            if i > 0:
                prev_id = milestones[i-1].get("id", f"m{i-1}")
                edges.append(Edge(source=prev_id, target=ms_id, type="CURVE_SMOOTH"))

        # Config updated to ensure label colors and highlighting work as expected
        config = AgConfig(
            width=700, 
            height=400, 
            directed=True,
            physics=False, 
            nodeHighlightBehavior=True,
            highlightColor="blue"
        )
        
        returned_id = agraph(nodes=nodes, edges=edges, config=config)
        if returned_id and returned_id != st.session_state.clicked_node:
            st.session_state.clicked_node = returned_id
            st.rerun()

    st.divider()
    st.subheader("🤖 Smart Assistant")
    
    selected_ms_text = ""
    if st.session_state.plan_json and st.session_state.clicked_node:
        ms_data = next((m for m in st.session_state.plan_json.get("milestones", []) 
                       if m.get("id") == st.session_state.clicked_node), None)
        if ms_data:
            st.info(f"Focused on: **{ms_data['title']}**")
            selected_ms_text = f"Title: {ms_data['title']}. Description: {ms_data['description']}."

    chat_box = st.container(height=500, border=True)
    with chat_box:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
        
        # Show loading animation when searching
        if st.session_state.is_searching:
            with st.chat_message("assistant"):
                st.markdown(
                    """
                    <div style="display: flex; justify-content: center; align-items: center; margin-bottom: 20px;">
                        <img src="https://i.gifer.com/ZKZg.gif" width="60" />
                    </div>
                    """, 
                    unsafe_allow_html=True
                )

    # UPDATED: Search input and buttons layout
    search_col, web_col, yt_col, doc_col = st.columns([0.7, 0.1, 0.1, 0.1])
    
    with search_col:
        # Search input that triggers on Enter
        search_query = st.text_input(
            "Search query...",
            value=st.session_state.search_query,
            key="search_input",
            on_change=perform_search,
            label_visibility="collapsed",
            placeholder="Search the web and YouTube..."
        )
        if search_query:
            st.session_state.search_query = search_query
    
    with web_col:
        # Search button
        if st.button("🔍", use_container_width=True, help="Search the web and YouTube"):
            if st.session_state.search_query:
                perform_search()
            else:
                st.warning("Please enter a search query first")
    
    with yt_col:
        st.button("📺", use_container_width=True, help="Direct YouTube search (coming soon)")
    
    with doc_col:
        uploaded_docs = st.file_uploader(
            "📄",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            help="Upload documents to use as reference"
        )
        # if st.session_state.uploaded_docs:
        #     for name in st.session_state.uploaded_docs:
        #         st.caption(f"• {name}")

    if uploaded_docs:
        for uploaded_doc in uploaded_docs:

            if uploaded_doc.name in st.session_state.uploaded_docs:
                continue  # already uploaded this session

            file_path = os.path.join(UPLOAD_DIR, uploaded_doc.name)

            with open(file_path, "wb") as f:
                f.write(uploaded_doc.getbuffer())
            
            st.session_state.uploaded_docs[uploaded_doc.name] = file_path

    if st.button("📚 Index Documents"):
        embeddings = OllamaEmbeddings(model="nomic-embed-text")

        chunk_docs = []
        file_docs = []

        for path in st.session_state.uploaded_docs.values():
            chunks, file_summary = ingest_pdf(path)
            chunk_docs.extend(chunks)
            file_docs.append(file_summary)

        st.session_state.vectorstore = FAISS.from_documents(
            chunk_docs, embeddings
        )

        st.session_state.file_vectorstore = FAISS.from_documents(
            file_docs, embeddings
        )

        st.success("Documents indexed successfully")

    # Regular chat input (separate from search)
    user_input = st.chat_input("Ask about your plan, request a quiz, or explain a topic...")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with chat_box:
            st.chat_message("user").write(user_input)

        with st.spinner("Assistant is working..."):
            history_context = ""
            if len(st.session_state.chat_history) > 1:
                previous_turns = st.session_state.chat_history[:-1]
                history_context = summarize_history(previous_turns)
                
            initial_chat_state = {
                "user_prompt": user_input,
                "messages": [],
                "plan_actions": [],
                "plan_instructions": [],
                "research_memory": [],
                "raw_data_storage": [],
                "execution_log": [],
                "validation_errors": [],
                "refinement_attempts": 0,
                "plan_data": st.session_state.plan_json,
                "selected_milestone_context": selected_ms_text,
                "conversation_summary": history_context
            }

            try:
                for chunk in study_buddy_graph.stream(initial_chat_state, stream_mode="updates"):
                    if "orchestrator" in chunk:
                        plan = chunk["orchestrator"]
                        actions = plan.get("plan_actions", [])
                        if "END" in actions:
                            with chat_box:
                                st.chat_message("ai").write("I cannot handle this request based on my capabilities.")
                                st.session_state.chat_history.append({"role": "ai", "content": "I cannot handle this request."})
                        else:
                            plan_msg = f"🔍 **Plan:** {', '.join(actions)}"
                            with chat_box: st.caption(plan_msg)

                    if "explain_node" in chunk:
                        response_messages = chunk["explain_node"].get("messages", [])
                        for msg in response_messages:
                            with chat_box: st.chat_message("ai").write(msg.content)
                            st.session_state.chat_history.append({"role": "ai", "content": msg.content})

                    if "quiz_generator" in chunk:
                        quiz_data = chunk["quiz_generator"].get("quiz_output")
                        if quiz_data:
                            quiz_str = f"**Quiz: {quiz_data.topic}**\n\n"
                            for q in quiz_data.mcq_questions:
                                quiz_str += f"❓ {q.question}\n"
                                for opt in q.options: quiz_str += f"- {opt}\n"
                                quiz_str += f"*(Answer: {q.correct_answer})*\n\n"
                            with chat_box: st.chat_message("ai").markdown(quiz_str)
                            st.session_state.chat_history.append({"role": "ai", "content": quiz_str})
                    
                    if "summarizer" in chunk:
                        response_messages = chunk["summarizer"].get("messages", [])
                        for msg in response_messages:
                            with chat_box: st.chat_message("ai").write(msg.content)
                            st.session_state.chat_history.append({"role": "ai", "content": msg.content})
            except Exception as e:
                st.error(f"Pipeline Error: {str(e)}")

with col2:
    st.subheader("🛠️ Roadmap Manager")
    editor_container = st.container(height=300, border=True)
    with editor_container:
        for msg in st.session_state.editor_chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
            
    editor_input = st.chat_input("Update plan structure...", key="editor_input")
    if editor_input:
        st.session_state.editor_chat_history.append({"role": "user", "content": editor_input})
        with st.spinner("Updating Plan..."):
            state_update = PlanState(
                current_plan=st.session_state.plan_json,
                messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.editor_chat_history],
                user_request=user_goal,
                ui_selected_node=st.session_state.clicked_node,
                attempt_count=0, feedback=None, search_context="", raw_output="", error=None
            )
            result = editor_graph.invoke(state_update)
            if not result.get("error"):
                st.session_state.plan_json = result["current_plan"]
                st.session_state.editor_chat_history.append({"role": "ai", "content": "Plan updated!"})
                st.rerun()
    
    st.divider()

    st.subheader("📝 Milestone Details")
    if st.session_state.clicked_node and st.session_state.plan_json:
        ms_data = next((m for m in st.session_state.plan_json.get("milestones", []) 
                       if m.get("id") == st.session_state.clicked_node), None)
        if ms_data:
            st.markdown(f"### {ms_data['title']}")
            st.caption(f"Status: {ms_data['status'].upper()}")
            st.write(ms_data['description'])
            
            st.markdown("#### Tasks")
            for task in ms_data.get("tasks", []):
                icon = "✅" if ms_data['status'] == 'done' else "⬜"
                st.markdown(f"**{icon} {task['name']}**")
                st.write(f"_{task['description']}_")
                if task.get('resources'): st.markdown(f"📚 [Read Resource]({task['resources']})")
                if task.get('youtube'): st.markdown(f"📺 [Watch Video]({task['youtube']})")
                st.divider()