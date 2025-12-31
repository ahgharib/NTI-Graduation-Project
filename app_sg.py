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
from search_agent import search_with_agent

# --- SETUP & CONFIGURATION ---
UPLOAD_DIR = "RAG/data"
os.makedirs(UPLOAD_DIR, exist_ok=True)
st.set_page_config(layout="wide", page_title="AI Planner Agent", page_icon="🎓")

# --- DARK MODE CSS STYLING ---
st.markdown("""
    <style>
        /* 1. Global Dark Theme & Fonts */
        .stApp {
            background-color: #0e1117;
            color: #e0e0e0;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        /* 2. Headers */
        h1, h2, h3, h4 { color: #ffffff !important; font-weight: 600; }
        p, .stMarkdown, .stText { color: #cfd8dc !important; line-height: 1.6; }
        
        /* 3. Panel Containers (Scrollable areas) */
        [data-testid="stVerticalBlockBorderWrapper"] {
            background-color: #1a1c24; 
            border-radius: 12px;
            border: 1px solid #2d3748;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        
        /* 4. Sidebar */
        [data-testid="stSidebar"] {
            background-color: #15171e;
            border-right: 1px solid #2d3748;
        }
        
        /* 5. Buttons */
        .stButton > button {
            border-radius: 8px;
            font-weight: 500;
            border: 1px solid #4a5568;
            background-color: #2d3748;
            color: #e2e8f0;
        }
        .stButton > button:hover {
            border-color: #63b3ed;
            color: #ffffff;
            background-color: #3182ce;
        }
        /* Primary Button Style */
        .stButton > button[kind="primary"] {
            background-color: #3182ce;
            border-color: #3182ce;
            color: white;
        }

        /* 6. Chat Bubbles */
        [data-testid="stChatMessage"] {
            background-color: transparent;
            padding: 0.5rem 1rem;
            border-radius: 8px;
        }
        [data-testid="stChatMessage"][data-testid="stChatMessageUser"] {
            background-color: #2d3748;
        }

        /* 7. Hide default file uploader junk */
        [data-testid="stFileUploaderDropzoneInstructions"] { display: none; }
        
        /* 8. Expander Styling */
        .streamlit-expanderHeader {
            background-color: #1a1c24;
            color: #e2e8f0;
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
if "plan_json" not in st.session_state:
    st.session_state.plan_json = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "editor_chat_history" not in st.session_state:
    st.session_state.editor_chat_history = []
if "clicked_node" not in st.session_state:
    st.session_state.clicked_node = None
if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = {}
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None 
if "file_vectorstore" not in st.session_state:
    st.session_state.file_vectorstore = None
    st.session_state.vectorstore = None
if "search_query" not in st.session_state:
    st.session_state.search_query = ""
if "is_searching" not in st.session_state:
    st.session_state.is_searching = False
if "active_quiz" not in st.session_state:
    st.session_state.active_quiz = None 
if "quiz_answers" not in st.session_state:
    st.session_state.quiz_answers = {}
if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False
if "quiz_thread_id" not in st.session_state:
    st.session_state.quiz_thread_id = str(uuid.uuid4())

# --- FUNCTIONS ---

def select_milestone(node_id):
    st.session_state.clicked_node = node_id

def perform_search_logic(query):
    """Refactored to be called from the modal."""
    if query.strip():
        # Get context
        context = ""
        if st.session_state.clicked_node and st.session_state.plan_json:
            ms_data = next((m for m in st.session_state.plan_json.get("milestones", []) 
                          if m.get("id") == st.session_state.clicked_node), None)
            if ms_data:
                context = ms_data['title']
        
        # Update history
        user_message = f"🔍 Search: {query}"
        st.session_state.chat_history.append({"role": "user", "content": user_message})
        st.session_state.is_searching = True
        
        try:
            # Logic runs here, wrapped by spinner in the caller function
            search_result = search_with_agent(query=query, context=context)
            st.session_state.chat_history.append({"role": "ai", "content": search_result})
        except Exception as e:
            st.session_state.chat_history.append({"role": "ai", "content": f"Search failed: {str(e)}"})
        
        st.session_state.is_searching = False
        st.session_state.search_query = ""

# --- MODALS (DIALOGS) ---

@st.dialog("Research Assistant")
def search_modal():
    st.caption("Ask a research question using the AI agent.")
    with st.form("search_form"):
        query = st.text_input("Enter topic to research...", placeholder="e.g. Best practices for REST APIs")
        submit_search = st.form_submit_button("Start Search")
        
        if submit_search and query:
            with st.spinner("🔎 Researching topic... this may take a moment"):
                perform_search_logic(query)
            st.rerun()

@st.dialog("Knowledge Base")
def doc_modal():
    st.caption("Upload documents to provide context for the AI.")
    
    st.markdown("##### 📄 Context Documents")
    uploaded_docs = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    
    if uploaded_docs:
        if st.button("Process & Index Documents", use_container_width=True, type="primary"):
            with st.spinner("Indexing Knowledge Base..."):
                for uploaded_doc in uploaded_docs:
                    if uploaded_doc.name not in st.session_state.uploaded_docs:
                        file_path = os.path.join(UPLOAD_DIR, uploaded_doc.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_doc.getbuffer())
                        st.session_state.uploaded_docs[uploaded_doc.name] = file_path

                embeddings = OllamaEmbeddings(model="nomic-embed-text")
                chunk_docs = []
                file_docs = []
                for path in st.session_state.uploaded_docs.values():
                    chunks, file_summary = ingest_pdf(path)
                    chunk_docs.extend(chunks)
                    file_docs.append(file_summary)

                st.session_state.vectorstore = FAISS.from_documents(chunk_docs, embeddings)
                st.session_state.file_vectorstore = FAISS.from_documents(file_docs, embeddings)
            st.success("✅ Documents Indexed Successfully!")
            st.rerun()

# --- SIDEBAR (FIXED) ---
with st.sidebar:
    st.header("🎓 AI Planner")
    
    st.markdown("### 🎯 Your Goal")
    user_goal = st.text_input("Goal", "Python Backend Development", label_visibility="collapsed", placeholder="E.g., Learn React")
    
    # 1. GENERATE BUTTON
    if st.button("🚀 Generate New Roadmap", use_container_width=True, type="primary"):
        with st.spinner("Generative AI is thinking..."):
            initial_state = PlanState(
                user_request=user_goal, attempt_count=0, messages=[], 
                current_plan={}, feedback=None, search_context="", 
                ui_selected_node=None, raw_output="", error=None
            )
            result = app_graph.invoke(initial_state)
            if result.get("error"):
                st.error(f"Error: {result['error']}")
            else:
                st.session_state.plan_json = result["current_plan"]
                st.session_state.chat_history.append({"role": "ai", "content": "Roadmap Generated Successfully!"})
                st.rerun()
    
    # 2. UPLOAD SECTION
    st.write("") # Spacer
    st.caption("OR")
    with st.expander("📂 Load Existing Plan"):
        uploaded_file = st.file_uploader("Upload JSON", type=["json"], label_visibility="collapsed")
        if uploaded_file is not None:
            if "last_uploaded_file" not in st.session_state or st.session_state.last_uploaded_file != uploaded_file.name:
                try:
                    uploaded_data = json.load(uploaded_file)
                    st.session_state.plan_json = uploaded_data
                    st.session_state.chat_history.append({"role": "ai", "content": f"Loaded plan: {uploaded_file.name}"})
                    st.session_state.last_uploaded_file = uploaded_file.name
                    st.rerun()
                except Exception as e:
                    st.error(f"File Error: {e}")

    st.divider()

    # 3. NAVIGATION SECTION
    st.markdown("### 📍 Navigation")
    
    if st.session_state.plan_json:
        # Download Button
        st.download_button(
            "📥 Export JSON", 
            data=json.dumps(st.session_state.plan_json, indent=2), 
            file_name="roadmap.json", 
            mime="application/json", 
            use_container_width=True
        )
        
        st.write("") # Spacer

        # Scrollable Navigation List
        with st.container(height=400):
            milestones = st.session_state.plan_json.get("milestones", [])
            for ms in milestones:
                ms_id = ms.get("id")
                is_selected = st.session_state.clicked_node == ms_id
                
                # Visual indicator for selection
                icon = "🔵" if is_selected else "⚪"
                label = ms['title']
                
                # Button for each milestone
                if st.button(f"{icon} {label}", key=f"nav_{ms_id}", use_container_width=True):
                    select_milestone(ms_id)
                    st.rerun()
    else:
        st.info("Generate a plan to see your milestones here.")

# --- MAIN UI LAYOUT ---
col1, col2 = st.columns([2.5, 1.2])

# LEFT COLUMN: Graph & Chat
with col1:
    # 1. Top Section: Visual Graph
    if st.session_state.plan_json:
        with st.expander("🗺️ Interactive Roadmap View", expanded=True):
            nodes = []
            edges = []
            milestones = st.session_state.plan_json.get("milestones", [])
            for i, ms in enumerate(milestones):
                ms_id = ms.get("id", f"m{i}")
                is_selected = st.session_state.clicked_node == ms_id
                
                # --- VISUALS ---
                node_color = "#2563eb" if is_selected else "#dc2626" 
                node_size = 30
                
                step_number = str(i + 1)
                
                nodes.append(Node(
                    id=ms_id, 
                    label=step_number,      
                    title=ms["title"],      
                    shape="circle",         
                    size=node_size, 
                    color=node_color,
                    # --- FONT CONFIGURATION FOR CODING STYLE ---
                    font={
                        'color': 'white',
                        'size': 16,
                        'face': 'Courier New, Courier, monospace', 
                        'align': 'center',
                        'bold': True
                    },
                    borderWidth=2,
                    borderWidthSelected=4
                ))
                
                if i > 0:
                    prev_id = milestones[i-1].get("id", f"m{i-1}")
                    edges.append(Edge(source=prev_id, target=ms_id, type="CURVE_SMOOTH", arrows="to", color="#6b7280"))

            config = AgConfig(
                width=None, 
                height=300, 
                directed=True, 
                physics=True, 
                hierarchical=False, 
                nodeHighlightBehavior=True, 
                highlightColor="#60a5fa", 
                backgroundColor="#0e1117",
                physicsOptions={
                    'stabilization': {'enabled': True, 'iterations': 1000},
                    'barnesHut': {
                        'gravitationalConstant': -2000,
                        'centralGravity': 0.3,
                        'springLength': 95,
                        'springConstant': 0.04,
                        'damping': 0.09,
                        'avoidOverlap': 0.1
                    }
                }
            )
            returned_id = agraph(nodes=nodes, edges=edges, config=config)
            if returned_id and returned_id != st.session_state.clicked_node:
                st.session_state.clicked_node = returned_id
                st.rerun()

    # 2. Bottom Section: Chat Interface
    st.subheader("💬 Study Assistant")
    
    # Context Banner
    selected_ms_text = ""
    if st.session_state.plan_json and st.session_state.clicked_node:
        ms_data = next((m for m in st.session_state.plan_json.get("milestones", []) if m.get("id") == st.session_state.clicked_node), None)
        if ms_data:
            st.caption(f"Currently focused on: **{ms_data['title']}**")
            selected_ms_text = f"Title: {ms_data['title']}. Description: {ms_data['description']}."

    # Chat History Container (Fixed height for scrolling)
    chat_container = st.container(height=500)
    with chat_container:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
        if st.session_state.is_searching:
            with st.chat_message("assistant"):
                st.write("🔍 Researching...")

    # Chat Tools Toolbar (Now with 3 buttons)
    # Adjust column ratios to fit 3 buttons nicely
    tool_c1, tool_c2, tool_c3, tool_space = st.columns([0.15, 0.15, 0.15, 0.55])
    
    with tool_c1:
        if st.button("🎤 Speech", help="Voice Input", use_container_width=True):
            st.toast("Voice input coming soon!")
    with tool_c2:
        if st.button("🔍 Search", help="Open Research Agent", use_container_width=True):
            search_modal()
    with tool_c3:
        if st.button("📂 Upload", help="Upload Documents", use_container_width=True):
            doc_modal()

    # Sticky Input
    user_input = st.chat_input("Ask about your plan, request a quiz, or explain a topic...")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with chat_container:
            st.chat_message("user").write(user_input)

        with st.spinner("Processing..."):
            history_context = ""
            if len(st.session_state.chat_history) > 1:
                history_context = summarize_history(st.session_state.chat_history[:-1])
                
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
                for chunk in study_buddy_graph.stream(initial_chat_state, stream_mode="updates", config={"configurable": {"thread_id": st.session_state.quiz_thread_id}}):
                    if "orchestrator" in chunk:
                        plan = chunk["orchestrator"]
                        actions = plan.get("plan_actions", [])
                        if "END" in actions:
                             st.session_state.chat_history.append({"role": "ai", "content": "I cannot handle this request."})
                        else:
                            with chat_container: st.caption(f"⚙️ Plan: {', '.join(actions)}")

                    if "explain_node" in chunk:
                        for msg in chunk["explain_node"].get("messages", []):
                            with chat_container: st.chat_message("ai").write(msg.content)
                            st.session_state.chat_history.append({"role": "ai", "content": msg.content})

                    if "quiz_generator" in chunk:
                        quiz_data = chunk["quiz_generator"].get("quiz_output")
                        if quiz_data:
                            st.session_state.active_quiz = quiz_data
                            st.session_state.quiz_answers = {}
                            st.session_state.quiz_submitted = False
                            msg_content = "📝 Quiz generated! Check the Quiz tab."
                            with chat_container: st.chat_message("ai").write(msg_content)
                            st.session_state.chat_history.append({"role": "ai", "content": msg_content})

                    if "summarizer" in chunk:
                         for msg in chunk["summarizer"].get("messages", []):
                            with chat_container: st.chat_message("ai").write(msg.content)
                            st.session_state.chat_history.append({"role": "ai", "content": msg.content})
            except Exception as e:
                st.error(f"Error: {str(e)}")


# RIGHT COLUMN: Manager & Details
with col2:
    tabs = st.tabs(["📝 Details", "🛠️ Manager", "❓ Quiz"])
    
    # TAB 1: Milestone Details
    with tabs[0]:
        st.subheader("Milestone Details")
        detail_container = st.container(height=650)
        with detail_container:
            if st.session_state.clicked_node and st.session_state.plan_json:
                ms_data = next((m for m in st.session_state.plan_json.get("milestones", []) if m.get("id") == st.session_state.clicked_node), None)
                if ms_data:
                    st.markdown(f"### {ms_data['title']}")
                    st.caption(f"Status: {ms_data['status'].upper()}")
                    st.info(ms_data['description'])
                    st.markdown("#### Tasks")
                    for task in ms_data.get("tasks", []):
                        icon = "✅" if ms_data['status'] == 'done' else "⬜"
                        st.markdown(f"**{icon} {task['name']}**")
                        st.caption(task['description'])
                        if task.get('resources'): st.markdown(f"📚 [Read]({task['resources']})")
                        if task.get('youtube'): st.markdown(f"📺 [Watch]({task['youtube']})")
                        st.divider()
            else:
                st.info("Select a milestone from the sidebar or graph to view details.")

    # TAB 2: Roadmap Editor
    with tabs[1]:
        editor_chat_container = st.container(height=550)
        with editor_chat_container:
            # 1. EMPTY STATE
            if not st.session_state.editor_chat_history:
                st.markdown("""
                <div style="text-align: center; color: #94a3b8; margin-top: 40px;">
                    <div style="font-size: 3rem; opacity: 0.8;">🛠️</div>
                    <h3 style="color: #e2e8f0; margin: 10px 0;">Roadmap Manager</h3>
                    <p>How would you like to adjust your plan?</p>
                    <div style="display: grid; gap: 10px; margin-top: 20px;">
                        <div style="background: #1e293b; padding: 12px; border-radius: 8px; border: 1px solid #334155; font-size: 0.9rem;">
                            "Add a milestone for Docker basics"
                        </div>
                        <div style="background: #1e293b; padding: 12px; border-radius: 8px; border: 1px solid #334155; font-size: 0.9rem;">
                            "Remove the last step"
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # 2. RENDER MESSAGES
            for msg in st.session_state.editor_chat_history:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])
        
        # 3. INPUT FIELD
        editor_input = st.chat_input("Type instructions to modify plan...", key="editor_input")
        
        if editor_input:
            st.session_state.editor_chat_history.append({"role": "user", "content": editor_input})
            
            with st.spinner("AI is updating your roadmap structure..."):
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
                    st.session_state.editor_chat_history.append({"role": "ai", "content": "✅ Plan updated successfully!"})
                    st.rerun()
                else:
                    st.error(f"Failed to update: {result.get('error')}")

    # TAB 3: Quiz Interface
    with tabs[2]:
        st.subheader("Active Quiz")
        quiz_container = st.container(height=650)
        with quiz_container:
            quiz = st.session_state.active_quiz
            if not quiz:
                st.info("Ask the Assistant to generate a quiz for a milestone!")
            else:
                st.markdown(f"**Topic:** {quiz.topic}")
                
                with st.form("quiz_form"):
                    st.markdown("##### Multiple Choice")
                    for idx, q in enumerate(quiz.mcq_questions):
                        st.session_state.quiz_answers[f"mcq_{idx}"] = st.radio(q.question, q.options, key=f"mcq_{idx}", index=None)
                    
                    if quiz.article_questions:
                        st.markdown("##### Short Answer")
                        for idx, q in enumerate(quiz.article_questions):
                            st.session_state.quiz_answers[f"article_{idx}"] = st.text_area(q.question, key=f"article_{idx}")
                    
                    if quiz.coding_questions:
                        st.markdown("##### Coding Challenge")
                        for idx, q in enumerate(quiz.coding_questions):
                            st.write(q.question)
                            st.session_state.quiz_answers[f"code_{idx}"] = st.text_area("Code", height=150, key=f"code_{idx}")

                    if st.form_submit_button("Submit Quiz"):
                        submission = {"quiz": quiz.model_dump(), "user_answers": st.session_state.quiz_answers}
                        st.session_state.quiz_submitted = True
                        st.success("Submitted! Analyzing...")
                        events = study_buddy_graph.stream(None, config={"configurable": {"thread_id": st.session_state.quiz_thread_id, "resume": submission}})
                        for event in events:
                             if isinstance(event, dict) and "user_profile_summary" in event:
                                 st.session_state.last_quiz_result = event["user_profile_summary"]
                        st.rerun()
                
                if st.session_state.get("last_quiz_result"):
                    res = st.session_state.last_quiz_result
                    st.metric("Score", f"{res['accuracy']}%")
                    st.write(res["summary_text"])