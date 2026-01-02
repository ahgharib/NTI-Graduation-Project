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
import requests
import base64
import time

# --- SETUP & CONFIGURATION ---
UPLOAD_DIR = "RAG/data"
NOTES_OUTPUT_DIR = "notes_output"
os.makedirs(UPLOAD_DIR, exist_ok=True)
# Keep this for browser metadata, but we will add a visible title below
st.set_page_config(layout="wide", page_title="Student Partner AI", page_icon="🎓")

# --- 🎨 CSS (KEPT AS REQUESTED - GRADIENT ANIMATION) ---
st.markdown("""
    <style>
        /* 1. KEYFRAMES */
        @keyframes fadeIn {
            0% { opacity: 0; transform: translateY(20px); filter: blur(10px); }
            100% { opacity: 1; transform: translateY(0); filter: blur(0); }
        }
        
        @keyframes gradientBG {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        /* Apply animation to main containers */
        .block-container {
            animation: fadeIn 0.6s cubic-bezier(0.22, 1, 0.36, 1) forwards;
            /* Adjusted padding for new header structure */
            padding-top: 2rem !important; 
            padding-bottom: 3rem !important;
        }

        /* 2. GLOBAL THEME */
        .stApp {
            background: linear-gradient(-45deg, #050a14, #0f172a, #1a1235, #092230);
            background-size: 400% 400%;
            animation: gradientBG 20s ease infinite;
            font-family: 'Inter', sans-serif;
            color: #e2e8f0;
        }

        /* 3. GLASS CARDS */
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(15, 23, 42, 0.6);
            backdrop-filter: blur(16px);
            border: 1px solid rgba(0, 240, 255, 0.1); 
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3);
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        }
        
        [data-testid="stVerticalBlockBorderWrapper"]:hover {
            border-color: rgba(0, 240, 255, 0.4);
            box-shadow: 0 0 25px rgba(0, 240, 255, 0.15), inset 0 0 10px rgba(0, 240, 255, 0.05); 
            transform: translateY(-3px) scale(1.005);
        }

        /* 4. SIDEBAR STYLING */
        [data-testid="stSidebar"] {
            background-color: rgba(5, 7, 12, 0.95);
            border-right: 1px solid rgba(0, 240, 255, 0.1);
        }
        /* Sidebar header specific style */
        [data-testid="stSidebar"] h1 {
            background: linear-gradient(to right, #00F0FF, #8A2BE2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800 !important;
            font-size: 1.8rem !important;
        }

        /* 5. MAIN PAGE HEADERS */
        /* Style for the main page title */
        .main-header {
            font-size: 3rem !important;
            font-weight: 800 !important;
            background: linear-gradient(to right, #E2E8F0, #00F0FF);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0px !important;
        }
        /* Style for subtitles */
        .sub-header {
            color: #94a3b8 !important;
            font-size: 1.2rem !important;
            font-weight: 400 !important;
            margin-top: 0px !important;
        }

        /* Standard headers */
        h1, h2, h3, h4 { 
            color: #F8FAFC !important; 
            letter-spacing: 0.5px;
            text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }

        /* 6. MODERN BUTTONS */
        .stButton > button {
            border-radius: 12px;
            font-weight: 600;
            border: 1px solid rgba(255, 255, 255, 0.05);
            background-color: rgba(30, 41, 59, 0.5);
            color: #E2E8F0;
            transition: all 0.3s ease;
            backdrop-filter: blur(5px);
        }
        .stButton > button:hover {
            background-color: rgba(0, 240, 255, 0.1);
            border-color: rgba(0, 240, 255, 0.3);
            color: #00F0FF;
            box-shadow: 0 0 15px rgba(0, 240, 255, 0.2);
        }
        .stButton > button[kind="primary"] {
            background: linear-gradient(90deg, #00C6FF 0%, #0072FF 100%);
            border: none;
            color: white;
            box-shadow: 0 4px 15px rgba(0, 114, 255, 0.3);
        }
        .stButton > button[kind="primary"]:hover {
            box-shadow: 0 6px 25px rgba(0, 198, 255, 0.5);
        }

        /* 7. CHAT & INPUTS */
        [data-testid="stChatMessage"][data-testid="stChatMessageUser"] {
            background: rgba(15, 23, 42, 0.8);
            border: 1px solid rgba(0, 240, 255, 0.15);
            border-radius: 12px;
            backdrop-filter: blur(5px);
        }
        [data-testid="stChatMessage"][data-testid="stChatMessageUser"] p { color: #00F0FF !important; }
        .stTextInput > div > div > input, .stTextArea > div > div > textarea {
            background-color: rgba(10, 14, 23, 0.8) !important;
            border: 1px solid rgba(0, 240, 255, 0.1) !important;
            border-radius: 10px;
            color: white !important;
        }
        .stTextInput > div > div > input:focus, .stTextArea > div > div > textarea:focus {
            border-color: #00F0FF !important;
            box-shadow: 0 0 0 2px rgba(0, 240, 255, 0.15) !important;
        }
        .stCaption, .stInfo { color: #94a3b8 !important; }
        strong { color: #00F0FF !important; }

        /* 8. MOBILE */
        @media (max-width: 768px) {
            .stButton > button { width: 100%; }
            .block-container { padding-top: 1rem !important; }
            .main-header { font-size: 2rem !important; }
        }
        
        /* 9. CLEANUP */
        [data-testid="stFileUploaderDropzoneInstructions"] { display: none; }
        [data-testid="stFileUploader"] section { padding: 0.5rem; border-color: rgba(0, 240, 255, 0.2); }
        hr { margin-top: 0; margin-bottom: 2rem; border: none; height: 1px; background: linear-gradient(90deg, transparent, rgba(0, 240, 255, 0.5), transparent); }
        
        /* HIDE DEFAULT SIDEBAR NAV */
        [data-testid="stSidebarNav"] {
            display: none !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if "view_mode" not in st.session_state:
    st.session_state.view_mode = "Dashboard"
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
if "last_quiz_result" not in st.session_state:
    st.session_state.last_quiz_result = None
# NEW SESSION STATE FOR GENERATED CONTENT
if "generated_video" not in st.session_state: 
    st.session_state.generated_video = None
if "generated_notes" not in st.session_state:
    st.session_state.generated_notes = []

# --- FUNCTIONS ---
def switch_view(view_name):
    st.session_state.view_mode = view_name
    st.rerun()

def select_milestone(node_id):
    st.session_state.clicked_node = node_id

def perform_search_logic(query):
    if query.strip():
        context = ""
        if st.session_state.clicked_node and st.session_state.plan_json:
            ms_data = next((m for m in st.session_state.plan_json.get("milestones", []) 
                          if m.get("id") == st.session_state.clicked_node), None)
            if ms_data:
                context = ms_data['title']
        
        user_message = f"🔍 Search: {query}"
        st.session_state.chat_history.append({"role": "user", "content": user_message})
        st.session_state.is_searching = True
        
        try:
            search_result = search_with_agent(query=query, context=context)
            st.session_state.chat_history.append({"role": "ai", "content": search_result})
        except Exception as e:
            st.session_state.chat_history.append({"role": "ai", "content": f"Search failed: {str(e)}"})
        
        st.session_state.is_searching = False
        st.session_state.search_query = ""

# --- MODALS ---
@st.dialog("Research Assistant")
def search_modal():
    st.caption("Ask a research question or upload documents to analyze.")
    st.markdown("##### 🌐 Web Search")
    with st.form("search_form"):
        query = st.text_input("Enter topic to research...", placeholder="e.g. Best practices for REST APIs")
        submit_search = st.form_submit_button("Start Search")
        if submit_search and query:
            with st.spinner("🔎 Researching topic..."):
                perform_search_logic(query)
            st.rerun()

@st.dialog("Knowledge Base")
def doc_modal():
    st.caption("Upload documents to provide context for the AI.")
    
    # --- HISTORY SECTION ---
    st.markdown("##### 📂 Upload History")
    if st.session_state.uploaded_docs:
        # Display history in a scrollable container
        with st.container(height=150, border=True):
            for filename in st.session_state.uploaded_docs.keys():
                st.text(f"📄 {filename}")
    else:
        st.info("No documents uploaded yet.")

    st.divider()

    # --- UPLOAD SECTION ---
    st.markdown("##### ➕ Add New Documents")
    
    # --- WARNING MESSAGE ---
    st.caption("⚠️ **Important:** Please do not close this form while processing to prevent cancellation.")
    
    uploaded_docs = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed")
    
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
            
            # --- DONE MESSAGE ---
            st.success("✅ Done! Documents Indexed Successfully!")
            time.sleep(2)
            st.rerun()


@st.dialog("Generate Study Video")
def video_modal():
    st.caption("Generate an AI video based on a topic.")
    with st.form("video_form"):
        video_topic = st.text_input("Topic for video", placeholder="e.g. Transformers")
        # --- WARNING MESSAGE ---
        st.caption("⚠️ **Important:** Please do not close this form while processing to prevent cancellation.")
        
        submit_video = st.form_submit_button("Generate Video", type="primary")
        
        if submit_video and video_topic:
            server_url = "https://9000-01kdwpt8sdbpx4p6pnre63abf7.cloudspaces.litng.ai"
            with st.spinner("🎬 Starting video generation..."):
                try:
                    r = requests.post(f"{server_url}/generate", json={"topic": video_topic}, timeout=60)
                    r.raise_for_status()
                    job_id = r.json()["job_id"]
                    
                    status_text = st.empty()
                    while True:
                        s = requests.get(f"{server_url}/jobs/{job_id}", timeout=60)
                        js = s.json()
                        status_text.write(f"Status: {js['status']} | Progress: {js['progress']}%")
                        if js["status"] == "done": break
                        if js["status"] == "error": raise Exception(js.get("error"))
                        time.sleep(5)
                    
                    # Download
                    video_path = f"video_{job_id}.mp4"
                    with requests.get(f"{server_url}/jobs/{job_id}/download", stream=True) as dl:
                        with open(video_path, "wb") as f:
                            for chunk in dl.iter_content(chunk_size=1024*1024): f.write(chunk)
                    
                    st.session_state.generated_video = video_path
                    # --- DONE MESSAGE ---
                    st.success("✅ Done! Video generated successfully.")
                    time.sleep(2) # Wait so user can see the message
                    st.rerun()
                except Exception as e:
                    st.error(f"Generation failed: {e}")

@st.dialog("Generate Visual Notes")
def notes_modal():
    st.caption("Create summarized visual pages for a topic.")
    with st.form("notes_form"):
        notes_topic = st.text_input("Topic for notes", placeholder="e.g. FastAPI Basics")
        # --- WARNING MESSAGE ---
        st.caption("⚠️ **Important:** Please do not close this form while processing to prevent cancellation.")
        
        submit_notes = st.form_submit_button("Generate Notes", type="primary")
        
        if submit_notes and notes_topic:
            cloud_url = "https://8000-01kdw6hxxpw7z5gt5aw3tn37am.cloudspaces.litng.ai/generate"
            with st.spinner("🎨 Designing your notes..."):
                try:
                    response = requests.post(cloud_url, json={"topic": notes_topic, "max_pages": 1}, timeout=600)
                    if response.status_code == 200:
                        data = response.json()
                        saved_paths = []
                        for i, img_b64 in enumerate(data["images"]):
                            path = os.path.join(NOTES_OUTPUT_DIR, f"note_{int(time.time())}_{i}.png")
                            with open(path, "wb") as f:
                                f.write(base64.b64decode(img_b64))
                            saved_paths.append(path)
                        st.session_state.generated_notes = saved_paths
                        # --- DONE MESSAGE ---
                        st.success("✅ Done! Notes generated successfully.")
                        time.sleep(2) # Wait so user can see the message
                        st.rerun()
                    else:
                        st.error(f"Error: {response.text}")
                except Exception as e:
                    st.error(f"Failed: {e}")

# --- SIDEBAR (FIXED) ---
with st.sidebar:
    st.header("🎓 Student Partner")
    
    st.markdown("### 🧭 Menu")
    col_nav1, col_nav2 = st.columns(2)
    with col_nav1:
        dashboard_type = "primary" if st.session_state.view_mode == "Dashboard" else "secondary"
        if st.button("📊 Board", key="nav_dashboard", use_container_width=True, type=dashboard_type):
            switch_view("Dashboard")
    with col_nav2:
        quiz_type = "primary" if st.session_state.view_mode == "Take Quiz" else "secondary"
        if st.button("📝 Quiz", key="nav_quiz", use_container_width=True, type=quiz_type):
            switch_view("Take Quiz")
            
    st.divider()
    
    st.markdown("### 🎯 Goal")
    user_goal = st.text_input("Goal", "Python Backend Development", label_visibility="collapsed", placeholder="E.g., Learn React")
    
    if st.button("🚀 Generate Roadmap", use_container_width=True):
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
    
    with st.expander("📂 Load Plan"):
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

    if st.session_state.plan_json:
        st.divider()
        st.markdown("### 📍 Milestones")
        st.download_button(
            "📥 Export JSON", 
            data=json.dumps(st.session_state.plan_json, indent=2), 
            file_name="roadmap.json", 
            mime="application/json", 
            use_container_width=True
        )
        st.write("") 
        with st.container(height=300):
            milestones = st.session_state.plan_json.get("milestones", [])
            for ms in milestones:
                ms_id = ms.get("id")
                is_selected = st.session_state.clicked_node == ms_id
                icon = "🔵" if is_selected else "⚪"
                label = ms['title']
                if st.button(f"{icon} {label}", key=f"nav_{ms_id}", use_container_width=True):
                    select_milestone(ms_id)
                    st.rerun()

# =========================================================
# === MAIN PAGE HEADER (NEW & ORGANIZED) ===
# =========================================================
# This section is now clearly defined within the page body, not the browser tab.
with st.container():
    # Use columns to organize the header elements nicely
    hc1, hc2 = st.columns([0.6, 4])
    with hc1:
        # A large, prominent icon
        st.markdown("<div style='font-size: 4.5rem; text-align: center; animation: float 3s ease-in-out infinite;'>🎓</div>", unsafe_allow_html=True)
    with hc2:
        # The main title using specific CSS classes for styling
        st.markdown('<h1 class="main-header">Student Partner AI</h1>', unsafe_allow_html=True)
        # A descriptive subtitle
        st.markdown('<p class="sub-header">Your intelligent companion for structured learning and roadmap planning.</p>', unsafe_allow_html=True)
    
    # A glowing divider to separate header from content
    st.markdown("<hr>", unsafe_allow_html=True)

# =========================================================
# === VIEW 1: DASHBOARD (Chat, Graph, Manager) ===
# =========================================================
if st.session_state.view_mode == "Dashboard":
    
    # Main Layout: 2 Columns on Desktop, Stacks on Mobile
    col1, col2 = st.columns([2.5, 1.2])

    # LEFT COLUMN: Graph & Chat
    with col1:
        # Slightly smaller header for this section
        st.markdown("#### 🚀 Roadmap View")
        st.caption("Visualize your path and track progress.")
        st.write("") 

        # GRAPH SECTION
        if st.session_state.plan_json:
            with st.expander("🗺️ Open Map View", expanded=True):
                nodes = []
                edges = []
                milestones = st.session_state.plan_json.get("milestones", [])
                for i, ms in enumerate(milestones):
                    ms_id = ms.get("id", f"m{i}")
                    is_selected = st.session_state.clicked_node == ms_id
                    
                    # UPDATE GRAPH COLORS TO MATCH NEW THEME
                    node_color = "#00F0FF" if is_selected else "#FF0080" 
                    node_size = 30
                    step_number = str(i + 1)
                    
                    nodes.append(Node(
                        id=ms_id, label=step_number, title=ms["title"], shape="circle", size=node_size, color=node_color,
                        font={'color': 'white', 'size': 16, 'face': 'Courier New', 'align': 'center', 'bold': True},
                        borderWidth=2, borderWidthSelected=4, 
                        shadow={'enabled': True, 'color': node_color, 'size': 10}
                    ))
                    if i > 0:
                        prev_id = milestones[i-1].get("id", f"m{i-1}")
                        edges.append(Edge(source=prev_id, target=ms_id, type="CURVE_SMOOTH", arrows="to", color="#5eead4"))

                config = AgConfig(
                    width=None, 
                    height=450, 
                    directed=True, 
                    physics=True, 
                    nodeHighlightBehavior=True, 
                    highlightColor="#00F0FF", 
                    backgroundColor="#0e1117",
                    physicsOptions={
                        'stabilization': {'enabled': True, 'iterations': 1000}, 
                        'barnesHut': {'gravitationalConstant': -3000, 'springLength': 130, 'springConstant': 0.04}
                    }
                )
                returned_id = agraph(nodes=nodes, edges=edges, config=config)
                if returned_id and returned_id != st.session_state.clicked_node:
                    st.session_state.clicked_node = returned_id
                    st.rerun()

        # CHAT SECTION
        st.divider()
        st.subheader("💬 Study Assistant")

        # DISPLAY GENERATED CONTENT IN CHAT AREA
        if st.session_state.generated_video:
            with st.expander("🎬 Generated Video", expanded=True):
                st.video(st.session_state.generated_video)
        
        if st.session_state.generated_notes:
            with st.expander("🎨 Generated Notes", expanded=True):
                for img_path in st.session_state.generated_notes:
                    st.image(img_path, use_container_width=True)
        
        selected_ms_text = ""
        if st.session_state.plan_json and st.session_state.clicked_node:
            ms_data = next((m for m in st.session_state.plan_json.get("milestones", []) if m.get("id") == st.session_state.clicked_node), None)
            if ms_data:
                st.info(f"**Focused on:** {ms_data['title']}")
                selected_ms_text = f"Title: {ms_data['title']}. Description: {ms_data['description']}."

        chat_container = st.container(height=500)
        with chat_container:
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])
            if st.session_state.is_searching:
                with st.chat_message("assistant"):
                    st.write("🔍 Researching...")

        # Toolbar Buttons (Responsive)
        # Speech button removed here as requested
        tool_c1, tool_c2, tool_c3, tool_c4 = st.columns(4)
        
        with tool_c1:
            if st.button("🔍 Search", help="Web Search", use_container_width=True):
                search_modal()
        with tool_c2:
            if st.button("📂 Upload", help="Upload Docs", use_container_width=True):
                doc_modal()
        with tool_c3:
            if st.button("🎬 Video", help="Generate AI Video", use_container_width=True):
                video_modal()
        with tool_c4:
            if st.button("🎨 Notes", help="Generate Visual Notes", use_container_width=True):
                notes_modal()

        user_input = st.chat_input("Ask about your plan, request a quiz, or explain a topic...")
        
        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with chat_container:
                st.chat_message("user").write(user_input)
            
            # KEYWORD TRIGGER
            if "quiz" in user_input.lower():
                notification_msg = "Go check the quiz page! 📝"
                st.session_state.chat_history.append({"role": "ai", "content": notification_msg})
                with chat_container:
                    st.chat_message("ai").write(notification_msg)
            else:
                with st.spinner("Processing..."):
                    history_context = ""
                    if len(st.session_state.chat_history) > 1:
                        history_context = summarize_history(st.session_state.chat_history[:-1])
                        
                    initial_chat_state = {
                        "user_prompt": user_input, "messages": [], "plan_actions": [], "plan_instructions": [], "research_memory": [],
                        "raw_data_storage": [], "execution_log": [], "validation_errors": [], "refinement_attempts": 0,
                        "plan_data": st.session_state.plan_json, "selected_milestone_context": selected_ms_text, "conversation_summary": history_context
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
                                    msg_content = "📝 **Quiz Generated!** Navigate to the **'Quiz'** page from the sidebar to start."
                                    with chat_container: st.chat_message("ai").write(msg_content)
                                    st.session_state.chat_history.append({"role": "ai", "content": msg_content})
                            if "summarizer" in chunk:
                                 for msg in chunk["summarizer"].get("messages", []):
                                    with chat_container: st.chat_message("ai").write(msg.content)
                                    st.session_state.chat_history.append({"role": "ai", "content": msg.content})
                    except Exception as e:
                        st.error(f"Connection Error: Ensure Ollama is running. ({str(e)})")

    # RIGHT COLUMN: Manager & Details
    with col2:
        st.write("") 
        st.write("") 
        tabs = st.tabs(["📝 Details", "🛠️ Manager"])
        
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
            st.subheader("Edit Roadmap")
            editor_chat_container = st.container(height=500)
            with editor_chat_container:
                if not st.session_state.editor_chat_history:
                    st.markdown("""
                    <div style="text-align: center; color: #94a3b8; margin-top: 40px;">
                        <div style="font-size: 3rem; opacity: 0.8;">🛠️</div>
                        <h3 style="color: #e2e8f0; margin: 10px 0;">Roadmap Manager</h3>
                        <p>How would you like to adjust your plan?</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                for msg in st.session_state.editor_chat_history:
                    with st.chat_message(msg["role"]):
                        st.write(msg["content"])
            
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

# =========================================================
# === VIEW 2: QUIZ PAGE (Dedicated Full Screen) ===
# =========================================================
elif st.session_state.view_mode == "Take Quiz":

    quiz_container = st.empty()

    with quiz_container.container():

        quiz = st.session_state.active_quiz
        submitted = st.session_state.get("quiz_submitted", False)
        results = st.session_state.get("last_quiz_result")

        if quiz:

            # =============================
            # HEADER
            # =============================
            st.markdown('<h3>📝 Knowledge Check</h3>', unsafe_allow_html=True)
            st.info(f"Topic: **{quiz.topic}** | Difficulty: **{quiz.proficiency_level}**")

            # =============================
            # OVERALL RESULT (TOP – AFTER SUBMISSION)
            # =============================
            if submitted and results:
                summary = results["summary"]

                st.divider()
                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "🎯 Total Score",
                        f"{summary['score']} / {summary['total']}"
                    )

                with col2:
                    st.metric(
                        "📊 Accuracy",
                        f"{summary['accuracy']}%"
                    )

                st.divider()
            else:
                st.divider()

            # =============================
            # QUIZ FORM (INPUTS ONLY)
            # =============================
            with st.form("quiz_form"):

                # ---------- MCQs ----------
                st.subheader("Multiple Choice")

                for idx, q in enumerate(quiz.mcq_questions):
                    key = f"mcq_{idx}"
                    user_answer = st.session_state.quiz_answers.get(key)

                    st.markdown(f"**{idx+1}. {q.question}**")

                    st.session_state.quiz_answers[key] = st.radio(
                        "Select Option",
                        q.options,
                        index=q.options.index(user_answer) if user_answer in q.options else None,
                        key=key,
                        disabled=submitted,
                        label_visibility="collapsed"
                    )
                    # ----- RESULT (AFTER SUBMISSION) -----
                    if submitted and results:
                        r = results["mcq_results"][idx]

                        border = "green" if r["is_correct"] else "red"
                        icon = "✅" if r["is_correct"] else "❌"

                        reasoning_html = (
                            "<i>Correct answer!</i>"
                            if r["is_correct"]
                            else f"<i>{r['reasoning']}</i>"
                        )

                        st.markdown(
                            f"""
                            <div style="
                                border:2px solid {border};
                                border-radius:8px;
                                padding:10px;
                                margin-top:6px;
                            ">
                            {icon} <b>Your answer:</b> {user_answer}<br>
                            {reasoning_html}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    st.write("")

                # ---------- ARTICLES ----------
                if quiz.article_questions:
                    st.subheader("Short Answer")

                    for idx, q in enumerate(quiz.article_questions):
                        key = f"article_{idx}"
                        user_answer = st.session_state.quiz_answers.get(key, "")

                        st.markdown(f"**{q.question}**")

                        st.session_state.quiz_answers[key] = st.text_area(
                            "Your Answer",
                            value=user_answer,
                            key=key,
                            disabled=submitted,
                            label_visibility="collapsed"
                        )

                        # ----- RESULT (AFTER SUBMISSION) -----
                        if submitted and results:
                            r = results["article_results"][idx]

                            if r["score"] == r["out_of"]:
                                border = "green"
                                icon = "✅"
                            elif r["score"] > 0:
                                border = "orange"
                                icon = "🟡"
                            else:
                                border = "red"
                                icon = "❌"

                            st.markdown(
                                f"""
                                <div style="
                                    border:2px solid {border};
                                    border-radius:8px;
                                    padding:10px;
                                    margin-top:6px;
                                ">
                                {icon} <b>Score:</b> {r["score"]}/{r["out_of"]}<br>
                                <i>{r["reasoning"]}</i>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                        st.write("")

                st.divider()

                submit_btn = st.form_submit_button(
                    "✅ Submit Quiz",
                    type="primary",
                    disabled=submitted
                )

            # =============================
            # SUBMISSION LOGIC (OUTSIDE FORM)
            # =============================
            if submit_btn and not submitted:

                submission = {
                    "quiz": quiz.model_dump(),
                    "user_answers": st.session_state.quiz_answers
                }

                study_buddy_graph.update_state(
                    {"configurable": {"thread_id": st.session_state.quiz_thread_id}},
                    {"user_submission": submission}
                )

                with st.spinner("Grading your answers..."):
                    for event in study_buddy_graph.stream(
                        None,
                        config={"configurable": {"thread_id": st.session_state.quiz_thread_id}}
                    ):
                        if isinstance(event, dict):
                            for node_name, node_output in event.items():
                                if (
                                    node_name == "user_summary_node"
                                    and isinstance(node_output, dict)
                                    and "grader_output" in node_output
                                ):
                                    st.session_state.last_quiz_result = node_output["grader_output"]

                st.session_state.quiz_submitted = True
                st.rerun()
            # =============================
            # FEEDBACK (BOTTOM)
            # =============================
            if submitted and results:
                st.divider()
                st.subheader("🧠 Overall Feedback")
                st.info(results["summary"]["feedback"])