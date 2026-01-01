import streamlit as st
import json
import uuid
# Import your graph so we can grade the quiz
from chat_graph import study_buddy_graph 

st.set_page_config(page_title="Active Quiz", page_icon="📝", layout="wide")

# --- GLOBAL STYLES (Match Main Page) ---
st.markdown("""
    <style>
        .stApp { background-color: #0e1117; color: #e0e0e0; font-family: 'Inter', sans-serif; }
        [data-testid="stVerticalBlockBorderWrapper"] { background-color: #1a1c24; border-radius: 12px; border: 1px solid #2d3748; padding: 2rem; }
    </style>
""", unsafe_allow_html=True)

st.title("📝 Knowledge Check")

# Ensure session state exists (in case user goes directly here)
if "active_quiz" not in st.session_state:
    st.session_state.active_quiz = None
if "quiz_answers" not in st.session_state:
    st.session_state.quiz_answers = {}
if "last_quiz_result" not in st.session_state:
    st.session_state.last_quiz_result = None

# --- QUIZ RENDER LOGIC ---
quiz = st.session_state.active_quiz

if not quiz:
    st.container(height=200, border=True).markdown("""
        <div style="text-align: center; padding-top: 20px;">
            <h2>🚫 No Active Quiz</h2>
            <p style="color: #94a3b8;">Go to the <b>Main Dashboard</b> and ask the assistant:<br>
            <i>"Make a quiz about [Topic]"</i></p>
        </div>
    """, unsafe_allow_html=True)

else:
    # Quiz Container
    with st.container(border=True):
        st.subheader(f"Topic: {quiz.topic}")
        st.caption(f"Proficiency Level: {quiz.proficiency_level}")
        st.divider()

        with st.form("quiz_page_form"):
            # 1. MCQs
            if quiz.mcq_questions:
                st.markdown("### 🔹 Multiple Choice")
                for idx, q in enumerate(quiz.mcq_questions):
                    st.markdown(f"**{idx+1}. {q.question}**")
                    st.session_state.quiz_answers[f"mcq_{idx}"] = st.radio(
                        "Select answer:", 
                        q.options, 
                        key=f"mcq_{idx}", 
                        index=None, 
                        label_visibility="collapsed"
                    )
                    st.write("") # Spacer

            # 2. Article Questions
            if quiz.article_questions:
                st.markdown("### 🔹 Short Answer")
                for idx, q in enumerate(quiz.article_questions):
                    st.markdown(f"**{q.question}**")
                    st.session_state.quiz_answers[f"article_{idx}"] = st.text_area(
                        "Your answer:", 
                        key=f"article_{idx}",
                        height=100,
                        label_visibility="collapsed"
                    )
                    st.write("")

            # 3. Code Questions
            if quiz.coding_questions:
                st.markdown("### 🔹 Coding Challenge")
                for idx, q in enumerate(quiz.coding_questions):
                    st.markdown(f"**{q.question}**")
                    st.session_state.quiz_answers[f"code_{idx}"] = st.text_area(
                        "Write code here:", 
                        key=f"code_{idx}",
                        height=200,
                        label_visibility="collapsed"
                    )
            
            st.divider()
            submitted = st.form_submit_button("✅ Submit Quiz", type="primary", use_container_width=True)

            if submitted:
                # Prepare Submission
                submission = {
                    "quiz": quiz.model_dump(),
                    "user_answers": st.session_state.quiz_answers
                }
                
                with st.spinner("🤖 Grading your answers..."):
                    # Call Graph to Grade
                    events = study_buddy_graph.stream(
                        None, 
                        config={
                            "configurable": {
                                "thread_id": st.session_state.quiz_thread_id, 
                                "resume": submission
                            }
                        }
                    )
                    
                    for event in events:
                        if isinstance(event, dict) and "user_profile_summary" in event:
                            st.session_state.last_quiz_result = event["user_profile_summary"]
                            
                st.success("Quiz Submitted!")
                st.rerun()

    # --- RESULTS SECTION ---
    if st.session_state.last_quiz_result:
        st.write("")
        res = st.session_state.last_quiz_result
        
        # Determine Color based on score
        score = res['accuracy']
        color = "green" if score > 80 else "orange" if score > 50 else "red"
        
        with st.container(border=True):
            col_score, col_text = st.columns([1, 3])
            with col_score:
                st.markdown(f"""
                    <div style="text-align: center; border: 4px solid {color}; border-radius: 50%; width: 120px; height: 120px; line-height: 110px; margin: auto;">
                        <h1 style="margin:0; font-size: 40px; color: {color};">{score}%</h1>
                    </div>
                """, unsafe_allow_html=True)
            with col_text:
                st.subheader("Performance Review")
                st.markdown(res["summary_text"])
                
        if st.button("Start New Quiz"):
            st.session_state.active_quiz = None
            st.session_state.last_quiz_result = None
            st.session_state.quiz_answers = {}
            st.rerun()