import streamlit as st
import json

# -------------------------------------------------
# Load Quiz
# -------------------------------------------------
with open("quiz_output.json", "r", encoding="utf-8") as f:
    quiz = json.load(f)

st.set_page_config(page_title="MCQ Quiz", layout="centered")

# -------------------------------------------------
# Header
# -------------------------------------------------
st.title("📘 MCQ Quiz")
st.subheader(f"Topic: {quiz['topic']}")
st.caption(f"Level: {quiz['proficiency_level']}")

st.divider()

# -------------------------------------------------
# Session State for answers
# -------------------------------------------------
if "answers" not in st.session_state:
    st.session_state.answers = {}

# -------------------------------------------------
# Display Questions
# -------------------------------------------------
for idx, q in enumerate(quiz["questions"]):
    st.markdown(f"### Question {idx + 1}")
    st.write(q["question"])

    selected = st.radio(
        label="Choose an answer:",
        options=q["options"],
        index=None,
        key=f"q_{idx}"
    )

    if selected is not None:
        if selected == q["correct_answer"]:
            st.success("✅ Correct!")
        else:
            st.error("❌ Incorrect")
            st.info(f"Correct answer: **{q['correct_answer']}**")

    st.divider()

