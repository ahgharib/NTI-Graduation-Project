import streamlit as st
import json

# -------------------------------------------------
# Load Quiz
# -------------------------------------------------
with open("quiz_output.json", "r", encoding="utf-8") as f:
    quiz = json.load(f)

st.set_page_config(page_title="MCQ & Python Quiz", layout="centered")

# -------------------------------------------------
# Header
# -------------------------------------------------
st.title("📘 Python Quiz")
st.subheader(f"Topic: {quiz['topic']}")
st.caption(f"Level: {quiz['proficiency_level']}")
st.divider()

# -------------------------------------------------
# Session State for answers
# -------------------------------------------------
if "answers" not in st.session_state:
    st.session_state.answers = {}

# -------------------------------------------------
# Display MCQ Questions
# -------------------------------------------------
st.header("Multiple Choice Questions")
for idx, q in enumerate(quiz["mcq_questions"]):
    st.markdown(f"### Question {idx + 1}")
    st.write(q["question"])

    selected = st.radio(
        label="Choose an answer:",
        options=q["options"],
        index=None,
        key=f"mcq_{idx}"
    )

    if selected is not None:
        if selected == q["correct_answer"]:
            st.success("✅ Correct!")
        else:
            st.error("❌ Incorrect")
            st.info(f"Correct answer: **{q['correct_answer']}**")

    st.divider()

# -------------------------------------------------
# Display Article Questions
# -------------------------------------------------
st.header("Article Questions")
for idx, q in enumerate(quiz.get("article_questions", [])):
    st.markdown(f"### Article Question {idx + 1}")
    st.write(q["question"])

    user_answer = st.text_area(
        label="Your answer:",
        placeholder="Write your answer here...",
        key=f"article_{idx}"
    )

    if st.button("Show Correct Answer", key=f"article_btn_{idx}"):
        st.info(f"Correct Answer: {q['answer']}")

    st.divider()

# -------------------------------------------------
# Display Coding Questions
# -------------------------------------------------
st.header("Coding Questions")
for idx, q in enumerate(quiz.get("coding_questions", [])):
    st.markdown(f"### Coding Question {idx + 1}")
    st.write(q["question"])
    st.code(q.get("code_snippet", ""), language="python")

    user_answer = st.text_area(
        label="Your code/answer:",
        placeholder="Write your solution here...",
        key=f"coding_{idx}"
    )

    if st.button("Show Correct Answer", key=f"coding_btn_{idx}"):
        st.info(f"Correct Answer:\n{q['code_snippet']}\n\nExplanation: {q.get('explanation', '')}")

    st.divider()
