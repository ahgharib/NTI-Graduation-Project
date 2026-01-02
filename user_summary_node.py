from langchain_core.messages import SystemMessage
from config import Config
import json
from pathlib import Path
from typing import Dict, Any
import unicodedata
import re

SUBMISSION_DIR = Path("quiz_submissions")

def debug(msg):
    print(f"[USER_GRADER_NODE][DEBUG] {msg}")


# --------------------------------------------------
# Utilities       يا رب يشتغل
# --------------------------------------------------

def merge_user_answers(quiz: dict, user_answers: dict):
    for i, q in enumerate(quiz.get("mcq_questions", [])):
        q["user_answer"] = user_answers.get(f"mcq_{i}")

    for i, q in enumerate(quiz.get("article_questions", [])):
        q["user_answer"] = user_answers.get(f"article_{i}")

    return quiz


def normalize_mcq_key(value):
    """
    Normalizes MCQ answers to a single uppercase letter (A-D).
    """
    if value is None:
        return None

    value = str(value)
    value = unicodedata.normalize("NFKC", value)
    value = value.strip().upper()

    return value


def parse_llm_json(response: str) -> dict:
    response = response.strip()
    response = re.sub(r"^```(json)?", "", response)
    response = re.sub(r"```$", "", response)

    match = re.search(r"\{[\s\S]*\}", response)
    if not match:
        raise ValueError("No JSON object found in LLM response")

    return json.loads(match.group())


# --------------------------------------------------
# LLM helpers
# --------------------------------------------------

def grade_article(llm, question: str, model_answer: str, user_answer: str):
    prompt = f"""
You are an educational evaluator.

Your task:
1. Use the model answer as the ideal reference.
2. Compare the user's answer against it.
3. Grade the answer from 0 to 3 using the rubric below.

GRADING RUBRIC:
- 3: Correct, complete, and well-explained
- 2: Mostly correct but missing minor details or clarity
- 1: Partially correct, shows limited understanding
- 0: Incorrect, irrelevant, or empty answer

Question:
{question}

Model answer:
{model_answer}

User answer:
{user_answer}

Return ONLY valid JSON:
{{
  "score": <integer 0-3>,
  "reasoning": "<brief explanation for the user explaining the score and what a good answer should include>"
}}
"""

    response = llm.invoke(prompt).content.strip()
    # debug(f"LLM response for grading article:\n{response}")

    try:
        data = parse_llm_json(response)
        score = max(0, min(3, int(data.get("score", 0))))
        reasoning = data.get("reasoning", "No reasoning provided.")
    except Exception as e:
        debug(f"JSON parsing failed: {e}")
        score = 0
        reasoning = "Failed to evaluate the answer due to invalid model output."

    return score, reasoning


def mcq_wrong_reasoning(llm, question, correct_key, options, user_key):
    prompt = f"""
You are an educational assistant providing feedback to a student's wrong answer.

Rules:
- Speak directly to the student.
- Explain why the correct answer is right.
- Be concise and educational (1-2 sentences).
- No introductions.

Question:
{question}

Correct answer:
{correct_key}: {options[correct_key]}

User's incorrect answer:
{user_key if user_key is not None else "No answer provided"}: {options.get(user_key, "No answer provided")}

Provide feedback explaining why the correct answer is right.
"""
    return llm.invoke(prompt).content.strip()


def performance_summary(llm, score, total, strong, weak):
    prompt = f"""
User quiz results:

Score: {score}/{total}

Strong skills:
{strong}

Weak skills:
{weak}

Generate a concise learning summary with:
- Overall performance
- Strengths
- Weaknesses
- Clear recommendations
"""
    return llm.invoke(prompt).content.strip()


# --------------------------------------------------
# Main Node
# --------------------------------------------------

def user_summary_node(state):

    llm = Config.get_ollama_llm()

    submission = state.get("user_submission")
    if not submission:
        raise ValueError("No user submission found in AgentState")

    quiz = merge_user_answers(
        submission["quiz"],
        submission["user_answers"]
    )

    results = {
        "mcq_results": [],
        "article_results": [],
        "summary": {}
    }

    earned_points = 0
    total_points = 0
    strong = []
    weak = []

    # -------------------------
    # MCQ grading (1 point)
    # -------------------------
    for q in quiz.get("mcq_questions", []):
        total_points += 1

        correct_key = normalize_mcq_key(q["correct_answer"])
        user_key = normalize_mcq_key(q.get("user_answer"))

        if user_key == correct_key:
            earned_points += 1
            strong.append(q["skill"])
            results["mcq_results"].append({
                "question": q["question"],
                "is_correct": True,
                "reasoning": None
            })
        else:
            weak.append(q["skill"])
            reasoning = None
            if user_key:
                reasoning = mcq_wrong_reasoning(
                    llm,
                    q["question"],
                    correct_key,
                    q["options"],
                    user_key
                )

            results["mcq_results"].append({
                "question": q["question"],
                "is_correct": False,
                "reasoning": reasoning
            })

    # -------------------------
    # Article grading (3 points)
    # -------------------------
    for q in quiz.get("article_questions", []):
        total_points += 3

        user = q.get("user_answer")
        if not user:
            score, reasoning = 0, "No answer submitted."
        else:
            score, reasoning = grade_article(
                llm,
                q["question"],
                q["model_answer"],
                user
            )

        earned_points += score

        if score >= 2:
            strong.append(q["skill"])
        else:
            weak.append(q["skill"])

        results["article_results"].append({
            "question": q["question"],
            "score": score,
            "out_of": 3,
            "reasoning": reasoning
        })

    # -------------------------
    # Summary
    # -------------------------
    summary_text = performance_summary(
        llm, earned_points, total_points, strong, weak
    )

    accuracy = round((earned_points / total_points) * 100, 2)

    results["summary"] = {
        "score": earned_points,
        "total": total_points,
        "accuracy": accuracy,
        "strong_points": strong,
        "weak_points": weak,
        "feedback": summary_text
    }

    # Persist (temporary)
    (SUBMISSION_DIR / "user_profile_summary.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8"
    )

    return {
        "grader_output": results,
        "messages": [
            SystemMessage(content="Quiz graded successfully")
        ]
    }