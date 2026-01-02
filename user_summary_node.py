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
# Utilities
# --------------------------------------------------

def merge_user_answers(quiz: dict, user_answers: dict):
    for i, q in enumerate(quiz.get("mcq_questions", [])):
        q["user_answer"] = user_answers.get(f"mcq_{i}")

    for i, q in enumerate(quiz.get("article_questions", [])):
        q["user_answer"] = user_answers.get(f"article_{i}")

    return quiz


def normalize_answer(value):
    if value is None:
        return None

    # Convert to string
    value = str(value)

    # Unicode normalization
    value = unicodedata.normalize("NFKC", value)

    # Strip whitespace
    value = value.strip()

    # Case normalization
    value = value.lower()

    return value


def parse_llm_json(response: str) -> dict:
    """
    Extracts and parses the first JSON object found in an LLM response.
    """

    # Remove markdown code fences if present
    response = response.strip()
    response = re.sub(r"^```(json)?", "", response)
    response = re.sub(r"```$", "", response)

    # Extract JSON object
    match = re.search(r"\{[\s\S]*\}", response)
    if not match:
        raise ValueError("No JSON object found in LLM response")

    return json.loads(match.group())


# --------------------------------------------------
# LLM helpers
# --------------------------------------------------

def grade_article(llm, question: str, model_answer: str, user_answer: str):
    """
    Grades an article-style answer using the LLM without a provided model answer.
    The LLM infers the expected answer from the question itself.
    """

    prompt = f"""
You are an educational evaluator.

Your task:
1. Infer what a high-quality answer to the question should contain.
2. Compare the user's answer against that inferred ideal.
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

Return ONLY valid JSON in the following format:
{{
  "score": <integer between 0 and 3>,
  "reasoning": "<brief explanation for the user to explain to they why this score was given to their answer and what the ideal answer is>"
}}
"""

    response = llm.invoke(prompt).content.strip()
    debug(f"LLM response for grading article: {response}")

    try:
        data = parse_llm_json(response)
        score = int(data.get("score", 0))
        score = max(0, min(3, score))
        reasoning = data.get("reasoning", "No reasoning provided.")
    except Exception:
        # Fail-safe in case the LLM output is malformed
        debug(f"JSON parsing failed: {e}")
        score = 0
        reasoning = "Failed to evaluate the answer due to invalid model output."

    return score, reasoning

def mcq_wrong_reasoning(llm, question, correct, user_answer):
    prompt = f"""
You are an educational assistant providing feedback to a student.

Task:
- Explain to the student why their selected answer is incorrect based on the provided model answer.
- Speak directly to the student.
- Use only the information in the question and the model answer.
- Avoid adding any external facts or making assumptions not in the question.
- Keep it concise, clear, and to the point (1-2 sentences).
- Don't use any introductions or Hey there, be direct and to the point.

Question:
{question}

Model answer:
{correct}

Student's answer:
{user_answer}

Provide your explanation directly to the student.
"""
    return llm.invoke(prompt).content.strip()


def performance_summary(llm, score, total, strong, weak):
    prompt = f"""
User quiz results:

Score: {score}/{total}

Strong areas:
{strong}

Weak areas:
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
    # submission = read_last_json(SUBMISSION_DIR)
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

        correct = q["correct_answer"]
        user = q.get("user_answer")
        norm_user = normalize_answer(user)
        norm_correct = normalize_answer(correct)

        if norm_user == norm_correct:
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
            if user is not None:
                reasoning = mcq_wrong_reasoning(
                    llm, q["question"], correct, user
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
                llm= llm,
                question= q["question"],
                model_answer= q["model_answer"],
                user_answer= user
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

    # Persist      ابقي شيلها بعدين 
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
