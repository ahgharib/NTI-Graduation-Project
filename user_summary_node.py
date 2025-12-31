from langchain_core.messages import HumanMessage, SystemMessage
from config import Config
import os
import time
import json
from pathlib import Path
from typing import Any

SUBMISSION_DIR = Path("quiz_submissions")
print(f"[USER_SUMMARY_NODE] Using submission directory: {str(SUBMISSION_DIR)}")

def debug(msg):
    print(f"[USER_SUMMARY_NODE][DEBUG] {msg}")

def read_last_json(directory: str | Path) -> Any:
    directory = Path(directory)

    if not directory.exists() or not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    json_files = list(directory.glob("*.json"))

    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {directory}")

    # Sort by modification time (most recent last)
    latest_file = max(json_files, key=lambda p: p.stat().st_mtime)

    with latest_file.open("r", encoding="utf-8") as f:
        return json.load(f)


# def wait_for_submission(
#     directory: str,
#     timeout: int = 300,
#     poll_interval: float = 12.0
# ):
#     """
#     Waits for a new JSON file to appear in directory.
#     Returns parsed JSON or None on timeout.
#     """
#     debug(f"Waiting for submission in directory: {directory}")
#     debug(f"Timeout: {timeout}s | Poll interval: {poll_interval}s")

#     start = time.time()

#     if os.path.exists(directory):
#         seen = set(os.listdir(directory))
#         debug(f"Initial files in directory: {seen}")
#     else:
#         seen = set()
#         debug("Directory does not exist yet")

#     while time.time() - start < timeout:
#         elapsed = round(time.time() - start, 2)
#         if elapsed % 60 < poll_interval:
#             debug(f"Polling... elapsed={elapsed}s")

#         if not os.path.exists(directory):
#             debug("Directory still missing, sleeping...")
#             time.sleep(poll_interval)
#             continue

#         current = set(os.listdir(directory))
#         new_files = [
#             f for f in current - seen
#             if f.endswith(".json")
#         ]
#         if current:    
#             debug(f"Current files: {current}")
#         if new_files:
#             debug(f"New JSON files detected: {new_files}")

#         if new_files:
#             path = os.path.join(directory, new_files[0])
#             debug(f"Loading submission file: {path}")

#             try:
#                 with open(path, "r") as f:
#                     data = json.load(f)
#                 debug("Submission JSON loaded successfully")
#                 return data
#             except Exception as e:
#                 debug(f"ERROR reading submission file: {e}")
#                 return None

#         time.sleep(poll_interval)

#     debug("Timeout reached - no submission received")
#     return None


def merge_user_answers(quiz: dict, user_answers: dict):
    """
    Injects user answers into quiz questions.
    """
    debug("Merging user answers into quiz structure")

    for i, q in enumerate(quiz.get("mcq_questions", [])):
        answer_key = f"mcq_{i}"
        q["user_answer"] = user_answers.get(answer_key)
        debug(
            f"MCQ {i}: user_answer={q['user_answer']} | correct={q.get('correct_answer')}"
        )

    return quiz


def user_summary_node(state):
    """
    Generates a summarized performance report for the user
    based on their quiz submission.
    """
    node_name = "USER_SUMMARY_NODE"
    debug(f"Node started: {node_name}")

    # -------------------------
    # Wait for submission
    # -------------------------
    submission = read_last_json(str(SUBMISSION_DIR))

    if not submission:
        debug("No submission received - exiting node")
        return {
            "error": "No user submission found",
            "next": "END"
        }

    debug("Submission payload keys:")
    debug(list(submission.keys()))

    if "quiz" not in submission or "user_answers" not in submission:
        debug("ERROR: Submission JSON missing required keys")
        debug(submission)
        return {
            "error": "Invalid submission format",
            "next": "END"
        }

    quiz = merge_user_answers(
        submission["quiz"],
        submission["user_answers"]
    )

    # -------------------------
    # Scoring
    # -------------------------
    correct = 0
    total = 0
    weak_questions = []
    strong_questions = []

    debug("Starting MCQ scoring loop")

    for idx, q in enumerate(quiz.get("mcq_questions", [])):
        debug(f"Evaluating MCQ {idx}: {q.get('question')}")

        if q.get("user_answer") is None:
            debug("→ Skipped (no answer)")
            continue

        total += 1

        if q["user_answer"] == q["correct_answer"]:
            correct += 1
            strong_questions.append(q["question"])
            debug("→ Correct")
        else:
            weak_questions.append(q["question"])
            debug("→ Incorrect")

    accuracy = round((correct / total) * 100, 2) if total else 0.0

    debug(f"Scoring completed: {correct}/{total} ({accuracy}%)")

    # -------------------------
    # LLM Summary
    # -------------------------
    debug("Invoking LLM for summary generation")
    llm = Config.get_ollama_llm()

    prompt = f"""
    You are an educational analyst.

    User quiz performance data:
    - Accuracy: {accuracy}%
    - Correct answers: {correct}/{total}

    Strength areas:
    {strong_questions}

    Weak areas:
    {weak_questions}

    TASK:
    Generate a concise learning summary with:
    1. Overall performance assessment
    2. Key strengths
    3. Key weaknesses
    4. Clear learning recommendations

    Keep it short, factual, and reusable.
    """

    try:
        summary_text = llm.invoke(prompt).content.strip()
        debug("LLM summary generated successfully")
    except Exception as e:
        debug(f"ERROR during LLM invocation: {e}")
        summary_text = "Summary generation failed."

    # -------------------------
    # Persist summary
    # -------------------------
    summary_payload = {
        "accuracy": accuracy,
        "strengths": strong_questions,
        "weaknesses": weak_questions,
        "summary_text": summary_text
    }

    debug("Final summary payload:")
    debug(summary_payload["summary_text"])

    summary_path = SUBMISSION_DIR / "user_profile_summary.json"

    try:
        with open(summary_path, "w") as f:
            json.dump(summary_payload, f, indent=2)
        debug(f"Summary saved to: {summary_path}")
    except Exception as e:
        debug(f"ERROR writing summary file: {e}")

    debug("Node execution completed successfully")

    return {
        "user_profile_summary": summary_payload,
        "messages": [
            SystemMessage(content="User quiz performance analysis completed")
        ],
    }
