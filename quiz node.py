# ===============================================================
# MCQ LangGraph App (UPDATED – WORKING)
# ===============================================================

import os
import json
import dotenv
dotenv.load_dotenv(".env")

from typing import List, TypedDict
from pydantic import BaseModel, Field

# ---------------- LangChain ----------------
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.prompts import PromptTemplate

# ---------------- LangGraph ----------------
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import InMemorySaver


# ===============================================================
# 1) ENV CHECK
# ===============================================================
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")


# ===============================================================
# 2) Pydantic Schemas
# ===============================================================
class MCQ(BaseModel):
    question: str = Field(..., description="The multiple-choice question")
    options: List[str] = Field(..., description="List of answer options")
    correct_answer: str = Field(..., description="The correct answer from the options")

class ArticleQuestion(BaseModel):
    question: str = Field(..., description="The article question")
    answer: str = Field(..., description="The article answer")

class CodingQuestion(BaseModel):
    question: str = Field(..., description="The coding question")
    code_snippet: str = Field(..., description="The code snippet for the coding question")
    explanation: str = Field(..., description="Explanation of the code snippet")

class Quiz(BaseModel):
    topic: str = Field(..., description="The topic of the quiz")
    proficiency_level: str = Field(..., description="The proficiency level of the quiz")
    mcq_questions: List[MCQ] = Field(..., description="List of multiple-choice questions")
    article_questions: List[ArticleQuestion] = Field(..., description="List of article questions")
    coding_questions: List[CodingQuestion] = Field(..., description="List of coding questions")


# ===============================================================
# 3) LangGraph State
# ===============================================================
class MCQState(TypedDict):
    topic: str
    proficiency: str
    quiz: Quiz | None


# ===============================================================
# 4) Prompt
# ===============================================================
PROMPT_TEMPLATE = """
Search the web for exactly 5 multiple-choice questions and 2 article questions and 2 coding questions about {topic}
with {proficiency} proficiency level.

Rules:
- Use web search
- 4 options per mcq question
- One correct answer
- Provide concise and short answers for article questions
- Provide code snippets for coding questions
"""

prompt = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["topic", "proficiency"]
)


# ===============================================================
# 5) Model + Tools
# ===============================================================
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=api_key,
)

model = model.bind_tools([{"google_search": {}}])


# ===============================================================
# 6) Agent
# ===============================================================
checkpointer = InMemorySaver()

agent = create_agent(
    model=model,
    system_prompt=(
        "You are a web-search-only questions collecting tool. "
        "Never use your pretrained knowledge."
    ),
    checkpointer=checkpointer,
    response_format=ToolStrategy(Quiz)
)


# ===============================================================
# 7) LangGraph Node Function
# ===============================================================
def mcq_agent_node(state: MCQState) -> MCQState:
    filled_prompt = prompt.format(
        topic=state["topic"],
        proficiency=state["proficiency"]
    )

    response = agent.invoke(
        {"messages": [{"role": "user", "content": filled_prompt}]},
        config={"thread_id": f"questions-{state['topic']}"}
    )

    return {
        **state,
        "quiz": response["structured_response"]
    }


# ===============================================================
# 8) Build Graph
# ===============================================================
builder = StateGraph(MCQState)

builder.add_node("questions_generator", mcq_agent_node)
builder.set_entry_point("questions_generator")
builder.set_finish_point("questions_generator")

app = builder.compile()


# ===============================================================
# 9) Run
# ===============================================================
if __name__ == "__main__":

    initial_state: MCQState = {
        "topic": "Python programming",
        "proficiency": "Advanced",
        "quiz": None
    }

    final_state = app.invoke(initial_state)

    quiz = final_state["quiz"]

    print("\n===== GENERATED QUIZ =====\n")
    print(quiz.model_dump_json(indent=2))

    with open("quiz_output.json", "w", encoding="utf-8") as f:
        json.dump(quiz.model_dump(), f, indent=2, ensure_ascii=False)

    print("\nSaved to quiz_output.json")
