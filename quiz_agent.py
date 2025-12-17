# Create a new file: quiz_agent.py
from state import AgentState, Quiz
from config import Config
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
import json

class QuizAgent:
    @staticmethod
    def quiz_node(state: AgentState) -> dict:
        print(f"\n[QUIZ AGENT NODE RUNNING]")
        
        # 1. Setup Gemini with structured output
        # llm = Config.get_gemini_llm()
        llm = Config.get_ollama_llm()
        structured_llm = llm.with_structured_output(Quiz)

        topic = state.get("user_prompt", "General Programming")
        
        prompt = PromptTemplate.from_template(
            "Generate a comprehensive quiz about {topic}. "
            "Include exactly 5 MCQs, 2 article questions, and 2 coding questions. "
            "Set the proficiency level based on the technical depth of the topic."
        )
        
        try:
            chain = prompt | structured_llm
            quiz_result = chain.invoke({"topic": topic})
            
            # Save to file (optional but good for consistency)
            filename = f"quiz_{topic.replace(' ', '_')[:10]}.json"
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(quiz_result.model_dump(), f, indent=2)

            return {
                "quiz_output": quiz_result,
                "last_node": "quiz_generator",
                "final_output": f"✅ Quiz generated successfully for '{topic}' and saved to {filename}."
            }
        except Exception as e:
            print(f"  Error in Quiz Node: {e}")
            return {"last_node": "quiz_generator", "next": "END"}