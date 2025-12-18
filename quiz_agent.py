from state import AgentState, Quiz
from config import Config
from langchain_core.prompts import PromptTemplate
import json

class QuizAgent:
    @staticmethod
    def quiz_node(state: AgentState) -> dict:
        print(f"\n[QUIZ AGENT NODE RUNNING]")
        
        # 1. Setup Gemini/Ollama
        llm = Config.get_ollama_llm() # Or Config.get_gemini_llm()
        structured_llm = llm.with_structured_output(Quiz)

        # --- KEY CHANGE: Use specific instructions ---
        topic_instruction = state.get("task_instructions")
        if not topic_instruction:
            topic_instruction = state.get("user_prompt", "General Programming")
            
        print(f"  👉 Generating quiz for: {topic_instruction}")
        
        prompt = PromptTemplate.from_template(
            "Generate a comprehensive quiz based on this request: '{topic}'. "
            "Include exactly 5 MCQs, 2 article questions, and 2 coding questions. "
            "Ensure the difficulty matches the topic complexity."
        )
        
        try:
            chain = prompt | structured_llm
            quiz_result = chain.invoke({"topic": topic_instruction})
            
            # Save locally
            filename = f"quiz_{str(hash(topic_instruction))[:8]}.json"
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(quiz_result.model_dump(), f, indent=2)

            return {
                "quiz_output": quiz_result,
                "last_node": "quiz_generator",
                # The orchestrator will read this 'quiz_output' from state
            }
        except Exception as e:
            print(f"  Error in Quiz Node: {e}")
            return {"last_node": "quiz_generator_error"}