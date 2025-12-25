# agents.py
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from config import Config
from schemas import Roadmap

# --- PARSER SETUP ---
parser = PydanticOutputParser(pydantic_object=Roadmap)
format_instructions = parser.get_format_instructions()

# --- PROMPTS ---

# 1. UPDATED: Added {feedback} so the loop actually improves the result
GEN_PROMPT = """You are an expert Educational Planner.
Context from web: {search_context}
User Goal: {user_request}

{feedback_section}

TASK: Generate a detailed learning roadmap.
{format_instructions}

IMPORTANT: Ensure valid JSON output matching the schema strictly.
"""

# 2. UPDATED: Added {user_request} so the fixer knows the topic context
FIX_PROMPT = """You are a JSON Repair Agent.
The user wanted a roadmap for: "{user_request}"

The following JSON failed to parse:
{error}

Bad JSON Input:
{bad_json}

TASK: Fix the JSON structure to match the required schema exactly.
Ensure the content matches the user's request.
{format_instructions}
"""

# 3. UPDATED: Added {user_request} so it checks relevance, not just logic
DISC_PROMPT = """You are a Logic Critic.
User Goal: {user_request}
Review this roadmap:
{current_plan}

Check logic (order of steps), completeness, and RELEVANCE to the user goal.
Output ONLY JSON: {{"approved": boolean, "feedback": "string"}}
"""

EDITOR_PROMPT = """You are a Roadmap Manager.
Current Plan: {current_plan}
History: {user_input}
Selected Node: {selected_node}

TASK: Update the plan based on the user's request (status change or structure change).
{format_instructions}
IMPORTANT: Striclly follow this Format and  Ensure That You Only Output the Json file No Extra text or '''json''' or any thing before or after the json file No "OK here is Your correct Json file" or "Here is the corrected JSON structure:", Your Output will go to a PydanticOutputParser and it will check the Foramt structure and it will fail if you do not get the right strucutre
"""
# IMPORTANT: Ensure That You Only Output the Json file No Extra text or '''json''' or any thing before or after the json file No "OK here is Your correct Json file"

# --- AGENTS ---
llm_gen = Config.get_gemini_llm()
llm_disc = Config.get_ollama_llm(Config.OLLAMA_MODEL_3)
llm_edit = Config.get_groq_llm()
llm_val = Config.get_groq_llm

def generator_node(state):
    prompt = ChatPromptTemplate.from_template(GEN_PROMPT)
    chain = prompt | llm_gen
    
    # LOGIC FIX: Check if there is feedback from a previous attempt
    feedback = state.get("feedback")
    feedback_section = ""
    if feedback:
        feedback_section = f"PREVIOUS ATTEMPT CRITIQUE: {feedback}\nFIX THESE ISSUES IN THE NEW PLAN."

    try:
        response = chain.invoke({
            "search_context": state.get("search_context", ""),
            "user_request": state["user_request"],
            "feedback_section": feedback_section, # Inject feedback
            "format_instructions": format_instructions
        })
        return {
            "raw_output": response.content, 
            "attempt_count": state.get("attempt_count", 0) + 1
        }
    except Exception as e:
        return {"error": str(e)}

def validator_node(state):
    """
    Validates and fixes JSON from Generator or Editor.
    """
    raw_content = state.get("raw_output", "")
    
    # 1. Try to parse immediately
    try:
        # cleanup markdown if present
        cleaned = raw_content.replace("```json", "").replace("```", "").strip()
        parsed_obj = parser.parse(cleaned)
        return {"current_plan": parsed_obj.dict(), "error": None}
    
    # 2. If failure, call LLM to fix
    except Exception as e:
        print(f"DEBUG: Parsing failed, attempting fix. Error: {e}")
        fix_prompt = ChatPromptTemplate.from_template(FIX_PROMPT)
        fix_chain = fix_prompt | llm_val | parser 
        
        try:
            # LOGIC FIX: Pass 'user_request' so the fixer knows the context
            fixed_obj = fix_chain.invoke({
                "error": str(e),
                "bad_json": raw_content,
                "user_request": state.get("user_request", "General Learning Plan"),
                "format_instructions": format_instructions
            })
            return {"current_plan": fixed_obj.dict(), "error": None}
        except Exception as final_e:
            return {"error": f"CRITICAL FAILURE: {str(final_e)}"}

def discriminator_node(state):
    if state.get("error"):
        return {"approved": False}
        
    prompt = ChatPromptTemplate.from_template(DISC_PROMPT)
    chain = prompt | llm_disc 
    
    try:
        # LOGIC FIX: Pass 'user_request' to ensure relevance
        res_str = chain.invoke({
            "current_plan": state["current_plan"],
            "user_request": state["user_request"]
        }).content
        
        res_str = res_str.replace("```json", "").replace("```", "").strip()
        response = json.loads(res_str)
        return {"feedback": response.get("feedback"), "approved": response.get("approved")}
    except:
        return {"feedback": "Format Error in Discriminator", "approved": False}

def editor_node(state):
    prompt = ChatPromptTemplate.from_template(EDITOR_PROMPT)
    chain = prompt | llm_edit
    
    # Handle chat history safely
    history = state.get("messages", [])
    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history]) if history else "No history"

    try:
        response = chain.invoke({
            "current_plan": state["current_plan"],
            "user_input": history_str,
            "selected_node": state.get("ui_selected_node", "None"),
            "format_instructions": format_instructions
        })
        return {"raw_output": response.content}
    except Exception as e:
        return {"error": str(e)}