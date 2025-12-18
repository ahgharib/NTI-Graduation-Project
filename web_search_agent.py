import os
from langchain_tavily import TavilySearch
from state import AgentState
from config import Config
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
    
class WebSearchAgent:
    """Agent that performs web searches and saves results to a file."""    
    
    @staticmethod
    def search_node(state: AgentState) -> dict:
        print(f"\n--- WEB SEARCH AGENT RUNNING ---")
        
        # Initialize Tools and LLM
        tool = TavilySearch(max_results=3)
        llm = ChatGroq(model="llama-3.3-70b-versatile").bind_tools([tool])
        
        # --- KEY CHANGE: Use task_instructions ---
        instruction = state.get("task_instructions")
        if not instruction:
            instruction = state.get("user_prompt", "")
            
        print(f"  🔎 Searching for: {instruction}")

        # Invoke LLM with the specific instruction
        # We wrap it in a Message list as ChatModels expect messages
        messages = [HumanMessage(content=instruction)]
        
        response = llm.invoke(messages)
        
        final_content = response.content
        if response.tool_calls:
            print("  > Executing Web Search Tool...")
            for tool_call in response.tool_calls:
                results = tool.invoke(tool_call["args"])
                summary_prompt = f"Summarize these search results specifically for this request: '{instruction}'. Results: {results}"
                # final_res = Config.get_gemini_llm().invoke(summary_prompt)
                final_res = Config.get_ollama_llm().invoke(summary_prompt)
                final_content = final_res.content

        # Save output to file
        filename = "search_results.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"SEARCH INSTRUCTION: {instruction}\n")
            f.write("-" * 30 + "\n")
            f.write(final_content)
        
        print(f"  ✅ Search complete.")

        return {
            "final_output": final_content,
            "last_node": "web_search",
            "saved_file": filename
        }