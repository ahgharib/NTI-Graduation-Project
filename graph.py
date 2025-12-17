"""
graph.py - LangGraph workflow for YouTube search with memory
"""
import os
import requests
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, List, Dict, Annotated
from operator import add


# ---------------------------
# Define State with Memory
# ---------------------------
class GraphState(TypedDict):
    query: str
    max_results: int
    refined_query: str
    videos: List[Dict]
    error: str
    # Memory fields
    search_history: Annotated[List[str], add]  # Accumulates all searches
    conversation_context: str  # Stores context for follow-up queries

# ---------------------------
# Node 1: Process query with memory context
# ---------------------------
def process_query_with_memory(state: GraphState) -> GraphState:
    """Process the query considering previous searches"""
    query = state["query"]
    search_history = state.get("search_history", [])
    
    # Build context from history
    context = ""
    if search_history:
        context = f"Previous searches: {', '.join(search_history[-3:])}. "  # Last 3 searches
    
    state["conversation_context"] = context
    
    # Add current query to history
    if "search_history" not in state:
        state["search_history"] = []
    state["search_history"] = [query]
    
    return state

# ---------------------------
# Node 2: Use Groq to refine search query with context
# ---------------------------
def refine_query_with_groq(state: GraphState) -> GraphState:
    """Use Groq to create a better YouTube search query with memory context"""
    query = state["query"]
    context = state.get("conversation_context", "")
    groq_api_key = os.getenv("GROQ_API_KEY")
    groq_api_url = "https://api.groq.com/openai/v1"
    
    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    }
    
    # Include context in the prompt if available
    user_prompt = f"{context}Create an optimized YouTube search query for: {query}"
    
    data = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that creates optimized YouTube search queries. Consider the user's search history to provide relevant queries. Return ONLY the search query, nothing else."},
            {"role": "user", "content": user_prompt}
        ]
    }

    try:
        response = requests.post(f"{groq_api_url}/chat/completions", json=data, headers=headers)
        response.raise_for_status()
        result = response.json()
        state["refined_query"] = result['choices'][0]['message']['content'].strip()
    except Exception as e:
        state["refined_query"] = query  # Fallback to original
        state["error"] = f"Groq API Error: {e}"
    
    return state

# ---------------------------
# Node 3: Search YouTube
# ---------------------------
def search_youtube(state: GraphState) -> GraphState:
    """Search YouTube using the API"""
    youtube_api_key = os.getenv("YOUTUBE_API_KEY")
    
    if not youtube_api_key:
        state["error"] = "YouTube API Key not found. Please add YOUTUBE_API_KEY to your .env file"
        state["videos"] = []
        return state
    
    query = state["refined_query"]
    max_results = state["max_results"]
    
    try:
        url = "https://www.googleapis.com/youtube/v3/search"
        params = {
            "part": "snippet",
            "q": query,
            "type": "video",
            "maxResults": max_results,
            "key": youtube_api_key
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        videos = []
        for item in data.get("items", []):
            video_id = item["id"]["videoId"]
            video = {
                "id": video_id,
                "title": item["snippet"]["title"],
                "description": item["snippet"]["description"],
                "thumbnail": item["snippet"]["thumbnails"]["medium"]["url"],
                "channel": item["snippet"]["channelTitle"]
            }
            videos.append(video)
        
        state["videos"] = videos
        state["error"] = ""
        
    except Exception as e:
        state["error"] = f"YouTube API Error: {e}"
        state["videos"] = []
    
    return state

# ---------------------------
# Build LangGraph with Memory
# ---------------------------
def build_youtube_graph():
    """Build and return the compiled LangGraph workflow with memory"""
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("process_memory", process_query_with_memory)
    workflow.add_node("refine_query", refine_query_with_groq)
    workflow.add_node("search_youtube", search_youtube)
    
    # Define the flow
    workflow.add_edge(START, "process_memory")
    workflow.add_edge("process_memory", "refine_query")
    workflow.add_edge("refine_query", "search_youtube")
    workflow.add_edge("search_youtube", END)
    
    # Add memory checkpointer
    memory = MemorySaver()
    
    return workflow.compile(checkpointer=memory)

# ---------------------------
# Helper function to create initial state
# ---------------------------
def create_initial_state(query: str, max_results: int = 5) -> GraphState:
    """Create the initial state for the graph"""
    return {
        "query": query,
        "max_results": max_results,
        "refined_query": "",
        "videos": [],
        "error": "",
        "search_history": [],
        "conversation_context": ""
    }