# from langchain_community.tools import DuckDuckGoSearchRun

# def search_web(query: str) -> str:
#     """
#     Searches the web for educational resources and roadmap steps.
#     Using DuckDuckGo (Free, no API key required).
#     """
#     search = DuckDuckGoSearchRun()
#     try:
#         # We limit the results to avoid token overflow
#         results = search.invoke(f"how to learn {query} roadmap steps resources")
#         return results
#     except Exception as e:
#         return f"Search failed: {str(e)}"

# # Wrapper for the graph node
# def search_node_func(state):
#     query = state["user_request"]
#     # If we are in a refinement loop, we might want to search based on feedback
#     if state.get("feedback"):
#         query = f"{query} fix: {state['feedback']}"
        
#     results = search_web(query)
#     return {"search_context": results}


import os
import requests # Added for optional validation if you really need it
from dotenv import load_dotenv, find_dotenv
from langchain_community.tools import TavilySearchResults
from googleapiclient.discovery import build

# 1. Load Environment Variables
load_dotenv(find_dotenv(), override=True)

# --- HELPER FUNCTIONS ---

def run_tavily_search(query: str):
    """
    Searches the web using Tavily.
    - Fetches 10 results.
    - Deduplicates links (removes repeats).
    """
    try:
        # 1. Fetch results
        tavily = TavilySearchResults(max_results=10) # 10 is a good number for variety
        results = tavily.invoke(query)
        
        # 2. Process & Deduplicate
        seen_urls = set()
        formatted_results = []
        
        for res in results:
            url = res['url']
            content = res['content']
            
            # Skip duplicates or empty content
            if url in seen_urls or len(content) < 50:
                continue
                
            seen_urls.add(url)
            formatted_results.append(f"- {content[:300]}... (Source: {url})")
            
        return "\n".join(formatted_results) if formatted_results else "No relevant web results found."

    except Exception as e:
        return f"Web search error: {str(e)}"

def run_youtube_search(query: str):
    """
    Searches YouTube using the Official API.
    - Fetches 5 videos.
    - Guaranteed to be 'real' links by the API.
    """
    try:
        api_key = os.environ.get("YOUTUBE_API_KEY")
        if not api_key:
            return "Error: YouTube API Key missing."
            
        youtube = build('youtube', 'v3', developerKey=api_key)
        
        # Search for video resources
        request = youtube.search().list(
            q=query, 
            part='snippet', 
            type='video', 
            maxResults=5 # Fetching 5 videos to give the user variety
        )
        response = request.execute()
        
        results = []
        for item in response.get('items', []):
            title = item['snippet']['title']
            channel = item['snippet']['channelTitle']
            video_id = item['id']['videoId']
            url = f"https://www.youtube.com/watch?v={video_id}"
            
            results.append(f"- [VIDEO] {title} by {channel}: {url}")
            
        return "\n".join(results) if results else "No videos found."

    except Exception as e:
        return f"YouTube search error: {str(e)}"

# --- MAIN NODE FUNCTION ---

def search_node_func(state):
    """
    The main function called by your Graph.
    This prepares the 'Context' that the AI reads.
    """
    # 1. Get the user's base topic
    base_query = state["user_request"]
    
    # 2. Check for feedback loop (Refinement)
    if state.get("feedback"):
        base_query = f"{base_query} fix: {state['feedback']}"
    
    # 3. Create the search query
    # We add "roadmap" and "tutorial" to ensure we get learning resources
    search_query = f"how to learn {base_query} roadmap tutorial"
    
    print(f"🔍 Searching for: {search_query}") 
    
    # 4. Run BOTH searches
    web_data = run_tavily_search(search_query)
    video_data = run_youtube_search(search_query)
    
    # 5. Combine into one big block of text for the AI
    final_context = f"""
    The user wants to learn '{base_query}'. Use these resources to build the roadmap:
    
    **📚 WEB ARTICLES (Select the best 2-3):**
    {web_data}
    
    **📺 VIDEO TUTORIALS (Select the best 2-3):**
    {video_data}
    """
    
    # Return to the graph state
    return {"search_context": final_context}