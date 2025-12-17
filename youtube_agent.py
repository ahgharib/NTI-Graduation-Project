import os
import json
import requests
from state import AgentState
from langchain_core.messages import HumanMessage

class YouTubeAgent:
    @staticmethod
    def youtube_node(state: AgentState) -> dict:
        print(f"\n[YOUTUBE SEARCH AGENT RUNNING]")
        
        # 1. Setup Parameters
        original_query = state.get("user_prompt", "Python programming")
        groq_api_key = os.getenv("GROQ_API_KEY")
        youtube_api_key = os.getenv("YOUTUBE_API_KEY")
        
        # 2. PHASE 1: Refine Query using Groq
        refined_query = original_query
        if groq_api_key:
            print(f"  Refining query with Groq...")
            try:
                headers = {
                    "Authorization": f"Bearer {groq_api_key}",
                    "Content-Type": "application/json"
                }
                data = {
                    "model": "llama-3.3-70b-versatile",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant that creates optimized YouTube search queries. Return ONLY the search query text, nothing else."},
                        {"role": "user", "content": f"Create an optimized YouTube search query for: {original_query}"}
                    ]
                }
                response = requests.post("https://api.groq.com/openai/v1/chat/completions", json=data, headers=headers)
                response.raise_for_status()
                refined_query = response.json()['choices'][0]['message']['content'].strip()
                print(f"  Optimized Query: {refined_query}")
            except Exception as e:
                print(f"  ⚠️ Groq Refinement failed: {e}. Using original query.")

        # 3. PHASE 2: Search YouTube
        print(f"  Searching YouTube...")
        videos = []
        try:
            url = "https://www.googleapis.com/youtube/v3/search"
            params = {
                "part": "snippet",
                "q": refined_query,
                "maxResults": 5,
                "type": "video",
                "key": youtube_api_key
            }
            yt_res = requests.get(url, params=params)
            yt_res.raise_for_status()
            items = yt_res.json().get("items", [])
            
            for item in items:
                videos.append({
                    "title": item["snippet"]["title"],
                    "link": f"https://www.youtube.com/watch?v={item['id']['videoId']}",
                    "channel": item["snippet"]["channelTitle"],
                    "description": item["snippet"]["description"]
                })
        except Exception as e:
            error_msg = f"❌ YouTube search failed: {str(e)}"
            return {"final_output": error_msg, "last_node": "youtube_search"}

        # 4. PHASE 3: Save to File
        filename = f"youtube_results_{original_query.replace(' ', '_')[:10]}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump({
                "original_query": original_query,
                "refined_query": refined_query,
                "results": videos
            }, f, indent=2)
            
        success_msg = f"✅ Found {len(videos)} videos using refined query: '{refined_query}'. Saved to {filename}."
        
        # Return to Orchestrator
        return {
            "videos": videos,
            "last_node": "youtube_search",
            "final_output": success_msg,
            "messages": [HumanMessage(content=success_msg, name="YouTubeAgent")]
        }