import os
import json
import requests
from state import AgentState
from langchain_core.messages import HumanMessage
from datetime import datetime

class YouTubeAgent:
    @staticmethod
    def youtube_node(state: AgentState) -> dict:
        print(f"\n[YOUTUBE SEARCH AGENT RUNNING]")
        
        # --- KEY CHANGE: Priority to specific task instructions ---
        original_query = state.get("task_instructions")
        if not original_query:
            original_query = state.get("user_prompt", "Python programming")
            
        print(f"  👉 processing query: {original_query}")
        
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
                        {"role": "system", "content": "Return ONLY the optimized YouTube search query text."},
                        {"role": "user", "content": f"Create an optimized YouTube search query for: {original_query}"}
                    ]
                }
                response = requests.post("https://api.groq.com/openai/v1/chat/completions", json=data, headers=headers)
                if response.status_code == 200:
                    refined_query = response.json()['choices'][0]['message']['content'].strip()
            except Exception as e:
                print(f"  ⚠️ Groq Refinement failed: {e}")

        # 3. PHASE 2: Search YouTube
        videos = []
        try:
            url = "https://www.googleapis.com/youtube/v3/search"
            params = {
                "part": "snippet",
                "q": refined_query,
                "maxResults": 3, # Reduced to 3 to keep context window cleaner
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
                    "channel": item["snippet"]["channelTitle"]
                })
        except Exception as e:
            return {"final_output": f"Error searching YouTube: {str(e)}", "last_node": "youtube_search"}

        # 4. Return formatted for Orchestrator Memory
        success_msg = f"Found {len(videos)} videos for '{refined_query}'."
        # SAVE VIDEO RESULTS TO JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        videos_filename = f"videos_{timestamp}.json"
        video_data = {
            "search_query": refined_query,
            "original_query": original_query,
            "search_date": timestamp,
            "videos": videos
        }

        with open(videos_filename, 'w', encoding='utf-8') as f:
            json.dump(video_data, f, indent=2, ensure_ascii=False)
        print(f"  ✅ Videos saved to: {videos_filename}")
        


        return {
            "videos": videos,
            "videos_data": video_data,  # NEW: Complete video data
            "videos_file": videos_filename,  # NEW: Filename
            "last_node": "youtube_search",
            "search_query": refined_query,
            "messages": [HumanMessage(content=success_msg, name="YouTubeAgent")]
        }