# search_agent.py
import os
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.tools import TavilySearchResults
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from googleapiclient.discovery import build
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv(), override=True)

# Check for required API keys
def check_api_keys():
    required_keys = ["GROQ_API_KEY", "TAVILY_API_KEY", "YOUTUBE_API_KEY"]
    missing_keys = []
    
    for key in required_keys:
        if not os.environ.get(key):
            missing_keys.append(key)
    
    if missing_keys:
        raise ValueError(f"Missing API keys: {', '.join(missing_keys)}. "
                        "Please add them to your .env file.")

# A. Define the Text Search Tool
web_search_tool = TavilySearchResults(max_results=5)

# B. Define the Official YouTube API Tool
@tool
def youtube_api_tool(query: str):
    """
    Searches YouTube using the official API to find real video links.
    Returns the Title, Channel, and URL of the top 2 videos.
    """
    try:
        api_key = os.environ["YOUTUBE_API_KEY"]
        youtube = build('youtube', 'v3', developerKey=api_key)
        
        request = youtube.search().list(
            q=query,
            part='snippet',
            type='video',
            maxResults=2
        )
        response = request.execute()
        
        results = []
        for item in response.get('items', []):
            title = item['snippet']['title']
            channel = item['snippet']['channelTitle']
            video_id = item['id']['videoId']
            url = f"https://www.youtube.com/watch?v={video_id}"
            results.append(f"Video: {title} | Channel: {channel} | Link: {url}")
            
        return "\n".join(results) if results else "No videos found."
        
    except Exception as e:
        return f"Error searching YouTube: {str(e)}"

# List of tools available to the agent
tools = [web_search_tool, youtube_api_tool]

# C. Define Model
llm = ChatGroq(model="llama-3.3-70b-versatile")
llm_with_tools = llm.bind_tools(tools)

# D. Define System Prompt
system_prompt = """You are a helpful research assistant that provides comprehensive information.

1. **DUAL SEARCH STRATEGY**: 
   - Use `tavily_search_results_json` for articles and text information.
   - Use `youtube_api_tool` for finding relevant videos.

2. **MANDATORY VIDEO SEARCH**: 
   - You MUST use the `youtube_api_tool` for every query to find relevant videos.
   - Include the video links provided by the tool in your response.

3. **FORMAT YOUR RESPONSE**:
   [Provide a comprehensive answer based on the search results]
   
   **📚 Sources:**
   * [Page Title](URL)
   
   **📺 Related Videos:**
   * [Video Title](YouTube Link) - *Channel Name*
   
4. **If the query is about a specific milestone/topic from a learning roadmap**, 
   tailor your search to provide resources specifically relevant to that topic.

"""
sys_msg = SystemMessage(content=system_prompt)

# E. Define State
class SearchState(TypedDict):
    messages: Annotated[list, add_messages]

# F. Define Nodes
def chatbot_node(state: SearchState):
    messages = [sys_msg] + state["messages"]
    return {"messages": [llm_with_tools.invoke(messages)]}

tool_node = ToolNode(tools)

# G. Define Logic
def should_continue(state: SearchState) -> Literal["tools", END]:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END

# H. Build Graph
def build_search_agent():
    """Build and return the search agent graph."""
    check_api_keys()  # Check API keys before building
    
    workflow = StateGraph(SearchState)
    workflow.add_node("agent", chatbot_node)
    workflow.add_node("tools", tool_node)
    
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", "agent")

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

# Create a singleton instance
_search_agent = None

def get_search_agent():
    """Get the search agent instance (creates it if needed)."""
    global _search_agent
    if _search_agent is None:
        _search_agent = build_search_agent()
    return _search_agent

def search_with_agent(query: str, context: str = ""):
    """
    Execute a search with the agent.
    
    Args:
        query: The search query
        context: Optional context about what the user is learning
        
    Returns:
        The agent's response as a string
    """
    try:
        # Add context to query if provided
        if context:
            full_query = f"{query} (Context: {context})"
        else:
            full_query = query
            
        agent = get_search_agent()
        config = {"configurable": {"thread_id": "streamlit_user"}}
        
        input_message = {"messages": [("user", full_query)]}
        
        # Run the graph
        for event in agent.stream(input_message, config, stream_mode="values"):
            if "messages" in event:
                last_msg = event["messages"][-1]
                
                # Return the final AI response
                if last_msg.type == "ai" and not last_msg.tool_calls:
                    raw_content = last_msg.content
                    
                    if isinstance(raw_content, list):
                        return "".join(
                            [block["text"] for block in raw_content if "text" in block]
                        )
                    else:
                        return raw_content
        
        return "I couldn't find information on that topic."
        
    except Exception as e:
        return f"Search error: {str(e)}"