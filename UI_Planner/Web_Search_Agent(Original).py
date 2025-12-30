import streamlit as st
import os
from dotenv import load_dotenv, find_dotenv

# --- 1. SETUP & CONFIG ---
st.set_page_config(page_title="Smart Search Agent", page_icon="🤖")
st.title("🤖 Agent with YouTube & Sand Clock")

# Load environment variables
load_dotenv(find_dotenv(), override=True)

# Check for keys
if not os.environ.get("GROQ_API_KEY") or not os.environ.get("TAVILY_API_KEY") or not os.environ.get("YOUTUBE_API_KEY"):
    st.error("⚠️ Keys missing! Make sure GROQ_API_KEY, TAVILY_API_KEY, and YOUTUBE_API_KEY are in your .env file.")
    st.stop()

# --- 2. IMPORTS ---
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

# --- 3. BUILD THE AGENT (Cached) ---
@st.cache_resource
def build_agent():
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
    system_prompt = """You are a helpful research assistant.
    
    1. **DUAL SEARCH STRATEGY**: 
       - Use `tavily_search_results_json` for articles and text.
       - Use `youtube_api_tool` for finding videos.
    
    2. **MANDATORY VIDEO**: 
       - You MUST use the `youtube_api_tool` for every query.
       - Include the video links provided by the tool.
    
    3. **FORMAT**:
       [Your text answer]
       
       **📚 Sources:**
       * [Page Title](URL)
       
       **📺 Related Videos:**
       * [Video Title](YouTube Link) - *Channel Name*
       

    """
    sys_msg = SystemMessage(content=system_prompt)

    # E. Define State
    class State(TypedDict):
        messages: Annotated[list, add_messages]

    # F. Define Nodes
    def chatbot_node(state: State):
        messages = [sys_msg] + state["messages"]
        return {"messages": [llm_with_tools.invoke(messages)]}

    tool_node = ToolNode(tools)

    # G. Define Logic
    def should_continue(state: State) -> Literal["tools", END]:
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools"
        return END

    # H. Build Graph
    workflow = StateGraph(State)
    workflow.add_node("agent", chatbot_node)
    workflow.add_node("tools", tool_node)
    
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", "agent")

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

# Initialize the agent
app = build_agent()
config = {"configurable": {"thread_id": "streamlit_user_1"}}


# --- 4. STREAMLIT CHAT UI ---

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What would you like to search?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # --- SAND CLOCK ANIMATION ---
        # 1. Create the placeholder
        loader_placeholder = st.empty()
        
        # 2. Display the animation
        loader_placeholder.markdown(
            """
            <div style="display: flex; justify-content: center; align-items: center; margin-bottom: 20px;">
                <img src="https://i.gifer.com/ZKZg.gif" width="60" />
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        input_message = {"messages": [("user", prompt)]}
        
        # Run the graph
        for event in app.stream(input_message, config, stream_mode="values"):
            if "messages" in event:
                last_msg = event["messages"][-1]
                
                # Check if this is the FINAL AI response (not a tool call)
                if last_msg.type == "ai" and not last_msg.tool_calls:
                    
                    # 3. ONLY clear the loader now that we have the final answer
                    if loader_placeholder:
                        loader_placeholder.empty()
                        loader_placeholder = None

                    raw_content = last_msg.content
                    
                    if isinstance(raw_content, list):
                        full_response = "".join(
                            [block["text"] for block in raw_content if "text" in block]
                        )
                    else:
                        full_response = raw_content
                    
                    message_placeholder.markdown(full_response)
        
        # Cleanup: Ensure loader is gone if the loop finishes
        if loader_placeholder:
            loader_placeholder.empty()
            
        st.session_state.messages.append({"role": "assistant", "content": full_response})