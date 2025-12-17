import streamlit as st
import os
from dotenv import load_dotenv, find_dotenv

# --- 1. SETUP & CONFIG ---
st.set_page_config(page_title="Smart Search Agent", page_icon="🤖")
st.title("🤖 Agent with Web Search & Memory")

# Load environment variables
load_dotenv(find_dotenv(), override=True)

# Check for keys (Simple UI warning)
if not os.environ.get("GROQ_API_KEY") or not os.environ.get("TAVILY_API_KEY"):
    st.error("⚠️ API Keys missing! Please check your .env file.")
    st.stop()

# --- 2. IMPORTS (Lazy loading) ---
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import TavilySearchResults
from langchain_groq import ChatGroq

# --- 3. BUILD THE AGENT (Cached) ---
# We use @st.cache_resource so we only build the agent ONCE, not on every refresh.
@st.cache_resource
def build_agent():
    # A. Define Tools
    tool = TavilySearchResults(max_results=2)
    tools = [tool]

    # B. Define Model (Gemini 2.5 Flash as we discussed)
    # llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    llm = ChatGroq(model="llama-3.3-70b-versatile")
    llm_with_tools = llm.bind_tools(tools)

    # C. Define State
    class State(TypedDict):
        messages: Annotated[list, add_messages]

    # D. Define Nodes
    def chatbot_node(state: State):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    tool_node = ToolNode(tools)

    # E. Define Logic (EDGE)
    def should_continue(state: State) -> Literal["tools", END]:
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools"
        return END

    # F. Build Graph
    workflow = StateGraph(State)
    workflow.add_node("agent", chatbot_node)
    workflow.add_node("tools", tool_node)
    
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", "agent")

    # G. Return the compiled app with Memory
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

# Initialize the agent
app = build_agent()
config = {"configurable": {"thread_id": "streamlit_user_1"}}

# --- 4. STREAMLIT CHAT UI ---

# Initialize chat history in session state if not present
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What would you like to search?"):
    # 1. Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # 2. Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 3. Run the Agent
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # We stream the events to show "Thinking..." or results in real-time
        # Note: We send the raw user input to LangGraph
        input_message = {"messages": [("user", prompt)]}
        
        # Run the graph
        for event in app.stream(input_message, config, stream_mode="values"):
            # We are interested in the final message content
            if "messages" in event:
                last_msg = event["messages"][-1]
                
                # Only show if it is an AI message and NOT a tool call
                if last_msg.type == "ai" and not last_msg.tool_calls:
                    raw_content = last_msg.content
                    
                    # --- NEW FIX START ---
                    # If content is a list (JSON structure), extract just the text
                    if isinstance(raw_content, list):
                        full_response = "".join(
                            [block["text"] for block in raw_content if "text" in block]
                        )
                    else:
                        full_response = raw_content
                    # --- NEW FIX END ---
                    
                    message_placeholder.markdown(full_response)
        
        # 4. Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})