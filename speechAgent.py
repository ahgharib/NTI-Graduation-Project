import streamlit as st
import os
from dotenv import load_dotenv
from groq import Groq
from streamlit_mic_recorder import mic_recorder
import base64
import asyncio
import edge_tts
import random
import uuid  # Used for unique HTML IDs

# --- LANGCHAIN IMPORTS ---
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults 
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

# Load environment variables
load_dotenv()

# --- VALIDATION CHECK ---
if not os.getenv("GROQ_API_KEY"):
    st.error("❌ GROQ_API_KEY is missing in .env file")
    st.stop()

# --- CONFIGURATION ---
client_groq = Groq(api_key=os.getenv("GROQ_API_KEY"))

st.set_page_config(page_title="AI Student Assistant", layout="centered")

# --- SYSTEM PROMPT ---
SYSTEM_PROMPT = """
You are a friendly and encouraging AI Student Assistant. 
Your goal is to help students learn by explaining complex topics in simple terms.
- Use analogies when possible.
- If the user asks a question, answer it clearly.
- If the answer involves code, provide examples.
"""

# --- AUDIO FUNCTIONS ---

def speech_to_text(audio_bytes):
    """Convert audio bytes to text using Groq Whisper (Fast & Cheap)."""
    try:
        with open("temp_input.wav", "wb") as f:
            f.write(audio_bytes)
        
        with open("temp_input.wav", "rb") as audio_file:
            transcript = client_groq.audio.transcriptions.create(
                model="whisper-large-v3",
                file=audio_file,
                response_format="text"
            )
        return transcript
    except Exception as e:
        st.error(f"STT Error: {e}")
        return None

async def edge_tts_generate(text, filename):
    """Async function to generate audio using Edge TTS."""
    voice = "en-US-MichelleNeural" # Options: en-US-ChristopherNeural (Male), en-GB-SoniaNeural (British)
    communicate = edge_tts.Communicate(text, voice, rate="-10%")
    await communicate.save(filename)

def text_to_speech(text):
    """Wrapper to call the async Edge TTS function."""
    try:
        # We use one static filename to keep your folder clean
        filename = "response_audio.mp3"
        
        # Remove old file if it exists to ensure fresh write
        if os.path.exists(filename):
            try:
                os.remove(filename)
            except PermissionError:
                pass # If file is in use, we just overwrite
        
        asyncio.run(edge_tts_generate(text, filename))
        return filename
    except Exception as e:
        st.error(f"Edge TTS Error: {e}")
        return None

def autoplay_audio(file_path: str):
    """
    Plays audio automatically, displays a synced waveform, and adds a Play/Pause button.
    """
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode()
        
        # We need a UNIQUE ID for the HTML element so the browser treats it as a new player
        # even though the filename is the same.
        unique_id = uuid.uuid4().hex[:8]
        audio_id = f"audio-{unique_id}"
        waveform_id = f"waveform-{unique_id}"
        btn_id = f"btn-{unique_id}"

        # 1. Generate Random Waveform Bars (Visuals)
        bars = "".join([f'<div class="bar" style="animation-duration: {random.uniform(0.5, 1.2)}s;"></div>' for _ in range(15)])

        # 2. The Complete HTML/JS Block
        html_code = f"""
        <style>
            .audio-widget {{
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                background-color: #0e1117;
                padding: 15px;
                border-radius: 12px;
                margin-top: 10px;
                border: 1px solid #303030;
                font-family: sans-serif;
            }}
            
            /* The Waveform Container - Hidden by default until playing */
            #{waveform_id} {{
                display: none; 
                align-items: center;
                justify-content: center;
                height: 50px;
                margin-bottom: 15px;
                transition: opacity 0.3s;
            }}
            
            .bar {{
                width: 6px;
                height: 10px;
                margin: 0 3px;
                background-color: #f55036;
                border-radius: 3px;
                animation: wave ease-in-out infinite;
            }}
            
            @keyframes wave {{
                0% {{ height: 10px; opacity: 0.5; }}
                50% {{ height: 40px; opacity: 1; }}
                100% {{ height: 10px; opacity: 0.5; }}
            }}
            
            /* Custom Button Styling */
            #{btn_id} {{
                background-color: transparent;
                border: 2px solid #f55036;
                color: #f55036;
                padding: 8px 20px;
                border-radius: 25px;
                cursor: pointer;
                font-size: 14px;
                font-weight: bold;
                transition: all 0.2s ease;
                display: flex;
                align-items: center;
                justify-content: center;
                min-width: 100px;
            }}
            #{btn_id}:hover {{
                background-color: #f55036;
                color: white;
            }}
        </style>

        <div class="audio-widget">
            <audio id="{audio_id}" autoplay style="display:none;">
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            
            <div id="{waveform_id}">
                {bars}
            </div>
            
            <button id="{btn_id}" onclick="toggleAudio('{unique_id}')">
                ⏸ Pause
            </button>
        </div>

        <script>
            function toggleAudio(uid) {{
                var audio = document.getElementById('audio-' + uid);
                var wave = document.getElementById('waveform-' + uid);
                var btn = document.getElementById('btn-' + uid);
                
                if (audio.paused) {{
                    audio.play();
                    wave.style.display = 'flex';
                    btn.innerHTML = '⏸ Pause';
                }} else {{
                    audio.pause();
                    wave.style.display = 'none';
                    btn.innerHTML = '▶ Play';
                }}
            }}

            (function() {{
                var audio = document.getElementById('{audio_id}');
                var wave = document.getElementById('{waveform_id}');
                var btn = document.getElementById('{btn_id}');
                
                // When audio plays (auto or manual), show wave
                audio.onplay = function() {{
                    wave.style.display = 'flex';
                    btn.innerHTML = '⏸ Pause';
                }};
                
                // When audio pauses, hide wave
                audio.onpause = function() {{
                    wave.style.display = 'none';
                    btn.innerHTML = '▶ Play';
                }};
                
                // When audio ends, hide wave and reset button
                audio.onended = function() {{
                    wave.style.display = 'none';
                    btn.innerHTML = '↺ Replay';
                }};
                
                // Attempt Autoplay
                audio.play().catch(e => {{
                    console.log("Autoplay blocked by browser");
                    btn.innerHTML = '▶ Play'; // Fallback state if blocked
                }});
            }})();
        </script>
        """
        
        st.components.v1.html(html_code, height=140)
        
    except Exception as e:
        st.error(f"Audio Playback Error: {e}")

# --- THE REACTIVE AGENT ---
search_tool = TavilySearchResults(max_results=2)
tools = [search_tool]

class State(TypedDict):
    messages: Annotated[list, add_messages]

llm = ChatGroq(
    model="llama-3.3-70b-versatile", 
    temperature=0.7,
    api_key=os.getenv("GROQ_API_KEY")
)
llm_with_tools = llm.bind_tools(tools)

def agent_node(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

workflow = StateGraph(State)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode(tools))

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", tools_condition)
workflow.add_edge("tools", "agent")
workflow.add_edge("agent", END)

app_graph = workflow.compile()

# --- UI/UX ---
st.markdown("""
    <style>
    .stChatMessage { font-family: 'Helvetica Neue', sans-serif; }
    h1 { color: #f55036; } 
    </style>
""", unsafe_allow_html=True)

st.title("⚡ AI Student Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = [SystemMessage(content=SYSTEM_PROMPT)]

# --- VOICE INPUT ---
with st.sidebar:
    st.header("🎙️ Voice Input")
    audio_data = mic_recorder(start_prompt="Start Recording", stop_prompt="Stop Recording", key='recorder')
    
    if audio_data:
        # Check if this specific recording ID has been processed
        if "last_audio_id" not in st.session_state or st.session_state.last_audio_id != audio_data['id']:
            with st.spinner("Listening..."):
                transcript = speech_to_text(audio_data['bytes'])
                if transcript:
                    st.session_state.user_input = transcript
                    st.session_state.last_audio_id = audio_data['id']

# --- CHAT INPUT ---
chat_container = st.container()
user_text = st.chat_input("Type your question here...")

final_input = None
if user_text:
    final_input = user_text
elif "user_input" in st.session_state and st.session_state.user_input:
    final_input = st.session_state.user_input
    st.session_state.user_input = None # Clear immediately

if final_input:
    st.session_state.messages.append(HumanMessage(content=final_input))
    
    # --- RENDER CHAT HISTORY (User Only First) ---
    with chat_container:
        for msg in st.session_state.messages:
            if isinstance(msg, HumanMessage):
                with st.chat_message("user"): st.write(msg.content)
            elif isinstance(msg, AIMessage):
                with st.chat_message("assistant"): st.write(msg.content)
    
    # --- GENERATE RESPONSE ---
    with st.spinner("Thinking..."):
        try:
            result = app_graph.invoke({"messages": st.session_state.messages})
            response_message = result["messages"][-1]
            st.session_state.messages.append(response_message)
            
            # Show the new AI response text immediately
            with chat_container:
                with st.chat_message("assistant"): st.write(response_message.content)
            
            # Generate Audio
            audio_file = text_to_speech(response_message.content)
            if audio_file:
                # This triggers the Smart Audio Player
                autoplay_audio(audio_file)
                
        except Exception as e:
            st.error(f"Error: {e}")

# --- PERSISTENT HISTORY RENDER (On Refresh) ---
if not final_input: 
    with chat_container:
        for msg in st.session_state.messages:
            if isinstance(msg, HumanMessage):
                with st.chat_message("user"): st.write(msg.content)
            elif isinstance(msg, AIMessage):
                with st.chat_message("assistant"): st.write(msg.content)