"""
app.py - Streamlit UI for YouTube Search with Memory
"""
import os
from dotenv import load_dotenv
import streamlit as st
from graph import build_youtube_graph, create_initial_state

# ---------------------------
# Load environment variables
# ---------------------------
load_dotenv()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# ---------------------------
# Initialize graph and session state
# ---------------------------
if "graph" not in st.session_state:
    st.session_state.graph = build_youtube_graph()
    st.session_state.thread_id = "user_session_1"  # Unique thread for memory
    st.session_state.search_count = 0
    st.session_state.all_searches = []

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="YouTube Search", page_icon="🎥", layout="wide")

st.title("🎥 YouTube Search with Groq AI")

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This app uses:
    - **Groq AI** to refine your search query
    - **YouTube Data API** to fetch videos
    - **LangGraph** to orchestrate the workflow
    - **Memory** to remember your search history
    
    ### Setup Required:
    Add to your `.env` file:
    ```
    GROQ_API_KEY=your_key
    YOUTUBE_API_KEY=your_key
    ```
    
    [Get YouTube API Key](https://console.cloud.google.com/apis/credentials)
    """)
    
    # Show API status
    st.markdown("---")
    st.subheader("API Status")
    if os.getenv("GROQ_API_KEY"):
        st.success("✅ Groq API Key found")
    else:
        st.error("❌ Groq API Key missing")
    
    if YOUTUBE_API_KEY:
        st.success("✅ YouTube API Key found")
    else:
        st.error("❌ YouTube API Key missing")
    
    # Show memory stats
    st.markdown("---")
    st.subheader("🧠 Memory Stats")
    st.metric("Searches in this session", st.session_state.search_count)
    
    if st.session_state.all_searches:
        st.markdown("**Recent searches:**")
        for i, search in enumerate(st.session_state.all_searches[-5:], 1):
            st.text(f"{i}. {search}")
    
    # Clear memory button
    if st.button("🗑️ Clear Memory"):
        st.session_state.graph = build_youtube_graph()
        st.session_state.thread_id = f"user_session_{st.session_state.search_count + 1}"
        st.session_state.all_searches = []
        st.session_state.search_count = 0
        st.success("Memory cleared!")
        st.rerun()

# Main content
query = st.text_input(
    "Enter your search query:", 
    placeholder="e.g., Python tutorials for beginners",
    help="The agent will remember your previous searches to provide better results!"
)
max_results = st.slider("Number of videos", 1, 10, 5)

col1, col2 = st.columns([3, 1])
with col1:
    search_button = st.button("🔍 Search Videos", type="primary")
with col2:
    if st.button("💡 Get Suggestions"):
        if st.session_state.all_searches:
            st.info(f"💭 Based on your history, you might also like: '{st.session_state.all_searches[-1]} advanced' or 'best {st.session_state.all_searches[-1]}'")
        else:
            st.info("💭 Start searching to get personalized suggestions!")

if search_button:
    if query.strip() == "":
        st.warning("⚠️ Please enter a search query!")
    elif not YOUTUBE_API_KEY:
        st.error("⚠️ YouTube API Key not found! Please add YOUTUBE_API_KEY to your .env file.")
        st.info("Get your API key from: https://console.cloud.google.com/apis/credentials")
    else:
        with st.spinner("🔄 Searching YouTube with memory context..."):
            # Create initial state
            initial_state = create_initial_state(query, max_results)
            
            # CRITICAL: Config with thread_id is REQUIRED when using checkpointer
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            
            # Run the graph with memory - MUST pass config parameter
            final_state = st.session_state.graph.invoke(initial_state, config)
            
            # Update session stats
            st.session_state.search_count += 1
            st.session_state.all_searches.append(query)
            
            # Show search context
            if final_state.get("conversation_context"):
                with st.expander("🧠 Memory Context Used"):
                    st.info(final_state["conversation_context"])
            
            # Show refined query
            if final_state["refined_query"]:
                st.info(f"🤖 Refined search query: **{final_state['refined_query']}**")
            
            # Show errors if any
            if final_state["error"]:
                st.error(final_state["error"])
            
            # Display videos
            videos = final_state["videos"]
            if videos:
                st.success(f"✅ Found {len(videos)} videos!")
                st.markdown("---")
                
                for i, video in enumerate(videos, 1):
                    st.markdown(f"### {i}. {video['title']}")
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Embed the video
                        video_url = f"https://www.youtube.com/embed/{video['id']}"
                        st.markdown(
                            f'<iframe width="100%" height="315" src="{video_url}" '
                            f'frameborder="0" allow="accelerometer; autoplay; clipboard-write; '
                            f'encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>',
                            unsafe_allow_html=True
                        )
                    
                    with col2:
                        st.image(video['thumbnail'], use_container_width=True)
                        st.markdown(f"**Channel:** {video['channel']}")
                        st.markdown(f"[🔗 Open in YouTube](https://www.youtube.com/watch?v={video['id']})")
                    
                    with st.expander("📝 Show description"):
                        st.write(video['description'])
                    
                    st.markdown("---")
            else:
                st.warning("No videos found. Try a different search query.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Built with ❤️ using Streamlit, LangGraph, Groq AI, and Memory"
    "</div>",
    unsafe_allow_html=True
)