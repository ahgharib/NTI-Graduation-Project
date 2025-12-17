import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import os

# Load environment variables
load_dotenv()
# GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")

# if not GOOGLE_KEY:
#     st.error("Google API key not found! Please set GOOGLE_API_KEY in your .env file.")
#     st.stop()

# # Initialize LLM
# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash",
#     google_api_key=GOOGLE_KEY
# )
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not HF_TOKEN:
    st.error("Hugging Face token not found!")
    st.stop()

# Initialize Hugging Face client
client = InferenceClient(
    model="meta-llama/Llama-3.1-8B-Instruct",
    token=HF_TOKEN
)

# Initialize DuckDuckGo search
web_search = DuckDuckGoSearchRun()

# Streamlit UI
st.title("🌐 Web Research & Explanation Assistant")
st.write("Enter any topic, it will search the web and explain it simply using Google Gemini.")

topic = st.text_input("Enter topic here:", "LangChain")

if st.button("Explain Topic"):
    with st.spinner("Searching and generating explanation..."):
        # Web search
        search_results = web_search.run(topic)

        # Prepare prompt
        prompt = f"""
        You are a helpful research assistant.

        Topic: {topic}

        Web search results:
        {search_results}

        Explain the topic clearly and simply.
        """

        # Get explanation from Gemini
        # response = llm.invoke(prompt)
        response = client.chat.completions.create(
    messages=[
        {"role": "system", "content": "You are a helpful research assistant."},
        {"role": "user", "content": prompt}
    ],
    max_tokens=512,
    temperature=0.4
    )

    st.subheader("📝 Explanation")
    # st.text(response.content)
    st.write(response.choices[0].message.content)


