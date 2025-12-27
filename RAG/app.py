import streamlit as st
import os
from ingest import ingest_pdf
from rag import run_rag

st.set_page_config(page_title="RAG Assistant", layout="wide")

st.title("📄 RAG Document Assistant")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    file_path = f"RAG/data/{uploaded_file.name}"

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("File uploaded successfully")

    if st.button("Ingest Document"):
        chunks = ingest_pdf(file_path)
        st.success(f"Document ingested into {chunks} chunks")

    mode = st.selectbox(
        "What do you want to do?",
        ["retrieve", "explain", "summary"]
    )

    user_query = st.text_input(
        "Optional: Ask a specific question",
        value="what is asu racing team?"
    )

    if st.button("Run"):
        response = run_rag(user_query, mode)
        st.write(response)
