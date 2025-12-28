import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from RAG.prompts import (
    EXPLAIN_PROMPT,
    SUMMARY_PROMPT,
    RETRIEVE_PROMPT
)

VECTORSTORE_PATH = "RAG/vectorstore/"
OLLAMA_BASE_URL = "http://localhost:11434"

llm = ChatOllama(model="llama3", base_url=OLLAMA_BASE_URL, temperature=0.0)

def load_vectorstore():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

def get_context_chunks(query: str, k: int = 3):      ### vectorstore is now in st.session_state
    # vectorstore = load_vectorstore()               ### it's not saved locally anymore
    vectorstore = st.session_state.vectorstore
    if vectorstore is None:
        print("#" *70,"Vectorstore is not loaded.", "#" *70)
        print("#-#" * 100)
        return ""
    docs = vectorstore.similarity_search(query, k=k)
    context = "\n\n".join(
        f"[File: {os.path.basename(doc.metadata.get('source', 'N/A'))} | "
        f"Page {doc.metadata.get('page', 'N/A')}] "
        f"{doc.metadata.get('original_content', doc.page_content)}"
        for doc in docs
    )
    # print("############################## Context ##############################:\n", context)
    # print('#' *70)
    return context


def run_rag(query: str, mode: str):
    # 1. Retrieve context
    context = get_context_chunks(query)
    
    # 2. Select prompt
    if mode == "explain":
        prompt = EXPLAIN_PROMPT
    elif mode == "retrieve":
        prompt = RETRIEVE_PROMPT
    elif mode == "summary":
        prompt = SUMMARY_PROMPT
    else:
        raise ValueError("Invalid mode")

    # 3. Generate answer
    final_prompt = prompt.format(
        query=query,
        context=context
    )
    response = llm.invoke(final_prompt)
    return response.content 

if __name__ == "__main__":
    response = run_rag("what is asu racing team?", "retrieve")
    print("############################## Response ##############################:\n", response)
    print('#' *70)
