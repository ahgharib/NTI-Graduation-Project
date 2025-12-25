import os
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from prompts import (
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

def run_rag(query: str, mode: str):
    vectorstore = load_vectorstore()

    # 1. Retrieve relevant chunks
    docs = vectorstore.similarity_search(query, k=3)
    

    context = "\n\n".join(
        f"[File: {os.path.basename(doc.metadata.get('source', 'N/A'))} | "
        f"Page {doc.metadata.get('page', 'N/A')}] "
        f"{doc.page_content}"
        for doc in docs
    )
    print("############################## Context ##############################:\n", context)
    print('#' *70)
    
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
