from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings   

VECTORSTORE_PATH = "RAG/vectorstore/"

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = FAISS.load_local(
            VECTORSTORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

for i, doc in enumerate(vectorstore.docstore._dict.values()):
    page_num = doc.metadata.get("page", "N/A")
    print(f"Chunk {i+1}: Page {page_num} - {doc.page_content[:100]}...")