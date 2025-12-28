import os
import logging
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from RAG.prompts import CHUNK_SUMMARY_PROMPT, FILE_SUMMARY_PROMPT
from RAG.OCR import extract_text_from_pdf, clean_text

VECTORSTORE_PATH = "RAG/vectorstore"

# -------------------
# Logging configuration
# -------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# -------------------
# LLM Setup
# -------------------
llm = ChatOllama(model="llama3.2:1b", temperature=0.0)

# -------------------
# PDF ingestion
# -------------------
def ingest_pdf(file_path: str):
    logger.info(f"Starting ingestion for: {file_path}")
    all_documents = []

    # 1️⃣ Try native text extraction first
    try:
        logger.info("Attempting text extraction using PyPDFLoader...")
        loader = PyPDFLoader(file_path)
        text_docs = loader.load()
        combined_text = " ".join(d.page_content for d in text_docs)

        if len(combined_text.strip()) > 500:
            logger.info(f"PyPDFLoader extracted {len(combined_text)} characters. Using native text.")
            for i, d in enumerate(text_docs):
                all_documents.append(
                    Document(
                        page_content=clean_text(d.page_content),
                        metadata={
                            "source": file_path,
                            "page": i + 1,
                            "loader": "pypdf"
                        }
                    )
                )
        else:
            logger.info(f"PyPDFLoader text too short ({len(combined_text)} chars). Will fallback to OCR.")
    except Exception as e:
        logger.warning(f"PyPDFLoader failed: {e}. Will fallback to OCR.")

    # 2️⃣ OCR fallback if text extraction failed
    if not all_documents:
        logger.info("Running OCR fallback with DocTR...")
        all_documents = extract_text_from_pdf(file_path)
        logger.info(f"OCR extracted {len(all_documents)} pages.")

    if not all_documents:
        logger.error(f"No text extracted from {file_path}. Aborting.")
        raise ValueError(f"No text extracted from {file_path}")

    # 3️⃣ Split per-page documents
    logger.info("Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(all_documents)
    logger.info(f"Created {len(chunks)} chunks from {len(all_documents)} pages.")

    for chunk in chunks:
        if "page" in chunk.metadata:
            chunk.metadata["page"] = str(chunk.metadata["page"])

    # -------------------
    # Print each chunk with page number(s) and source file
    # -------------------
    logger.info("Printing chunks with their page numbers:")
    for idx, chunk in enumerate(chunks):
        page_num = chunk.metadata.get("page", "N/A")
        file_name = os.path.basename(chunk.metadata.get("source", "N/A"))
        print(f"\n--- Chunk {idx+1} (Page: {page_num}, Source: {file_name}) ---\n{chunk.page_content[:800]}\n")


    # 4️⃣ Embeddings

    logger.info("Summarizing each chunk and embedding summaries...")

    chunk_summary_documents = []
    file_chunk_summaries = []

    for idx, chunk in enumerate(chunks):
        prompt = CHUNK_SUMMARY_PROMPT.format(content=chunk.page_content)

        response = llm.invoke([HumanMessage(content=prompt)])
        summary_text = response.content.strip()

        # 🔍 DEBUG PRINT
        print(f"\n[DEBUG] Chunk {idx+1} summary:\n{summary_text}\n")

        file_chunk_summaries.append(summary_text)

        chunk_summary_documents.append(
            Document(
                page_content=summary_text,
                metadata={**chunk.metadata,
                          "original_content": chunk.page_content}
            )
        )

    logger.info("Storing chunk summary embeddings...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # 5️⃣ Vector store (append-safe)
    if st.session_state.vectorstore is None:
        st.session_state.vectorstore = FAISS.from_documents(chunk_summary_documents, embeddings)
    else:
        st.session_state.vectorstore.add_documents(chunk_summary_documents)
        logger.info(f"Vectorstore now contains {st.session_state.vectorstore.index.ntotal} vectors")

    logger.info(
        f"Chunk-summary vectorstore size: "
        f"{st.session_state.vectorstore.index.ntotal}")

    # st.session_state.vectorstore.save_local(VECTORSTORE_PATH)
    # logger.info(f"Vector store saved at {VECTORSTORE_PATH}.")


    # 6️⃣ File-level summary
    logger.info("Generating file-level summary...")
    file_prompt = FILE_SUMMARY_PROMPT.format(summaries="\n".join(file_chunk_summaries))

    file_response = llm.invoke([HumanMessage(content=file_prompt)])
    file_summary_text = file_response.content.strip()

    # 🔍 DEBUG PRINT
    print(f"\n[DEBUG] File summary:\n{file_summary_text}\n")

    file_summary_doc = Document(
    page_content=file_summary_text,
    metadata={
        "source": file_path,
        "type": "file_summary"
    })

    logger.info("Storing file summary embedding...")

    if st.session_state.file_vectorstore is None:
        st.session_state.file_vectorstore = FAISS.from_documents(
            [file_summary_doc],
            embeddings
        )
    else:
        st.session_state.file_vectorstore.add_documents(
            [file_summary_doc]
        )

    logger.info(
        f"File vectorstore size: "
        f"{st.session_state.file_vectorstore.index.ntotal}"
    )



    logger.info(f"Ingestion complete. Total chunks: {len(chunks)}")
    return len(chunks)

if __name__ == "__main__":
    ingest_pdf("/home/sg/Desktop/Serag_Ehab_2 pages.pdf")