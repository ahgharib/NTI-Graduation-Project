import os
from pydoc import doc
from scipy import io
# import streamlit as st
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pdf2image import convert_from_path
import numpy as np
import torch
import re
import tempfile

def extract_text_from_pdf(pdf_path):
    
    # poppler_path = r"C:\poppler-25.12.0\Library\bin" ## needed for Windows users only
    pages = convert_from_path(pdf_path, dpi=300)

    with tempfile.TemporaryDirectory() as tmpdir:
        image_paths = []
        for i, page in enumerate(pages):
           path = f"{tmpdir}/page_{i}.png"
           page.save(path)
           image_paths.append(path)

        doc = DocumentFile.from_images(image_paths)
    # هنا DocTR يشتغل بدون مشاكل


    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ocr_predictor(
        det_arch="db_resnet50",
        reco_arch="sar_resnet31",
        pretrained=True
    ).to(device)

    result = model(doc)

    # Extract text
    text = " ".join(
    word.value
    for page in result.pages
    for block in page.blocks
    for line in block.lines
    for word in line.words
)


    return clean_text(text)

def clean_text(text):
    text = re.sub(r'\b[A-Z]{2,}\b', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()



# -----------------------------
# Summarizer
# -----------------------------
def load_summarizer():
    model_name = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model


def chunk_text(text, tokenizer, max_tokens=1000):
    words = text.split()
    chunks, current = [], []
    current_tokens = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for word in words:
        tokens = len(tokenizer.tokenize(word))
        if current_tokens + tokens > max_tokens:
            chunks.append(" ".join(current))
            current, current_tokens = [], 0
        current.append(word)
        current_tokens += tokens

    if current:
        chunks.append(" ".join(current))
    return chunks

PROMPT_INSTRUCTION = (
    "Summarize the following text completely as bullet points. Be clear, concise, and avoid opinions"
    "Do NOT include instructions or task descriptions in the output."
    "The text: "
)

def summarize_text(text, tokenizer, model):
    prompt = PROMPT_INSTRUCTION + text

    inputs = tokenizer(prompt, truncation=True, max_length=1024, return_tensors="pt")

    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=300,
        min_length=150,
        num_beams=6,
        no_repeat_ngram_size=3,
        length_penalty=1.0,
        early_stopping=True
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


# =============================
# Streamlit UI
# =============================
# st.set_page_config(page_title="PDF Summarizer", layout="wide")

# st.title("📄 PDF OCR & Summarization App")
# st.write("Upload PDF → DocTR OCR → Clean → BART Summary")
# uploaded_file = st.file_uploader("Upload PDF file", type=["pdf"])

# if uploaded_file:
#     with open("temp.pdf", "wb") as f:
#         f.write(uploaded_file.read())

#     if st.button("🔍 Extract & Summarize"):
#         with st.spinner("Running OCR..."):
#             raw_text = extract_text_from_pdf("temp.pdf")
#             clean_text = clean_text(raw_text)

#         st.success("OCR completed!")

#         st.subheader("📜 Extracted Text (Preview)")
#         st.text_area("", clean_text[:5000], height=300)

#         with st.spinner("Summarizing..."):
#             tokenizer, model = load_summarizer()
#             chunks = chunk_text(clean_text, tokenizer)


#             summaries = [
#                 summarize_text(chunk, tokenizer, model)
#                 for chunk in chunks
#             ]

#             final_summary = " ".join(summaries)

#         st.subheader("📝 Summary")
#         st.write(final_summary)

#         st.download_button(
#             "⬇️ Download Summary",
#             final_summary,
#             file_name="summary.txt"
#         )