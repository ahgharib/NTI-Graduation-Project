from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from langchain_core.documents import Document
from pdf2image import convert_from_path
import torch
import tempfile
import re


def clean_text(text):
    text = re.sub(r'\b[A-Z]{2,}\b', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_text_from_pdf(pdf_path: str):
    """
    Returns: List[Document] — one Document per page with metadata
    """

    pages = convert_from_path(pdf_path, dpi=300)

    with tempfile.TemporaryDirectory() as tmpdir:
        image_paths = []
        for i, page in enumerate(pages):
            path = f"{tmpdir}/page_{i}.png"
            page.save(path)
            image_paths.append(path)

        doc = DocumentFile.from_images(image_paths)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ocr_predictor(
        det_arch="db_resnet50",
        reco_arch="sar_resnet31",
        pretrained=True,
        assume_straight_pages=False,   # better for Arabic
        export_as_straight_boxes=False
    ).to(device)

    result = model(doc)

    documents = []

    for page_idx, page in enumerate(result.pages):
        page_text = " ".join(
            word.value
            for block in page.blocks
            for line in block.lines
            for word in line.words
        )

        page_text = clean_text(page_text)

        if page_text:
            documents.append(
                Document(
                    page_content=page_text,
                    metadata={
                        "source": pdf_path,
                        "page": page_idx + 1,
                        "loader": "doctr_ocr"
                    }
                )
            )

    return documents
