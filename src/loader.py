import os
import io
import logging
import fitz  # PyMuPDF
import pytesseract
from PIL import Image

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts digital text page-by-page or falls back to OCR if scanned."""
    doc = fitz.open(pdf_path)
    full_text = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text").strip()

        if len(text) > 20:
            full_text.append(text)
        else:
            try:
                pix = page.get_pixmap(dpi=300)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                ocr_text = pytesseract.image_to_string(img, lang="eng")
                full_text.append(ocr_text)
            except Exception as e:
                logging.warning(f"OCR skipped for page {page_num + 1}: {e}")

    doc.close()
    return "\n\n".join(full_text)

def process_and_chunk_user_doc(file_path: str, doc_id: str):
    """
    Parses an uploaded file once, tags every chunk with doc_id (UUID),
    and returns chunked LangChain documents for persistent DB storage.
    """
    logging.info(f"📥 Extracting text for Document UUID: {doc_id} ({file_path})")
    
    ext = file_path.lower().split('.')[-1]
    raw_text = ""

    if ext in ["png", "jpg", "jpeg"]:
        try:
            img = Image.open(file_path)
            raw_text = pytesseract.image_to_string(img, lang="eng").strip()
        except Exception as e:
            logging.warning(f"Image OCR failed: {e}")
    elif ext == "pdf":
        raw_text = extract_text_from_pdf(file_path).strip()

    # Safety Guard: Fallback text if OCR or text extraction failed
    if not raw_text:
        logging.warning("⚠️ Text extraction returned empty content. Inserting fallback placeholder.")
        raw_text = f"Document uploaded from file {os.path.basename(file_path)}. Could not automatically extract text."

    doc = Document(
        page_content=raw_text,
        metadata={"source": os.path.basename(file_path), "doc_id": doc_id, "doc_type": "user_doc"}
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", " "]
    )
    chunks = splitter.split_documents([doc])

    # Ensure chunks list is never empty
    if not chunks:
        chunks = [doc]

    # Tag individual chunks with sequence indices and doc_id
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = idx
        chunk.metadata["doc_id"] = doc_id

    logging.info(f"✅ Generated {len(chunks)} chunks for UUID: {doc_id}")
    
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            logging.info(f"🗑️ Removed temporary disk file: {file_path}")
        except Exception as e:
            logging.warning(f"Failed to delete temp file {file_path}: {e}")
            
    return chunks