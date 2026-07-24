import os
import io
import logging
import fitz  # PyMuPDF
import pytesseract
from PIL import Image, ImageEnhance, ImageOps

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Windows Tesseract Executable Fallback
if os.name == 'nt':
    win_tesseract = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    if os.path.exists(win_tesseract):
        pytesseract.pytesseract.tesseract_cmd = win_tesseract

def run_enhanced_ocr(pil_img: Image.Image) -> str:
    """
    Robust OCR Engine for Screenshots, Dark Mode UI, and Image PDF Pages.
    - Strips alpha channels (RGBA -> RGB).
    - Enhances contrast and converts to Grayscale.
    - Retries with inverted colors if initial pass fails (handles Dark Mode text).
    """
    try:
        # Convert RGBA / Palette mode images to standard RGB
        if pil_img.mode in ("RGBA", "P"):
            pil_img = pil_img.convert("RGB")

        # Convert to Grayscale
        gray_img = ImageOps.grayscale(pil_img)

        # Enhance Contrast for low-DPI screenshots
        enhancer = ImageEnhance.Contrast(gray_img)
        enhanced_img = enhancer.enhance(2.0)

        # Pass 1: Standard Grayscale OCR
        text = pytesseract.image_to_string(enhanced_img, lang="eng", config="--psm 6").strip()

        # Pass 2: Dark Mode Inversion Fallback (if Pass 1 extracted under 10 chars)
        if not text or len(text) < 10:
            inverted_img = ImageOps.invert(gray_img)
            text = pytesseract.image_to_string(inverted_img, lang="eng", config="--psm 6").strip()

        return text
    except Exception as e:
        logging.warning(f"Enhanced OCR failed: {e}")
        return ""

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts digital text page-by-page. If text is missing/sparse 
    (e.g., scanned PDF or embedded screenshot PDF), renders the page as a high-DPI image and runs OCR.
    """
    doc = fitz.open(pdf_path)
    full_text = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text").strip()

        # If page contains healthy digital text, use it directly
        if len(text) > 30:
            full_text.append(text)
        else:
            # Otherwise render page to 300 DPI image and run OCR
            try:
                pix = page.get_pixmap(dpi=300)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                ocr_text = run_enhanced_ocr(img)
                if ocr_text:
                    full_text.append(ocr_text)
            except Exception as e:
                logging.warning(f"OCR skipped for PDF page {page_num + 1}: {e}")

    doc.close()
    return "\n\n".join(full_text)

def process_and_chunk_user_doc(file_path: str, doc_id: str):
    """
    Parses an uploaded file (PDF/PNG/JPG), tags every chunk with doc_id (UUID),
    returns chunked LangChain documents for persistent DB storage,
    and removes the physical staging file from disk.
    """
    logging.info(f"📥 Extracting text for Document UUID: {doc_id} ({file_path})")
    
    file_name = os.path.basename(file_path)
    ext = file_path.lower().split('.')[-1]
    raw_text = ""

    if ext in ["png", "jpg", "jpeg"]:
        try:
            img = Image.open(file_path)
            raw_text = run_enhanced_ocr(img)
        except Exception as e:
            logging.warning(f"Image load/OCR failed: {e}")
    elif ext == "pdf":
        raw_text = extract_text_from_pdf(file_path).strip()

    # Safety Guard: Fallback text if OCR or text extraction failed completely
    if not raw_text:
        logging.warning("⚠️ Text extraction returned empty content. Inserting fallback placeholder.")
        raw_text = f"Document uploaded from file {file_name}. Could not automatically extract text."

    doc = Document(
        page_content=raw_text,
        metadata={"source": file_name, "doc_id": doc_id, "doc_type": "user_doc"}
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents([doc])

    # Ensure chunks list is never empty
    if not chunks:
        chunks = [doc]

    # Tag individual chunks with sequence indices and doc_id
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = idx
        chunk.metadata["doc_id"] = doc_id
        chunk.metadata["source"] = file_name

    logging.info(f"✅ Generated {len(chunks)} chunks for UUID: {doc_id}")
    
    # Clean up temporary staging PDF/Image file immediately
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            logging.info(f"🗑️ Removed temporary disk file: {file_path}")
        except Exception as e:
            logging.warning(f"Failed to delete temp file {file_path}: {e}")
            
    return chunks