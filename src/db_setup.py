import os
import time
import logging
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from .parser import parse_statutory_text

load_dotenv()

CHROMA_PATH = os.getenv("VECTOR_DB_PATH", "./chroma_db")
STATUTORY_COLLECTION = "statutory_laws"

def get_embedding_function():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def initialize_statutory_db(file_configs: list[dict]):
    embeddings = get_embedding_function()
    
    vector_store = Chroma(
        collection_name=STATUTORY_COLLECTION,
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH
    )

    all_lc_docs = []
    all_ids = []

    for config in file_configs:
        path = config["path"]
        if not os.path.exists(path):
            logging.warning(f"File not found: {path}")
            continue

        logging.info(f"📄 Processing: {path}...")
        
        if path.lower().endswith(".pdf"):
            loader = PyPDFLoader(path)
            raw_pdf_pages = loader.load()
            raw_text = "\n\n".join([page.page_content for page in raw_pdf_pages])
        else:
            with open(path, "r", encoding="utf-8") as f:
                raw_text = f.read()

        parsed_legal_docs = parse_statutory_text(
            raw_text=raw_text,
            act_name=config["act"],
            year=config["year"],
            source_filename=os.path.basename(path)
        )
        
        for legal_doc in parsed_legal_docs:
            lc_doc = Document(
                page_content=legal_doc.content,
                metadata=legal_doc.to_metadata()
            )
            all_lc_docs.append(lc_doc)
            all_ids.append(legal_doc.stable_id)

    if not all_lc_docs:
        logging.error("No valid document chunks generated.")
        return vector_store

    total_chunks = len(all_lc_docs)
    logging.info(f"⚡ Upserting {total_chunks} sections into ChromaDB [{STATUTORY_COLLECTION}] in batches...")
    
    # Deduplicate IDs in list if any collisions occur
    seen_ids = set()
    unique_ids = []
    for doc_id in all_ids:
        final_id = doc_id
        counter = 1
        while final_id in seen_ids:
            final_id = f"{doc_id}_{counter}"
            counter += 1
        seen_ids.add(final_id)
        unique_ids.append(final_id)

    # Use unique_ids instead of all_ids in batch processing loop
    
    # Batch processing (100 docs per batch to stay safely within API limits)
    BATCH_SIZE = 30
    for i in range(0, total_chunks, BATCH_SIZE):
        batch_docs = all_lc_docs[i : i + BATCH_SIZE]
        batch_ids = unique_ids[i : i + BATCH_SIZE]
        
        # Retry loop for rate limits (429)
        max_retries = 5
        for attempt in range(max_retries):
            try:
                vector_store.add_documents(documents=batch_docs, ids=batch_ids)
                logging.info(f"  └─ Batch {i // BATCH_SIZE + 1} / {(total_chunks + BATCH_SIZE - 1) // BATCH_SIZE} complete ({len(batch_docs)} docs)")
                
                # Pace requests to respect Gemini's 100 requests/min free tier rate limit
                time.sleep(2) 
                break
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    wait_time = (attempt + 1) * 15  # Exponential backoff (15s, 30s, 45s...)
                    logging.warning(f"⚠️ Rate limit hit. Waiting {wait_time}s before retrying batch {i // BATCH_SIZE + 1}...")
                    time.sleep(wait_time)
                else:
                    logging.error(f"❌ Batch {i // BATCH_SIZE + 1} failed: {e}")
                    raise e

    logging.info(f"✅ ChromaDB successfully updated at '{CHROMA_PATH}'!")
    return vector_store