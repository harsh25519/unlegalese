import re
import logging
from typing import List
from .models import LegalDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def parse_statutory_text(raw_text: str, act_name: str, year: int, source_filename: str) -> List[LegalDocument]:
    """
    Parses statutory law text using regex to extract full Sections, 
    and sub-splits exceptionally long paragraphs while preserving metadata.
    """
    documents = []
    
    section_pattern = re.compile(
        r'((?:Section|Article|Clause|\b\d{1,3}\.)\s+.*?)(?=\n\s*(?:Section|Article|Clause|\b\d{1,3}\.)|\Z)', 
        re.DOTALL | re.IGNORECASE
    )
    
    matches = section_pattern.findall(raw_text)
    
    if not matches:
        logging.warning(f"No specific section headers matched in {source_filename}. Treating whole text as general.")
        matches = [raw_text]

    sub_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " "]
    )

    for match in matches:
        content = match.strip()
        sec_num_match = re.search(r'Section\s+(\d+[A-Z]?)', content, re.IGNORECASE)
        sec_number = sec_num_match.group(1) if sec_num_match else "General"
        
        if len(content) <= 1200:
            doc = LegalDocument(
                act=act_name,
                year=year,
                section=sec_number,
                chapter="General",
                content=content,
                source=source_filename,
                doc_type="statute"
            )
            documents.append(doc)
        else:
            sub_chunks = sub_splitter.split_text(content)
            for idx, chunk in enumerate(sub_chunks):
                doc = LegalDocument(
                    act=act_name,
                    year=year,
                    section=f"{sec_number}_p{idx+1}",
                    chapter="General",
                    content=chunk,
                    source=source_filename,
                    doc_type="statute"
                )
                documents.append(doc)
        
    logging.info(f"Successfully extracted {len(documents)} structured section/paragraph chunks from {source_filename}.")
    return documents