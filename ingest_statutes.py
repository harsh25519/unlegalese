import os
import logging
from dotenv import load_dotenv
from src.parser import parse_statutory_text
from src.db_setup import initialize_statutory_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
load_dotenv()

STATUTORY_CONFIGS = [
    {
        "path": "data/statutes/BNS_2023.pdf",
        "act": "Bharatiya Nyaya Sanhita",
        "year": 2023
    },
    {
        "path": "data/statutes/BNSS_2023.pdf",
        "act": "Bharatiya Nagarik Suraksha Sanhita",
        "year": 2023
    },
    {
        "path": "data/statutes/BSA_2023.pdf",
        "act": "Bharatiya Sakshya Adhiniyam",
        "year": 2023
    },
    {
        "path": "data/statutes/Indian_Contract_Act_1872.pdf",
        "act": "Indian Contract Act",
        "year": 1872
    },
    {
        "path": "data/statutes/Consumer_Protection_Act_2019.pdf",
        "act": "Consumer Protection Act",
        "year": 2019
    }
]

def run_statutory_ingestion():
    """
    Scans data/statutes, parses all official PDFs, and writes 
    structured section embeddings into the ChromaDB statutory collection.
    """
    valid_configs = []
    
    for config in STATUTORY_CONFIGS:
        if os.path.exists(config["path"]):
            valid_configs.append(config)
            logging.info(f" Found statutory document: {config['path']}")
        else:
            logging.warning(f" Missing file: {config['path']} (Drop the PDF into data/statutes/)")

    if not valid_configs:
        logging.error("No statutory PDFs found in data/statutes/. Please place your downloaded files there.")
        return

    logging.info(f"⚡ Starting ingestion for {len(valid_configs)} acts...")
    initialize_statutory_db(valid_configs)
    logging.info(" All statutory reference documents ingested into ChromaDB!")

if __name__ == "__main__":
    run_statutory_ingestion()