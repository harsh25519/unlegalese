# ⚖️ UnLegalese

**Demystifying Complex Legal Documents into Plain English**

UnLegalese is an AI-powered Retrieval-Augmented Generation (RAG) platform engineered to parse, summarize, and explain complex legal notices and contracts under Indian Law. Built with Streamlit, ChromaDB, and Google Gemini, it pairs document clause analysis with relevant statutory provisions—such as the *Bharatiya Nyaya Sanhita (BNS)*, *Consumer Protection Act*, and *Indian Contract Act*—to provide verified, plain-English legal breakdowns.

- [Live API](https://unlegalese.streamlit.app/)

---

## Tech Stack

* **Frontend**: Streamlit
* **Core Languages**: Python
* **Parsing & OCR**: PyMuPDF, tesseract
* **Vector Search & Indexing**: Chroma DB
* **NLP & Reranking**: Sentence Transformers (Embedding & Cross-Encoder)

---

## 🌟 Key Features

* **📄 Hybrid Document Ingestion:** Uses PyMuPDF for digital PDFs and falls back to **Tesseract OCR** for scanned PDFs and image uploads (`.png`, `.jpg`, `.jpeg`).
* **🔍 Dual-Collection MMR Retrieval:** Employs Max Marginal Relevance (MMR) search across two isolated vector databases:
  1. **User Document Store:** Query-scoped vector search filtered strictly by document UUID (`doc_id`).
  2. **Statutory Reference Store:** Pre-populated collection of Indian statutory laws.
* **🔍Hybrid Retrieval Pipeline**: Combines dense semantic search (ChromaticDb) with exact keyword matching (BM25) using Reciprocal Rank Fusion (RRF) for highly accurate document retrieval.
* **🔄 Legal Query Translation (Sub-Query Rewriter):** Automatically translates informal user questions (*"Can I sue them back?"*) into formal Indian legal terms (*"criminal intimidation BNS 351 extortion defamation"*) to ensure precise statutory section retrieval.
* **🛡️ Anti-Hallucination Guardrail:** A two-stage LLM verification chain audits every output against retrieved context before displaying it to the user.
* **💬 Conversational Memory:** Full multi-turn chat interface via `st.chat_message` allowing users to request draft reply notices or follow-up legal strategies based on previous context.
* **🧹 Zero-Leakage Privacy & Storage Management:** Physical upload files are cleaned up from disk immediately after chunking, and vector embeddings are purged upon file switching.

---

## 🏗️ System Architecture

- Unlegalese HLD

![alt text](images/UnLegalese.png)

```text
[ User Upload (PDF/Image) ]
            │
            ▼
[ OCR / Text Parser (PyMuPDF + Tesseract) ]
            │
            ▼
[ Chunking & Metadata Tagging (LangChain) ]
            │
            ├──► [ Immediate Disk Cleanup ]
            │
            ▼
[ Vector Storage (ChromaDB - User Collection) ]
            │
            ├─────────────────────────────────────────┐
            ▼                                         ▼
[ Document MMR Retrieval ]               [ Query Translation ]
            │                                         │
            │                                         ▼
            │                       [ Statutory MMR Retrieval ]
            │                             (BNS / Contract Act)
            └────────────────────┬────────────────────┘
                                 ▼
                     [ Gemini Primary Chain ]
                                 │
                                 ▼
                    [ Anti-Hallucination Audit ]
                                 │
                                 ▼
                      [ Streamlit Chat UI ]

```

 **File Structure 
 ```
Projects\unlegalese
├── data/
├── images/
│   └── UnLegalese.png
├── src/
│   ├── __init__.py
│   ├── db_setup.py
│   ├── loader.py
│   ├── models.py
│   ├── parser.py
│   └── rag_chain.py
├── .env
├── .gitignore
├── app.py
├── dockerfile
├── ingest_statutes.py
├── LICENSE
├── packages.txt
├── README.md
└── requirements.txt

```
