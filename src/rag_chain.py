import os
import logging
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Suppress Hugging Face verbosity & unauthenticated warnings in terminal
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"

load_dotenv()

CHROMA_PATH = os.getenv("VECTOR_DB_PATH", "./chroma_db")
STATUTORY_COLLECTION = "statutory_laws"
USER_DOCS_COLLECTION = "uploaded_user_docs"

def get_embedding_function():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        max_retries=3,
        temperature=0.2
    )

def generate_statutory_search_query(user_query: str, chat_history: list = None) -> str:
    """
    Translates informal user questions or chat follow-ups into formal 
    Indian legal concepts (BNS, BNSS, Contract Act terms) for ChromaDB vector search.
    """
    history_context = ""
    if chat_history:
        history_context = "\n".join([f"{m['role']}: {m['content']}" for m in chat_history[-2:]])

    rewrite_prompt = f"""
    Given the user's question and recent chat history, convert the user's query into 
    3-5 formal legal concepts under Indian Law (e.g. BNS, Contract Act, CPC).
    Focus on specific legal terms such as: criminal intimidation, extortion, defamation, malicious prosecution, breach of contract, abuse of legal process.

    CHAT HISTORY:
    {history_context}

    USER QUESTION: {user_query}

    OUTPUT ONLY 3-5 LEGAL SEARCH CONCEPTS SEPARATED BY SPACES (NO OTHER TEXT):
    """
    
    llm = get_llm()
    try:
        legal_search_query = llm.invoke(rewrite_prompt).content.strip()
        logging.info(f"🔄 Rewritten Statutory Search Terms: '{legal_search_query}'")
        return legal_search_query
    except Exception as e:
        logging.warning(f"Query rewriter failed, falling back to raw user query: {e}")
        return user_query

def store_user_doc_embeddings(chunks: list, doc_id: str):
    """Stores user document chunks in ChromaDB under the USER_DOCS_COLLECTION."""
    if not chunks:
        logging.error(f"❌ Cannot store empty chunks for doc_id: {doc_id}")
        return

    logging.info(f"💾 Persisting embeddings for doc_id {doc_id} to ChromaDB...")
    embeddings = get_embedding_function()
    
    user_db = Chroma(
        collection_name=USER_DOCS_COLLECTION,
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH
    )
    
    # Generate unique chunk IDs (e.g., DOC123_chunk_0, DOC123_chunk_1)
    chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
    
    # Add non-empty documents safely
    user_db.add_documents(documents=chunks, ids=chunk_ids)
    logging.info(f"✅ Stored {len(chunks)} chunks under doc_id: {doc_id}")

def query_unlegalese(doc_id: str, user_query: str, chat_history: list = None):
    """
    Dual Retrieval RAG Pipeline with Chat Memory and Query Translation:
    1. MMR Search on user document collection filtered by doc_id (UUID).
    2. Query Translation -> MMR Search on statutory law collection (BNS, Contract Act).
    3. Primary synthesis via Gemini.
    4. Anti-hallucination verification pass.
    """
    logging.info(f"🔎 Querying dual collections via MMR for doc_id: {doc_id}")
    embeddings = get_embedding_function()

    # 1. RETRIEVE CLAUSES VIA MMR (User Document)
    user_db = Chroma(
        collection_name=USER_DOCS_COLLECTION,
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH
    )
    user_docs = user_db.max_marginal_relevance_search(
        user_query, k=4, fetch_k=20, lambda_mult=0.5, filter={"doc_id": doc_id}
    )
    user_doc_context = "\n\n".join([f"[Clause/Chunk {d.metadata.get('chunk_index', '?')}]: {d.page_content}" for d in user_docs])

    # 2. RETRIEVE STATUTES VIA MMR (Using Formal Legal Query Rewriter)
    statutory_search_terms = generate_statutory_search_query(user_query, chat_history)

    statutory_db = Chroma(
        collection_name=STATUTORY_COLLECTION,
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH
    )
    statute_docs = statutory_db.max_marginal_relevance_search(
        statutory_search_terms, 
        k=4, 
        fetch_k=20, 
        lambda_mult=0.5
    )
    statute_context = "\n\n".join([f"[{d.metadata.get('act', 'Statute')} Sec {d.metadata.get('section', '')}]: {d.page_content}" for d in statute_docs])

    # 3. CONVERT CHAT HISTORY TO STRING CONTEXT
    formatted_history = ""
    if chat_history:
        formatted_history = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in chat_history[-4:]])  # Last 2 turns

    # 4. SYSTEM PROMPT WITH CHAT HISTORY
    # 4. SYSTEM PROMPT WITH DOCUMENT TYPE CHECK
    system_prompt = """
    You are UnLegalese, an expert AI legal assistant specializing in Indian Law.
    Analyze the provided document clauses, statutory references, and ongoing conversation history to answer the user's request.

    --- CONVERSATION HISTORY ---
    {chat_history}

    --- RETRIEVED USER DOCUMENT CLAUSES ---
    {user_doc_context}

    --- RETRIEVED STATUTORY REFERENCE ---
    {statute_context}

    --- INSTRUCTIONS ---
    1. **DOCUMENT TYPE CHECK:** First, determine if the uploaded document is a legal document (e.g., notice, contract, court order, act, summons, lease, terms of service).
       - If the document is NON-LEGAL (e.g., a college syllabus, curriculum, technical report, resume, or recipe), explicitly state that the uploaded document is not a legal document. Summarize what it actually is in 2-3 sentences and state that legal/statutory analysis does not apply.
       - DO NOT invent or force-fit legal claims, statutory penalties, or legal risks onto non-legal text.
    2. If it IS a legal document:
       - Cite specific statutory sections (e.g., BNS, Contract Act) retrieved from the statutory reference context when explaining legal remedies or cross-cases.
       - Quote or reference specific document clauses when applicable.
       - If the user asks to draft a response or legal reply based on prior turns, use the conversation history context directly.
    3. Maintain an objective, calm, and professional tone.
    """

    primary_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])

    llm = get_llm()
    primary_chain = primary_prompt | llm | StrOutputParser()

    draft_response = primary_chain.invoke({
        "chat_history": formatted_history,
        "statute_context": statute_context,
        "user_doc_context": user_doc_context,
        "question": user_query
    })

    # 5. GUARDRAIL VERIFICATION
    guardrail_prompt = """
    You are a strict Legal Compliance & Anti-Hallucination Auditor.
    Review the Draft AI Response against the Source Statutory Context, User Document Context, and Conversation History.

    --- CONVERSATION HISTORY ---
    {chat_history}

    --- SOURCE USER DOCUMENT CONTEXT ---
    {user_doc_context}

    --- SOURCE STATUTORY LAWS ---
    {statute_context}

    --- DRAFT AI RESPONSE TO AUDIT ---
    {draft_response}

    --- INSTRUCTIONS ---
    - Ensure non-legal documents (e.g., college curricula, syllabi, software docs) are NOT assigned fake statutory violations or legal risk clauses.
    - For actual legal notices/contracts, verify that cited sections match the statutory reference context.
    - If the draft correctly flags the document as non-legal or accurately summarizes a legal document, return it as-is.
    - Return ONLY the finalized, verified text response.
    """

    guardrail_chain = ChatPromptTemplate.from_template(guardrail_prompt) | llm | StrOutputParser()
    verified_response = guardrail_chain.invoke({
        "chat_history": formatted_history,
        "user_doc_context": user_doc_context,
        "statute_context": statute_context,
        "draft_response": draft_response
    })

    return verified_response, statute_docs, user_docs


def delete_user_doc_embeddings(doc_id: str):
    """Deletes all chunks associated with a specific doc_id from ChromaDB."""
    if not doc_id:
        return
    try:
        embeddings = get_embedding_function()
        user_db = Chroma(
            collection_name=USER_DOCS_COLLECTION,
            embedding_function=embeddings,
            persist_directory=CHROMA_PATH
        )
        # Delete documents matching the doc_id filter
        user_db.delete(where={"doc_id": doc_id})
        logging.info(f"🧹 Successfully purged ChromaDB embeddings for doc_id: {doc_id}")
    except Exception as e:
        logging.error(f"Failed to delete ChromaDB embeddings for doc_id {doc_id}: {e}")