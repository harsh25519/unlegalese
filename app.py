import os
import sys
import urllib.parse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import uuid
import streamlit as st
from src.loader import process_and_chunk_user_doc
from src.rag_chain import store_user_doc_embeddings, query_unlegalese, delete_user_doc_embeddings
from src.rag_chain import purge_all_user_docs

st.set_page_config(page_title="UnLegalese - Legal Notice Decoder", page_icon="⚖️", layout="wide")

if "initialized" not in st.session_state:
    purge_all_user_docs()
    st.session_state.initialized = True

st.title("⚖️ UnLegalese")
st.subheader("Demystifying Complex Legal Documents into Plain English")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_doc_id" not in st.session_state:
    st.session_state.active_doc_id = None
if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None

with st.sidebar:
    st.header("📄 Upload Document")
    uploaded_file = st.file_uploader(
        "Upload Notice or Contract (PDF/Image)", 
        type=["pdf", "png", "jpg", "jpeg"]
    )

    if uploaded_file and st.session_state.uploaded_filename != uploaded_file.name:
        clean_filename = urllib.parse.unquote(uploaded_file.name)

        if st.session_state.active_doc_id:
            delete_user_doc_embeddings(st.session_state.active_doc_id)
            
        temp_dir = "data/uploaded"
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, clean_filename)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        doc_id = str(uuid.uuid4())
        
        with st.spinner("Indexing document vectors..."):
            chunks = process_and_chunk_user_doc(file_path, doc_id)
            
            if chunks and len(chunks) > 0:
                store_user_doc_embeddings(chunks, doc_id)
                st.session_state.active_doc_id = doc_id
                st.session_state.uploaded_filename = uploaded_file.name
                st.session_state.messages = []

                st.success(f"Indexed: **{clean_filename}**")

                initial_prompt = "Summarize this legal document, highlight potential risks/clauses, and suggest practical next steps."
                analysis, statute_refs, user_clause_refs = query_unlegalese(
                    doc_id=doc_id, 
                    user_query=initial_prompt,
                    chat_history=[]
                )
                st.session_state.messages.append({"role": "assistant", "content": analysis})
                st.rerun()
            else:
                st.error("Failed to extract readable text from the uploaded file.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question or request a draft reply based on this notice..."):
    if not st.session_state.active_doc_id:
        st.error("Please upload a legal document in the sidebar first!")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing document & conversation history..."):
                analysis, statute_refs, user_clause_refs = query_unlegalese(
                    doc_id=st.session_state.active_doc_id, 
                    user_query=prompt,
                    chat_history=st.session_state.messages[:-1]
                )
                
                st.markdown(analysis)

                with st.expander("📚 View Retrieved Context & Statutes"):
                    st.write("**Matched Clauses:**", [f"Chunk #{c.metadata.get('chunk_index', '?')}: {c.page_content[:150]}..." for c in user_clause_refs])
                    st.write("**Matched Statutes:**", [f"[{d.metadata.get('act', 'Statute')} Sec {d.metadata.get('section', '')}]" for d in statute_refs])

        st.session_state.messages.append({"role": "assistant", "content": analysis})