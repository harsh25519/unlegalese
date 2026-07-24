import os
import uuid
import streamlit as st
from src.loader import process_and_chunk_user_doc
from src.rag_chain import store_user_doc_embeddings, query_unlegalese, delete_user_doc_embeddings

st.set_page_config(page_title="UnLegalese - Legal Notice Decoder", page_icon="⚖️", layout="wide")

st.title("⚖️ UnLegalese")
st.subheader("Demystifying Complex Legal Documents into Plain English")

# Initialize Chat & Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_doc_id" not in st.session_state:
    st.session_state.active_doc_id = None
if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None

# Sidebar Upload
with st.sidebar:
    st.header("📄 Upload Document")
    uploaded_file = st.file_uploader(
        "Upload Notice or Contract (PDF/Image)", 
        type=["pdf", "png", "jpg", "jpeg"]
    )

    # File Ingestion & Purge Logic
    if uploaded_file and st.session_state.uploaded_filename != uploaded_file.name:
        # Purge previous document vectors if switching files
        if st.session_state.active_doc_id:
            delete_user_doc_embeddings(st.session_state.active_doc_id)
            
        temp_dir = "data/uploaded"
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, uploaded_file.name)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        doc_id = str(uuid.uuid4())
        
        with st.spinner("Indexing document vectors..."):
            chunks = process_and_chunk_user_doc(file_path, doc_id)
            store_user_doc_embeddings(chunks, doc_id)
            
            st.session_state.active_doc_id = doc_id
            st.session_state.uploaded_filename = uploaded_file.name
            st.session_state.messages = []  # Clear chat history for new doc

        st.success(f"Indexed: **{uploaded_file.name}**")

        # Automatically trigger initial summary prompt
        initial_prompt = "Summarize this legal document, highlight potential risks/clauses, and suggest practical next steps."
        analysis, statute_refs, user_clause_refs = query_unlegalese(
            doc_id=doc_id, 
            user_query=initial_prompt,
            chat_history=[]
        )
        st.session_state.messages.append({"role": "assistant", "content": analysis})

# Display Existing Chat Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input Box
if prompt := st.chat_input("Ask a question or request a draft reply based on this notice..."):
    if not st.session_state.active_doc_id:
        st.error("Please upload a legal document in the sidebar first!")
    else:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate Assistant Response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing document & conversation history..."):
                analysis, statute_refs, user_clause_refs = query_unlegalese(
                    doc_id=st.session_state.active_doc_id, 
                    user_query=prompt,
                    chat_history=st.session_state.messages[:-1]  # Pass previous context
                )
                
                st.markdown(analysis)

                with st.expander("📚 View Retrieved Context & Statutes"):
                    st.write("**Matched Clauses:**", [f"Chunk #{c.metadata.get('chunk_index', '?')}: {c.page_content[:150]}..." for c in user_clause_refs])
                    st.write("**Matched Statutes:**", [f"[{d.metadata.get('act', 'Statute')} Sec {d.metadata.get('section', '')}]" for d in statute_refs])

        # Save assistant message to history
        st.session_state.messages.append({"role": "assistant", "content": analysis})