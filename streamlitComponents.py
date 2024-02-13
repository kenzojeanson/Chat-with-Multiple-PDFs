import streamlit as st


def initialUI():
    st.set_page_config(
        page_title="Chatting with Multiple PDF Files",
        page_icon=":star:"
    )

    st.header("Chat with Multiple PDFs")
    st.caption(
        f''' Below are the instructions for this POC: \n
        1. Upload your document(s) as a PDF File
        2. Press 'Upload' 
        3. Ask your question to do with the documents
        '''
    )


# Session Storage
if 'text_chunks' not in st.session_state:
    st.session_state.text_chunks = "NULL"

if 'text_chunks_embeddings' not in st.session_state:
    st.session_state.text_chunks_embeddings = "NULL"
