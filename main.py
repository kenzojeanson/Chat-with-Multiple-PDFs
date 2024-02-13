from functions import *
from prompts import *
import streamlit as st
from streamlitComponents import *

initialUI()

with st.sidebar:
    st.subheader("Your Current Documents:")
    pdf_docs = st.file_uploader("Upload PDFs Here", accept_multiple_files=True)

    if st.button("Upload") == True:
        with st.spinner():
            print('\n', '\n', 'START:', datetime.now().strftime(
                "%d/%m/%Y %H:%M:%S"), '\n', '\n')
            raw_text = get_PDF_text(pdf_docs)

            st.session_state.text_chunks = get_text_chunks(raw_text)

            st.session_state.text_chunks_embeddings = get_text_chunk_embeddings(
                st.session_state.text_chunks)  # Returns a NP array with all the embeddings

            st.write("Upload Complete")

with st.spinner():
    if prompt := st.chat_input("Ask a question on the documents"):
        # Returns a NP array of the prompt embedding (only one)
        prompt_embeddings = get_prompt_embeddings(prompt)
        most_similar_chunk_index, most_similar_chunk_text = get_most_similar_chunk(
            prompt_embeddings, st.session_state.text_chunks_embeddings, st.session_state.text_chunks)  # Returns index of the most similar chunk + respective text chunk

        answer = get_answer(prompt, most_similar_chunk_text)
        st.write(answer)