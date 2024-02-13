
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from sentence_transformers import SentenceTransformer
import numpy as np
from langchain.text_splitter import CharacterTextSplitter
from prompts import *
from PyPDF2 import PdfReader
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------- Get PDF Text ---------- #


def get_PDF_text(pdf_docs):
    text = ""

    for pdf in pdf_docs:
        pages = PdfReader(pdf)
        for page in pages.pages:
            text += page.extract_text()
        return text


# ---------- Chunk Text ----------#
def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=5000,
        chunk_overlap=500,
        length_function=len
    )

    chunks = text_splitter.split_text(raw_text)
    return chunks


# ---------- Text Chunks Embeddings ---------- #
def get_text_chunk_embeddings(text_chunks):
    print('\n', '\n', '----------',
          datetime.now().strftime("%d/%m/%Y %H:%M:%S"), '----------', '\n', '\n')
    print('- Text Chunk Embeddings -')

    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings_list = []

    for chunk in text_chunks:
        embedding = model.encode(chunk)
        embedding_np = np.array(embedding)
        embeddings_list.append(embedding_np)

    embeddings_list_np = np.array(embeddings_list)

    print("Shape of Embeddings List:", embeddings_list_np.shape, '\n')

    return embeddings_list_np


# ---------- Prompt Embeddings ---------- #
def get_prompt_embeddings(prompt):

    print('- Prompt Embeddings -')

    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embedding = model.encode(prompt)
    embedding_np = np.array(embedding)

    print(embedding_np.shape, '\n')

    return embedding_np


# ---------- Get Most Similar Chunk ---------- #
def get_most_similar_chunk(prompt_embeddings, text_chunk_embeddings, text_chunks):

    print('- Get Most Similar Chunk -')

    similarity_list = []

    for text_chunk_embedding in text_chunk_embeddings:
        A = np.array([prompt_embeddings])
        B = np.array([text_chunk_embedding])

        print("A.Shape:", A.shape)
        print("B.Shape:", B.shape)

        similarity = cosine_similarity(A, B)
        similarity_list.append(similarity)

    print(similarity_list)
    most_similar_chunk_embedding = max(similarity_list)

    most_similar_chunk_index = 0
    index = 0
    for similarity in similarity_list:
        if most_similar_chunk_embedding == similarity:
            most_similar_chunk_index = index
        index += 1

    most_similar_chunk_text = text_chunks[most_similar_chunk_index]

    print('\n', most_similar_chunk_text)
    print('\n')
    return most_similar_chunk_index, most_similar_chunk_text
