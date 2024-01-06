from datetime import datetime
import openai
from torch import cosine_similarity
import env
API_KEY = env.API_KEY
openai.api_key = API_KEY

# ---------- Get PDF Text ---------- #
import PyPDF2
from PyPDF2 import PdfReader
def get_PDF_text(pdf_docs):
    text = ""
    
    for pdf in pdf_docs:
        pages = PdfReader(pdf)
        for page in pages.pages:
            text += page.extract_text()
        return text
    
# ---------- Chunk Text ----------#
from langchain.text_splitter import CharacterTextSplitter
def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
    separator = '\n',
    chunk_size = 3000,
    chunk_overlap = 500,
    length_function = len
    )
    
    chunks = text_splitter.split_text(raw_text)
    return chunks
        
# ---------- Text Chunks Embeddings ---------- #
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
def get_text_chunk_embeddings(text_chunks):  
    print('\n','\n','----------', datetime.now().strftime("%d/%m/%Y %H:%M:%S"), '----------', '\n','\n' )  
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
    import numpy as np
    from sentence_transformers import SentenceTransformer
    
    print('- Prompt Embeddings -')           
               
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')        
    embedding = model.encode(prompt)
    embedding_np = np.array(embedding)
    
    print(embedding_np.shape, '\n')
       
    return embedding_np

# ---------- Get Most Similar Chunk ---------- #
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity
def get_most_similar_chunk(prompt_embeddings, text_chunk_embeddings):
    
    print('- Get Most Similar Chunk -')
    
    similarity_list = []
    
    for text_chunk_embedding in text_chunk_embeddings:
        A = np.array([prompt_embeddings])
        B = np.array([text_chunk_embedding])
        
        print("A.Shape:", A.shape)
        print("B.Shape:", B.shape)
        
        similarity = cosine_similarity(A,B)
        similarity_list.append(similarity)
        
    print(similarity_list)  
    most_similar_chunk_embedding = max(similarity_list)
    
    most_similar_chunk_index = 0    
    index = 0
    for similarity in similarity_list:
        if most_similar_chunk_embedding == similarity:
            most_similar_chunk_index = index
        index += 1
    
    print('\n')
    return most_similar_chunk_index

# ---------- Get Answer ---------- #
def get_answer(question, text_chunk):
    template = f'''
       
    [Text Document]: {text_chunk} \n
    [Question]: {question} \n
    
    [Template] \n
    Question:\n    
    Answer: \n
    \n
    [Instructions] \n
    1. You are a professional in the related field. \n
    2. You have been given a text document. \n
    3. To the best of your ability, you will answer the question only using the information provided in the text document. \n
    4. Be sure to structure your answer. Sound as professional as possible. \n
    5. In your response, please follow the template mentioned above. So, be sure to repeat the question. \n
    
    Let's think step by step.
    '''
    
    response = openai.Completion.create(
    model="text-davinci-003",
    prompt = template,
    temperature = 0,
    max_tokens = 1000,
)  
   
    answer = ""
    for result in response.choices: #type: ignore
        answer = result.text
       
    return answer

# ---------- Streamlit ---------- #
import streamlit as st

st.set_page_config(
    page_title = "Chatting with Multiple PDF Files",
    page_icon = ":star:"
)

st.header("Chat with Multiple PDFs")  
st.caption(
    f''' Below are the instructions for this POC: \n
    1. Upload your documents 
    2. Press 'Upload' 
    3. Ask your question to do with the documents
    '''
    )

with st.sidebar:
    st.subheader("Your Current Documents:")
    pdf_docs = st.file_uploader("Upload PDFs Here", accept_multiple_files = True)
    
    if st.button("Upload") == True:
        with st.spinner():
            print('\n','\n','START:', datetime.now().strftime("%d/%m/%Y %H:%M:%S"), '\n','\n' )  
            raw_text = get_PDF_text(pdf_docs)
                            
            st.session_state.text_chunks = get_text_chunks(raw_text)
                   
            st.session_state.text_chunks_embeddings = get_text_chunk_embeddings(st.session_state.text_chunks) # Returns a NP array with all the embeddings
                           
            st.write("Upload Complete")  
  
if prompt := st.chat_input("Ask a question on the documents"):
      
    prompt_embeddings = get_prompt_embeddings(prompt) # Returns a NP array of the prompt embedding (only one)
    
    most_similar_chunk_index = get_most_similar_chunk(prompt_embeddings, st.session_state.text_chunks_embeddings) # Returns index of the most similar chunk
    
    most_similar_chunk_text = st.session_state.text_chunks[most_similar_chunk_index]   
        
    answer = get_answer(prompt, most_similar_chunk_text) 
    st.write(answer)
    
        
if 'text_chunks' not in st.session_state:
    st.session_state.text_chunks = "NULL"
                
if 'text_chunks_embeddings' not in st.session_state:
    st.session_state.text_chunks_embeddings = "NULL"
    