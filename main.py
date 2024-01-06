from datetime import datetime
from email import message
from pkgutil import resolve_name
from random import choices
import openai
from openai import OpenAI
from torch import cosine_similarity
import env
API_KEY = env.API_KEY
client = OpenAI(api_key = API_KEY) #API Key

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
    chunk_size = 5000,
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
    [Document]: \n
    {text_chunk} \n
    
    [Question]: \n
    {question} \n \n    
    
    [Instructions]: \n
    Step 1: Pick one of the following Templates depending on the [Question], Do not add this as part of your response. But follow the template: \n
        1. If the question is relevant to the text documents, use [Template #1] \n
        2. If the question is not relevant to the text documents, use [Template #2] \n
        3. If the question is not in the form of a question, or you do not know how to reply, use [Template #3] \n \n
                       
    [Template #1]: \n
        First; Repeat back the user question \n
        Next; Answer the question confidently, accurately, and with detail \n
        
        Example: \n    
        User Question: "What is the mass of the sun?" \n
        Structure your output like this: \n
        
        Question: What is the mass of the sun? \n  
        Answer: The mass of the sun is 3.955 × 10^30 kg. \n \n

    [Template #2]: \n
        First; Repeat back the user question \n
        Next; Answer with: "Please ensure that your message is a question relevant to the information provided in the document." \n
        
        Example: \n    
        User Question: "Irrelevant question" \n
        Structure your output like this: \n
        
        Question: "Irrelevant Question" \n  
        Answer: "Please ensure that your message is a question relevant to the information provided in the document." \n \n

    [Template #3]: \n
        First; Repeat back the user question \n
        Next; Answer with: "Please ensure your message in the form of a question relevant to the information provided in the document." \n
        
        Example: \n    
        User Question: "Not a question" \n
        Structure your output like this: \n
        
        Question: "Not a Question" \n  
        Answer: "Please ensure your message in the form of a question relevant to the information provided in the document." \n \n
        
        
    Step 2: In your response, ensure the following: \n
        1. The name of the template is not in the response. \n
        2. Only use what is specifically provided in the templates. Only use one template at a time. \n \n
        
       
    Let's think step by step    
    '''
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": template},        
    ],
    temperature = 0,
    max_tokens = 1000,
    )     
   
    answer = response.choices[0].message.content
       
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
  
with st.spinner():
    if prompt := st.chat_input("Ask a question on the documents"):
        
        print(prompt)
        
        prompt_embeddings = get_prompt_embeddings(prompt) # Returns a NP array of the prompt embedding (only one)
        
        most_similar_chunk_index = get_most_similar_chunk(prompt_embeddings, st.session_state.text_chunks_embeddings) # Returns index of the most similar chunk
        
        most_similar_chunk_text = st.session_state.text_chunks[most_similar_chunk_index]   
            
        answer = get_answer(prompt, most_similar_chunk_text) 
        st.write(most_similar_chunk_text)
        st.write(answer)
    
        
if 'text_chunks' not in st.session_state:
    st.session_state.text_chunks = "NULL"
                
if 'text_chunks_embeddings' not in st.session_state:
    st.session_state.text_chunks_embeddings = "NULL"
    