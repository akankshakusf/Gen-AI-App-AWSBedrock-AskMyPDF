import json
import os
import sys
import boto3
import streamlit as st
import numpy as np
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA


# Retrieve AWS credentials from Streamlit secrets
aws_access_key = st.secrets["AWS_ACCESS_KEY_ID"]
aws_secret_key = st.secrets["AWS_SECRET_ACCESS_KEY"]

# Initialize the Bedrock client with credentials
bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key
)

# Initialize embedding from Amazon
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

# Initialize session state for caching responses
if "claude_query_cache" not in st.session_state:
    st.session_state.claude_query_cache = {}
if "llama_query_cache" not in st.session_state:
    st.session_state.llama_query_cache = {}
if "claude_response" not in st.session_state:
    st.session_state.claude_response = ""
if "llama_response" not in st.session_state:
    st.session_state.llama_response = ""

## Data Ingestion
def data_ingestion():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_documents(documents)

## Update the vector embeddings
def update_vector_store():
    docs = data_ingestion()
    vectorstores_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    vectorstores_faiss.save_local("faiss_index")
    st.session_state.faiss_index = vectorstores_faiss  # Update session state to avoid reloading

def get_claude_llm():
    return Bedrock(model_id="ai21.j2-mid-v1", client=bedrock, model_kwargs={"maxTokens": 400})

def get_llama3_llm():
    return Bedrock(model_id="meta.llama3-70b-instruct-v1:0", client=bedrock, model_kwargs={"max_gen_len": 512})

## Prepare prompt so LLM can understand
prompt_template = """
Human: Use the following pieces of context to provide a concise 
answer to the question at the end but do at least summarize with 
250 words with a detailed explanation. If you don't know the answer,
just say that you don't know, don't try to make up an answer.

(context)
{context}
(\context)

Question: {question}
Assistant: """

prompt = PromptTemplate(input_variables=["question", "context"], template=prompt_template)

# Load FAISS only once and store in session state
if "faiss_index" not in st.session_state:
    try:
        st.session_state.faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.session_state.faiss_index = None
        st.error(f"Error loading FAISS index: {str(e)}")

if "claude_llm" not in st.session_state:
    st.session_state.claude_llm = get_claude_llm()

if "llama3_llm" not in st.session_state:
    st.session_state.llama3_llm = get_llama3_llm()

# Get Response from LLM (Avoiding Duplicate Calls for Same Questions)
def get_response_from_llm(llm, vectorstores_faiss, query, model_type):
    cache = st.session_state.claude_query_cache if model_type == "claude" else st.session_state.llama_query_cache
    
    if query in cache:
        return cache[query]
    
    retriever = vectorstores_faiss.as_retriever(search_type="similarity", search_kwargs={"k": 3})  # Top 3 similarities
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, chain_type_kwargs={"prompt": prompt})

    response = qa({"query": query})["result"]
    cache[query] = response  # Cache/store responses from LLM 
    return response

##------------------------- DESIGN STREAMLIT APP -------------------------##
def main():
    st.set_page_config(page_title="AWSBedrock AskMyPDF")

    ## Display the image and title
    col1, col2 = st.columns([1, 8])
    with col1:
        st.image("bedrock.jpg", width=100)  # Adjust the width as needed
    with col2:
        st.title("AskMyPDF with AWS Bedrock")

    # User question dialogue box
    user_question = st.text_input("Ask a Question from the PDF Files")

    with st.sidebar:
        st.title("🧠 Update Or Create Vector Store:")

        if st.button("Vectors Update", use_container_width=True):
            with st.spinner("Processing..."):
                update_vector_store()
                st.success("Vector store updated!")

    vectorstores_faiss = st.session_state.faiss_index
    if not vectorstores_faiss:
        st.warning("No vector store found. Please update the vector store first.")
        return

    if user_question:
        if st.button("Claude Output"):
            with st.spinner("Processing..."):
                st.session_state.claude_response = get_response_from_llm(st.session_state.claude_llm, vectorstores_faiss, user_question, "claude")
        
        if st.button("Llama Output"):
            with st.spinner("Processing..."):
                st.session_state.llama_response = get_response_from_llm(st.session_state.llama3_llm, vectorstores_faiss, user_question, "llama")
        
        if st.session_state.claude_response:
            st.subheader("Claude Output")
            st.write(st.session_state.claude_response)
            st.success("Done")
        
        if st.session_state.llama_response:
            st.subheader("Llama Output")
            st.write(st.session_state.llama_response)
            st.success("Done")

if __name__ == "__main__":
    main()
