#import packages 
import json 
import os 
import sys
import boto3
import streamlit as st

## import Data Ingestion packages 
import numpy as np
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

##  import Data Transformation (embedding and Vectorstoredb) packages
from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.vectorstores import FAISS

## import Data Modelling packages
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

## Bedrock Clients call
bedrock=boto3.client(service_name="bedrock-runtime")

#initialize the embedding
bedrock_embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)


## Data Ingestion
def data_ingestion():
    #initialize the load
    loader=PyPDFDirectoryLoader("data")
    documents=loader.load()

    #initialize splitter
    splitter=RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=1000)
    docs=splitter.split_documents(documents)
    return docs

## vector embeddings
def get_vector_stores(docs):
    vectorstores_faiss=FAISS.from_documents(docs,bedrock_embeddings)
    vectorstores_faiss.save_local("faiss_index")

def get_claude_llm():
    ## import the AI21 Labs : Jurassic-2 Mid model
    llm=Bedrock(model_id="ai21.j2-mid-v1", client=bedrock,
                model_kwargs={"maxTokens": 400})
    return llm

def get_llama3_llm():
    ## import the Meta Llama 3 model
    llm=Bedrock(model_id = "meta.llama3-70b-instruct-v1:0", client=bedrock,
                model_kwargs={"max_gen_len": 512})
    return llm


## prepare prompt so llm can understand
prompt_template="""
Human: Use the following pieces of context to provide a concise 
answer to the question at the end but do atleast summarize with 
250 words with detailed explaination. If you don't know the answer,
just say that you don't know, don't try to make up an answer.

(context)
{context}
(\context)

Question: {question}
Assistant: """

prompt=PromptTemplate(
    input_variables=["question","context"],
    template=prompt_template
)


def get_reponse_llm(llm,vectorstores_faiss,query):
    qa=RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstores_faiss.as_retriever(search_type="similarity",search_kwargs={"k":3}),
        chain_type_kwargs={"prompt":prompt}      
)
    answer=qa({"query":query})
    return answer['result']
    
##------------------------- DESIGN STREAMLIT APP -------------------------##

def main():
    st.set_page_config("AWSBedrock AskMyPDF")
    #st.header("Chat with Pdf using AWS Bedrock ")  

    # Add a blank line to move the content upward
    st.write("<br>", unsafe_allow_html=True)  # Adjust the number of <br> tags as needed

    # Display the image and title
    col1, col2 = st.columns([1, 8])

    with col1:
        st.image("bedrock.jpg", width=100)  # Adjust the width as needed

    with col2:
        st.title("AskMyPDF with AWSBedrock")

    user_question=st.text_input("Ask a Question from the PDF Files")

    with st.sidebar:
        st.title(" 🛒🧠 Update Or Create Vector Store:")

        if st.button("Vectors Update",use_container_width=True):
            with st.spinner("Processing..."):
                docs=data_ingestion()
                get_vector_stores(docs)
                st.success("Done")
    
    if st.button("Claude Output"):
        with st.spinner("Processing..."):
            faiss_index=FAISS.load_local("faiss_index",bedrock_embeddings,allow_dangerous_deserialization=True)
            llm=get_claude_llm()

            st.write(get_reponse_llm(llm,faiss_index,user_question))
            st.success("Done")


    if st.button("Llama Output"):
        with st.spinner("Processing..."):
            faiss_index=FAISS.load_local("faiss_index",bedrock_embeddings,allow_dangerous_deserialization=True)
            llm=get_llama3_llm()

            st.write(get_reponse_llm(llm,faiss_index,user_question))
            st.success("Done")

if __name__=="__main__":
    main()


        
    



    
    






