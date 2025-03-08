# AskMyPDF with AWS Bedrock

## Overview
AskMyPDF is a powerful document-based Q&A system built using **AWS Bedrock**, **LangChain**, **FAISS**, and **Streamlit**. It allows users to query a set of PDF documents and receive intelligent responses from **Claude** and **Llama 3** models, leveraging Amazon Bedrock for inference.

The project implements **Retrieval-Augmented Generation (RAG)** to enhance the accuracy of responses by grounding LLM outputs in retrieved document knowledge.

## Features
- **RAG-based Question Answering**: Uses FAISS vector retrieval to enhance LLM responses.
- **Multi-Model Support**: Claude and Llama 3 models via AWS Bedrock.
- **Efficient Caching**: Avoids redundant API calls with session-based caching.
- **Dynamic PDF Processing**: Ingest and vectorize PDF content for efficient retrieval.
- **Scalable & Extensible**: Businesses can add their own logic and customizations.
- 
## How It Works
### Data Ingestion & Vectorization
- PDF documents are loaded using `PyPDFDirectoryLoader`.
- Documents are chunked and embedded using Amazon Titan Embeddings (`amazon.titan-embed-text-v1`).
- FAISS stores embeddings locally for efficient retrieval.

### Query Processing
- User queries trigger a similarity search against the FAISS index.
- The retrieved context is passed to the LLM using a structured prompt.
- The response is generated and cached to avoid redundant API calls.

## Business Use Cases & Customization
### **How Businesses Can Integrate Their Logic**
1. **Domain-Specific Tuning**: Replace `amazon.titan-embed-text-v1` with a custom embedding model trained on domain-specific data.
2. **Enterprise Data Sources**: Extend document ingestion to include data from SQL, NoSQL, or cloud storage (e.g., S3, SharePoint, or Google Drive).
3. **Hybrid Search**: Combine FAISS with keyword-based search for better context retrieval.
4. **Model Selection**: Choose from Amazon Bedrock-supported models based on response quality and cost-effectiveness.
5. **Custom Response Formatting**: Modify prompt templates for structured responses like summaries, reports, or bullet points.

### **Leveraging RAG for Business Applications**
Retrieval-Augmented Generation (RAG) is key for enhancing LLMs with real-world, up-to-date knowledge. Businesses can:
- **Improve chatbot responses**: Build customer support bots grounded in company documentation.
- **Boost compliance & legal research**: Extract precise regulatory insights from legal texts.
- **Enhance enterprise search**: Empower employees to query internal knowledge bases.
- **Optimize financial reporting**: Summarize quarterly reports using dynamic data retrieval.

## Future Enhancements
- Add support for **multi-turn conversational memory**.
- Integrate **real-time API data retrieval** for live updates.
- Implement **fine-tuning** options for domain-specific LLMs.
- Enable **user authentication & access control** for enterprise use.

