# 🌍 AskMyPDF with AWS Bedrock 🚀

App link- https://gen-ai-app-awsbedrock-askmypdf-gmgzapiekszkkdo5bn7vva.streamlit.app/

## 📌 Overview
**AskMyPDF** is an AI-powered document-based Q&A system leveraging **Amazon Bedrock**, **LangChain**, **FAISS**, and **Streamlit**. It enables users to extract insights from PDF documents using **Claude** and **Llama 3** models with Amazon Bedrock inference.

> 🔥 **Why is this important?** This project implements **Retrieval-Augmented Generation (RAG)** to enhance LLM responses by grounding answers in **retrieved document knowledge** for higher accuracy.

## 🌟 Features
✅ **RAG-based Question Answering** - Enhances LLM responses with document knowledge.  
✅ **Multi-Model Support** - Query Claude and Llama 3 seamlessly.  
✅ **Efficient Caching** - Reduces redundant API calls, improving performance.  
✅ **Dynamic PDF Processing** - Automatically ingest and vectorize PDFs for retrieval.  
✅ **Scalable & Extensible** - Businesses can integrate their own logic & domain-specific models.

## 🛠️ How It Works
### 📥 Data Ingestion & Vectorization
- 📂 PDF documents are loaded using `PyPDFDirectoryLoader`.
- 🔄 Documents are chunked & embedded using **Amazon Titan Embeddings (`amazon.titan-embed-text-v1`)**.
- 📊 FAISS stores embeddings for fast and efficient similarity-based retrieval.

### 🔍 Query Processing
1️⃣ User enters a query.  
2️⃣ FAISS searches for the most relevant document snippets.  
3️⃣ Retrieved context is sent to **Claude/Llama 3** via Amazon Bedrock.  
4️⃣ Response is generated & displayed to the user.  

---

## 🏢 Business Use Cases & Customization
### ✨ **How Businesses Can Integrate Their Logic**
🔹 **Domain-Specific Tuning**: Use custom embeddings trained on company data.  
🔹 **Enterprise Data Sources**: Extend PDF ingestion to include SQL, NoSQL, or S3 storage.  
🔹 **Hybrid Search**: Combine FAISS retrieval with keyword search for better accuracy.  
🔹 **Model Selection**: Pick an optimal Amazon Bedrock model based on cost & performance.  
🔹 **Custom Response Formatting**: Modify prompts for structured outputs (e.g., reports, summaries).  

### 💡 **Leveraging RAG for Business Applications**
📌 **Enhancing Chatbot Responses**: Power AI-driven customer support bots.  
📌 **Legal & Compliance Research**: Extract key insights from legal documents.  
📌 **Enterprise Knowledge Retrieval**: Enable employees to query internal documentation.  
📌 **Automated Financial Reporting**: Summarize financial reports using dynamic RAG retrieval.  

---

## 🔮 Future Enhancements
✨ Add **multi-turn conversational memory** for better context retention.  
✨ Integrate **real-time API data retrieval** for continuous updates.  
✨ Implement **fine-tuning options** for domain-specific LLMs.  
✨ Enable **user authentication & role-based access** for enterprises.  
