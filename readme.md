🤖 RAG Chatbot - Intelligent Document Q&A System

A production-ready Retrieval-Augmented Generation (RAG) chatbot that enables intelligent question-answering from PDF documents using state-of-the-art AI models.

⚠️ Note: This is currently the backend API only. Frontend development is planned for future phases.

📋 Table of Contents

Overview
Key Features
Tech Stack
RAG Architecture
Project Structure
Installation
Usage
API Endpoints
How It Works
Future Enhancements


🎯 Overview
This RAG (Retrieval-Augmented Generation) chatbot allows users to upload PDF documents and ask natural language questions about their content. The system intelligently retrieves relevant information from the documents and generates accurate, context-aware answers using a large language model.
What makes this special:

🔍 Semantic Search: Finds information based on meaning, not just keywords
🧠 AI-Powered Answers: Generates human-like responses using LLM
📄 Source Citations: Shows exactly where information comes from (page numbers)
💾 Persistent Storage: Documents and embeddings stored permanently
⚡ Fast & Scalable: Optimized for production use


✨ Key Features
Core Functionality

✅ PDF Document Upload & Processing
✅ Intelligent Text Chunking with overlap for context preservation
✅ Semantic Embeddings using sentence transformers
✅ Vector Database Storage with ChromaDB
✅ Semantic Search for relevant content retrieval
✅ LLM-based Answer Generation with source attribution
✅ RESTful API with interactive documentation

Technical Features

✅ Batch processing for large documents
✅ Automatic text cleaning and preprocessing
✅ Persistent vector storage (no re-indexing required)
✅ Configurable retrieval parameters (top-k, temperature, etc.)
✅ Error handling and logging throughout
✅ Memory-optimized for CPU inference


🛠️ Tech Stack
Backend Framework

FastAPI - Modern, high-performance web framework
Uvicorn - ASGI server for production deployment
Pydantic - Data validation using Python type annotations

AI/ML Stack

LangChain - Framework for LLM application development
Sentence Transformers - State-of-the-art text embeddings

Model: all-MiniLM-L6-v2 (384-dimensional embeddings)


Hugging Face Transformers - LLM infrastructure

Model: TinyLlama-1.1B-Chat-v1.0 (efficient chat model)


PyTorch - Deep learning framework

Vector Database

ChromaDB - Open-source embedding database with persistent storage

Document Processing

pypdf - PDF text extraction
RecursiveCharacterTextSplitter - Intelligent text chunking


🏗️ RAG Architecture
High-Level Flow
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERACTION                         │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DOCUMENT INGESTION PHASE                      │
├─────────────────────────────────────────────────────────────────┤
│  1. PDF Upload                                                   │
│  2. Text Extraction        → pypdf                               │
│  3. Text Chunking          → LangChain (800 chars, 150 overlap) │
│  4. Generate Embeddings    → Sentence Transformers              │
│  5. Store in Vector DB     → ChromaDB                            │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     QUERY PROCESSING PHASE                       │
├─────────────────────────────────────────────────────────────────┤
│  1. User Question                                                │
│  2. Generate Query Embedding                                     │
│  3. Similarity Search (Cosine)                                   │
│  4. Retrieve Top-K Chunks                                        │
│  5. Build Context from Retrieved Chunks                          │
│  6. Create Prompt: Context + Question                            │
│  7. LLM Generation                                               │
│  8. Return Answer + Sources                                      │
└─────────────────────────────────────────────────────────────────┘
Detailed RAG Pipeline
python# Step-by-step RAG workflow:

1. INDEXING (One-time per document):
   PDF → Text Extraction → Chunking → Embeddings → Vector DB

2. RETRIEVAL (Per query):
   Question → Query Embedding → Similarity Search → Top-K Chunks

3. AUGMENTATION:
   Retrieved Chunks + Question → Structured Prompt

4. GENERATION:
   Prompt → LLM → Answer + Source Citations
Why RAG?
Traditional LLMs have limitations:

❌ Limited context window
❌ No access to private/recent documents
❌ Can hallucinate information

RAG solves this by:

✅ Grounding answers in actual document content
✅ Providing source citations
✅ Scaling to unlimited documents
✅ Always using up-to-date information


📁 Project Structure
rag-chatbot/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                 # FastAPI application & endpoints
│   │   ├── pdf_processor.py        # PDF text extraction logic
│   │   ├── text_chunker.py         # Text splitting & cleaning
│   │   ├── embeddings_service.py   # Embedding generation
│   │   ├── vector_store.py         # ChromaDB operations
│   │   ├── llm_service.py          # LLM inference & prompt handling
│   │   └── rag_pipeline.py         # Complete RAG workflow
│   ├── data/
│   │   ├── uploads/                # Uploaded PDF storage
│   │   └── chroma_db/              # Vector database persistence
│   ├── requirements.txt            # Python dependencies
│   ├── .env                        # Environment variables
│   └── preload_documents.py        # Batch PDF processing script
├── .gitignore
└── README.md

🚀 Installation
Prerequisites

Python 3.10 or higher
8GB+ RAM (16GB recommended for optimal performance)
5GB free disk space (for models and data)

Setup Instructions

Clone the repository

bashgit clone <repository-url>
cd rag-chatbot

Create virtual environment

bashcd backend
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

Install dependencies

bashpip install -r requirements.txt

Create data directories

bashmkdir -p data/uploads data/chroma_db

Start the server

bashuvicorn app.main:app --reload --host 0.0.0.0 --port 8000
