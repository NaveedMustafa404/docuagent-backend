# 🤖 RAG Chatbot - Intelligent Document Q&A System

> A production-ready Retrieval-Augmented Generation (RAG) chatbot that enables intelligent question-answering from PDF documents using state-of-the-art AI models.

**⚠️ Note**: This is currently the **backend API only**. Frontend development is planned for future phases.
<img width="1912" height="1038" alt="image" src="https://github.com/user-attachments/assets/c9f49900-4fcd-4838-b65c-3d67e95fcdc3" />
<img width="1921" height="1041" alt="image" src="https://github.com/user-attachments/assets/5ee8bd35-bbf7-455a-8388-b8436448c8a6" />
<img width="1917" height="1038" alt="image" src="https://github.com/user-attachments/assets/1704ebc2-14be-4288-8402-0df8987b2bfa" />
<img width="1918" height="1044" alt="image" src="https://github.com/user-attachments/assets/eadb19da-c9e1-4438-8586-dadc23cf04f8" />
<img width="1918" height="1041" alt="image" src="https://github.com/user-attachments/assets/a54cbc5c-e79d-437b-98ca-5f910a2c38fe" />

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [RAG Architecture](#rag-architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [How It Works](#how-it-works)
- [Future Enhancements](#future-enhancements)

---

## 🎯 Overview

This RAG (Retrieval-Augmented Generation) chatbot allows users to upload PDF documents and ask natural language questions about their content. The system intelligently retrieves relevant information from the documents and generates accurate, context-aware answers using a large language model.

**What makes this special:**
- 🔍 **Semantic Search**: Finds information based on meaning, not just keywords
- 🧠 **AI-Powered Answers**: Generates human-like responses using LLM
- 📄 **Source Citations**: Shows exactly where information comes from (page numbers)
- 💾 **Persistent Storage**: Documents and embeddings stored permanently
- ⚡ **Fast & Scalable**: Optimized for production use

---

## ✨ Key Features

### Core Functionality
- ✅ **PDF Document Upload & Processing**
- ✅ **Intelligent Text Chunking** with overlap for context preservation
- ✅ **Semantic Embeddings** using sentence transformers
- ✅ **Vector Database Storage** with ChromaDB
- ✅ **Semantic Search** for relevant content retrieval
- ✅ **LLM-based Answer Generation** with source attribution
- ✅ **RESTful API** with interactive documentation

### Technical Features
- ✅ Batch processing for large documents
- ✅ Automatic text cleaning and preprocessing
- ✅ Persistent vector storage (no re-indexing required)
- ✅ Configurable retrieval parameters (top-k, temperature, etc.)
- ✅ Error handling and logging throughout
- ✅ Memory-optimized for CPU inference

---

## 🛠️ Tech Stack

### Backend Framework
- **FastAPI** - Modern, high-performance web framework
- **Uvicorn** - ASGI server for production deployment
- **Pydantic** - Data validation using Python type annotations

### AI/ML Stack
- **LangChain** - Framework for LLM application development
- **Sentence Transformers** - State-of-the-art text embeddings
  - Model: `all-MiniLM-L6-v2` (384-dimensional embeddings)
- **Hugging Face Transformers** - LLM infrastructure
  - Model: `TinyLlama-1.1B-Chat-v1.0` (efficient chat model)
- **PyTorch** - Deep learning framework

### Vector Database
- **ChromaDB** - Open-source embedding database with persistent storage

### Document Processing
- **pypdf** - PDF text extraction
- **RecursiveCharacterTextSplitter** - Intelligent text chunking

---

## 🏗️ RAG Architecture

### High-Level Flow
```
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
```

### Detailed RAG Pipeline
```python
# Step-by-step RAG workflow:

1. INDEXING (One-time per document):
   PDF → Text Extraction → Chunking → Embeddings → Vector DB

2. RETRIEVAL (Per query):
   Question → Query Embedding → Similarity Search → Top-K Chunks

3. AUGMENTATION:
   Retrieved Chunks + Question → Structured Prompt

4. GENERATION:
   Prompt → LLM → Answer + Source Citations
```

### Why RAG?

Traditional LLMs have limitations:
- ❌ Limited context window
- ❌ No access to private/recent documents
- ❌ Can hallucinate information

**RAG solves this by:**
- ✅ Grounding answers in actual document content
- ✅ Providing source citations
- ✅ Scaling to unlimited documents
- ✅ Always using up-to-date information

---

## 📁 Project Structure
```
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
```

---

## 🚀 Installation

### Prerequisites
- Python 3.10 or higher
- 8GB+ RAM (16GB recommended for optimal performance)
- 5GB free disk space (for models and data)

### Setup Instructions

1. **Clone the repository**
```bash
git clone <repository-url>
cd rag-chatbot
```

2. **Create virtual environment**
```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Create data directories**
```bash
# Windows PowerShell
New-Item -ItemType Directory -Path data/uploads, data/chroma_db -Force

# Mac/Linux
mkdir -p data/uploads data/chroma_db
```

5. **Start the server**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

6. **Access the API**
- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/

---

## 💻 Usage

### 1. Upload a PDF Document
```bash
curl -X POST "http://localhost:8000/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/document.pdf"
```

**Response:**
```json
{
  "filename": "document.pdf",
  "status": "success",
  "num_pages": 50,
  "num_chunks": 125,
  "message": "Successfully processed 50 pages into 125 chunks and stored in vector DB"
}
```

### 2. Ask Questions (RAG Chat)
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the main topics discussed in this document?",
    "top_k": 3,
    "max_length": 300,
    "temperature": 0.7
  }'
```

**Response:**
```json
{
  "question": "What are the main topics discussed in this document?",
  "answer": "Based on the provided context, the document discusses three main topics: machine learning fundamentals, neural network architectures, and practical applications in computer vision. The content covers both theoretical concepts and implementation details.",
  "sources": [
    {
      "content": "Machine learning has revolutionized...",
      "page": 5,
      "document": "document.pdf",
      "relevance_score": 0.89
    }
  ]
}
```

### 3. Semantic Search (Retrieval Only)
```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "neural networks",
    "top_k": 5
  }'
```

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/health` | GET | Service status |
| `/upload` | POST | Upload and process PDF |
| `/documents` | GET | List all uploaded documents |
| `/document/{filename}/chunks` | GET | View document chunks |
| `/search` | POST | Semantic search (retrieve only) |
| `/chat` | POST | **RAG chat** (retrieve + generate) |
| `/vector-store/stats` | GET | Vector database statistics |
| `/vector-store/reset` | DELETE | Clear all documents |

**Interactive API Documentation**: http://localhost:8000/docs

---

## 🔍 How It Works

### Document Processing Pipeline

1. **PDF Upload**
   - User uploads PDF via API
   - File saved to `data/uploads/`

2. **Text Extraction**
   - PyPDF extracts text from each page
   - Preserves page numbers for citation

3. **Intelligent Chunking**
   - Text split into ~800 character chunks
   - 150 character overlap to maintain context
   - Splits on paragraph/sentence boundaries
   - Filters out boilerplate content

4. **Embedding Generation**
   - Each chunk converted to 384-dim vector
   - Batch processing for efficiency
   - Uses `all-MiniLM-L6-v2` model

5. **Vector Storage**
   - Embeddings stored in ChromaDB
   - Metadata preserved (page number, document name)
   - Persistent storage (survives restarts)

### Query Processing Pipeline

1. **Question Embedding**
   - User question → 384-dim vector
   - Same model as document embeddings

2. **Semantic Search**
   - Cosine similarity search in ChromaDB
   - Retrieves top-k most relevant chunks
   - Returns with relevance scores

3. **Context Assembly**
   - Retrieved chunks formatted with metadata
   - Combined into structured context

4. **Prompt Engineering**
   - Context + Question → Structured prompt
   - Model-specific formatting (TinyLlama/Mistral)

5. **LLM Generation**
   - TinyLlama generates contextual answer
   - Temperature controls creativity
   - Max length controls response size

6. **Response Formatting**
   - Answer extracted from LLM output
   - Sources attached with page numbers
   - Relevance scores included

---

## 🎛️ Configuration

### Adjustable Parameters

**Chunking:**
- `chunk_size`: 800 (characters per chunk)
- `chunk_overlap`: 150 (overlap between chunks)

**Retrieval:**
- `top_k`: 3 (number of chunks to retrieve)

**Generation:**
- `max_length`: 512 (max tokens in response)
- `temperature`: 0.7 (0.0 = deterministic, 1.0 = creative)

**Models:**
- Embedding: `sentence-transformers/all-MiniLM-L6-v2`
- LLM: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

---

## 🙏 Acknowledgments

- **Hugging Face** - For open-source models
- **LangChain** - For RAG framework
- **ChromaDB** - For vector database
- **FastAPI** - For excellent API framework

---

**⭐ If you find this project interesting, please star the repository!**
