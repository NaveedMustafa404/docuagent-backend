
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import os

from app.pdf_processor import PDFProcessor
from app.text_chunker import TextChunker
from app.embeddings_service import EmbeddingsService
from app.vector_store import VectorStore
from app.llm_service import LLMService
from app.rag_pipeline import RAGPipeline


app = FastAPI(
    title="RAG Chatbot API",
    description="A simple RAG-based chatbot API",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize PDF processor
print("Initializing services...")
pdf_processor = PDFProcessor(upload_dir="./data/uploads")
text_chunker = TextChunker(chunk_size=800, chunk_overlap=150)  # Slightly smaller for better sentence breaks
embeddings_service = EmbeddingsService(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = VectorStore(persist_directory="./data/chroma_db", collection_name="documents")

# # Initialize LLM (this will take time!)
# print("\n  Loading LLM - This may take 3-5 minutes on first run...")
# llm_service = LLMService(model_name="mistralai/Mistral-7B-Instruct-v0.1")

print("\n  Loading LLM - This may take 2-3 minutes on first run...")
# Using TinyLlama - smaller, faster, more reliable
llm_service = LLMService(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Initialize RAG pipeline
rag_pipeline = RAGPipeline(
    embeddings_service=embeddings_service,
    vector_store=vector_store,
    llm_service=llm_service
)

print("\n✅ All services initialized and ready!")
print("🚀 RAG Chatbot is ready to answer questions!\n")


# Response models
class UploadResponse(BaseModel):
    filename: str
    status: str
    num_pages: int
    num_chunks: int
    message: str

class PageContent(BaseModel):
    page: int
    content: str
    
class ChunkData(BaseModel):
    content: str
    metadata: Dict

class SearchRequest(BaseModel):
    query: str
    top_k: int = 3
    
class SearchResult(BaseModel):
    content: str
    metadata: Dict
    distance: Optional[float] = None
    
class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    count: int
    
class ChatRequest(BaseModel):
    question: str
    top_k: int = 3
    max_length: int = 512
    temperature: float = 0.7

class SourceInfo(BaseModel):
    content: str
    page: Optional[int] = None
    document: Optional[str] = None
    relevance_score: float

class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceInfo]

# Simple health check endpoint
@app.get("/")
async def root():
    return {
        "message": "RAG Chatbot API is running!",
        "status": "healthy"
    }

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file, extract text, chunk it, generate embeddings, and store in vector DB
    """
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        print(f"\n📄 Processing: {file.filename}")
        
        # Read file content
        content = await file.read()
        
        # Save file
        file_path = pdf_processor.save_uploaded_file(content, file.filename)
        print(f"✅ File saved")
        
        # Get PDF info
        pdf_info = pdf_processor.get_pdf_info(file_path)
        
        # Extract text
        pages_content = pdf_processor.extract_text(file_path)
        print(f"✅ Extracted text from {len(pages_content)} pages")
        
        # Chunk the text
        chunks = text_chunker.chunk_pages(pages_content)
        # Get chunk statistics
        stats = text_chunker.get_chunk_stats(chunks)
        print(f"✅ Created {stats['total_chunks']} chunks")
        
        # Generate embeddings
        print(f"🧠 Generating embeddings...")
        chunk_texts = [chunk["content"] for chunk in chunks]
        embeddings = embeddings_service.embed_batch(chunk_texts)
        print(f"✅ Generated {len(embeddings)} embeddings")
        
        # Store in vector database
        print(f"💾 Storing in vector database...")
        vector_store.add_documents(chunks, embeddings, file.filename)
        print(f"✅ Stored in ChromaDB")
        
        return UploadResponse(
            filename=file.filename,
            status="success",
            num_pages=pdf_info["num_pages"],
            num_chunks=stats["total_chunks"],
            message=f"Successfully processed {len(pages_content)} pages into {stats['total_chunks']} chunks and stored in vector DB"
        )
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/document/{filename}", response_model=List[PageContent])
async def get_document_content(filename: str):
    """
    Get extracted text content from an uploaded PDF
    """
    file_path = os.path.join(pdf_processor.upload_dir, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Document not found")
    
    try:
        pages_content = pdf_processor.extract_text(file_path)
        return pages_content
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def list_documents():
    """
    List all uploaded documents
    """
    try:
        files = os.listdir(pdf_processor.upload_dir)
        pdf_files = [f for f in files if f.endswith('.pdf')]
        return {"documents": pdf_files, "count": len(pdf_files)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/document/{filename}/chunks", response_model=List[ChunkData])
async def get_document_chunks(filename: str):
    """
    Get text chunks from an uploaded PDF
    """
    file_path = os.path.join(pdf_processor.upload_dir, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Document not found")
    
    try:
        # Extract text from PDF
        pages_content = pdf_processor.extract_text(file_path)
        
        # Chunk the text
        chunks = text_chunker.chunk_pages(pages_content)
        
        return chunks
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    
    try:
        print(f"\n🔍 Searching for: '{request.query}'")
        
        # Generate embedding for the query
        query_embedding = embeddings_service.embed_text(request.query)
        print(f"✅ Query embedding generated")
        
        # Search in vector store
        search_results = vector_store.search(query_embedding, top_k=request.top_k)
        print(f"✅ Found {search_results['count']} results")
        
        # Format response
        results = [
            SearchResult(
                content=result["content"],
                metadata=result["metadata"],
                distance=result.get("distance")
            )
            for result in search_results["results"]
        ]
        
        return SearchResponse(
            query=request.query,
            results=results,
            count=len(results)
        )
        
    except Exception as e:
        print(f"❌ Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/vector-store/stats")
async def get_vector_store_stats():
   
    try:
        stats = vector_store.get_collection_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/vector-store/reset")
async def reset_vector_store():
    
    try:
        vector_store.reset_collection()
        return {"status": "success", "message": "Vector store has been reset"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    
    try:
        # Check if vector store has documents
        stats = vector_store.get_collection_stats()
        if stats["total_documents"] == 0:
            raise HTTPException(
                status_code=400, 
                detail="No documents uploaded yet. Please upload a PDF first."
            )
        
        print(f"\n💬 Chat request received: '{request.question}'")
        
        # Use RAG pipeline to get answer
        result = rag_pipeline.query(
            question=request.question,
            top_k=request.top_k,
            max_length=request.max_length,
            temperature=request.temperature
        )
        
        # Format response
        sources = [
            SourceInfo(
                content=source["content"],
                page=source.get("page"),
                document=source.get("document"),
                relevance_score=source.get("relevance_score", 0.0)
            )
            for source in result["sources"]
        ]
        
        return ChatResponse(
            question=request.question,
            answer=result["answer"],
            sources=sources
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))