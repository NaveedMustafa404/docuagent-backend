from typing import List, Dict
from app.embeddings_service import EmbeddingsService
from app.vector_store import VectorStore
from app.llm_service import LLMService

class RAGPipeline:
    """Complete RAG pipeline: Retrieve + Generate"""
    
    def __init__(
        self,
        embeddings_service: EmbeddingsService,
        vector_store: VectorStore,
        llm_service: LLMService
    ):
        
        self.embeddings_service = embeddings_service
        self.vector_store = vector_store
        self.llm_service = llm_service
        print("✅ RAG Pipeline initialized")
    
    def query(
        self,
        question: str,
        top_k: int = 3,
        max_length: int = 512,
        temperature: float = 0.7
    ) -> Dict[str, any]:
       
        print(f"\n RAG Query: '{question}'")
        
        # Step 1: Generate query embedding
        print("1️⃣ Generating query embedding...")
        query_embedding = self.embeddings_service.embed_text(question)
        
        # Step 2: Retrieve relevant documents
        print(f"2️⃣ Retrieving top {top_k} relevant documents...")
        search_results = self.vector_store.search(query_embedding, top_k=top_k)
        
        if not search_results["results"]:
            return {
                "answer": "I couldn't find any relevant information to answer your question.",
                "sources": [],
                "context_used": ""
            }
        
        # Step 3: Prepare context from retrieved documents
        print("3️⃣ Preparing context...")
        context = self._prepare_context(search_results["results"])
        
        # Step 4: Generate answer using LLM
        print("4️⃣ Generating answer with LLM...")
        answer = self.llm_service.generate_response(
            query=question,
            context=context,
            max_length=max_length,
            temperature=temperature
        )
        
        # Step 5: Format sources
        sources = self._format_sources(search_results["results"])
        
        print("✅ RAG query complete!")
        
        return {
            "answer": answer,
            "sources": sources,
            "context_used": context
        }
    
    def _prepare_context(self, retrieved_docs: List[Dict]) -> str:
        
        context_parts = []
        
        for idx, doc in enumerate(retrieved_docs, 1):
            content = doc["content"]
            page = doc["metadata"].get("page", "Unknown")
            
            context_parts.append(
                f"[Source {idx} - Page {page}]\n{content}\n"
            )
        
        return "\n".join(context_parts)
    
    def _format_sources(self, retrieved_docs: List[Dict]) -> List[Dict]:
        
        sources = []
        
        for doc in retrieved_docs:
            sources.append({
                "content": doc["content"][:200] + "..." if len(doc["content"]) > 200 else doc["content"],
                "page": doc["metadata"].get("page"),
                "document": doc["metadata"].get("document_name"),
                "relevance_score": 1 - doc.get("distance", 0)  # Convert distance to similarity
            })
        
        return sources