import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import uuid

class VectorStore:
    """Handle ChromaDB operations for vector storage and retrieval"""
    
    def __init__(self, persist_directory: str = "./data/chroma_db", collection_name: str = "documents"):
        """
        Initialize ChromaDB vector store
        
        Args:
            persist_directory: Directory to persist the database
            collection_name: Name of the collection to use
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        
        print(f"Vector store initialized. Collection '{collection_name}' ready.")
        print(f"Current documents in collection: {self.collection.count()}")
    
    # def add_documents(self, chunks: List[Dict[str, any]], embeddings: List[List[float]], document_name: str):
    #     """
    #     Add document chunks with embeddings to the vector store
        
    #     Args:
    #         chunks: List of chunk dictionaries with 'content' and 'metadata'
    #         embeddings: List of embedding vectors
    #         document_name: Name of the source document
    #     """
    #     if len(chunks) != len(embeddings):
    #         raise ValueError("Number of chunks must match number of embeddings")
        
    #     # Prepare data for ChromaDB
    #     ids = []
    #     documents = []
    #     metadatas = []
        
    #     for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
    #         # Generate unique ID
    #         chunk_id = f"{document_name}_{idx}_{uuid.uuid4().hex[:8]}"
    #         ids.append(chunk_id)
            
    #         # Extract content
    #         documents.append(chunk["content"])
            
    #         # Prepare metadata
    #         metadata = chunk.get("metadata", {})
    #         metadata["document_name"] = document_name
    #         metadata["chunk_id"] = idx
    #         metadatas.append(metadata)
        
    #     # Add to collection
    #     self.collection.add(
    #         ids=ids,
    #         embeddings=embeddings,
    #         documents=documents,
    #         metadatas=metadatas
    #     )
        
    #     print(f"Added {len(chunks)} chunks from '{document_name}' to vector store")
    #     print(f"Total documents in collection: {self.collection.count()}")
    
    def add_documents(self, chunks: List[Dict[str, any]], embeddings: List[List[float]], document_name: str):
        """
        Add document chunks with embeddings to the vector store
        
        Args:
            chunks: List of chunk dictionaries with 'content' and 'metadata'
            embeddings: List of embedding vectors
            document_name: Name of the source document
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        # ChromaDB has a batch size limit, so we'll process in batches
        batch_size = 100  # Safe batch size for ChromaDB
        total_chunks = len(chunks)
        
        for i in range(0, total_chunks, batch_size):
            batch_end = min(i + batch_size, total_chunks)
            batch_chunks = chunks[i:batch_end]
            batch_embeddings = embeddings[i:batch_end]
            
            # Prepare data for ChromaDB
            ids = []
            documents = []
            metadatas = []
            
            for idx, (chunk, embedding) in enumerate(zip(batch_chunks, batch_embeddings)):
                # Generate unique ID with global index
                global_idx = i + idx
                chunk_id = f"{document_name}_{global_idx}_{uuid.uuid4().hex[:8]}"
                ids.append(chunk_id)
                
                # Extract content
                documents.append(chunk["content"])
                
                # Prepare metadata
                metadata = chunk.get("metadata", {})
                metadata["document_name"] = document_name
                metadata["chunk_id"] = global_idx
                metadatas.append(metadata)
            
            # Add batch to collection
            self.collection.add(
                ids=ids,
                embeddings=batch_embeddings,
                documents=documents,
                metadatas=metadatas
            )
            
            print(f"  → Added batch {i//batch_size + 1}: chunks {i+1}-{batch_end} of {total_chunks}")
        
        print(f"✅ Added {total_chunks} chunks from '{document_name}' to vector store")
        print(f"📊 Total documents in collection: {self.collection.count()}")
        
        
    def search(self, query_embedding: List[float], top_k: int = 3) -> Dict[str, any]:
        """
        Search for similar documents using query embedding
        
        Args:
            query_embedding: Embedding vector of the query
            top_k: Number of top results to return
            
        Returns:
            Dictionary with search results
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Format results
        formatted_results = []
        
        if results and results['documents'] and len(results['documents']) > 0:
            for idx in range(len(results['documents'][0])):
                formatted_results.append({
                    "content": results['documents'][0][idx],
                    "metadata": results['metadatas'][0][idx],
                    "distance": results['distances'][0][idx] if 'distances' in results else None,
                    "id": results['ids'][0][idx]
                })
        
        return {
            "results": formatted_results,
            "count": len(formatted_results)
        }
    
    def delete_document(self, document_name: str):
        """
        Delete all chunks from a specific document
        
        Args:
            document_name: Name of the document to delete
        """
        # Query for all chunks with this document name
        results = self.collection.get(
            where={"document_name": document_name}
        )
        
        if results and results['ids']:
            self.collection.delete(ids=results['ids'])
            print(f"Deleted {len(results['ids'])} chunks from '{document_name}'")
        else:
            print(f"No chunks found for document '{document_name}'")
    
    def get_collection_stats(self) -> Dict[str, any]:
        """Get statistics about the collection"""
        return {
            "total_documents": self.collection.count(),
            "collection_name": self.collection_name
        }
    
    def reset_collection(self):
        """Delete all documents from the collection"""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"Collection '{self.collection_name}' has been reset")