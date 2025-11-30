from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict
import re


class TextChunker:
    """Handle text chunking for RAG"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize the text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]  # Try to split on paragraphs first
        )
    
    def clean_text(self, text: str) -> str:
        
        # Remove multiple spaces
        text = re.sub(r' +', ' ', text)
        
        # Remove multiple newlines (keep max 2)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Fix common PDF extraction issues
        text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)  # Fix hyphenated words split across lines
        
        return text.strip()
    
    def is_meaningful_chunk(self, text: str) -> bool:
        
        # Skip very short chunks
        if len(text) < 50:
            return False
        
        # Skip chunks that are mostly copyright/legal boilerplate
        copyright_keywords = ['copyright', '©', 'all rights reserved', 'license notice', 
                            'legal responsibility', 'terms and conditions']
        text_lower = text.lower()
        copyright_count = sum(1 for keyword in copyright_keywords if keyword in text_lower)
        
        # If more than 2 copyright keywords, likely boilerplate
        if copyright_count > 2:
            return False
        
        return True
    
    
    def chunk_pages(self, pages_content: List[Dict[str, any]]) -> List[Dict[str, any]]:
        
        all_chunks = []
        global_chunk_idx = 0
        
        for page_data in pages_content:
            page_num = page_data["page"]
            content = page_data["content"]
            
            # Clean the text first
            cleaned_content = self.clean_text(content)
            
            # Skip empty pages
            if not cleaned_content:
                continue
            
            
            # Split the page content into chunks
            chunks = self.text_splitter.split_text(cleaned_content)
            
            # Add metadata to each chunk
            for chunk_idx, chunk_text in enumerate(chunks):
                if self.is_meaningful_chunk(chunk_text):
                    all_chunks.append({
                        "content": chunk_text,
                        "metadata": {
                            "page": page_num,
                            "chunk_index": global_chunk_idx,
                            "page_chunk_index": chunk_idx,
                            "source": "pdf",
                            "char_count": len(chunk_text)
                        }
                    })
                    global_chunk_idx += 1
        
        return all_chunks
    
    def chunk_text(self, text: str, metadata: dict = None) -> List[Dict[str, any]]:
        
        chunks = self.text_splitter.split_text(text)
        
        result = []
        for idx, chunk in enumerate(chunks):
            chunk_data = {
                "content": chunk,
                "metadata": {
                    "chunk_index": idx,
                    **(metadata or {})
                }
            }
            result.append(chunk_data)
        
        return result
    
    def get_chunk_stats(self, chunks: List[Dict[str, any]]) -> Dict[str, any]:
        
        print(("Hello from get_chunk_stats"))
        if not chunks:
            return {
                "total_chunks": 0,
                "avg_chunk_size": 0,
                "min_chunk_size": 0,
                "max_chunk_size": 0
            }
        
        chunk_sizes = [len(chunk["content"]) for chunk in chunks]
        
        return {
            "total_chunks": len(chunks),
            "avg_chunk_size": sum(chunk_sizes) / len(chunk_sizes),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes)
        }