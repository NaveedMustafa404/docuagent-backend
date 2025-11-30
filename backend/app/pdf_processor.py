from pypdf import PdfReader
from typing import List, Dict
import os

class PDFProcessor:
    """Handle PDF text extraction"""
    
    def __init__(self, upload_dir: str = "./data/uploads"):
        self.upload_dir = upload_dir
        # Create upload directory if it doesn't exist
        os.makedirs(upload_dir, exist_ok=True)
    
    def extract_text(self, pdf_path: str) -> List[Dict[str, any]]:
        
        try:
            reader = PdfReader(pdf_path)
            pages_content = []
            
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                
                if text.strip():  # Only add pages with content
                    pages_content.append({
                        "page": page_num,
                        "content": text.strip()
                    })
            
            return pages_content
            
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")
    
    def save_uploaded_file(self, file_content: bytes, filename: str) -> str:
        
        file_path = os.path.join(self.upload_dir, filename)
        
        with open(file_path, "wb") as f:
            f.write(file_content)
        
        return file_path
    
    def get_pdf_info(self, pdf_path: str) -> Dict[str, any]:
        
        try:
            reader = PdfReader(pdf_path)
            
            return {
                "num_pages": len(reader.pages),
                "metadata": reader.metadata if reader.metadata else {},
                "file_size": os.path.getsize(pdf_path)
            }
        except Exception as e:
            raise Exception(f"Error getting PDF info: {str(e)}")