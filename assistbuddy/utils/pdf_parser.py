"""
PDF Parser for text extraction
Supports text-based and scanned PDFs
"""

import pdfplumber
from PyPDF2 import PdfReader
from typing import List, Dict, Tuple, Optional
import io
from PIL import Image
import numpy as np


class PDFParser:
    """
    Parse PDF documents
    Extract text, tables, and images
    """
    
    def __init__(self, use_ocr: bool = True):
        """
        Args:
            use_ocr: Whether to use OCR for scanned PDFs
        """
        self.use_ocr = use_ocr
        
        if use_ocr:
            from .ocr_engine import OCREngine
            self.ocr = OCREngine(backend='tesseract', languages=['en', 'hi'])
    
    def extract_text(
        self,
        pdf_path: str,
        pages: Optional[List[int]] = None
    ) -> Dict[int, str]:
        """
        Extract text from PDF
        
        Args:
            pdf_path: Path to PDF file
            pages: List of page numbers to extract (1-indexed), None for all
            
        Returns:
            Dictionary mapping page number to text
        """
        text_by_page = {}
        
        try:
            # Try pdfplumber first (better for complex layouts)
            with pdfplumber.open(pdf_path) as pdf:
                pages_to_process = pages or range(1, len(pdf.pages) + 1)
                
                for page_num in pages_to_process:
                    if page_num < 1 or page_num > len(pdf.pages):
                        continue
                    
                    page = pdf.pages[page_num - 1]  # pdfplumber uses 0-indexing
                    text = page.extract_text()
                    
                    if text and len(text.strip()) > 0:
                        text_by_page[page_num] = text
                    elif self.use_ocr:
                        # Try OCR if no text found
                        im = page.to_image(resolution=200)
                        pil_image = im.original
                        
                        # Convert PIL to numpy
                        img_array = np.array(pil_image)
                        if img_array.shape[-1] == 4:
                            img_array = img_array[:, :, :3]  # Remove alpha
                        
                        ocr_text, confidence = self.ocr.extract_text(img_array)
                        if ocr_text:
                            text_by_page[page_num] = f"[OCR, confidence: {confidence:.2f}] {ocr_text}"
        
        except Exception as e:
            # Fallback to PyPDF2
            try:
                reader = PdfReader(pdf_path)
                pages_to_process = pages or range(1, len(reader.pages) + 1)
                
                for page_num in pages_to_process:
                    if page_num < 1 or page_num > len(reader.pages):
                        continue
                    
                    page = reader.pages[page_num - 1]
                    text = page.extract_text()
                    
                    if text:
                        text_by_page[page_num] = text
            
            except Exception as e2:
                print(f"Error extracting PDF: {e2}")
        
        return text_by_page
    
    def extract_tables(
        self,
        pdf_path: str,
        pages: Optional[List[int]] = None
    ) -> Dict[int, List[List]]:
        """
        Extract tables from PDF
        
        Args:
            pdf_path: Path to PDF
            pages: Pages to extract from
            
        Returns:
            Dictionary mapping page number to list of tables
        """
        tables_by_page = {}
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                pages_to_process = pages or range(1, len(pdf.pages) + 1)
                
                for page_num in pages_to_process:
                    if page_num < 1 or page_num > len(pdf.pages):
                        continue
                    
                    page = pdf.pages[page_num - 1]
                    tables = page.extract_tables()
                    
                    if tables:
                        tables_by_page[page_num] = tables
        
        except Exception as e:
            print(f"Error extracting tables: {e}")
        
        return tables_by_page
    
    def get_page_count(self, pdf_path: str) -> int:
        """Get number of pages in PDF"""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                return len(pdf.pages)
        except:
            try:
                reader = PdfReader(pdf_path)
                return len(reader.pages)
            except:
                return 0
    
    def extract_full_document(
        self,
        pdf_path: str,
        max_pages: int = 100
    ) -> Dict:
        """
        Extract everything from PDF
        
        Returns:
            Dictionary with 'text', 'tables', 'page_count'
        """
        page_count = self.get_page_count(pdf_path)
        
        # Limit pages for large documents
        pages_to_process = list(range(1, min(page_count + 1, max_pages + 1)))
        
        # Extract text
        text_by_page = self.extract_text(pdf_path, pages=pages_to_process)
        
        # Extract tables
        tables_by_page = self.extract_tables(pdf_path, pages=pages_to_process)
        
        # Combine all text
        full_text = "\n\n".join([
            f"[Page {page_num}]\n{text}"
            for page_num, text in sorted(text_by_page.items())
        ])
        
        return {
            'text': full_text,
            'text_by_page': text_by_page,
            'tables_by_page': tables_by_page,
            'page_count': page_count,
            'pages_processed': len(pages_to_process)
        }
    
    def is_scanned_pdf(self, pdf_path: str, sample_pages: int = 3) -> bool:
        """
        Check if PDF is scanned (image-based)
        
        Args:
            pdf_path: Path to PDF
            sample_pages: Number of pages to sample
            
        Returns:
            True if likely scanned
        """
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text_lengths = []
                
                for i in range(min(sample_pages, len(pdf.pages))):
                    page = pdf.pages[i]
                    text = page.extract_text()
                    text_lengths.append(len(text) if text else 0)
                
                # If average text length is very low, likely scanned
                avg_length = sum(text_lengths) / len(text_lengths) if text_lengths else 0
                return avg_length < 50  # Threshold
        
        except:
            return False


# Example usage
if __name__ == "__main__":
    parser = PDFParser(use_ocr=True)
    
    # Extract from PDF
    result = parser.extract_full_document("invoice.pdf")
    
    print(f"Pages: {result['page_count']}")
    print(f"Text length: {len(result['text'])}")
    print(f"\nFirst 500 chars:\n{result['text'][:500]}")
    
    if result['tables_by_page']:
        print(f"\nTables found on {len(result['tables_by_page'])} pages")
