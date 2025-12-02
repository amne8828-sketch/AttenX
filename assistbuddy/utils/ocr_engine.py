"""
OCR Engine for text extraction from images
Supports Tesseract and EasyOCR
"""

import cv2
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional
import pytesseract


class OCREngine:
    """
    OCR engine with multiple backend support
    """
    
    def __init__(
        self,
        backend: str = 'tesseract',  # 'tesseract' or 'easyocr'
        languages: List[str] = ['en', 'hi']  # English and Hindi
    ):
        """
        Args:
            backend: OCR backend to use
            languages: Languages to detect
        """
        self.backend = backend
        self.languages = languages
        
        if backend == 'easyocr':
            try:
                import easyocr
                self.reader = easyocr.Reader(languages)
            except ImportError:
                print("Warning: easyocr not installed. Falling back to tesseract.")
                self.backend = 'tesseract'
    
    def extract_text(
        self,
        image: np.ndarray,
        preprocess: bool = True
    ) -> Tuple[str, float]:
        """
        Extract text from image
        
        Args:
            image: Image as numpy array (BGR format from cv2)
            preprocess: Whether to preprocess image
            
        Returns:
            (extracted_text, confidence)
        """
        # Preprocess if requested
        if preprocess:
            image = self._preprocess_image(image)
        
        # Run OCR based on backend
        if self.backend == 'tesseract':
            return self._tesseract_ocr(image)
        elif self.backend == 'easyocr':
            return self._easyocr_ocr(image)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR
        - Convert to grayscale
        - Denoise
        - Increase contrast
        - Binarize
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Increase contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast = clahe.apply(denoised)
        
        # Adaptive threshold
        binary = cv2.adaptiveThreshold(
            contrast,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )
        
        return binary
    
    def _tesseract_ocr(self, image: np.ndarray) -> Tuple[str, float]:
        """Run Tesseract OCR"""
        # Get detailed data
        data = pytesseract.image_to_data(
            image,
            output_type=pytesseract.Output.DICT,
            lang='+'.join(self.languages)
        )
        
        # Extract text and calculate average confidence
        text_parts = []
        confidences = []
        
        for i, conf in enumerate(data['conf']):
            if conf > 0:  # Valid detection
                text = data['text'][i].strip()
                if text:
                    text_parts.append(text)
                    confidences.append(float(conf) / 100.0)
        
        full_text = ' '.join(text_parts)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return full_text, avg_confidence
    
    def _easyocr_ocr(self, image: np.ndarray) -> Tuple[str, float]:
        """Run EasyOCR"""
        results = self.reader.readtext(image)
        
        if not results:
            return "", 0.0
        
        # Extract text and confidence
        text_parts = [item[1] for item in results]
        confidences = [item[2] for item in results]
        
        full_text = ' '.join(text_parts)
        avg_confidence = sum(confidences) / len(confidences)
        
        return full_text, avg_confidence
    
    def extract_text_with_boxes(
        self,
        image: np.ndarray
    ) -> List[Dict]:
        """
        Extract text with bounding boxes
        
        Returns:
            List of dicts with 'text', 'bbox', 'confidence'
        """
        if self.backend == 'tesseract':
            data = pytesseract.image_to_data(
                image,
                output_type=pytesseract.Output.DICT,
                lang='+'.join(self.languages)
            )
            
            results = []
            for i in range(len(data['text'])):
                if data['conf'][i] > 0:
                    text = data['text'][i].strip()
                    if text:
                        results.append({
                            'text': text,
                            'bbox': (
                                data['left'][i],
                                data['top'][i],
                                data['width'][i],
                                data['height'][i]
                            ),
                            'confidence': float(data['conf'][i]) / 100.0
                        })
            
            return results
        
        elif self.backend == 'easyocr':
            easyocr_results = self.reader.readtext(image)
            
            results = []
            for bbox, text, conf in easyocr_results:
                # Convert polygon bbox to x,y,w,h
                x_coords = [p[0] for p in bbox]
                y_coords = [p[1] for p in bbox]
                x, y = min(x_coords), min(y_coords)
                w, h = max(x_coords) - x, max(y_coords) - y
                
                results.append({
                    'text': text,
                    'bbox': (int(x), int(y), int(w), int(h)),
                    'confidence': conf
                })
            
            return results
    
    def is_text_readable(self, image: np.ndarray, min_confidence: float = 0.4) -> bool:
        """
        Check if image contains readable text
        
        Args:
            image: Input image
            min_confidence: Minimum confidence threshold
            
        Returns:
            True if text is readable
        """
        text, confidence = self.extract_text(image)
        return len(text.strip()) > 0 and confidence >= min_confidence


# Example usage
if __name__ == "__main__":
    # Test with sample image
    ocr = OCREngine(backend='tesseract', languages=['en'])
    
    # Create test image with text
    img = np.ones((100, 400, 3), dtype=np.uint8) * 255
    cv2.putText(img, "Hello World", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    text, conf = ocr.extract_text(img)
    print(f"Extracted: '{text}' (confidence: {conf:.2f})")
