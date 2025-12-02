import re
from typing import List, Dict, Tuple
import spacy
from dataclasses import dataclass


@dataclass
class PIIDetection:
    """Container for detected PII"""
    type: str  # 'name', 'email', 'phone', 'address', 'ssn', 'aadhaar', 'credit_card'
    value: str
    start: int
    end: int
    confidence: float


class PIIDetector:
    
    def __init__(self, use_ner: bool = True):
        
        self.use_ner = use_ner
        
        if use_ner:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except:
                print("Warning: spaCy model not found. Install with: python -m spacy download en_core_web_sm")
                self.use_ner = False
        
        # Regex patterns for common PII
        self.patterns = {
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'phone_india': re.compile(r'\b(?:\+91|0)?[6-9]\d{9}\b'),
            'phone_us': re.compile(r'\b(?:\+1\s?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'),
            'aadhaar': re.compile(r'\b\d{4}\s?\d{4}\s?\d{4}\b'),
            'ssn': re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            'credit_card': re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b'),
            'pan_india': re.compile(r'\b[A-Z]{5}\d{4}[A-Z]\b'),
            'address_number': re.compile(r'\b\d{1,5}\s+[\w\s]{5,50}(?:Street|St|Avenue|Ave|Road|Rd|Lane|Ln|Drive|Dr)\b', re.IGNORECASE)
        }
        
    def detect(self, text: str) -> List[PIIDetection]:
        
        detections = []
        
        # Regex-based detection
        for pii_type, pattern in self.patterns.items():
            for match in pattern.finditer(text):
                detections.append(PIIDetection(
                    type=pii_type,
                    value=match.group(),
                    start=match.start(),
                    end=match.end(),
                    confidence=0.9  # High confidence for regex matches
                ))
        
        # NER-based detection for names and organizations
        if self.use_ner:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'ORG', 'GPE', 'LOC']:
                    detections.append(PIIDetection(
                        type=f'name_{ent.label_.lower()}',
                        value=ent.text,
                        start=ent.start_char,
                        end=ent.end_char,
                        confidence=0.7  # Lower confidence for NER
                    ))
        
        # Remove duplicates and sort by position
        detections = self._remove_overlaps(detections)
        detections.sort(key=lambda x: x.start)
        
        return detections
    
    def _remove_overlaps(self, detections: List[PIIDetection]) -> List[PIIDetection]:
        """Remove overlapping detections, keeping higher confidence ones"""
        if not detections:
            return []
        
        # Sort by confidence (descending)
        sorted_dets = sorted(detections, key=lambda x: x.confidence, reverse=True)
        
        kept = []
        for det in sorted_dets:
            # Check if overlaps with any kept detection
            overlaps = False
            for kept_det in kept:
                if self._overlaps(det, kept_det):
                    overlaps = True
                    break
            
            if not overlaps:
                kept.append(det)
        
        return kept
    
    def _overlaps(self, det1: PIIDetection, det2: PIIDetection) -> bool:
        """Check if two detections overlap"""
        return not (det1.end <= det2.start or det2.end <= det1.start)
    
    def has_pii(self, text: str) -> bool:
        """Quick check if text contains any PII"""
        return len(self.detect(text)) > 0
    
    def get_pii_types(self, text: str) -> List[str]:
        """Get list of PII types found in text"""
        detections = self.detect(text)
        return list(set(det.type for det in detections))


# Example usage
if __name__ == "__main__":
    detector = PIIDetector(use_ner=True)
    
    sample_text = """
    Contact John Smith at john.smith@example.com or call +91 9876543210.
    His Aadhaar is 1234 5678 9012 and he lives at 123 Main Street, Delhi.
    """
    
    detections = detector.detect(sample_text)
    
    for det in detections:
        print(f"{det.type}: '{det.value}' (confidence: {det.confidence:.2f})")
