"""
PII Redaction Module
Redacts detected PII from text and outputs
"""

from typing import List, Dict
from .pii_detector import PIIDetector, PIIDetection


class PIIRedactor:
    """
    Redacts PII from text with appropriate placeholders
    """
    
    def __init__(self, detector: PIIDetector = None):
        self.detector = detector or PIIDetector(use_ner=True)
        
        # Redaction templates
        self.redaction_templates = {
            'email': '[REDACTED_EMAIL]',
            'phone_india': '[REDACTED_PHONE]',
            'phone_us': '[REDACTED_PHONE]',
            'aadhaar': '[REDACTED_AADHAAR]',
            'ssn': '[REDACTED_SSN]',
            'credit_card': '[REDACTED_CARD]',
            'pan_india': '[REDACTED_PAN]',
            'address_number': '[REDACTED_ADDRESS]',
            'name_person': '[REDACTED_NAME]',
            'name_org': '[REDACTED_ORG]',
            'name_gpe': '[REDACTED_LOCATION]',
            'name_loc': '[REDACTED_LOCATION]'
        }
    
    def redact(
        self,
        text: str,
        mask_mode: str = 'placeholder',  # 'placeholder', 'partial', 'hash'
        authorized_types: List[str] = None
    ) -> tuple[str, List[PIIDetection]]:
        """
        Redact PII from text
        
        Args:
            text: Input text
            mask_mode: How to mask PII
                - 'placeholder': Replace with [REDACTED_TYPE]
                - 'partial': Show partial (e.g., j***@example.com)
                - 'hash': Replace with hash
            authorized_types: PII types that are authorized to display (skip redaction)
            
        Returns:
            (redacted_text, list of detections)
        """
        detections = self.detector.detect(text)
        
        if not detections:
            return text, []
        
        # Build redacted text by replacing from end to start
        # (to preserve positions)
        redacted = text
        for det in reversed(detections):
            # Skip if authorized
            if authorized_types and det.type in authorized_types:
                continue
            
            # Get replacement text
            if mask_mode == 'placeholder':
                replacement = self.redaction_templates.get(det.type, '[REDACTED]')
            elif mask_mode == 'partial':
                replacement = self._partial_mask(det.value, det.type)
            elif mask_mode == 'hash':
                replacement = f'[HASH_{hash(det.value) % 10000:04d}]'
            else:
                replacement = '[REDACTED]'
            
            # Replace in text
            redacted = redacted[:det.start] + replacement + redacted[det.end:]
        
        return redacted, detections
    
    def _partial_mask(self, value: str, pii_type: str) -> str:
        """Create partial mask showing some characters"""
        if pii_type == 'email':
            # Show first char and domain: j***@example.com
            parts = value.split('@')
            if len(parts) == 2:
                return f"{parts[0][0]}***@{parts[1]}"
        
        elif pii_type in ['phone_india', 'phone_us']:
            # Show last 4 digits: ***-***-1234
            digits = ''.join(c for c in value if c.isdigit())
            if len(digits) >= 4:
                return f"***-***-{digits[-4:]}"
        
        elif pii_type == 'aadhaar':
            # Show last 4 digits: XXXX XXXX 1234
            digits = ''.join(c for c in value if c.isdigit())
            if len(digits) == 12:
                return f"XXXX XXXX {digits[-4:]}"
        
        elif pii_type.startswith('name_'):
            # Show first letter: J*** S***
            words = value.split()
            return ' '.join(f"{w[0]}***" if w else "" for w in words)
        
        # Default: show first and last char
        if len(value) > 2:
            return f"{value[0]}{'*' * (len(value) - 2)}{value[-1]}"
        return "***"
    
    def generate_privacy_warning(self, detections: List[PIIDetection]) -> str:
        """
        Generate warning message about detected PII
        
        Returns:
            Warning string for admin
        """
        if not detections:
            return ""
        
        pii_types = {}
        for det in detections:
            pii_types[det.type] = pii_types.get(det.type, 0) + 1
        
        type_strs = [f"{count} {pii_type}" for pii_type, count in pii_types.items()]
        
        warning = f"⚠️ Privacy Warning: Detected {', '.join(type_strs)}. "
        warning += "All PII has been redacted. Obtain explicit consent before processing identifiable data."
        
        return warning
    
    def redact_json_output(self, output_dict: Dict, authorized: bool = False) -> Dict:
        """
        Redact PII from JSON output dictionary
        
        Args:
            output_dict: The output JSON (with tldr, key_details, etc.)
            authorized: Whether user has authorized PII display
            
        Returns:
            Redacted dictionary
        """
        if authorized:
            return output_dict
        
        redacted = output_dict.copy()
        
        # Redact TL;DR
        if 'tldr' in redacted:
            redacted['tldr'], _ = self.redact(redacted['tldr'])
        
        # Redact key details
        if 'key_details' in redacted:
            for detail in redacted['key_details']:
                if 'text' in detail:
                    detail['text'], _ = self.redact(detail['text'])
                if 'source' in detail:
                    detail['source'], _ = self.redact(detail['source'])
        
        # Redact actions
        if 'actions' in redacted:
            for action in redacted['actions']:
                if 'text' in action:
                    action['text'], _ = self.redact(action['text'])
                if 'why' in action:
                    action['why'], _ = self.redact(action['why'])
        
        # Add privacy note
        all_text = str(output_dict)
        _, detections = self.redact(all_text)
        if detections:
            warning = self.generate_privacy_warning(detections)
            redacted['notes'] = redacted.get('notes', '') + f"\n{warning}"
        
        return redacted


# Example usage
if __name__ == "__main__":
    redactor = PIIRedactor()
    
    sample = "Contact John Smith at john.smith@example.com or +91 9876543210"
    
    redacted, detections = redactor.redact(sample, mask_mode='placeholder')
    print(f"Original: {sample}")
    print(f"Redacted: {redacted}")
    print(f"\nDetections: {len(detections)}")
    
    warning = redactor.generate_privacy_warning(detections)
    print(f"\nWarning: {warning}")
