"""
Voice-Enabled OCR Assistant
Extracts text from images and provides spoken answers to questions.
"""

import re
import numpy as np
from PIL import Image
import asyncio
import edge_tts
from pathlib import Path


class VoiceAssistant:
    """OCR + Voice Assistant for invoices and CCTV analysis"""
    
    def __init__(self, use_gpu=True):
        """Initialize OCR reader"""
        try:
            import easyocr
            self.reader = easyocr.Reader(['en'], gpu=use_gpu)
        except ImportError:
            raise ImportError("Install easyocr: pip install easyocr")
    
    def extract_text(self, image):
        """Extract text from image using OCR"""
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        result = self.reader.readtext(image)
        return " ".join([item[1] for item in result])
    
    def parse_invoice(self, text):
        """Parse invoice information from text"""
        info = {}
        
        # Amount: flexible patterns
        amount_patterns = [
            r'\$([\\d,]+\\.\\d{2})',
            r'[Aa]mount[:\\s]+\\$?([\\d,]+\\.\\d{2})',
            r'[Tt]otal[:\\s]+\\$?([\\d,]+\\.\\d{2})',
            r'\\$?([\\d,]+\\.\\d{2})'
        ]
        for pattern in amount_patterns:
            match = re.search(pattern, text)
            if match:
                info['amount'] = f"${match.group(1)}"
                break
        
        # Vendor
        vendor_match = re.search(r'INVOICE[-:\\s]*([A-Za-z]+)', text, re.IGNORECASE)
        if vendor_match:
            info['vendor'] = vendor_match.group(1)
        
        # Status
        if 'PAID' in text.upper():
            info['status'] = 'PAID'
        elif 'PENDING' in text.upper():
            info['status'] = 'PENDING'
        
        return info
    
    def answer_question(self, text, question):
        """Answer questions about extracted text"""
        if not text or not text.strip():
            return "No text detected in this image."
        
        q_lower = question.lower()
        info = self.parse_invoice(text)
        
        # Handle different question types
        if any(word in q_lower for word in ['amount', 'total', 'cost', 'price']):
            return info.get('amount', 'Amount not found')
        
        if any(word in q_lower for word in ['vendor', 'company', 'who', 'issued']):
            return info.get('vendor', 'Vendor not found')
        
        if any(word in q_lower for word in ['status', 'paid']):
            return info.get('status', 'Status not found')
        
        if any(word in q_lower for word in ['person', 'people', 'many']):
            match = re.search(r'(\\d+)\\s*PERSON', text.upper())
            if match:
                return f"{match.group(1)} persons"
            return "No person count found"
        
        # Default: return all info
        parts = []
        if 'vendor' in info:
            parts.append(f"Vendor: {info['vendor']}")
        if 'amount' in info:
            parts.append(f"Amount: {info['amount']}")
        if 'status' in info:
            parts.append(f"Status: {info['status']}")
        
        return ". ".join(parts) if parts else f"I found: {text[:100]}"
    
    def analyze_image(self, image, question):
        """Complete pipeline: OCR -> Answer"""
        text = self.extract_text(image)
        return self.answer_question(text, question)
    
    async def text_to_speech(self, text, output_file="response.mp3"):
        """Convert text to speech"""
        if not text or len(text) < 3:
            text = "No response available"
        
        communicate = edge_tts.Communicate(text, "en-US-AriaNeural")
        await communicate.save(output_file)
        return output_file
    
    def ask_with_voice(self, image, question, save_audio=True):
        """Ask question and get spoken answer"""
        answer = self.analyze_image(image, question)
        
        if save_audio:
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                audio_file = loop.run_until_complete(
                    self.text_to_speech(answer)
                )
                loop.close()
                return answer, audio_file
            except Exception as e:
                print(f"Voice generation failed: {e}")
                return answer, None
        
        return answer, None


# Example usage
if __name__ == "__main__":
    from PIL import Image, ImageDraw
    
    # Create test image
    img = Image.new('RGB', (800, 600), 'white')
    d = ImageDraw.Draw(img)
    d.text((50, 50), "INVOICE - Amazon", fill='black')
    d.text((50, 100), "Amount: $1,299.00", fill='black')
    d.text((50, 150), "Status: PAID", fill='green')
    
    # Use assistant
    assistant = VoiceAssistant(use_gpu=False)  # Set to True if GPU available
    answer, audio_file = assistant.ask_with_voice(img, "What is the amount?")
    
    print(f"Answer: {answer}")
    if audio_file:
        print(f"Audio saved to: {audio_file}")
