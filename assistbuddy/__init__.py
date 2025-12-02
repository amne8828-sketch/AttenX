"""
AssistBuddy Package
Multimodal LLM for admin-facing file summarization
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from .model import AssistBuddyModel, VisionEncoder, TextEncoder, AudioEncoder, MultimodalFusion, StyleControlledDecoder
from .privacy import PIIDetector, PIIRedactor, ProvenanceTracker
from .utils import OCREngine, PDFParser, ExcelParser, VideoProcessor, WebScraper
from .data import DatasetGenerator
from .inference import get_system_prompt, ADMIN_TEMPLATE, FRIEND_TEMPLATE

__all__ = [
    # Model components
    'AssistBuddyModel',
    'VisionEncoder',
    'TextEncoder',
    'AudioEncoder',
    'MultimodalFusion',
    'StyleControlledDecoder',
    
    # Privacy
    'PIIDetector',
    'PIIRedactor',
    'ProvenanceTracker',
    
    # Utils
    'OCREngine',
    'PDFParser',
    'ExcelParser',
    'VideoProcessor',
    'WebScraper',
    
    # Data
    'DatasetGenerator',
    
    # Inference
    'get_system_prompt',
    'ADMIN_TEMPLATE',
    'FRIEND_TEMPLATE'
]
