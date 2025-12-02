
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import json


@dataclass
class ProvenanceRecord:
    """Single provenance record"""
    file: str
    type: str  # 'pdf', 'image', 'video', 'audio', 'excel', 'word', 'web'
    method: str  # 'OCR', 'ASR', 'scrape', 'parse', 'vision', 'manual'
    page_or_ts: str  # Page number or timestamp
    confidence: float = 1.0
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self):
        return asdict(self)
    
    def to_citation(self) -> str:
        """Generate human-readable citation"""
        if self.type in ['pdf', 'word', 'excel']:
            return f"{self.file} (page {self.page_or_ts})"
        elif self.type in ['video', 'audio']:
            return f"{self.file} (@{self.page_or_ts})"
        elif self.type == 'image':
            return f"{self.file}"
        elif self.type == 'web':
            return f"{self.file}"
        return f"{self.file}"


class ProvenanceTracker:
    
    def __init__(self):
        self.records: List[ProvenanceRecord] = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def add_record(
        self,
        file: str,
        file_type: str,
        method: str,
        page_or_timestamp: str = "",
        confidence: float = 1.0
    ) -> ProvenanceRecord:
        """
        Add a provenance record
        
        Args:
            file: Filename or URL
            file_type: Type of file
            method: Extraction method used
            page_or_timestamp: Page number or timestamp
            confidence: Confidence in extraction (0-1)
            
        Returns:
            Created record
        """
        record = ProvenanceRecord(
            file=file,
            type=file_type,
            method=method,
            page_or_ts=page_or_timestamp,
            confidence=confidence
        )
        
        self.records.append(record)
        return record
    
    def get_citations(self) -> List[str]:
        """Get all citations in human-readable format"""
        return [rec.to_citation() for rec in self.records]
    
    def get_provenance_summary(self) -> List[Dict]:
        """Get provenance summary for JSON output"""
        # Deduplicate by file + page/timestamp
        seen = set()
        unique_records = []
        
        for rec in self.records:
            key = (rec.file, rec.page_or_ts)
            if key not in seen:
                seen.add(key)
                unique_records.append(rec)
        
        return [rec.to_dict() for rec in unique_records]
    
    def generate_sources_block(self) -> str:
        
        if not self.records:
            return "Sources & Files:\n- None"
        
        lines = ["Sources & Files:"]
        
        # Group by file
        by_file = {}
        for rec in self.records:
            if rec.file not in by_file:
                by_file[rec.file] = []
            by_file[rec.file].append(rec)
        
        for file, recs in by_file.items():
            # Get unique pages/timestamps
            pages = list(set(rec.page_or_ts for rec in recs if rec.page_or_ts))
            methods = list(set(rec.method for rec in recs))
            
            page_str = f" — {', '.join(pages)}" if pages else ""
            method_str = f" — {', '.join(methods)}"
            
            lines.append(f"- {file}{page_str}{method_str}")
        
        return '\n'.join(lines)
    
    def save_audit_log(self, output_file: str):
        """Save audit log to JSON file"""
        log_data = {
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'records': [rec.to_dict() for rec in self.records]
        }
        
        with open(output_file, 'w') as f:
            json.dump(log_data, f, indent=2)
    
    def get_confidence_score(self) -> float:
        
        if not self.records:
            return 0.0
        
        return sum(rec.confidence for rec in self.records) / len(self.records)
    
    def verify_claim(self, claim: str, source_file: str, page_or_ts: str = "") -> Optional[ProvenanceRecord]:
       
        for rec in self.records:
            if rec.file == source_file:
                if not page_or_ts or rec.page_or_ts == page_or_ts:
                    return rec
        return None


# Example usage
if __name__ == "__main__":
    tracker = ProvenanceTracker()
    
    # Add some records
    tracker.add_record(
        file="invoice_104.pdf",
        file_type="pdf",
        method="OCR",
        page_or_timestamp="1",
        confidence=0.92
    )
    
    tracker.add_record(
        file="cctv.mp4",
        file_type="video",
        method="vision",
        page_or_timestamp="00:14:32",
        confidence=0.34
    )
    
    # Print sources block
    print(tracker.generate_sources_block())
    print(f"\nOverall confidence: {tracker.get_confidence_score() * 100:.0f}")
    
    # Get JSON format
    print("\nProvenance JSON:")
    print(json.dumps(tracker.get_provenance_summary(), indent=2))
