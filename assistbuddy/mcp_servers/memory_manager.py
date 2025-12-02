"""
Memory Manager using ChromaDB
Provides Long-Term Memory and RAG capabilities for the Agentic Orchestrator
"""

import chromadb
from chromadb.config import Settings
import uuid
from datetime import datetime
from typing import List, Dict, Any

class MemoryManager:
    def __init__(self, persistence_path: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=persistence_path)
        self.collection = self.client.get_or_create_collection(name="agent_memory")

    def add_memory(self, text: str, metadata: Dict[str, Any] = None):
        """Add a new memory entry"""
        if metadata is None:
            metadata = {}
        
        # Add timestamp if not present
        if "timestamp" not in metadata:
            metadata["timestamp"] = datetime.utcnow().isoformat()
            
        self.collection.add(
            documents=[text],
            metadatas=[metadata],
            ids=[str(uuid.uuid4())]
        )

    def query_memory(self, query_text: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant memories"""
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        
        memories = []
        if results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i]
                memories.append({
                    "content": doc,
                    "metadata": meta
                })
        
        return memories

    def clear_memory(self):
        """Clear all memories"""
        self.client.delete_collection("agent_memory")
        self.collection = self.client.get_or_create_collection(name="agent_memory")
