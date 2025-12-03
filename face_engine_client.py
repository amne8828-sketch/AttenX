import requests
import os
import base64
import cv2
import numpy as np
import json
from typing import List, Dict, Any, Optional, Tuple

class FaceEngineClient:
    def __init__(self, base_url: str = None):
        self.base_url = base_url or os.environ.get("FACE_ENGINE_URL", "http://localhost:7860")
        self.api_url = f"{self.base_url}/api/v1"
        self.timeout = 10  # seconds

    def _encode_image(self, image: np.ndarray) -> str:
        """Encode numpy image to base64 string"""
        _, buffer = cv2.imencode('.jpg', image)
        img_str = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{img_str}"

    def is_available(self) -> bool:
        """Check if Face Engine is online"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=2)
            return response.status_code == 200
        except:
            return False

    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces in an image"""
        try:
            img_str = self._encode_image(image)
            response = requests.post(
                f"{self.api_url}/detect",
                json={"image": img_str},
                timeout=self.timeout
            )
            if response.status_code == 200:
                return response.json().get("faces", [])
            else:
                print(f"Face Engine Error: {response.text}")
                return []
        except Exception as e:
            print(f"Face Engine Connection Error: {e}")
            return []

    def get_embeddings(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Get embeddings for faces in image"""
        try:
            img_str = self._encode_image(image)
            response = requests.post(
                f"{self.api_url}/embed",
                json={"image": img_str},
                timeout=self.timeout
            )
            if response.status_code == 200:
                return response.json().get("results", [])
            else:
                print(f"Face Engine Error: {response.text}")
                return []
        except Exception as e:
            print(f"Face Engine Connection Error: {e}")
            return []

    def verify_faces(self, img1: np.ndarray, img2: np.ndarray) -> Dict[str, Any]:
        """Verify if two images contain the same face"""
        try:
            img1_str = self._encode_image(img1)
            img2_str = self._encode_image(img2)
            
            response = requests.post(
                f"{self.api_url}/verify",
                json={"image1": img1_str, "image2": img2_str},
                timeout=self.timeout
            )
            if response.status_code == 200:
                return response.json()
            else:
                return {"match": False, "error": response.text}
        except Exception as e:
            return {"match": False, "error": str(e)}

# Singleton instance
face_client = FaceEngineClient()
