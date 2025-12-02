import cv2 as cv
import mediapipe as mp
import numpy as np
import time
# Reuse existing embedding and search logic for DB compatibility
try:
    from function import get_face_embedding, search_face_faiss
    DB_AVAILABLE = True
except ImportError:
    print("WARNING: function.py not found, recognition disabled")
    DB_AVAILABLE = False

class SimpleFaceRecognizer:
    def __init__(self):
        print("Initializing SimpleFaceRecognizer with MediaPipe...")
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1, # 0 for short range, 1 for full range (5 meters)
            min_detection_confidence=0.5
        )
        self.last_recognition_time = {}
        
    def detect_and_recognize(self, image):
        """
        Detect faces using MediaPipe and recognize using existing DB.
        """
        if image is None:
            return {"status": "failed", "faces": []}
            
        # Convert to RGB for MediaPipe
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = self.face_detection.process(image_rgb)
        
        faces_data = []
        
        if results.detections:
            h, w, _ = image.shape
            
            for detection in results.detections:
                # Get bounding box
                bboxC = detection.location_data.relative_bounding_box
                x = int(bboxC.xmin * w)
                y = int(bboxC.ymin * h)
                width = int(bboxC.width * w)
                height = int(bboxC.height * h)
                
                # Ensure bbox is within bounds
                x = max(0, x)
                y = max(0, y)
                width = min(w - x, width)
                height = min(h - y, height)
                
                # Get score
                score = detection.score[0]
                
                face_info = {
                    "bbox": {"x": x, "y": y, "width": width, "height": height},
                    "confidence": float(score),
                    "name": "Unknown",
                    "recognized": False,
                    "match_confidence": 0.0
                }
                
                # Perform recognition if DB is available
                if DB_AVAILABLE and width > 20 and height > 20:
                    try:
                        # Crop face
                        face_img = image[y:y+height, x:x+width]
                        
                        # Get embedding (reusing existing logic for compatibility)
                        # This uses DeepFace internally but we wrap it simply
                        embedding = get_face_embedding(face_img, handle_occlusion=False)
                        
                        if embedding is not None:
                            name, match_score = search_face_faiss(embedding, threshold=0.50)
                            
                            if name:
                                face_info["name"] = name
                                face_info["recognized"] = True
                                face_info["match_confidence"] = float(match_score)
                    except Exception as e:
                        print(f"Recognition error: {e}")
                
                faces_data.append(face_info)
                
        return {
            "status": "success",
            "faces_detected": len(faces_data),
            "faces": faces_data
        }

    def detect_faces(self, image):
        """Legacy method for compatibility"""
        result = self.detect_and_recognize(image)
        # Convert to list of dicts format if needed by legacy code, 
        # but the new API expects the full dict structure.
        return result
