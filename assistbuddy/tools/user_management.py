
import os
import datetime
import base64
from typing import Dict, Any, Optional
from werkzeug.security import generate_password_hash
from werkzeug.utils import secure_filename
from database_utils import get_db_collections
import re

# Import face recognition
try:
    from simple_face_recognition import SimpleFaceRecognizer
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    print("[WARNING] Face recognition not available")


class UserManagementTool:
    """Tool for creating and managing users via chat"""
    
    def __init__(self):
        self.db = get_db_collections()
        self.face_recognizer = None
        if FACE_RECOGNITION_AVAILABLE:
            try:
                self.face_recognizer = SimpleFaceRecognizer()
                print("Face recognizer initialized in UserManagementTool")
            except Exception as e:
                print(f"Failed to initialize face recognizer: {e}")
    
    def create_user(self, 
                    name: str,
                    email: str,
                    password: str,
                    department: str = "General",
                    photo_path: Optional[str] = None,
                    role: str = "user") -> Dict[str, Any]:
       
        try:
            # Validate inputs
            if not name or not email or not password:
                return {
                    "success": False,
                    "message": "Name, email, and password are required"
                }
            
            # Validate email format
            if not re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', email):
                return {
                    "success": False,
                    "message": "Invalid email format"
                }
            
            # Check if user already exists
            existing_user = self.db['users_col'].find_one({"email": email})
            if existing_user:
                return {
                    "success": False,
                    "message": f"User with email {email} already exists"
                }
            
            # Process face enrollment if photo is provided
            embeddings = None
            profile_image = None
            
            if photo_path and os.path.exists(photo_path):
                if self.face_recognizer:
                    try:
                        # Generate embeddings from photo
                        result = self.face_recognizer.register_face(photo_path, email)
                        
                        if result.get('success'):
                            embeddings = result.get('embeddings')
                            
                            # Read image as base64 for storage
                            with open(photo_path, 'rb') as f:
                                img_data = f.read()
                                profile_image = base64.b64encode(img_data).decode('utf-8')
                                
                            print(f"Face enrolled for {email}: {len(embeddings) if embeddings else 0} embeddings")
                        else:
                            return {
                                "success": False,
                                "message": f"Face enrollment failed: {result.get('message', 'Unknown error')}"
                            }
                    except Exception as e:
                        return {
                            "success": False,
                            "message": f"Face processing error: {str(e)}"
                        }
                else:
                    return {
                        "success": False,
                        "message": "Face recognition not available"
                    }
            
            # Create user document
            user_doc = {
                "name": name.strip(),
                "email": email.strip().lower(),
                "password_hash": generate_password_hash(password),
                "department": department.strip(),
                "role": role,
                "status": "active",
                "is_enrolled": embeddings is not None,
                "created_at": datetime.datetime.utcnow(),
                "created_by": "assistbuddy",
                "last_login": None
            }
            
            # Add embeddings if available
            if embeddings:
                user_doc["embeddings"] = embeddings
                user_doc["profile_image"] = profile_image
            
            # Insert into database
            result = self.db['users_col'].insert_one(user_doc)
            user_id = str(result.inserted_id)
            
            # Log the action
            self.db['system_logs_col'].insert_one({
                "action": "create_user",
                "user_id": user_id,
                "user_email": email,
                "performed_by": "assistbuddy",
                "timestamp": datetime.datetime.utcnow(),
                "details": {
                    "name": name,
                    "department": department,
                    "enrolled": embeddings is not None
                }
            })
            
            return {
                "success": True,
                "message": f"User {name} created successfully" + 
                          (" with face enrollment" if embeddings else ""),
                "user_id": user_id,
                "email": email,
                "enrolled": embeddings is not None
            }
            
        except Exception as e:
            print(f"Error creating user: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "message": f"Failed to create user: {str(e)}"
            }
    
    def parse_user_details(self, text: str) -> Dict[str, str]:
        
        details = {
            "name": None,
            "email": None,
            "password": None,
            "department": "General"
        }
        
        text_lower = text.lower()
        
        # Extract name
        name_patterns = [
            r'(?:name[:\s]+|user\s+)([a-zA-Z\s]+?)(?:,|email|password|\s+dept)',
            r'create\s+user\s+([a-zA-Z\s]+?)(?:,|email|password)',
        ]
        for pattern in name_patterns:
            match = re.search(pattern, text_lower)
            if match:
                details['name'] = match.group(1).strip().title()
                break
        
        # Extract email
        email_match = re.search(r'email[:\s]+([^\s,]+@[^\s,]+)', text_lower)
        if email_match:
            details['email'] = email_match.group(1).strip()
        
        # Extract password
        password_patterns = [
            r'password[:\s]+([^\s,]+)',
            r'pass[:\s]+([^\s,]+)',
            r'pwd[:\s]+([^\s,]+)',
        ]
        for pattern in password_patterns:
            match = re.search(pattern, text_lower)
            if match:
                details['password'] = match.group(1).strip()
                break
        
        # Extract department
        dept_patterns = [
            r'dept[:\s]+([a-zA-Z\s]+?)(?:,|$)',
            r'department[:\s]+([a-zA-Z\s]+?)(?:,|$)',
        ]
        for pattern in dept_patterns:
            match = re.search(pattern, text_lower)
            if match:
                details['department'] = match.group(1).strip().title()
                break
        
        return details


# Global instance
_user_management_tool = None

def get_user_management_tool() -> UserManagementTool:
    """Get global user management tool instance"""
    global _user_management_tool
    if _user_management_tool is None:
        _user_management_tool = UserManagementTool()
    return _user_management_tool
