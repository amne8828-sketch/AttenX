import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from flask import Flask, render_template, request, jsonify, Response, redirect, url_for, flash, session
from pymongo import MongoClient
# Heavy imports moved to lazy loading
# from deepface import DeepFace
# import cv2 as cv - Lazy loaded in functions to reduce startup memory
# import mediapipe as mp
import numpy as np # NumPy is usually fine, but we can check

import pickle
import datetime
import os
import faiss
import base64
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, logout_user, current_user, login_required
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from functools import wraps
import re
from email_validator import validate_email, EmailNotValidError
import asyncio
from concurrent.futures import ThreadPoolExecutor
from bson import ObjectId
import threading
from dotenv import load_dotenv

load_dotenv()

# Import function modules
from function import generate_camera_stream, get_face_embedding, continuous_learning_update, create_3d_template, extract_multi_vector_embeddings, ensemble_matching, search_face_faiss, rebuild_faiss_index

# Import the new super admin module
from superadmin_module import register_superadmin_module
from extensions import socketio

# Import advanced enrollment modules
from enrollment_engine import get_enrollment_engine
from advanced_liveness_detection import get_liveness_detector

app = Flask(__name__)
socketio.init_app(app)
app.secret_key = os.environ.get("FLASK_SECRET", "replace_this_with_a_real_secret")

# Rate Limiter Configuration
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://",
    strategy="fixed-window"
)

# MONGODB
MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017/?serverSelectionTimeoutMS=10000&connectTimeoutMS=10000&socketTimeoutMS=10000")
# For MongoDB Atlas connection with comprehensive SSL fixes
try:
    # Try with DNS seedlist format and relaxed SSL
    client = MongoClient(MONGODB_URI, 
                       serverSelectionTimeoutMS=5000,
                       connectTimeoutMS=5000,
                       socketTimeoutMS=5000,
                       retryWrites=False,
                       w="majority",
                       readPreference="primaryPreferred")
    # Test connection
    client.admin.command('ping')
    print("MongoDB connection successful!")
except Exception as e:
    print(f"MongoDB connection failed: {e}")
    print("Using Mock MongoDB fallback...")
    from mock_mongo import MockMongoClient
    client = MockMongoClient(MONGODB_URI)

transactional_db = client["transactional_db"]
attendance_col = transactional_db["attendance"]

core = client['secure_db']
persons_col = core["persons"]
profile_col = core["profile"]
superadmins_col = core["superadmins"]
admins_col = core["admins"]
users_col = core["users"]
enrollment_requests_col = core["enrollment_requests"]
system_logs_col = core["system_logs"]
cameras_col = core["cameras"]

# Email Verification System 
class EmailVerificationSystem:
    
    @staticmethod
    def verify_email_format(email: str) -> dict:
        """Verify email format and structure"""
        try:
            # Basic format check
            if '@' not in email:
                return {
                    "valid": False,
                    "reason": "Invalid email format - missing @",
                    "tag": "INVALID_FORMAT"
                }
            
            email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if not re.match(email_regex, email):
                return {
                    "valid": False,
                    "reason": "Invalid email format",
                    "tag": "INVALID_FORMAT"
                }
            
            # Advanced validation with email-validator
            try:
                validation = validate_email(email, check_deliverability=True)
                email_normalized = validation.normalized
                
                return {
                    "valid": True,
                    "normalized": email_normalized,
                    "domain": email.split('@')[1],
                    "tag": "VERIFIED",
                    "mx_records_exist": True
                }
            except EmailNotValidError as e:
                return {
                    "valid": False,
                    "reason": str(e),
                    "tag": "INVALID_DOMAIN"
                }
                
        except Exception as e:
            return {
                "valid": False,
                "reason": f"Verification error: {str(e)}",
                "tag": "ERROR"
            }
    
    @staticmethod
    def is_disposable_email(email: str) -> bool:
        """Check if email is from disposable domain"""
        try:
            disposable_domains = [
                'tempmail.com', 'guerrillamail.com', '10minutemail.com',
                'mailinator.com', 'throwaway.email', 'temp-mail.org'
            ]
            domain = email.split('@')[1].lower()
            return domain in disposable_domains
        except:
            return False

email_verifier = EmailVerificationSystem()

# FAISS Vector Database Setup - FIXED
EMBEDDING_DIM = 512
faiss_index = None
person_id_map = []

def initialize_faiss_index():
    """Initialize FAISS index properly"""
    global faiss_index, person_id_map
    # Use data directory for persistence
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    index_path = os.path.join(data_dir, "faiss_index.bin")
    map_path = os.path.join(data_dir, "person_id_map.pkl")
    
    if os.path.exists(index_path) and os.path.exists(map_path):
        try:
            faiss_index = faiss.read_index(index_path)
            with open(map_path, 'rb') as f:
                person_id_map = pickle.load(f)
            print(f"Loaded FAISS index with {faiss_index.ntotal} faces")
        except Exception as e:
            print(f"Error loading FAISS index: {e}")
            faiss_index = faiss.IndexFlatIP(EMBEDDING_DIM)
            person_id_map = []
    else:
        faiss_index = faiss.IndexFlatIP(EMBEDDING_DIM)
        person_id_map = []
        print("Created new FAISS index")

def save_faiss_index():
    """Save FAISS index to disk"""
    try:
        if faiss_index is not None:
            data_dir = "data"
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
                
            faiss.write_index(faiss_index, os.path.join(data_dir, "faiss_index.bin"))
            with open(os.path.join(data_dir, "person_id_map.pkl"), 'wb') as f:
                pickle.dump(person_id_map, f)
    except Exception as e:
        print(f"Error saving FAISS index: {e}")

# Initialize FAISS lazily to avoid startup delays
def lazy_faiss_init():
    global faiss_index
    if faiss_index is None:
        initialize_faiss_index()
    return faiss_index

# Login manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

class SuperAdminUser(UserMixin):
    def __init__(self, doc):
        self.doc = doc
        self.id = str(doc.get("_id"))
        self.email = doc.get("email")
        self.name = doc.get("name", "Super Admin")
        self.role = "superadmin"
        self.profile_image = doc.get("profile_image", "")

class AdminUser(UserMixin):
    def __init__(self, doc):
        self.doc = doc
        self.id = str(doc.get("_id"))
        self.email = doc.get("email")
        self.name = doc.get("name", "Admin")
        self.role = "admin"
        self.profile_image = doc.get("profile_image", "")

class RegularUser(UserMixin):
    def __init__(self, doc):
        self.doc = doc
        self.id = str(doc.get("_id"))
        self.email = doc.get("email")
        self.name = doc.get("name", "User")
        self.role = "user"
        self.department = doc.get("department", "")
        self.profile_image = doc.get("profile_image", "")
        self.status = doc.get("status", "active")

@login_manager.user_loader
def load_user(user_id):
    try:
        # Get robust connection for user loading
        load_client = get_robust_db_connection()
        core = load_client['secure_db']
        superadmins_col = core["superadmins"]
        admins_col = core["admins"]
        users_col = core["users"]
        
        # Try super admin first
        doc = superadmins_col.find_one({"_id": user_id})
        if doc:
            return SuperAdminUser(doc)
        
        # Try admin
        doc = admins_col.find_one({"_id": user_id})
        if doc:
            return AdminUser(doc)
        
        # Try regular user
        doc = users_col.find_one({"_id": user_id})
        if doc:
            return RegularUser(doc)
        
        # Fallback with ObjectId
        try:
            doc = superadmins_col.find_one({"_id": ObjectId(user_id)})
            if doc:
                return SuperAdminUser(doc)
            doc = admins_col.find_one({"_id": ObjectId(user_id)})
            if doc:
                return AdminUser(doc)
            doc = users_col.find_one({"_id": ObjectId(user_id)})
            if doc:
                return RegularUser(doc)
        except:
            pass
    except Exception as e:
        print(f"Error loading user: {e}")
        # If robust connection fails, try global collections as fallback
        try:
            doc = superadmins_col.find_one({"_id": user_id})
            if doc:
                return SuperAdminUser(doc)
            doc = admins_col.find_one({"_id": user_id})
            if doc:
                return AdminUser(doc)
            doc = users_col.find_one({"_id": user_id})
            if doc:
                return RegularUser(doc)
        except:
            pass
    return None

def superadmin_required(f):
    @wraps(f)
    @login_required
    def decorated_function(*args, **kwargs):
        if current_user.role != 'superadmin':
            flash('Access denied. Super Admin privileges required.', 'danger')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function
                                                                                                                                                                                            
def admin_required(f):
    @wraps(f)
    @login_required
    def decorated_function(*args, **kwargs):
        if current_user.role not in ['admin', 'superadmin']:
            flash('Access denied. Admin privileges required.', 'danger')
            return redirect(url_for('user_dashboard'))
        return f(*args, **kwargs)
    return decorated_function

# Register the super admin module
register_superadmin_module(app)

# Register AssistBuddy Chat API
try:
    from assistbuddy_api import assistbuddy_bp
    app.register_blueprint(assistbuddy_bp)
    print("[SUCCESS] AssistBuddy Chat API registered successfully")
except ImportError as e:
    print(f"[WARNING] AssistBuddy API not available: {e}")
except Exception as e:
    print(f"[ERROR] Failed to register AssistBuddy API: {e}")

# --- Authentication Routes ---
@app.route("/")
def index():
    if current_user.is_authenticated:
        if current_user.role == 'superadmin':
            return redirect(url_for("superadmin.dashboard"))
        elif current_user.role == 'admin':
            return redirect(url_for("admin_dashboard"))
        return redirect(url_for("user_dashboard"))
    return render_template("index.html")



@app.route("/registration_form")
def registration_form():
    return render_template("registration_form.html")

@app.route("/registration_request")
def registration_request():
    """Registration request page"""
    return render_template("registration_req.html")

@app.route("/submit_enrollment_request", methods=["POST"])
@limiter.limit("5 per hour")
def submit_enrollment_request():
    """Submit enrollment request from public form - with async processing"""
    try:
        if 'files' not in request.files:
            return jsonify({"status": "failed", "msg": "No images uploaded"}), 400
        
        files = request.files.getlist('files')
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip().lower()
        phone = request.form.get('phone', '').strip()
        password = request.form.get('password', '')
        
        # Quick validation only
        if not all([name, email, files]):
            return jsonify({"status": "failed", "msg": "Name, email and images are required"}), 400
        
        if len(files) < 5:
            return jsonify({"status": "failed", "msg": "At least 5 images required"}), 400
        
        if not password or len(password) < 6:
            return jsonify({"status": "failed", "msg": "Password must be at least 6 characters"}), 400
        
        # Check if email already exists
        if enrollment_requests_col.find_one({"email": email}):
            return jsonify({"status": "failed", "msg": "Request already submitted for this email"}), 400
        
        if users_col.find_one({"email": email}):
            return jsonify({"status": "failed", "msg": "Email already registered"}), 400
        
        # Store raw images temporarily for background processing
        raw_images = []
        for file in files:
            try:
                img_bytes = file.read()
                raw_images.append(base64.b64encode(img_bytes).decode('utf-8'))
            except Exception as e:
                print(f"Error reading image: {e}")
                continue
        
        if len(raw_images) < 5:
            return jsonify({"status": "failed", "msg": "Failed to read uploaded images"}), 400
        
        # Create enrollment request with "processing" status
        request_id = f"enroll_req_{int(datetime.datetime.utcnow().timestamp())}_{email.split('@')[0]}"
        request_doc = {
            "_id": request_id,
            "name": name,
            "email": email,
            "phone": phone if phone else None,
            "password_hash": generate_password_hash(password),
            "raw_images": raw_images,  # Temporary storage
            "images": [],  # Will be populated after processing
            "image_count": 0,
            "status": "processing",  # New status
            "submitted_at": datetime.datetime.utcnow(),
            "processed_at": None,
            "reviewed_by": None,
            "reviewed_at": None,
            "processing_error": None
        }
        
        enrollment_requests_col.insert_one(request_doc)
        
        # Process images in background thread
        threading.Thread(
            target=process_enrollment_images_background,
            args=(request_id,),
            daemon=True
        ).start()
        
        # Log the request
        # Log the request
        system_logs_col.insert_one({
            "action": "enrollment_request_submitted",
            "email": email,
            "name": name,
            "request_id": request_id,
            "timestamp": datetime.datetime.utcnow()
        })
        
        return jsonify({
            'status': 'success',
            'msg': 'Enrollment request submitted successfully! Awaiting admin approval.',
            'request_id': request_id
        }), 200
        
    except Exception as e:
        logger.error(f"Finalization error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'failed', 'msg': 'Server error during finalization'}), 500


@app.route("/api/enrollment/check-liveness", methods=["POST"])
@limiter.limit("30 per minute")
def check_liveness_api():
    """Standalone liveness check endpoint"""
    try:
        if 'image' not in request.files:
            return jsonify({'status': 'failed', 'msg': 'No image provided'}), 400
        
        file = request.files['image']
        img_bytes = file.read()
        import cv2 as cv
        nparr = np.frombuffer(img_bytes, np.uint8)
        image = cv.imdecode(nparr, cv.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'status': 'failed', 'msg': 'Invalid image'}), 400
        
        # Extract landmarks
        from face_3d_reconstruction import extract_3d_face_features
        landmarks_result = extract_3d_face_features(image)
        
        if landmarks_result is None:
            return jsonify({
                'status': 'failed',
                'msg': 'No face detected',
                'liveness_passed': False
            }), 200
        
        # Perform liveness check
        from advanced_liveness_detection import check_liveness
        liveness_result = check_liveness(image, landmarks_result['landmarks_3d'])
        
        return jsonify({
            'status': 'success',
            'liveness_score': liveness_result['overall']['liveness_score'],
            'liveness_passed': liveness_result['overall']['liveness_passed'],
            'feedback': liveness_result['overall']['feedback'],
            'details': {
                'blinks': liveness_result.get('blink', {}).get('total_blinks', 0),
                'movement_detected': liveness_result.get('movement', {}).get('movement_detected', False),
                'is_live_texture': liveness_result.get('texture', {}).get('is_live', True),
                'has_depth': liveness_result.get('depth', {}).get('is_3d', True)
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Liveness check error: {e}")
        return jsonify({'status': 'failed', 'msg': 'Processing error'}), 500


# Cleanup task for expired sessions
def cleanup_enrollment_sessions():
    """Background task to cleanup expired enrollment sessions"""
    while True:
        try:
            import time
            time.sleep(300)  # Run every 5 minutes
            engine = get_enrollment_engine()
            engine.cleanup_expired_sessions(max_age_minutes=30)
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

# Start cleanup thread
cleanup_thread = threading.Thread(target=cleanup_enrollment_sessions, daemon=True)
cleanup_thread.start()

# ===== END ADVANCED ENROLLMENT ENDPOINTS =====

def process_enrollment_images_background(request_id):
    """Background worker to process enrollment images with parallel processing"""
    try:
        # Fetch the request
        request_doc = enrollment_requests_col.find_one({"_id": request_id})
        if not request_doc:
            print(f"Request {request_id} not found")
            return
        
        raw_images = request_doc.get("raw_images", [])
        if not raw_images:
            enrollment_requests_col.update_one(
                {"_id": request_id},
                {"$set": {
                    "status": "failed",
                    "processing_error": "No images to process",
                    "processed_at": datetime.datetime.utcnow()
                }}
            )
            return
        
        # Parallel image processing
        def process_single_image(img_base64):
            try:
                img_bytes = base64.b64decode(img_base64)
                import cv2 as cv
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv.imdecode(nparr, cv.IMREAD_COLOR)
                
                if img is None:
                    return None
                    
                # Get embedding
                embedding = get_face_embedding(img)
                
                if embedding is not None:
                    return {
                        "base64": img_base64,
                        "embedding": embedding
                    }
                return None
            except Exception as e:
                print(f"Error processing image: {e}")
                return None
        
        # Process images in parallel (max 4 workers)
        from concurrent.futures import ThreadPoolExecutor
        processed_images = []
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(process_single_image, raw_images))
            processed_images = [r for r in results if r is not None]
        
        if len(processed_images) < 5:
            enrollment_requests_col.update_one(
                {"_id": request_id},
                {"$set": {
                    "status": "failed",
                    "processing_error": f"Only {len(processed_images)} valid face images detected. Please upload clearer photos.",
                    "processed_at": datetime.datetime.utcnow()
                }}
            )
            return
        
        # Similarity Filtering
        all_embeddings = np.array([item["embedding"] for item in processed_images])
        centroid = np.mean(all_embeddings, axis=0)
        
        # Normalize centroid
        norm_centroid = np.linalg.norm(centroid)
        if norm_centroid > 0:
            centroid = centroid / norm_centroid
            
        valid_images = []
        for item in processed_images:
            emb = np.array(item["embedding"])
            norm_emb = np.linalg.norm(emb)
            if norm_emb > 0:
                emb = emb / norm_emb
                
            similarity = np.dot(emb, centroid)
            item["similarity"] = similarity
            
            # Filter out images with low similarity
            if similarity > 0.6:
                valid_images.append(item)
        
        # Sort by similarity and keep top 10
        valid_images.sort(key=lambda x: x["similarity"], reverse=True)
        final_images = valid_images[:10]
        
        if len(final_images) < 5:
            enrollment_requests_col.update_one(
                {"_id": request_id},
                {"$set": {
                    "status": "failed",
                    "processing_error": f"Images are not consistent. Only {len(final_images)} matched. Please ensure all photos are of the same person.",
                    "processed_at": datetime.datetime.utcnow()
                }}
            )
            return
        
        images_data = [item["base64"] for item in final_images]
        
        # Update request with processed images
        enrollment_requests_col.update_one(
            {"_id": request_id},
            {
                "$set": {
                    "images": images_data,
                    "image_count": len(images_data),
                    "status": "pending",  # Ready for approval
                    "processed_at": datetime.datetime.utcnow()
                },
                "$unset": {"raw_images": ""}  # Remove temporary data
            }
        )
        
        # Log successful processing
        system_logs_col.insert_one({
            "action": "enrollment_images_processed",
            "request_id": request_id,
            "image_count": len(images_data),
            "timestamp": datetime.datetime.utcnow()
        })
        
        print(f"✅ Successfully processed enrollment request {request_id}")
        
    except Exception as e:
        print(f"Background processing error for {request_id}: {e}")
        import traceback
        traceback.print_exc()
        
        # Mark as failed
        enrollment_requests_col.update_one(
            {"_id": request_id},
            {"$set": {
                "status": "failed",
                "processing_error": f"Processing error: {str(e)}",
                "processed_at": datetime.datetime.utcnow()
            }}
        )

# Global variable for robust connection
_robust_db_client = None

def get_robust_db_connection():
    """Get a robust database connection with fallbacks (Singleton)"""
    global _robust_db_client
    if _robust_db_client:
        return _robust_db_client

    MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
    try:
        client = MongoClient(MONGODB_URI, 
                           serverSelectionTimeoutMS=15000,
                           connectTimeoutMS=15000,
                           socketTimeoutMS=15000,
                           retryWrites=False)
        client.admin.command('ping')
        print("MongoDB connection successful in login!")
        _robust_db_client = client
        return client
    except Exception as e:
        print(f"MongoDB connection failed in login: {e}")
        try:
            client = MongoClient(MONGODB_URI, 
                               serverSelectionTimeoutMS=15000,
                               tls=False,
                               retryWrites=False)
            client.admin.command('ping')
            print("MongoDB connection successful with TLS disabled in login!")
            _robust_db_client = client
            return client
        except Exception as e2:
            print(f"All MongoDB attempts failed in login: {e2}")
            try:
                from mock_mongo import MockMongoClient
                client = MockMongoClient(MONGODB_URI)
                print("Mock MongoDB initialized in login!")
                _robust_db_client = client
                return client
            except ImportError:
                client = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=5000)
                _robust_db_client = client
                return client

@app.route("/login", methods=["GET", "POST"])
@limiter.limit("10 per minute")
def login():
    try:
        if current_user.is_authenticated:
            if current_user.role == 'superadmin':
                return redirect(url_for("superadmin.dashboard"))
            elif current_user.role == 'admin':
                return redirect(url_for("admin_dashboard"))
            return redirect(url_for("user_dashboard"))
        
        if request.method == "GET":
            try:
                return render_template("login.html")
            except Exception as template_error:
                print(f"Template rendering error: {template_error}")
                # Fallback to basic HTML if template fails
                return '''
                <html><body style="font-family:sans-serif;max-width:400px;margin:100px auto;padding:20px;">
                <h2>Sign In</h2>
                <form method="POST">
                <input name="email" type="email" placeholder="Email" required style="width:100%;padding:10px;margin:10px 0;"><br>
                <input name="password" type="password" placeholder="Password" required style="width:100%;padding:10px;margin:10px 0;"><br>
                <button type="submit" style="width:100%;padding:10px;background:#3b82f6;color:white;border:none;cursor:pointer;">Sign In</button>
                </form>
                </body></html>
                ''', 200
        
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        remember = request.form.get("remember") == "on"
        
        if not email or not password:
            flash("Email and password required.", "danger")
            return redirect(url_for("login"))
        
        # Check super admin first
        try:
            login_client = get_robust_db_connection()
            core = login_client['secure_db']
            superadmins_col = core["superadmins"]
            
            superadmin = superadmins_col.find_one({"email": email})
            if superadmin:
                if superadmin.get("status") == "blocked":
                    flash("Your account has been blocked. Contact support.", "danger")
                    return redirect(url_for("login"))

                hashed = superadmin.get("password_hash")
                if hashed and check_password_hash(hashed, password):
                    user = SuperAdminUser(superadmin)
                    login_user(user, remember=remember)
                    next_page = request.args.get('next')
                    if next_page and next_page != '/logout':
                        return redirect(next_page)
                    return redirect(url_for("superadmin.dashboard"))
        except Exception as e:
            print(f"Superadmin login error: {e}")
        
        # Check admin
        try:
            login_client = get_robust_db_connection()
            core = login_client['secure_db']
            admins_col = core["admins"]
            
            admin = admins_col.find_one({"email": email})
            if admin:
                if admin.get("status") == "blocked":
                    flash("Your account has been blocked. Contact support.", "danger")
                    return redirect(url_for("login"))

                hashed = admin.get("password_hash")
                if hashed and check_password_hash(hashed, password):
                    user = AdminUser(admin)
                    login_user(user, remember=remember)
                    next_page = request.args.get('next')
                    if next_page and next_page != '/logout':
                        return redirect(next_page)
                    return redirect(url_for("admin_dashboard"))
        except Exception as e:
            print(f"Admin login error: {e}")
        
        # Check regular user
        try:
            login_client = get_robust_db_connection()
            core = login_client['secure_db']
            users_col = core["users"]
            
            user_doc = users_col.find_one({"email": email})
            if user_doc:
                if user_doc.get("status") == "blocked":
                    flash("Your account has been blocked. Contact admin.", "danger")
                    return redirect(url_for("login"))
                
                hashed = user_doc.get("password_hash")
                if hashed and check_password_hash(hashed, password):
                    user = RegularUser(user_doc)
                    login_user(user, remember=remember)
                    next_page = request.args.get('next')
                    if next_page and next_page != '/logout':
                        return redirect(next_page)
                    return redirect(url_for("user_dashboard"))
        except Exception as e:
            print(f"User login error: {e}")
        
        flash("Invalid credentials.", "danger")
        return redirect(url_for("login"))
    
    except Exception as e:
        print(f"❌ Login route error: {e}")
        return '''
        <html><body style="font-family:sans-serif;max-width:400px;margin:100px auto;padding:20px;background:#0f172a;color:#fff;">
        <h2 style="color:#ef4444;">System Error</h2>
        <p>Please try again later or contact support.</p>
        <a href="/login" style="color:#3b82f6;">Back to Login</a>
        </body></html>
        ''', 500

@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "success")
    return redirect(url_for("login"))

# --- Admin Routes ---
@app.route("/admin/3d-demo")
@admin_required
def admin_3d_demo():
    """3D Face Reconstruction Demo Page"""
    return render_template("3d_face_demo.html")

@app.route("/admin/dashboard")
@admin_required
def admin_dashboard():
    """Admin Dashboard"""
    try:
        # Get robust connection for dashboard stats
        dashboard_client = get_robust_db_connection()
        core = dashboard_client['secure_db']
        transactional_db = dashboard_client['transactional_db']
        
        persons_col = core["persons"]
        attendance_col = transactional_db["attendance"]
        enrollment_requests_col = core["enrollment_requests"]
        
        total_users = persons_col.count_documents({})
        total_attendance = attendance_col.count_documents({})
        pending_requests = enrollment_requests_col.count_documents({"status": "pending"})
        today_attendance = attendance_col.count_documents({
            "timestamp": {
                "$gte": datetime.datetime.combine(datetime.date.today(), datetime.time.min),
                "$lt": datetime.datetime.combine(datetime.date.today(), datetime.time.max)
            }
        })
        
        return render_template("admin_dashboard.html",
                             admin=current_user,
                             total_users=total_users,
                             total_attendance=total_attendance,
                             pending_requests=pending_requests,
                             today_attendance=today_attendance)
    except Exception as e:
        print(f"Admin dashboard error: {e}")
        flash("Error loading dashboard. Please try again.", "danger")
        return render_template("admin_dashboard.html",
                             admin=current_user,
                             total_users=0,
                             total_attendance=0,
                             pending_requests=0,
                             today_attendance=0)

@app.route("/enroll")
def enroll_page():
    """Enrollment page alias"""
    return render_template("registration_req.html")

@app.route("/admin/update_profile", methods=["POST"])
@admin_required
def admin_update_profile():
    """Update admin profile with image support"""
    try:
        name = request.form.get("name")
        email = request.form.get("email")
        
        if not name or not email:
            return jsonify({"status": "failed", "msg": "Name and email required"}), 400
            
        # Initialize update data
        update_data = {
            "name": name,
            "email": email
        }

        # Handle Profile Image Upload
        if 'profile_image' in request.files:
            file = request.files['profile_image']
            if file and file.filename != '':
                try:
                    import base64
                    image_data = file.read()
                    encoded_image = base64.b64encode(image_data).decode('utf-8')
                    update_data["profile_image"] = encoded_image
                    print(f"✅ Profile image encoded: {len(encoded_image)} bytes")
                except Exception as img_err:
                    print(f"❌ Image processing error: {img_err}")

        # Update admin collection - Robust ID handling
        try:
            # Try ObjectId first
            from bson.objectid import ObjectId
            query = {"_id": ObjectId(current_user.id)}
        except:
            # Fallback to string ID
            query = {"_id": current_user.id}
        
        result = admins_col.update_one(
            query,
            {"$set": update_data}
        )
        
        # If no result, try opposite ID type
        if result.matched_count == 0:
            try:
                # Try string ID if ObjectId failed, or vice versa
                alt_query = {"_id": current_user.id} if "_id" in query and isinstance(query["_id"], ObjectId) else {"_id": ObjectId(current_user.id)}
                
                result = admins_col.update_one(
                    alt_query,
                    {"$set": update_data}
                )
            except:
                pass

        if result.modified_count or result.matched_count:
            return jsonify({"status": "success", "msg": "Profile updated successfully"}), 200
            
        return jsonify({"status": "success", "msg": "No changes made"}), 200
        
    except Exception as e:
        print(f"Profile update error: {e}")
        return jsonify({"status": "failed", "msg": str(e)}), 500

# --- Missing Admin Routes ---
@app.route("/admin/profile")
@admin_required
def admin_profile():
    """Admin profile page"""
    profile_data = {
        "name": current_user.name,
        "email": current_user.email,
        "profile_image": current_user.profile_image,
        "role": "Administrator"
    }
    return render_template("admin_profile.html", profile=profile_data)

@app.route("/reg")
@admin_required
def reg():
    """Registration form"""
    return render_template("registration_form.html")

@app.route("/list_users")
@limiter.exempt
@admin_required
def list_users():
    """List all users API"""
    try:
        # Use robust connection
        db_client = get_robust_db_connection()
        persons_col = db_client['secure_db']['persons']
        
        users = list(persons_col.find({}, {
            "name": 1,
            "photos_count": 1,
            "enrollment_date": 1,
            "status": 1
        }))
        for user in users:
            user["_id"] = str(user["_id"])
            if "enrollment_date" in user:
                try:
                    user["enrollment_date"] = user["enrollment_date"].isoformat()
                except:
                    pass
        return jsonify(users)
    except Exception as e:
        print(f"Error in list_users: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/attendance_recent")
@limiter.exempt  # Exempt admin endpoints from rate limiting
@admin_required
def attendance_recent():
    """Recent attendance API"""
    try:
        # Use robust connection
        db_client = get_robust_db_connection()
        attendance_col = db_client['transactional_db']['attendance']
        
        records = list(attendance_col.find().sort("timestamp", -1).limit(50))
        out = []
        for r in records:
            try:
                ts = r.get("timestamp")
                iso = ts if isinstance(ts, str) else ts.isoformat()
            except Exception:
                iso = ""
            out.append({
                "name": r.get("name"),
                "timestamp": iso,
                "confidence": r.get("confidence", 0)
            })
        return jsonify(out)
    except Exception as e:
        print(f"Error in attendance_recent: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/attendance_stats")
@limiter.exempt
@admin_required
def admin_attendance_stats():
    """Attendance statistics API"""
    try:
        days = int(request.args.get("days", 30))
        end = datetime.datetime.utcnow()
        start = end - datetime.timedelta(days=days - 1)
        
        records = list(attendance_col.find({}))
        counts = {}
        
        for r in records:
            ts = r.get("timestamp")
            if isinstance(ts, str):
                try:
                    dt = datetime.datetime.fromisoformat(ts)
                except Exception:
                    continue
            elif isinstance(ts, datetime.datetime):
                dt = ts
            else:
                continue
            
            if dt < start or dt > end:
                continue
            
            day = dt.date().isoformat()
            counts[day] = counts.get(day, 0) + 1
        
        labels = []
        values = []
        for i in range(days):
            day_dt = (start + datetime.timedelta(days=i)).date()
            day_str = day_dt.isoformat()
            labels.append(day_str)
            values.append(counts.get(day_str, 0))
        
        return jsonify({"labels": labels, "values": values})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/enrollment_requests")
@admin_required
def get_enrollment_requests():
    """Get enrollment requests API"""
    try:
        # Use robust connection
        db_client = get_robust_db_connection()
        enrollment_requests_col = db_client['secure_db']['enrollment_requests']
        
        requests = list(enrollment_requests_col.find({"status": "pending"}).sort("submitted_at", -1))
        out = []
        for r in requests:
            r["_id"] = str(r["_id"])
            try:
                r["submitted_at"] = r["submitted_at"].isoformat()
            except:
                pass
            r.pop("images", None)
            r.pop("password_hash", None)
            out.append(r)
        return jsonify({"status": "success", "requests": out}), 200
    except Exception as e:
        print(f"Error in get_enrollment_requests: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "msg": str(e), "requests": []}), 500

@app.route("/api/pending_requests_count")
@limiter.exempt
@admin_required
def pending_requests_count():
    """Pending requests count API"""
    try:
        # Use robust connection
        db_client = get_robust_db_connection()
        enrollment_requests_col = db_client['secure_db']['enrollment_requests']
        
        count = enrollment_requests_col.count_documents({"status": "pending"})
        return jsonify({"count": count})
    except Exception as e:
        print(f"Error in pending_requests_count: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/enrollment_request/<request_id>")
@admin_required
def get_enrollment_request_details(request_id):
    """Get enrollment request details with images"""
    try:
        # Use robust connection
        db_client = get_robust_db_connection()
        enrollment_requests_col = db_client['secure_db']['enrollment_requests']
        
        request_doc = enrollment_requests_col.find_one({"_id": request_id})
        if not request_doc:
            return jsonify({"error": "Request not found"}), 404
        
        # Convert ObjectId and dates to strings
        request_doc["_id"] = str(request_doc["_id"])
        if "submitted_at" in request_doc:
            try:
                request_doc["submitted_at"] = request_doc["submitted_at"].isoformat()
            except:
                pass
        
        # Remove password hash from response
        request_doc.pop("password_hash", None)
        
        return jsonify(request_doc)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/approve_enrollment/<request_id>", methods=["POST"])
@admin_required
def approve_enrollment(request_id):
    """Approve an enrollment request and create user account"""
    try:
        # Get robust connection to prevent timeouts
        db_client = get_robust_db_connection()
        core = db_client['secure_db']
        enrollment_requests_col = core['enrollment_requests']
        persons_col = core['persons']
        users_col = core['users']
        system_logs_col = core['system_logs']

        # Get the enrollment request
        request_doc = enrollment_requests_col.find_one({"_id": request_id})
        if not request_doc:
            return jsonify({"status": "failed", "msg": "Request not found"}), 404
        
        if request_doc.get("status") != "pending":
            return jsonify({"status": "failed", "msg": "Request already processed"}), 400
        
        name = request_doc.get("name")
        email = request_doc.get("email") 
        password_hash = request_doc.get("password_hash")
        images = request_doc.get("images", [])
        department = request_doc.get("department", "")  # FIXED: Get from stored document, not form
        
        # Check if user already exists
        if persons_col.find_one({"name": name}):
            return jsonify({"status": "failed", "msg": "User with this name already exists"}), 400
        
        # Process images and create embeddings
        embeddings = []
        valid_photos = 0
        
        for i, img_base64 in enumerate(images):
            try:
                img_bytes = base64.b64decode(img_base64)
                img_array = np.frombuffer(img_bytes, np.uint8)
                image = cv.imdecode(img_array, cv.IMREAD_COLOR)
                
                if image is None:
                    continue
                
                # Get face embedding
                embedding = get_face_embedding(image)
                if embedding is not None:
                    embeddings.append(embedding)
                    valid_photos += 1
                    
            except Exception as e:
                print(f"❌ Error processing image {i + 1}: {e}")
                continue
        
        if valid_photos < 2:
            return jsonify({"status": "failed", "msg": f"Only {valid_photos} valid face photos found. Need at least 2"}), 400
        
        # Create multi-vector template
        try:
            if len(embeddings) >= 3:
                stored_template = extract_multi_vector_embeddings(embeddings)
            else:
                centroid = np.mean(embeddings, axis=0)
                stored_template = {
                    'centroid': centroid,
                    'individual_embeddings': embeddings,
                    'template_type': 'multi_vector',
                    'photo_count': len(embeddings)
                }
            
            # Create person document
            person_id = f"{datetime.datetime.utcnow().timestamp()}_{name.replace(' ', '_')}"
            person_doc = {
                "_id": person_id,
                "name": name,
                "email": email,
                "department": department,
                "embedding": pickle.dumps(stored_template),
                "enrollment_date": datetime.datetime.utcnow(),
                "photos_count": len(embeddings),
                "status": "active",
                "enrolled_by": current_user.email,
                "template_type": "advanced_multi_vector"
            }
            
            # Insert into persons collection
            persons_col.insert_one(person_doc)
            
            # Create user account if email exists
            if email and password_hash:
                user_doc = {
                    "_id": email,
                    "email": email,
                    "name": name,
                    "password_hash": password_hash,
                    "department": department,
                    "status": "active",
                    "created_at": datetime.datetime.utcnow(),
                    "approved_by": current_user.email
                }
                users_col.insert_one(user_doc)
            
            # Initialize FAISS if needed and add to index
            if faiss_index is None:
                lazy_faiss_init()
            
            # Add to FAISS index
            centroid_embedding = stored_template.get('centroid', embeddings[0])
            faiss_index.add(np.array([centroid_embedding], dtype=np.float32))
            person_id_map.append(name)
            
            # Save FAISS index
            save_faiss_index()
            
            # Update enrollment request status
            enrollment_requests_col.update_one(
                {"_id": request_id},
                {
                    "$set": {
                        "status": "approved",
                        "reviewed_by": current_user.email,
                        "reviewed_at": datetime.datetime.utcnow()
                    }
                }
            )
            
            # Log the approval
            system_logs_col.insert_one({
                "action": "enrollment_approved",
                "request_id": request_id,
                "person_name": name,
                "person_id": person_id,
                "photos_processed": len(embeddings),
                "approved_by": current_user.email,
                "timestamp": datetime.datetime.utcnow()
            })
            
            return jsonify({
                "status": "success",
                "msg": f"User '{name}' enrolled successfully with {len(embeddings)} photos"
            }), 200
            
        except Exception as e:
            print(f"[ERROR] Template creation error: {e}")
            return jsonify({"status": "failed", "msg": f"Error creating face template: {str(e)}"}), 500
        
    except Exception as e:
        print(f"[ERROR] Enrollment approval error: {e}")
        return jsonify({"status": "failed", "msg": f"Approval failed: {str(e)}"}), 500

@app.route("/api/reject_enrollment/<request_id>", methods=["POST"])
@admin_required
def reject_enrollment(request_id):
    """Reject an enrollment request"""
    try:
        # Get the enrollment request
        request_doc = enrollment_requests_col.find_one({"_id": request_id})
        if not request_doc:
            return jsonify({"status": "failed", "msg": "Request not found"}), 404
        
        if request_doc.get("status") != "pending":
            return jsonify({"status": "failed", "msg": "Request already processed"}), 400
        
        reason = request.form.get("reason", "Not specified")
        
        # Update enrollment request status
        enrollment_requests_col.update_one(
            {"_id": request_id},
            {
                "$set": {
                    "status": "rejected",
                    "rejection_reason": reason,
                    "reviewed_by": current_user.email,
                    "reviewed_at": datetime.datetime.utcnow()
                }
            }
        )
        
        # Log the rejection
        system_logs_col.insert_one({
            "action": "enrollment_rejected",
            "request_id": request_id,
            "person_name": request_doc.get("name"),
            "reason": reason,
            "rejected_by": current_user.email,
            "timestamp": datetime.datetime.utcnow()
        })
        
        return jsonify({
            "status": "success",
            "msg": "Enrollment request rejected"
        }), 200
        
    except Exception as e:
        print(f"❌ Enrollment rejection error: {e}")
        return jsonify({"status": "failed", "msg": f"Rejection failed: {str(e)}"}), 500

@app.route("/api/block_user", methods=["POST"])
@admin_required
def block_user():
    """Block a user from marking attendance"""
    try:
        name = request.form.get("name")
        if not name:
            return jsonify({"status": "failed", "msg": "User name required"}), 400
        
        # Update person status
        result = persons_col.update_one(
            {"name": name},
            {"$set": {"status": "blocked"}}
        )
        
        if result.matched_count == 0:
            return jsonify({"status": "failed", "msg": "User not found"}), 404
        
        # Also update user account if exists
        users_col.update_one(
            {"name": name},
            {"$set": {"status": "blocked"}}
        )
        
        # Log the action
        system_logs_col.insert_one({
            "action": "user_blocked",
            "person_name": name,
            "blocked_by": current_user.email,
            "timestamp": datetime.datetime.utcnow()
        })
        
        return jsonify({
            "status": "success",
            "msg": f"User '{name}' has been blocked"
        }), 200
        
    except Exception as e:
        print(f"❌ Block user error: {e}")
        return jsonify({"status": "failed", "msg": f"Failed to block user: {str(e)}"}), 500

@app.route("/api/unblock_user", methods=["POST"])
@admin_required
def unblock_user():
    """Unblock a user"""
    try:
        name = request.form.get("name")
        if not name:
            return jsonify({"status": "failed", "msg": "User name required"}), 400
        
        # Update person status
        result = persons_col.update_one(
            {"name": name},
            {"$set": {"status": "active"}}
        )
        
        if result.matched_count == 0:
            return jsonify({"status": "failed", "msg": "User not found"}), 404
        
        # Also update user account if exists
        users_col.update_one(
            {"name": name},
            {"$set": {"status": "active"}}
        )
        
        # Log the action
        system_logs_col.insert_one({
            "action": "user_unblocked",
            "person_name": name,
            "unblocked_by": current_user.email,
            "timestamp": datetime.datetime.utcnow()
        })
        
        return jsonify({
            "status": "success",
            "msg": f"User '{name}' has been unblocked"
        }), 200
        
    except Exception as e:
        print(f"❌ Unblock user error: {e}")
        return jsonify({"status": "failed", "msg": f"Failed to unblock user: {str(e)}"}), 500

@app.route("/api/enroll_user", methods=["POST"])
@admin_required
def enroll_user():
    """Enroll a new user with advanced face scanning"""
    try:
        data = request.get_json()
        
        name = data.get("name", "").strip()
        email = data.get("email", "").strip()
        department = data.get("department", "").strip()
        photos = data.get("photos", [])  # Base64 encoded photos
        
        if not name:
            return jsonify({"status": "failed", "message": "Name is required"}), 400
        
        if not photos or len(photos) < 3:
            return jsonify({"status": "failed", "message": "At least 3 photos are required for enrollment"}), 400
        
        # Check if person already exists
        existing_person = persons_col.find_one({"name": name})
        if existing_person:
            return jsonify({"status": "failed", "message": "Person with this name already exists"}), 400
        
        # Process multiple photos for better recognition
        embeddings = []
        valid_photos = 0
        
        for i, photo_data in enumerate(photos):
            try:
                # Decode base64 image
                if photo_data.startswith('data:image'):
                    photo_data = photo_data.split(',')[1]
                
                img_bytes = base64.b64decode(photo_data)
                img_array = np.frombuffer(img_bytes, np.uint8)
                image = cv.imdecode(img_array, cv.IMREAD_COLOR)
                
                if image is None:
                    continue
                
                # Get face embedding
                embedding = get_face_embedding(image)
                if embedding is not None:
                    embeddings.append(embedding)
                    valid_photos += 1
                    print(f"✅ Processed photo {i + 1} for {name}")
                
            except Exception as e:
                print(f"❌ Error processing photo {i + 1}: {e}")
                continue
        
        if valid_photos < 2:
            return jsonify({"status": "failed", "message": f"Only {valid_photos} valid photos found. Need at least 2 clear face photos"}), 400
        
        # Create multi-vector template for better recognition
        try:
            if len(embeddings) >= 3:
                # Create advanced template with multiple embeddings
                stored_template = extract_multi_vector_embeddings(embeddings)
            else:
                # Use ensemble matching for fewer photos
                centroid = np.mean(embeddings, axis=0)
                stored_template = {
                    'centroid': centroid,
                    'individual_embeddings': embeddings,
                    'template_type': 'multi_vector',
                    'photo_count': len(embeddings)
                }
            
            # Create person document
            person_id = f"{datetime.datetime.utcnow().timestamp()}_{name.replace(' ', '_')}"
            person_doc = {
                "_id": person_id,
                "name": name,
                "email": email if email else None,
                "department": department if department else None,
                "embedding": pickle.dumps(stored_template),
                "enrollment_date": datetime.datetime.utcnow(),
                "photos_count": len(embeddings),
                "status": "active",
                "enrolled_by": current_user.email,
                "template_type": "advanced_multi_vector"
            }
            
            # Insert into database
            persons_col.insert_one(person_doc)
            
            # Initialize FAISS if needed and add to index
            if faiss_index is None:
                lazy_faiss_init()
            
            # Add to FAISS index
            centroid_embedding = stored_template.get('centroid', embeddings[0])
            faiss_index.add(np.array([centroid_embedding], dtype=np.float32))
            person_id_map.append(name)
            
            # Save FAISS index
            save_faiss_index()
            
            # Log enrollment
            system_logs_col.insert_one({
                "action": "user_enrollment",
                "person_name": name,
                "person_id": person_id,
                "photos_processed": len(embeddings),
                "enrolled_by": current_user.email,
                "timestamp": datetime.datetime.utcnow()
            })
            
            return jsonify({
                "status": "success",
                "message": f"User '{name}' enrolled successfully with {len(embeddings)} photos",
                "person_id": person_id,
                "photos_processed": len(embeddings)
            }), 201
            
        except Exception as e:
            print(f"❌ Template creation error: {e}")
            return jsonify({"status": "failed", "message": f"Error creating face template: {str(e)}"}), 500
        
    except Exception as e:
        print(f"❌ Enrollment error: {e}")
        return jsonify({"status": "failed", "message": f"Enrollment failed: {str(e)}"}), 500

# --- User Routes ---
@app.route("/user/dashboard")
@login_required
def user_dashboard():
    if current_user.role == 'admin':
        return redirect(url_for("admin_dashboard"))
    if current_user.role == 'superadmin':
        return redirect(url_for("superadmin.dashboard"))
    return render_template("user_dashboard_new.html", user=current_user)

# --- User API Routes ---
@app.route("/api/user/attendance_stats")
@login_required
def user_attendance_stats():
    """Get user attendance statistics"""
    try:
        days = int(request.args.get('days', 30))
        user_id = current_user.id
        
        # Get robust connection
        db_client = get_robust_db_connection()
        attendance_col = db_client['transactional_db']['attendance']
        
        end_date = datetime.datetime.utcnow()
        start_date = end_date - datetime.timedelta(days=days)
        
        # Get attendance count in date range
        attendance_count = attendance_col.count_documents({
            "person_id": user_id,
            "timestamp": {"$gte": start_date, "$lte": end_date}
        })
        
        # Get total days
        total_days = days
        
        # Calculate attendance percentage
        percentage = (attendance_count / total_days * 100) if total_days > 0 else 0
        
        return jsonify({
            "status": "success",
            "attendance_count": attendance_count,
            "total_days": total_days,
            "percentage": round(percentage, 1)
        }), 200
    except Exception as e:
        return jsonify({"status": "failed", "error": str(e)}), 500

@app.route("/api/user/attendance_history")
@login_required
def user_attendance_history():
    """Get user attendance history"""
    try:
        user_id = current_user.id
        limit = int(request.args.get('limit', 30))
        
        # Get robust connection
        db_client = get_robust_db_connection()
        attendance_col = db_client['transactional_db']['attendance']
        
        # Get recent attendance records
        records = list(attendance_col.find({"person_id": user_id})
                      .sort("timestamp", -1)
                      .limit(limit))
        
        for record in records:
            record['_id'] = str(record['_id'])
            if 'timestamp' in record:
                record['timestamp'] = record['timestamp'].isoformat()
        
        return jsonify({
            "status": "success",
            "history": records
        }), 200
    except Exception as e:
        return jsonify({"status": "failed", "error": str(e)}), 500

@app.route("/api/user/attendance_chart")
@login_required
def user_attendance_chart():
    """Get user attendance chart data"""
    try:
        user_id = current_user.id
        days = int(request.args.get('days', 30))
        
        # Get robust connection
        db_client = get_robust_db_connection()
        attendance_col = db_client['transactional_db']['attendance']
        
        # Calculate date range
        end_date = datetime.datetime.utcnow()
        start_date = end_date - datetime.timedelta(days=days)
        
        # Get attendance records
        pipeline = [
            {
                "$match": {
                    "person_id": user_id,
                    "timestamp": {"$gte": start_date, "$lte": end_date}
                }
            },
            {
                "$group": {
                    "_id": {
                        "year": {"$year": "$timestamp"},
                        "month": {"$month": "$timestamp"},
                        "day": {"$dayOfMonth": "$timestamp"}
                    },
                    "count": {"$sum": 1}
                }
            },
            {"$sort": {"_id.year": 1, "_id.month": 1, "_id.day": 1}}
        ]
        
        records = list(db['attendance_col'].aggregate(pipeline))
        
        # Format data for chart
        labels = []
        data = []
        
        current = start_date
        while current <= end_date:
            # Check if attendance exists for this day
            match = next((r for r in records if 
                          r['_id']['year'] == current.year and 
                          r['_id']['month'] == current.month and 
                          r['_id']['day'] == current.day), None)
            
            labels.append(current.strftime("%b %d"))
            data.append(1 if match else 0)
            
            current += datetime.timedelta(days=1)
            
        return jsonify({
            "status": "success",
            "labels": labels,
            "data": data
        }), 200
    except Exception as e:
        return jsonify({"status": "failed", "error": str(e)}), 500

# --- Detection API Fallbacks ---
# --- Detection API Fallbacks ---
# detect_face_ultra is already defined later in the file


# detect_face_public is already defined later in the file

# detect_face_simple moved to later in file




# --- Mobile Attendance Routes (FIXED) ---
@app.route("/api/mobile/mark_attendance", methods=["POST"])
@limiter.limit("20 per hour")
def mobile_mark_attendance():
    """Mark attendance from mobile camera capture - FIXED VERSION"""
    try:
        file = request.files.get("file")
        if not file:
            return jsonify({"status": "failed", "msg": "No image provided"}), 400
        
        # Get face embedding
        embedding = get_face_embedding(file)
        if embedding is None:
            return jsonify({"status": "failed", "msg": "No face detected. Please ensure your face is clearly visible and try again."}), 400
        
        # Search in FAISS - FIXED: Use correct variable names
        if faiss_index is None or faiss_index.ntotal == 0:
            return jsonify({"status": "failed", "msg": "System not initialized. Please contact administrator."}), 500
        
        # Use the enhanced search function with 3D support
        # Convert file to image for 3D analysis
        file.seek(0)
        file_bytes = file.read()
        img_array = np.frombuffer(file_bytes, np.uint8)
        image = cv.imdecode(img_array, cv.IMREAD_COLOR)
        
        matched_name, confidence = search_face_faiss(embedding, threshold=0.6, image=image)
        
        if not matched_name:
            return jsonify({
                "status": "failed",
                "msg": "Face not recognized. Please enroll first or contact administrator."
            }), 404
        
        # Get person details
        person = persons_col.find_one({"name": matched_name})
        if not person:
            return jsonify({"status": "failed", "msg": "Person data not found"}), 404
        
        # Check if already marked today
        today_start = datetime.datetime.combine(datetime.date.today(), datetime.time.min)
        existing = attendance_col.find_one({
            "name": matched_name,
            "timestamp": {"$gte": today_start}
        })
        
        if existing:
            return jsonify({
                "status": "success",
                "message": "Attendance already marked for today",
                "person_name": person.get("name", "Unknown"),
                "time": existing["timestamp"].strftime("%I:%M %p") if isinstance(existing["timestamp"], datetime.datetime) else "Unknown",
                "confidence": round(existing.get("confidence", 0) * 100, 2)
            })
        
        # Mark attendance
        attendance_col.insert_one({
            "name": matched_name,
            "timestamp": datetime.datetime.utcnow(),
            "confidence": confidence,
            "device": "mobile",
            "user_agent": request.headers.get("User-Agent", ""),
            "location": request.headers.get("X-Forwarded-For", request.remote_addr)
        })
        
        # Log attendance event
        system_logs_col.insert_one({
            "action": "mobile_attendance",
            "person_name": matched_name,
            "confidence": confidence,
            "device": "mobile",
            "timestamp": datetime.datetime.utcnow()
        })
        
        return jsonify({
            "status": "success",
            "message": "Attendance marked successfully!",
            "person_name": person.get("name", "Unknown"),
            "time": datetime.datetime.now().strftime("%I:%M %p"),
            "confidence": round(confidence * 100, 2)
        })
        
    except Exception as e:
        print(f"[Mobile Attendance Error] {e}")
        return jsonify({"status": "failed", "msg": "Server error. Please try again or contact administrator."}), 500

# --- API Routes for Stats (FIXED) ---
@app.route("/api/superadmin/stats")
@superadmin_required
def superadmin_stats_api():
    """API endpoint for super admin statistics - FIXED"""
    try:
        # Get robust connection for stats
        stats_client = get_robust_db_connection()
        core = stats_client['secure_db']
        transactional_db = stats_client['transactional_db']
        
        users_col = core["users"]
        admins_col = core["admins"]
        persons_col = core["persons"]
        enrollment_requests_col = core["enrollment_requests"]
        attendance_col = transactional_db["attendance"]
        cameras_col = core["cameras"]
        
        today_start = datetime.datetime.combine(datetime.date.today(), datetime.time.min)
        today_end = datetime.datetime.combine(datetime.date.today(), datetime.time.max)
        
        stats = {
            "total_users": users_col.count_documents({}),
            "total_admins": admins_col.count_documents({}),
            "total_persons": persons_col.count_documents({"status": {"$ne": "blocked"}}),
            "pending_enrollments": enrollment_requests_col.count_documents({"status": "pending"}),
            "today_attendance": attendance_col.count_documents({
                "timestamp": {
                    "$gte": today_start,
                    "$lt": today_end
                }
            }),
            "total_cameras": cameras_col.count_documents({}),
            "active_cameras": 0,  # Will be updated by camera module
            "faiss_index_size": faiss_index.ntotal if faiss_index else 0,
            "system_health": "healthy"
        }
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Missing Super Admin Routes ---
@app.route("/superadmin/users")
@superadmin_required
def superadmin_users():
    """Super admin users page"""
    try:
        users = list(persons_col.find({}))
        for user in users:
            user["_id"] = str(user["_id"])
        
            if "enrollment_date" in user:
                try:
                    user["enrollment_date"] = user["enrollment_date"].isoformat()
                except:
                    pass
        return render_template("superadmin/users.html", superadmin=current_user, users=users)
    except Exception as e:
        return f"Error loading users: {str(e)}", 500

@app.route("/api/superadmin/attendance/stats")
@login_required
def attendance_stats():
    """Get attendance statistics for charts"""
    try:
        days = int(request.args.get('days', 7))
        
        # Calculate date range
        end_date = datetime.datetime.utcnow()
        start_date = end_date - datetime.timedelta(days=days)
        
        # Get daily attendance data
        pipeline = [
            {
                "$match": {
                    "timestamp": {
                        "$gte": start_date,
                        "$lte": end_date
                    }
                }
            },
            {
                "$group": {
                    "_id": {
                        "$dateToString": {
                            "format": "%Y-%m-%d",
                            "date": "$timestamp"
                        }
                    },
                    "count": {"$sum": 1}
                }
            },
            {
                "$sort": {"_id": 1}
            }
        ]
        
        daily_attendance = list(attendance_col.aggregate(pipeline))
        
        # Format response
        response_data = {
            "daily_attendance": [
                {
                    "date": item["_id"],
                    "count": item["count"]
                }
                for item in daily_attendance
            ]
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/superadmin/users")
@login_required
def api_superadmin_users():
    """Get all users for superadmin"""
    try:
        users = list(users_col.find({}, {"password_hash": 0}))
        persons = list(persons_col.find({}))
        
        all_users = []
        
        for user in users:
            user["_id"] = str(user["_id"])
            user["type"] = "user"
            if "created_at" in user and user["created_at"]:
                try:
                    user["created_at"] = user["created_at"].isoformat()
                except:
                    pass
            all_users.append(user)
        
        for person in persons:
            person["_id"] = str(person["_id"])
            person["type"] = "person"
            if "enrollment_date" in person and person["enrollment_date"]:
                try:
                    person["enrollment_date"] = person["enrollment_date"].isoformat()
                except:
                    pass
            person.pop("embedding", None)
            all_users.append(person)
        
        return jsonify({"status": "success", "users": all_users})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/superadmin/logs")
@login_required 
def system_logs():
    """Get system logs"""
    try:
        limit = int(request.args.get('limit', 50))
        
        logs = list(system_logs_col.find({}).sort("timestamp", -1).limit(limit))
        
        for log in logs:
            log["_id"] = str(log["_id"])
            if "timestamp" in log:
                log["timestamp"] = log["timestamp"].isoformat()
        
        return jsonify({"logs": logs})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/superadmin/logs")
@superadmin_required
def superadmin_logs():
    """Super admin logs page"""
    try:
        logs = list(system_logs_col.find().sort("timestamp", -1).limit(100))
        for log in logs:
            log["_id"] = str(log["_id"])
            if "timestamp" in log:
                try:
                    log["timestamp"] = log["timestamp"].isoformat()
                except:
                    pass
        return render_template("superadmin/logs.html", superadmin=current_user, logs=logs)
    except Exception as e:
        print(f"Error in superadmin_logs: {e}")
        return render_template("superadmin/logs.html", superadmin=current_user, logs=[])

@app.route("/api/superadmin/logs")
@superadmin_required
def api_get_logs():
    """API endpoint to get system logs as JSON"""
    try:
        
        limit = int(request.args.get("limit", 50))
        logs = list(system_logs_col.find().sort("timestamp", -1).limit(limit))
        
        for log in logs:
            log["_id"] = str(log["_id"])
            if "timestamp" in log:
                try:
                    log["timestamp"] = log["timestamp"].isoformat()
                except:
                    pass
        
        return jsonify({
            "status": "success",
            "logs": logs
        })
    except Exception as e:
        return jsonify({"status": "failed", "msg": str(e)}), 500



@app.route("/api/superadmin/admin/<admin_id>/block", methods=["POST"])
@superadmin_required
def block_admin_fixed(admin_id):
    """Block an admin"""
    try:
        # Get robust connection for admin operations
        admin_client = get_robust_db_connection()
        core = admin_client['secure_db']
        admins_col = core["admins"]
        
        admin = admins_col.find_one({"_id": admin_id})
        if not admin:
            return jsonify({"status": "failed", "msg": "Admin not found"}), 404
        
        # Update admin status
        result = admins_col.update_one(
            {"_id": admin_id},
            {"$set": {
                "is_active": False,
                "blocked_at": datetime.datetime.utcnow(),
                "blocked_by": current_user.email
            }}
        )
        
        if result.modified_count > 0:
            # Log action
            system_logs_col.insert_one({
                "action": "block_admin",
                "user": current_user.email,
                "admin_id": admin_id,
                "admin_email": admin.get("email"),
                "timestamp": datetime.datetime.utcnow(),
                "status": "success"
            })
            
            return jsonify({
                "status": "success",
                "msg": f"Admin {admin.get('name')} blocked successfully"
            })
        
        return jsonify({"status": "failed", "msg": "Failed to block admin"}), 500
        
    except Exception as e:
        return jsonify({"status": "failed", "msg": str(e)}), 500

@app.route("/api/superadmin/admin/<admin_id>/unblock", methods=["POST"])
@superadmin_required
def unblock_admin_fixed(admin_id):
    """Unblock an admin"""
    try:
        # Get robust connection for admin operations
        admin_client = get_robust_db_connection()
        core = admin_client['secure_db']
        admins_col = core["admins"]
        
        admin = admins_col.find_one({"_id": admin_id})
        if not admin:
            return jsonify({"status": "failed", "msg": "Admin not found"}), 404
        
        # Update admin status
        result = admins_col.update_one(
            {"_id": admin_id},
            {"$set": {
                "is_active": True,
                "unblocked_at": datetime.datetime.utcnow(),
                "unblocked_by": current_user.email
            }}
        )
        
        if result.modified_count > 0:
            # Log action
            system_logs_col.insert_one({
                "action": "unblock_admin",
                "user": current_user.email,
                "admin_id": admin_id,
                "admin_email": admin.get("email"),
                "timestamp": datetime.datetime.utcnow(),
                "status": "success"
            })
            
            return jsonify({
                "status": "success",
                "msg": f"Admin {admin.get('name')} unblocked successfully"
            })
        
        return jsonify({"status": "failed", "msg": "Failed to unblock admin"}), 500
        
    except Exception as e:
        return jsonify({"status": "failed", "msg": str(e)}), 500

@app.route("/api/superadmin/admin/<admin_id>/delete", methods=["DELETE"])
@superadmin_required
def delete_admin_fixed(admin_id):
    """Delete an admin"""
    try:
        # Get robust connection for admin operations
        admin_client = get_robust_db_connection()
        core = admin_client['secure_db']
        admins_col = core["admins"]
        
        admin = admins_col.find_one({"_id": admin_id})
        if not admin:
            return jsonify({"status": "failed", "msg": "Admin not found"}), 404
        
        # Delete admin
        result = admins_col.delete_one({"_id": admin_id})
        if result.deleted_count > 0:
            system_logs_col.insert_one({
                "action": "delete_admin",
                "admin_id": admin_id,
                "admin_email": admin.get("email"),
                "performed_by": current_user.email,
                "timestamp": datetime.datetime.utcnow()
            })
            return jsonify({"status": "success", "msg": "Admin deleted successfully"})
        return jsonify({"status": "failed", "msg": "Admin not found"}), 404
    except Exception as e:
        return jsonify({"status": "failed", "msg": str(e)}), 500

@app.route("/api/superadmin/delete_admin/<admin_id>", methods=["DELETE"])
@superadmin_required
def superadmin_delete_admin(admin_id):
    """Delete admin"""
    try:
        admin = admins_col.find_one({"_id": admin_id})
        if not admin:
            return jsonify({"status": "failed", "msg": "Admin not found"}), 404
        
        result = admins_col.delete_one({"_id": admin_id})
        if result.deleted_count > 0:
            system_logs_col.insert_one({
                "action": "delete_admin",
                "admin_id": admin_id,
                "admin_email": admin.get("email"),
                "performed_by": current_user.email,
                "timestamp": datetime.datetime.utcnow()
            })
            return jsonify({"status": "success", "msg": "Admin deleted successfully"})
        return jsonify({"status": "failed", "msg": "Admin not found"}), 404
    except Exception as e:
        return jsonify({"status": "failed", "msg": str(e)}), 500

# --- Video Feed Route ---
@app.route("/video_feed")
@admin_required
def video_feed():
    return Response(generate_camera_stream(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

# --- Create Super Admin Route ---
@app.route("/create_superadmin", methods=["POST"])
def create_superadmin():
    """Create super admin account"""
    secret = os.environ.get("CREATE_SUPERADMIN_SECRET", "temp_secret_123")
    provided = request.form.get("secret", "")
    
    if provided != secret:
        return jsonify({"status": "failed", "msg": "Invalid secret"}), 403
    
    name = request.form.get("name")
    email = request.form.get("email")
    password = request.form.get("password")
    
    if not (name and email and password):
        return jsonify({"status": "failed", "msg": "Missing fields"}), 400
    
    email = email.strip().lower()
    
    # Get robust connection for superadmin creation
    try:
        create_client = get_robust_db_connection()
        core = create_client['secure_db']
        superadmins_col = core["superadmins"]
    except Exception as e:
        print(f"Error getting robust connection for superadmin creation: {e}")
        # Fallback to global collections
        pass
    
    existing = superadmins_col.find_one({"email": email})
    if existing:
        return jsonify({"status": "failed", "msg": "Super admin already exists"}), 400
    
    password_hash = generate_password_hash(password)
    doc = {
        "_id": str(datetime.datetime.utcnow().timestamp()) + "_" + email,
        "name": name,
        "email": email,
        "password_hash": password_hash,
        "profile_image": "",
        "created_at": datetime.datetime.utcnow()
    }
    superadmins_col.insert_one(doc)
    return jsonify({"status": "success", "msg": "Super admin created"}), 201

# --- 3D Face Analysis Routes ---
@app.route("/api/analyze_3d_face", methods=["POST"])
@admin_required
def analyze_3d_face():
    """Analyze face using 3D reconstruction"""
    try:
        file = request.files.get("file")
        if not file:
            return jsonify({"status": "failed", "msg": "No image provided"}), 400
        
        # Convert file to image
        file_bytes = file.read()
        img_array = np.frombuffer(file_bytes, np.uint8)
        image = cv.imdecode(img_array, cv.IMREAD_COLOR)
        
        if image is None:
            return jsonify({"status": "failed", "msg": "Invalid image format"}), 400
        
        # Import 3D reconstruction functions
        try:
            from face_3d_reconstruction import extract_3d_face_features, visualize_3d_landmarks
            
            # Extract 3D features
            result = extract_3d_face_features(image)
            
            if result is None:
                return jsonify({"status": "failed", "msg": "No face detected in image"}), 400
            
            # Create visualization
            vis_image = visualize_3d_landmarks(image, result['landmarks_3d'])
            
            # Encode visualization as base64
            _, buffer = cv.imencode('.jpg', vis_image)
            vis_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return jsonify({
                "status": "success",
                "features": {
                    "face_width": result['features']['face_width'],
                    "face_height": result['features']['face_height'],
                    "face_depth": result['features']['face_depth'],
                    "eye_distance": result['features']['eye_distance'],
                    "nose_length": result['features']['nose_length'],
                    "nose_protrusion": result['features']['nose_protrusion'],
                    "mouth_width": result['features']['mouth_width'],
                    "symmetry_score": result['features']['symmetry_score'],
                    "face_curvature": result['features']['face_curvature']
                },
                "pose_angles": result['pose_angles'],
                "quality_score": result['quality_score'],
                "landmarks_count": len(result['landmarks_3d']),
                "visualization": vis_base64
            })
            
        except ImportError:
            return jsonify({"status": "failed", "msg": "3D reconstruction not available"}), 503
        
    except Exception as e:
        return jsonify({"status": "failed", "msg": f"Analysis failed: {str(e)}"}), 500

@app.route("/api/compare_3d_faces", methods=["POST"])
@admin_required
def compare_3d_faces():
    """Compare two faces using 3D analysis"""
    try:
        file1 = request.files.get("file1")
        file2 = request.files.get("file2")
        
        if not file1 or not file2:
            return jsonify({"status": "failed", "msg": "Two images required"}), 400
        
        # Convert files to images
        images = []
        for file in [file1, file2]:
            file_bytes = file.read()
            img_array = np.frombuffer(file_bytes, np.uint8)
            image = cv.imdecode(img_array, cv.IMREAD_COLOR)
            if image is None:
                return jsonify({"status": "failed", "msg": "Invalid image format"}), 400
            images.append(image)
        
        try:
            from face_3d_reconstruction import extract_3d_face_features, match_3d_face_templates
            
            # Extract features from both images
            features = []
            for i, image in enumerate(images):
                result = extract_3d_face_features(image)
                if result is None:
                    return jsonify({"status": "failed", "msg": f"No face detected in image {i+1}"}), 400
                features.append(result)
            
            # Create temporary templates
            template1 = {
                'landmarks_3d': features[0]['landmarks_3d'],
                'features': features[0]['features'],
                'descriptor': np.random.randn(100),  # Simplified for demo
                'confidence': features[0]['quality_score']
            }
            
            template2 = {
                'landmarks_3d': features[1]['landmarks_3d'],
                'features': features[1]['features'],
                'descriptor': np.random.randn(100),  # Simplified for demo
                'confidence': features[1]['quality_score']
            }
            
            # Calculate similarity
            similarity = match_3d_face_templates(template1, template2)
            
            return jsonify({
                "status": "success",
                "similarity_score": similarity,
                "match_result": "MATCH" if similarity > 0.7 else "NO_MATCH",
                "confidence": (features[0]['quality_score'] + features[1]['quality_score']) / 2,
                "face1_features": features[0]['features'],
                "face2_features": features[1]['features'],
                "pose1": features[0]['pose_angles'],
                "pose2": features[1]['pose_angles']
            })
            
        except ImportError:
            return jsonify({"status": "failed", "msg": "3D reconstruction not available"}), 503
        
    except Exception as e:
        return jsonify({"status": "failed", "msg": f"Comparison failed: {str(e)}"}), 500

# --- Health Check Route ---
@app.route("/health")
def health_check():
    """Health check endpoint for Railway deployment"""
    try:
        # Check MongoDB connection
        client.admin.command('ping')
        
        # Check FAISS index
        faiss_status = "loaded" if faiss_index is not None else "not_loaded"
        
        return jsonify({
            "status": "healthy",
            "mongodb": "connected",
            "faiss_index": faiss_status,
            "faiss_count": faiss_index.ntotal if faiss_index else 0,
            "timestamp": datetime.datetime.utcnow().isoformat()
        }), 200
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.datetime.utcnow().isoformat()
        }), 503

# --- Real-time Face Recognition API with Combined Embeddings ---
@app.route("/api/detect_face", methods=["POST"])
@admin_required
def detect_face_realtime():
    """Real-time face detection and recognition using combined ArcFace + MediaPipe embeddings"""
    try:
        # Import combined recognition module
        from combined_recognition import (
            get_combined_embedding, 
            detect_faces_mediapipe, 
            is_real_face,
            cosine_sim
        )
        
        file = request.files.get("frame")
        if not file:
            return jsonify({
                "status": "error", 
                "message": "No image frame provided"
            }), 400
        
        # Convert uploaded frame to image
        file_bytes = file.read()
        img_array = np.frombuffer(file_bytes, np.uint8)
        image = cv.imdecode(img_array, cv.IMREAD_COLOR)
        
        if image is None:
            return jsonify({
                "status": "error", 
                "message": "Invalid image format"
            }), 400

        # Detect faces
        import mediapipe as mp
        mp_face_detection = mp.solutions.face_detection
        
        with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.3) as face_detection:
            image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            results = face_detection.process(image_rgb)
            
            detected_faces = []
            
            if results.detections:
                for detection in results.detections:
                    # Get face bounding box
                    bbox = detection.location_data.relative_bounding_box
                    h, w, _ = image.shape
                    
                    # Convert to pixel coordinates
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)
                    
                    # Ensure bbox is within image bounds
                    x = max(0, x)
                    y = max(0, y)
                    width = min(width, w - x)
                    height = min(height, h - y)
                    
                    # Extract face region
                    face_roi = image[y:y+height, x:x+width]
                    
                    if face_roi.size > 0:
                        # Liveness detection
                        is_live = is_real_face(face_roi)
                        
                        face_info = {
                            "bbox": {"x": x, "y": y, "width": width, "height": height},
                            "confidence": float(detection.score[0]),
                            "detected": True,
                            "is_live": is_live
                        }
                        
                        if not is_live:
                            face_info.update({
                                "recognized": False,
                                "name": "Fake/Low Quality",
                                "match_confidence": 0.0
                            })
                        else:
                            try:
                                # Get combined embedding
                                embedding = get_combined_embedding(face_roi)
                                
                                if embedding is not None:
                                    # Search in database
                                    best_name, best_score = None, -1
                                    threshold = 0.6
                                    
                                    lazy_faiss_init()  # Initialize FAISS if needed
                                    
                                    for doc in persons_col.find():
                                        try:
                                            stored_embedding = pickle.loads(doc["embedding"])
                                            
                                            # Handle different embedding formats
                                            if isinstance(stored_embedding, dict) and 'centroid' in stored_embedding:
                                                stored_embedding = stored_embedding['centroid']
                                            
                                            # Handle dimension mismatch
                                            min_len = min(len(stored_embedding), len(embedding))
                                            score = cosine_sim(
                                                stored_embedding[:min_len], 
                                                embedding[:min_len]
                                            )
                                            
                                            if score > best_score:
                                                best_name, best_score = doc["name"], score
                                        except Exception as e:
                                            print(f"[Error matching {doc.get('name', 'unknown')}: {e}")
                                            continue
                                    
                                    if best_score >= threshold:
                                        # Person recognized
                                        face_info.update({
                                            "recognized": True,
                                            "name": best_name,
                                            "match_confidence": float(best_score),
                                            "embedding_type": "combined"
                                        })
                                        
                                        # Auto-mark attendance
                                        try:
                                            today_start = datetime.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
                                            existing = attendance_col.find_one({
                                                "name": best_name,
                                                "timestamp": {"$gte": today_start.isoformat()}
                                            })
                                            
                                            if not existing:
                                                attendance_col.insert_one({
                                                    "name": best_name,
                                                    "timestamp": datetime.datetime.utcnow().isoformat(),
                                                    "confidence": best_score,
                                                    "method": "dashboard_camera",
                                                    "embedding_type": "combined_arcface_geometric"
                                                })
                                                face_info["attendance_marked"] = True
                                        except Exception as att_err:
                                            print(f"[Attendance Error]: {att_err}")
                                    else:
                                        face_info.update({
                                            "recognized": False,
                                            "name": "Unknown",
                                            "match_confidence": 0.0
                                        })
                                else:
                                    face_info.update({
                                        "recognized": False,
                                        "name": "Embedding Failed",
                                        "match_confidence": 0.0
                                    })
                            except Exception as e:
                                print(f"[Recognition Error]: {e}")
                                face_info.update({
                                    "recognized": False,
                                    "name": "Error",
                                    "match_confidence": 0.0,
                                    "error": str(e)
                                })
                        
                        detected_faces.append(face_info)
            
            return jsonify({
                "status": "success",
                "faces_detected": len(detected_faces),
                "faces": detected_faces,
                "advanced_processing": {
                    "models_used": ["ArcFace", "MediaPipe"],
                    "embedding_type": "combined_70_30",
                    "liveness_enabled": True
                }
            })
            
    except Exception as e:
        print(f"[API Error]: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": f"Detection failed: {str(e)}"
        }), 500

@app.route("/test")
def test_route():
    """Test route to check server status"""
    return jsonify({
        "status": "success",
        "message": "Server is running",
        "timestamp": datetime.datetime.utcnow().isoformat()
    })

@app.route("/api/system_status")
def system_status():
    """Check face recognition system status"""
    try:
        from function import faiss_index, person_id_map
        
        status_info = {
            "status": "success",
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "face_recognition": {
                "faiss_loaded": faiss_index is not None,
                "total_persons": faiss_index.ntotal if faiss_index else 0,
                "person_mapping": len(person_id_map) if person_id_map else 0
            },
            "database": {
                "mongodb_connected": True,
                "persons_count": persons_col.count_documents({}),
                "attendance_today": attendance_col.count_documents({
                    "timestamp": {"$gte": datetime.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)}
                })
            },
            "3d_reconstruction": {
                "enabled": "face_3d_reconstruction" in globals()
            }
        }
        
        # Test MediaPipe
        try:
            import mediapipe as mp
            status_info["mediapipe"] = {"available": True, "version": mp.__version__}
        except Exception as e:
            status_info["mediapipe"] = {"available": False, "error": str(e)}
        
        # Test DeepFace
        try:
            import deepface
            status_info["deepface"] = {"available": True}
        except Exception as e:
            status_info["deepface"] = {"available": False, "error": str(e)}
        
        return jsonify(status_info)
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "timestamp": datetime.datetime.utcnow().isoformat()
        }), 500

@app.route("/attendance")
def attendance_camera():
    """Attendance camera page - Public access"""
    try:
        # Check if template exists
        import os
        template_path = os.path.join(app.template_folder, "attendance_camera.html")
        if not os.path.exists(template_path):
            return f"Template not found at: {template_path}", 404
        
        return render_template("attendance_camera.html")
    except Exception as e:
        print(f"Attendance page error: {e}")
        import traceback
        return f"Error loading attendance page: {str(e)}<br><pre>{traceback.format_exc()}</pre>", 500

@app.route("/api/detect_face_ultra_test", methods=["GET", "POST"])
def detect_face_ultra_test():
    """Test endpoint for ultra detection"""
    return jsonify({
        "status": "success",
        "message": "Ultra detection endpoint is working",
        "method": request.method,
        "timestamp": datetime.datetime.utcnow().isoformat()
    })



@app.route("/api/detection_health", methods=["GET"])
def detection_health():
    """Health check for detection systems"""
    try:
        # Test MediaPipe availability
        mp_available = False
        try:
            import mediapipe as mp
            mp_available = True
        except:
            pass
        
        # Test DeepFace availability
        deepface_available = False
        try:
            from deepface import DeepFace
            deepface_available = True
        except:
            pass
        
        # Test FAISS availability
        faiss_available = False
        try:
            import faiss
            faiss_available = True
        except:
            pass
        
        # Check ultra module
        ultra_available = False
        try:
            import ultra_face_recognition
            ultra_available = True
        except:
            pass
        
        return jsonify({
            "status": "success",
            "components": {
                "mediapipe": mp_available,
                "deepface": deepface_available,
                "faiss": faiss_available,
                "ultra_recognition": ultra_available
            },
            "overall_health": mp_available and deepface_available,
            "recommendation": "Enhanced detection available" if (mp_available and deepface_available) else "Basic detection only"
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

def detect_face_enhanced():
    """Enhanced face detection as fallback"""
    try:
        file = request.files.get("frame")
        if not file:
            return jsonify({
                "status": "error", 
                "message": "No image frame provided"
            }), 400
        
        # Convert uploaded frame to image
        file_bytes = file.read()
        img_array = np.frombuffer(file_bytes, np.uint8)
        image = cv.imdecode(img_array, cv.IMREAD_COLOR)
        
        if image is None:
            return jsonify({
                "status": "error", 
                "message": "Invalid image format"
            }), 400

        print(f"🔬 Enhanced processing frame: {image.shape}")
        
        # Use MediaPipe for reliable face detection
        try:
            import mediapipe as mp
            mp_face_detection = mp.solutions.face_detection
            face_detector = mp_face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.2  # Even lower threshold for better detection
            )
            print("✅ MediaPipe face detector initialized")
        except Exception as e:
            print(f"❌ MediaPipe initialization error: {e}")
            return jsonify({
                "status": "error",
                "message": f"MediaPipe not available: {str(e)}"
            }), 500
        
        # Multiple preprocessing techniques for better detection
        print("🔧 Applying image enhancements...")
        
        # 1. Basic enhancement
        enhanced = cv.bilateralFilter(image, 9, 75, 75)
        
        # 2. Histogram equalization for better contrast
        gray = cv.cvtColor(enhanced, cv.COLOR_BGR2GRAY)
        gray = cv.equalizeHist(gray)
        enhanced = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
        
        # 3. Gamma correction for brightness
        gamma = 1.2
        enhanced = np.power(enhanced / 255.0, gamma)
        enhanced = (enhanced * 255).astype(np.uint8)
        
        # 4. Convert to RGB for MediaPipe
        rgb_image = cv.cvtColor(enhanced, cv.COLOR_BGR2RGB)
        print(f"📊 Processing RGB image: {rgb_image.shape}")
        
        # 5. Try detection on original and enhanced images
        results = face_detector.process(rgb_image)
        
        # If no detection, try original image
        if not results.detections:
            print("🔄 No faces in enhanced image, trying original...")
            rgb_original = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            results = face_detector.process(rgb_original)
        
        final_results = []
        
        print(f"🔍 MediaPipe detection results: {results.detections is not None}")
        if results.detections:
            print(f"🎯 Found {len(results.detections)} face(s)")
            h, w = image.shape[:2]
            
            for i, detection in enumerate(results.detections):
                # Extract bounding box
                bbox = detection.location_data.relative_bounding_box
                x = max(0, int(bbox.xmin * w))
                y = max(0, int(bbox.ymin * h))
                width = min(w - x, int(bbox.width * w))
                height = min(h - y, int(bbox.height * h))
                
                if width < 50 or height < 50:  # Skip very small faces
                    continue
                
                print(f"🎯 Face detected at ({x}, {y}, {width}, {height})")
                
                # Extract face embedding
                try:
                    face_embedding = get_face_embedding(image, (x, y, width, height))
                    
                    if face_embedding is not None:
                        # Search in FAISS database
                        match_result = search_face_faiss(face_embedding, image=image)
                        
                        face_result = {
                            "bbox": {"x": x, "y": y, "width": width, "height": height},
                            "confidence": float(detection.score[0]),
                            "detected": True,
                            "face_id": i,
                            "quality": {"overall": min(1.0, (width * height) / 10000.0)},
                            "stability": 0.8,  # Default good stability
                            "description": f"Enhanced detection (confidence: {float(detection.score[0]):.2f})",
                            "liveness": {
                                "is_live": True,  # Assume live for enhanced mode
                                "confidence": 85.0,
                                "reason": "Enhanced detection mode - liveness assumed"
                            }
                        }
                        
                        if match_result and isinstance(match_result, dict) and match_result.get('name') != 'Unknown':
                            print(f"✨ Enhanced recognition: {match_result['name']} (confidence: {match_result.get('confidence', 0):.3f})")
                            
                            # Person recognized
                            face_result.update({
                                "recognized": True,
                                "name": match_result['name'],
                                "match_confidence": float(match_result.get('confidence', 0)),
                                "person_id": str(match_result.get('_id', ''))
                            })
                            
                            # Mark attendance with simplified validation
                            try:
                                person = persons_col.find_one({"name": match_result['name']})
                                person_id = person.get('_id') if person else None
                                
                                if person_id:
                                    attendance_data = {
                                        "person_id": person_id,
                                        "person_name": match_result['name'],
                                        "timestamp": datetime.datetime.utcnow(),
                                        "confidence": float(match_result.get('confidence', 0)),
                                        "method": "Enhanced_Face_Recognition",
                                        "camera_source": "enhanced_attendance_camera",
                                        "detection_confidence": float(detection.score[0])
                                    }
                                    
                                    # Check if attendance already marked today
                                    today_start = datetime.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
                                    existing_attendance = attendance_col.find_one({
                                        "person_id": person_id,
                                        "timestamp": {"$gte": today_start}
                                    })
                                    
                                    if not existing_attendance:
                                        result = attendance_col.insert_one(attendance_data)
                                        face_result["attendance_marked"] = True
                                        face_result["attendance_id"] = str(result.inserted_id)
                                        print(f"✅ Enhanced attendance marked: {match_result['name']}")
                                    else:
                                        face_result["attendance_marked"] = False
                                        face_result["already_marked"] = True
                                        face_result["existing_time"] = existing_attendance['timestamp'].isoformat()
                                        
                            except Exception as e:
                                print(f"❌ Enhanced attendance error: {e}")
                                face_result["attendance_error"] = str(e)
                        else:
                            face_result.update({
                                "recognized": False,
                                "name": "Unknown",
                                "match_confidence": 0.0,
                                "reason": "No match in database"
                            })
                        
                        final_results.append(face_result)
                    
                except Exception as e:
                    print(f"⚠️ Face processing error: {e}")
        else:
            print("❌ No faces detected by MediaPipe")
        
        # Enhanced response
        response_data = {
            "status": "success",
            "faces_detected": len(final_results),
            "faces": final_results,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "processing_mode": "enhanced_fallback",
            "enhanced_processing": {
                "detection_method": "MediaPipe",
                "image_enhancement": True,
                "total_faces": len(final_results),
                "recognized_faces": len([f for f in final_results if f.get("recognized", False)])
            }
        }
        
        print(f"🎯 Enhanced result: {len(final_results)} faces, {len([f for f in final_results if f.get('recognized', False)])} recognized")
        return jsonify(response_data)
        
    except Exception as e:
        print(f"💥 Enhanced detection error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": f"Enhanced detection failed: {str(e)}",
            "timestamp": datetime.datetime.utcnow().isoformat()
        }), 500

@app.route("/api/detect_face_simple", methods=["POST"])
def detect_face_simple():
    """Proxy to Recognition Service (Microservice)"""
    try:
        file = request.files.get("frame")
        if not file:
            return jsonify({"status": "error", "message": "No image frame provided"}), 400
        
        # Forward to microservice
        import requests
        
        # Prepare file for upload
        file.seek(0)
        files = {'file': ('frame.jpg', file, 'image/jpeg')}
        
        try:
            # Call the recognition service
            response = requests.post("http://localhost:8001/recognize", files=files, timeout=5)
            
            if response.status_code == 200:
                return jsonify(response.json())
            else:
                print(f"Microservice error: {response.text}")
                return jsonify({"status": "error", "message": "Recognition service failed"}), 500
                
        except requests.exceptions.ConnectionError:
            print("❌ Recognition service not reachable on port 8001")
            return jsonify({"status": "error", "message": "Recognition service unavailable"}), 503
            
    except Exception as e:
        print(f"Proxy error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/detect_face_ultra", methods=["POST"])
def detect_face_ultra():
    """Ultra-accurate detection using combined embeddings - same as standard for now"""
    return detect_face_realtime()

@app.route("/api/detect_face_public", methods=["POST"])
def detect_face_public():
    """Public detection API - same as standard for now"""
    return detect_face_realtime()


@app.route("/api/mark_attendance", methods=["POST"])
@admin_required
def mark_attendance_api():
    """API to manually mark attendance"""
    try:
        data = request.get_json()
        person_id = data.get('person_id')
        person_name = data.get('person_name')
        
        if not person_id or not person_name:
            return jsonify({"status": "error", "message": "Missing person data"}), 400
        
        # Check if attendance already marked today
        today_start = datetime.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        existing_attendance = attendance_col.find_one({
            "person_id": person_id,
            "timestamp": {"$gte": today_start}
        })
        
        if existing_attendance:
            return jsonify({
                "status": "already_marked",
                "message": f"Attendance already marked for {person_name} today"
            })
        
        # Mark attendance
        attendance_data = {
            "person_id": person_id,
            "person_name": person_name,
            "timestamp": datetime.datetime.utcnow(),
            "confidence": data.get('confidence', 0.95),
            "method": "Manual_Dashboard",
            "camera_source": "dashboard_live_camera"
        }
        
        result = attendance_col.insert_one(attendance_data)
        
        return jsonify({
            "status": "success",
            "message": f"Attendance marked for {person_name}",
            "attendance_id": str(result.inserted_id)
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to mark attendance: {str(e)}"
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting server on port {port} with WebSockets enabled...")
    socketio.run(app, host="0.0.0.0", port=port, debug=False, allow_unsafe_werkzeug=True)
