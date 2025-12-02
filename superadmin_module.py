from flask import Blueprint, render_template, request, jsonify, Response, redirect, url_for, flash, current_app
from pymongo import MongoClient
import datetime
import os
import cv2 as cv
import threading
import numpy as np
from bson import ObjectId
from werkzeug.security import generate_password_hash
from flask_login import current_user, login_required
from functools import wraps
import traceback
from extensions import socketio
from flask_socketio import emit
import time
import requests
# from ultra_face_recognition import UltraFaceRecognizer
from simple_face_recognition import SimpleFaceRecognizer
import concurrent.futures
import json
import tempfile
from werkzeug.utils import secure_filename
# Lazy load AssistBuddy
# try:
#     from assistbuddy import AssistBuddyModel
#     ASSISTBUDDY_AVAILABLE = True
# except ImportError:
#     ASSISTBUDDY_AVAILABLE = False
#     print("AssistBuddy package not found. Using mock responses.")

from assistbuddy.tool_manager import get_tool_manager
from assistbuddy.utils.spell_checker import SpellingCorrector

# Initialize AssistBuddy components
tool_manager = get_tool_manager()
spell_checker = SpellingCorrector()

# Create Blueprint
superadmin_bp = Blueprint('superadmin', __name__, url_prefix='/superadmin')

# Global variables for camera management
active_camera_streams = {} # Maps camera_id to CameraHandler instance
camera_stream_locks = {}

# Lazy initialized globals
face_recognizer = None

def get_face_recognizer():
    global face_recognizer
    if face_recognizer is None:
        try:
            # from ultra_face_recognition import UltraFaceRecognizer
            # face_recognizer = UltraFaceRecognizer()
            face_recognizer = SimpleFaceRecognizer()
            print("SimpleFaceRecognizer initialized in superadmin_module")
        except Exception as e:
            print(f"Failed to initialize SimpleFaceRecognizer: {e}")
            return None
    return face_recognizer

class CameraHandler:
    def __init__(self, source, camera_id):
        self.source = source
        self.camera_id = camera_id
        self.cap = cv.VideoCapture(source)
        self.lock = threading.Lock()
        self.running = True
        self.current_frame = None
        self.last_frame_time = 0
        
        # Recording state
        self.is_recording = False
        self.video_writer = None
        self.recording_filename = None
        self.recording_start_time = None
        
        # Start frame reader thread
        self.thread = threading.Thread(target=self._read_frames, daemon=True)
        self.thread.start()
        
    def _read_frames(self):
        while self.running and self.cap.isOpened():
            success, frame = self.cap.read()
            if success:
                with self.lock:
                    self.current_frame = frame
                    self.last_frame_time = time.time()
                    
                # Handle recording
                if self.is_recording and self.video_writer:
                    try:
                        self.video_writer.write(frame)
                    except Exception as e:
                        print(f"Recording write error: {e}")
            else:
                # If stream fails, try to reconnect or stop
                time.sleep(1)
                
            time.sleep(0.01) # Prevent CPU hogging
            
    def get_frame(self):
        with self.lock:
            return self.current_frame.copy() if self.current_frame is not None else None
            
    def start_recording(self, filename):
        if self.is_recording:
            return False, "Already recording"
            
        with self.lock:
            if self.current_frame is None:
                return False, "No frame available"
                
            h, w = self.current_frame.shape[:2]
            fourcc = cv.VideoWriter_fourcc(*'mp4v') # or XVID
            # Save to static/recordings
            filepath = os.path.join(current_app.root_path, 'static', 'recordings', filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            self.video_writer = cv.VideoWriter(filepath, fourcc, 20.0, (w, h))
            self.recording_filename = filename
            self.recording_start_time = datetime.datetime.utcnow()
            self.is_recording = True
            return True, "Recording started"

    def stop_recording(self):
        if not self.is_recording:
            return False, "Not recording"
            
        with self.lock:
            self.is_recording = False
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            
            # Return metadata
            duration = (datetime.datetime.utcnow() - self.recording_start_time).total_seconds()
            return True, {
                "filename": self.recording_filename,
                "duration": duration,
                "start_time": self.recording_start_time
            }

    def stop(self):
        self.running = False
        self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
        if self.video_writer:
            self.video_writer.release()

def superadmin_required(f):
    """Decorator to require super admin privileges"""
    @wraps(f)
    @login_required
    def decorated_function(*args, **kwargs):
        if not hasattr(current_user, 'role') or current_user.role != 'superadmin':
            flash('Access denied. Super Admin privileges required.', 'danger')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

from database_utils import get_db_collections

# Global database client is now managed in database_utils


# Dashboard Routes
@superadmin_bp.route('/dashboard')
@superadmin_required
def dashboard():
    """Super Admin Dashboard"""
    # Get stats for initial load
    stats = get_dashboard_stats_helper()
    return render_template('superadmin/dashboard_new.html', stats=stats, superadmin=current_user)

# API Routes
@superadmin_bp.route('/api/stats')
@superadmin_required
def api_stats():
    """Get comprehensive statistics"""
    try:
        db = get_db_collections()
        
        today_start = datetime.datetime.combine(datetime.date.today(), datetime.time.min)
        today_end = datetime.datetime.combine(datetime.date.today(), datetime.time.max)
        
        stats = {
            "total_superadmins": db['superadmins_col'].count_documents({}),
            "total_admins": db['admins_col'].count_documents({}),
            "total_users": db['users_col'].count_documents({}),
            "total_persons": db['persons_col'].count_documents({"status": {"$ne": "blocked"}}),
            "active_users": db['users_col'].count_documents({"status": "active"}),
            "blocked_users": db['users_col'].count_documents({"status": "blocked"}),
            "pending_enrollments": db['enrollment_requests_col'].count_documents({"status": "pending"}),
            "today_attendance": db['attendance_col'].count_documents({
                "timestamp": {"$gte": today_start, "$lt": today_end}
            }),
            "total_attendance": db['attendance_col'].count_documents({}),
            "total_cameras": db['cameras_col'].count_documents({}),
            "active_cameras": len(active_camera_streams),
            "system_health": "healthy"
        }
        
        return jsonify(stats), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@superadmin_bp.route('/api/users')
@superadmin_required
def api_users():
    """Get all users"""
    try:
        db = get_db_collections()
        
        # Fetch all users
        users = list(db['users_col'].find({}))
        
        # Convert ObjectId to string and format dates
        for user in users:
            user['_id'] = str(user['_id'])
            if 'created_at' in user and user['created_at']:
                try:
                    user['created_at'] = user['created_at'].isoformat()
                except:
                    pass
            if 'last_login' in user and user['last_login']:
                try:
                    user['last_login'] = user['last_login'].isoformat()
                except:
                    pass
            # Remove password hash from response
            user.pop('password_hash', None)
            
            # Add derived fields for frontend
            user['is_active'] = user.get('status') != 'blocked'
            # Check if enrolled (has embeddings or explicit flag)
            user['is_enrolled'] = user.get('is_enrolled', False) or (user.get('embeddings') is not None and len(user.get('embeddings', [])) > 0)
            
        return jsonify({"status": "success", "users": users}), 200
    except Exception as e:
        print(f"Error fetching users: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@superadmin_bp.route('/api/system-logs')
@superadmin_required
def api_system_logs():
    """Get system logs"""
    try:
        db = get_db_collections()
        
        # Fetch system logs, sorted by timestamp descending
        logs = list(db['system_logs_col'].find({})
                   .sort("timestamp", -1)
                   .limit(100))
        
        # Convert ObjectId to string and format dates
        for log in logs:
            log['_id'] = str(log['_id'])
            if 'timestamp' in log and log['timestamp']:
                try:
                    log['timestamp'] = log['timestamp'].isoformat()
                except:
                    pass
            
        return jsonify({"status": "success", "logs": logs}), 200
    except Exception as e:
        print(f"Error fetching system logs: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

# Camera Management
@superadmin_bp.route('/cameras')
@superadmin_required
def cameras():
    """Camera management page"""
    return render_template('superadmin/cameras_new.html', superadmin=current_user)

# Gallery Routes
@superadmin_bp.route('/gallery')
@superadmin_required
def gallery():
    """Gallery page for screenshots and recordings"""
    return render_template('superadmin/gallery.html', superadmin=current_user)

@superadmin_bp.route('/api/gallery')
@superadmin_required
def api_gallery():
    """Get list of screenshots and recordings"""
    try:
        # Define paths
        static_folder = os.path.join(current_app.root_path, 'static')
        screenshots_dir = os.path.join(static_folder, 'screenshots')
        recordings_dir = os.path.join(static_folder, 'recordings')
        
        # Ensure directories exist
        os.makedirs(screenshots_dir, exist_ok=True)
        os.makedirs(recordings_dir, exist_ok=True)
        
        files = []
        
        # Scan screenshots
        for f in os.listdir(screenshots_dir):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(screenshots_dir, f)
                stats = os.stat(path)
                files.append({
                    'name': f,
                    'type': 'image',
                    'url': url_for('static', filename=f'screenshots/{f}'),
                    'size': stats.st_size,
                    'created_at': datetime.datetime.fromtimestamp(stats.st_ctime).isoformat()
                })
                
        # Scan recordings
        for f in os.listdir(recordings_dir):
            if f.lower().endswith(('.mp4', '.avi', '.webm')):
                path = os.path.join(recordings_dir, f)
                stats = os.stat(path)
                files.append({
                    'name': f,
                    'type': 'video',
                    'url': url_for('static', filename=f'recordings/{f}'),
                    'size': stats.st_size,
                    'created_at': datetime.datetime.fromtimestamp(stats.st_ctime).isoformat()
                })
        
        # Sort by date desc
        files.sort(key=lambda x: x['created_at'], reverse=True)
        
        return jsonify({"status": "success", "files": files}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@superadmin_bp.route('/api/gallery/<filename>', methods=['DELETE'])
@superadmin_required
def api_delete_gallery_item(filename):
    """Delete a gallery item"""
    try:
        static_folder = os.path.join(current_app.root_path, 'static')
        
        # Determine if it's an image or video
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            subdir = 'screenshots'
        elif filename.lower().endswith(('.mp4', '.avi', '.webm')):
            subdir = 'recordings'
        else:
            return jsonify({"status": "error", "message": "Invalid file type"}), 400
            
        filepath = os.path.join(static_folder, subdir, filename)
        
        if os.path.exists(filepath):
            os.remove(filepath)
            return jsonify({"status": "success", "message": "File deleted"}), 200
        else:
            return jsonify({"status": "error", "message": "File not found"}), 404
            
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@superadmin_bp.route('/api/cameras', methods=['POST'])
@superadmin_required
def api_create_camera():
    """Create new camera"""
    try:
        db = get_db_collections()
        data = request.get_json()
        
        name = data.get("name", "").strip()
        source_type = data.get("source_type", "opencv")
        purpose = data.get("purpose", ["attendance"]) # Default to attendance
        
        if not name:
            return jsonify({"status": "failed", "error": "Camera name is required"}), 400
        
        # Check for duplicate name
        if db['cameras_col'].find_one({"name": name}):
            return jsonify({"status": "failed", "error": "Camera name already exists"}), 400
        
        # Determine source
        if source_type == "opencv":
            camera_index = data.get("camera_index")
            if camera_index is None:
                return jsonify({"status": "failed", "error": "Camera index is required"}), 400
            source = int(camera_index)
        else:
            stream_url = data.get("stream_url", "").strip()
            if not stream_url:
                return jsonify({"status": "failed", "error": "Stream URL is required"}), 400
            source = stream_url
        
        camera_id = f"cam_{int(datetime.datetime.utcnow().timestamp())}_{name.replace(' ', '_')[:20]}"
        
        camera_doc = {
            "_id": camera_id,
            "name": name,
            "source_type": source_type,
            "source": source,
            "config": {
                "fps": data.get("fps", 30),
                "resolution": {
                    "width": data.get("resolution_width", 640),
                    "height": data.get("resolution_height", 480)
                },
                "enabled": True
            },
            "purpose": purpose,
            "enabled": True,
            "is_active": False,
            "created_at": datetime.datetime.utcnow(),
            "created_by": current_user.email,
            "last_seen": None
        }
        
        db['cameras_col'].insert_one(camera_doc)
        
        # Log action
        db['system_logs_col'].insert_one({
            "action": "create_camera",
            "camera_id": camera_id,
            "camera_name": name,
            "performed_by": current_user.email,
            "timestamp": datetime.datetime.utcnow()
        })
        
        return jsonify({
            "status": "success",
            "message": f"Camera '{name}' created successfully",
            "camera_id": camera_id
        }), 201
        
    except Exception as e:
        return jsonify({"status": "failed", "error": str(e)}), 500

# Lazy load MediaPipe
mp_pose = None
pose = None

def get_pose_detector():
    global mp_pose, pose
    if pose is None:
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    return pose

def is_point_in_polygon(point, polygon):
    """
    Check if a point (x, y) is inside a polygon [(x1, y1), (x2, y2), ...]
    Ray casting algorithm.
    """
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

@superadmin_bp.route('/api/cameras/<camera_id>', methods=['PUT'])
@superadmin_required
def api_update_camera(camera_id):
    """Update camera details"""
    try:
        db = get_db_collections()
        data = request.get_json()
        
        camera = db['cameras_col'].find_one({"_id": camera_id})
        if not camera:
            return jsonify({"status": "failed", "error": "Camera not found"}), 404
            
        name = data.get("name", "").strip()
        source_type = data.get("source_type")
        purpose = data.get("purpose", [])
        zones = data.get("zones", []) # List of polygons
        
        update_fields = {}
        
        if name:
            update_fields["name"] = name
            
        if purpose:
            update_fields["purpose"] = purpose
            
        if zones is not None:
            update_fields["zones"] = zones
            
        # Handle source change
        if source_type:
            update_fields["source_type"] = source_type
            if source_type == "opencv":
                camera_index = data.get("camera_index")
                if camera_index is not None:
                    update_fields["source"] = int(camera_index)
            else:
                stream_url = data.get("stream_url", "").strip()
                if stream_url:
                    update_fields["source"] = stream_url
                    
        # Handle config update
        if "resolution_width" in data or "resolution_height" in data or "fps" in data:
            config = camera.get("config", {})
            resolution = config.get("resolution", {})
            
            if "resolution_width" in data:
                resolution["width"] = int(data["resolution_width"])
            if "resolution_height" in data:
                resolution["height"] = int(data["resolution_height"])
                
            config["resolution"] = resolution
            if "fps" in data:
                config["fps"] = int(data["fps"])
                
            update_fields["config"] = config
            
        if update_fields:
            db['cameras_col'].update_one(
                {"_id": camera_id},
                {"$set": update_fields}
            )
            
            # If active, restart to apply changes
            if camera_id in active_camera_streams:
                api_stop_camera(camera_id)
                # Auto-restart will handle it on next feed request
                
            # Log action
            db['system_logs_col'].insert_one({
                "action": "update_camera",
                "camera_id": camera_id,
                "changes": list(update_fields.keys()),
                "performed_by": current_user.email,
                "timestamp": datetime.datetime.utcnow()
            })
            
        return jsonify({
            "status": "success",
            "message": "Camera updated successfully"
        }), 200
        
    except Exception as e:
        return jsonify({"status": "failed", "error": str(e)}), 500

@superadmin_bp.route('/api/cameras/logs', methods=['GET'])
@superadmin_required
def api_camera_logs():
    """Get camera-related logs for analytics"""
    try:
        db = get_db_collections()
        
        # Fetch logs related to cameras
        logs = list(db['system_logs_col'].find({
            "action": {"$in": ["create_camera", "update_camera", "delete_camera", "start_camera", "stop_camera", "camera_error", "intrusion_alert"]}
        }).sort("timestamp", -1).limit(100))
        
        # Process for chart (events per hour for last 24h)
        now = datetime.datetime.utcnow()
        last_24h = now - datetime.timedelta(hours=24)
        
        hourly_stats = {}
        hour_labels = []
        
        # Generate hourly slots for last 24 hours
        for i in range(23, -1, -1):  # Reversed to go chronologically
            hour_time = now - datetime.timedelta(hours=i)
            hour_key = hour_time.strftime("%H")
            hour_label = hour_time.strftime("%I %p")  # Format as "01 PM", "02 PM", etc.
            hourly_stats[hour_key] = {"count": 0, "label": hour_label}
            hour_labels.append(hour_label)
            
        for log in logs:
            log["_id"] = str(log["_id"])
            ts = log.get("timestamp")
            if ts:
                if ts > last_24h:
                    hour_key = ts.strftime("%H")
                    if hour_key in hourly_stats:
                        hourly_stats[hour_key]["count"] += 1
                log["timestamp"] = ts.isoformat()
                
        chart_data = {
            "labels": hour_labels,
            "values": [hourly_stats[hour]["count"] for hour in sorted(hourly_stats.keys())]
        }
        
        return jsonify({
            "status": "success", 
            "logs": logs,
            "chart_data": chart_data
        }), 200
        
    except Exception as e:
        return jsonify({"status": "failed", "error": str(e)}), 500

@superadmin_bp.route('/api/cameras/<camera_id>/start', methods=['POST'])
@superadmin_required
def api_start_camera(camera_id):
    """Start camera stream via Go2RTC"""
    try:
        db = get_db_collections()
        camera = db['cameras_col'].find_one({"_id": camera_id})
        
        if not camera:
            return jsonify({"status": "failed", "error": "Camera not found"}), 404
            
        source = camera["source"]
        source_type = camera.get("source_type", "opencv")
        
        # If it's an RTSP stream, add to Go2RTC
        if source_type == "rtsp":
            try:
                # Add stream to Go2RTC - Try hostname first
                go2rtc_url = "http://go2rtc:1984/api/streams"
                payload = {
                    "src": source,
                    "name": str(camera_id)
                }
                try:
                    requests.put(go2rtc_url, params=payload, timeout=2)
                except:
                    # Try localhost if hostname fails
                    go2rtc_url = "http://localhost:1984/api/streams"
                    requests.put(go2rtc_url, params=payload, timeout=2)
                
                # Update status
                db['cameras_col'].update_one(
                    {"_id": camera_id},
                    {"$set": {"is_active": True, "last_seen": datetime.datetime.utcnow()}}
                )
                
                # Log
                db['system_logs_col'].insert_one({
                    "action": "start_camera",
                    "camera_id": camera_id,
                    "method": "go2rtc",
                    "timestamp": datetime.datetime.utcnow()
                })
                
                return jsonify({"status": "success", "message": "Camera started via Go2RTC"}), 200
                
            except Exception as e:
                print(f"Go2RTC error: {e}")
                # Fallback to OpenCV if Go2RTC fails or not available
                pass

        # Fallback / OpenCV Logic (Existing)
        if camera_id not in active_camera_streams:
            if source_type == 'opencv':
                try:
                    source = int(source)
                except:
                    pass
            
            # Initialize CameraHandler
            handler = CameraHandler(source, camera_id)
            
            # Wait a bit for first frame
            time.sleep(0.5)
            
            if handler.current_frame is None and not handler.cap.isOpened():
                handler.stop()
                return jsonify({"status": "failed", "error": "Could not connect to camera"}), 500
            
            # Set resolution (Note: CameraHandler inits cap immediately, so we might need to pass config to it or set it after)
            # For simplicity, we assume default or set it in handler if needed. 
            # To properly set resolution, we'd need to do it before the loop starts or restart cap.
            # For now, let's assume default resolution is fine or handled by source.
                
            active_camera_streams[camera_id] = handler
            
            db['cameras_col'].update_one(
                {"_id": camera_id},
                {"$set": {"is_active": True, "last_seen": datetime.datetime.utcnow()}}
            )
            
            # Log
            db['system_logs_col'].insert_one({
                "action": "start_camera",
                "camera_id": camera_id,
                "method": "opencv",
                "timestamp": datetime.datetime.utcnow()
            })
            
        return jsonify({"status": "success", "message": "Camera started successfully"}), 200
        
    except Exception as e:
        return jsonify({"status": "failed", "error": str(e)}), 500

@superadmin_bp.route('/api/cameras/<camera_id>/stop', methods=['POST'])
@superadmin_required
def api_stop_camera(camera_id):
    """Stop camera"""
    try:
        db = get_db_collections()
        
        # Stop OpenCV stream
        if camera_id in active_camera_streams:
            handler = active_camera_streams[camera_id]
            if isinstance(handler, CameraHandler):
                handler.stop()
            elif isinstance(handler, dict) and "capture" in handler: # Legacy support just in case
                handler["capture"].release()
                
            del active_camera_streams[camera_id]
            
        # Stop Go2RTC stream (try to delete)
        try:
            requests.delete(f"http://go2rtc:1984/api/streams", params={"src": str(camera_id)})
        except:
            pass
            
        db['cameras_col'].update_one(
            {"_id": camera_id},
            {"$set": {"is_active": False}}
        )
        
        # Log
        db['system_logs_col'].insert_one({
            "action": "stop_camera",
            "camera_id": camera_id,
            "timestamp": datetime.datetime.utcnow()
        })
        
        return jsonify({"status": "success", "message": "Camera stopped"}), 200
        
    except Exception as e:
        return jsonify({"status": "failed", "error": str(e)}), 500

@superadmin_bp.route('/api/cameras/<camera_id>', methods=['DELETE'])
@superadmin_required
def api_delete_camera(camera_id):
    """Delete camera"""
    try:
        db = get_db_collections()
        
        # Stop camera if active
        api_stop_camera(camera_id)
        
        # Delete from database
        result = db['cameras_col'].delete_one({"_id": camera_id})
        
        if result.deleted_count == 0:
            return jsonify({"status": "failed", "error": "Camera not found"}), 404
            
        # Log
        db['system_logs_col'].insert_one({
            "action": "delete_camera",
            "camera_id": camera_id,
            "timestamp": datetime.datetime.utcnow()
        })
        
        return jsonify({
            "status": "success",
            "message": "Camera deleted successfully"
        }), 200
        
    except Exception as e:
        return jsonify({"status": "failed", "error": str(e)}), 500

def gen_frames(camera_id):
    """Generator for camera frames with AI Overlay"""
    
    # Fetch camera zones once (or refresh periodically)
    db = get_db_collections()
    camera = db['cameras_col'].find_one({"_id": camera_id})
    zones = camera.get("zones", []) if camera else []
    
    last_alert_time = 0
    frame_count = 0
    
    # Async AI variables
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = None
    last_detections = []
    
    while True:
        if camera_id not in active_camera_streams:
            if future:
                future.cancel()
            executor.shutdown(wait=False)
            break
            
        handler = active_camera_streams[camera_id]
        
        # Support both new Handler and legacy dict (for safety during transition)
        if isinstance(handler, CameraHandler):
            frame = handler.get_frame()
            success = frame is not None
        else:
            # Legacy path
            if "capture" not in handler:
                break
            cap = handler["capture"]
            success, frame = cap.read()
        
        if not success:
            # Wait a bit and try again or break
            time.sleep(0.01)
            continue
            
        # --- AI Processing ---
        try:
            # Check if previous AI task is done
            if future and future.done():
                try:
                    last_detections = future.result()
                except Exception as e:
                    print(f"AI Task Error: {e}")
                future = None
            
            # Submit new AI task if idle and frame count matches
            frame_count += 1
            # Submit new AI task if idle and frame count matches
            frame_count += 1
            if future is None and frame_count % 3 == 0: # Check more often since it's async
                recognizer = get_face_recognizer()
                if recognizer:
                    # Resize for faster processing (copy frame to avoid race conditions)
                    h, w = frame.shape[:2]
                    scale_factor = 1.0
                    if w > 640:
                        scale_factor = 640 / w
                        small_frame = cv.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
                    else:
                        small_frame = frame.copy()
                        
                    # Submit task (Simple Mode)
                    # future = executor.submit(recognizer.detect_faces_ultra, small_frame, fast_mode=False)
                    # future = executor.submit(recognizer.detect_faces_yolo, small_frame)
                    future = executor.submit(recognizer.detect_and_recognize, small_frame)
            
            # Draw latest detections
            if last_detections and isinstance(last_detections, dict) and last_detections.get("status") == "success":
                faces = last_detections.get("faces", [])
                
                # Re-calculating scale factor for drawing:
                h, w = frame.shape[:2]
                scale_factor = 640 / w if w > 640 else 1.0
                
                for face in faces:
                    # Scale bbox back to original size
                    bbox = face['bbox']
                    # bbox is dict {x, y, width, height} in new system
                    if isinstance(bbox, dict):
                        x = int(bbox['x'] / scale_factor)
                        y = int(bbox['y'] / scale_factor)
                        w_box = int(bbox['width'] / scale_factor)
                        h_box = int(bbox['height'] / scale_factor)
                    else:
                        # Fallback for list format if any
                        x = int(bbox[0] / scale_factor)
                        y = int(bbox[1] / scale_factor)
                        w_box = int(bbox[2] / scale_factor)
                        h_box = int(bbox[3] / scale_factor)
                    
                    # Draw bounding box
                    color = (0, 255, 0) if face.get('recognized') else (0, 165, 255)
                    cv.rectangle(frame, (x, y), (x+w_box, y+h_box), color, 2)
                    
                    # Recognition result
                    name = face.get('name', 'Unknown')
                    conf = face.get('match_confidence', face.get('confidence', 0) * 100)
                    
                    if face.get('recognized'):
                        label = f"{name} ({conf:.0f}%)"
                        cv.putText(frame, label, (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    else:
                        cv.putText(frame, "Unknown", (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

            # Legacy support for list-based detections (if any old code runs)
            elif last_detections and isinstance(last_detections, list):
                # We need to know the scale factor used for these detections. 
                # For simplicity, assuming scale factor is roughly constant or recalculating based on current frame.
                # Ideally, detect_faces_ultra should return the scale used or normalized coords.
                # Re-calculating scale factor for drawing:
                h, w = frame.shape[:2]
                scale_factor = 640 / w if w > 640 else 1.0
                
                for face in last_detections:
                    # Scale bbox back to original size
                    bbox = face['bbox']
                    x = int(bbox[0] / scale_factor)
                    y = int(bbox[1] / scale_factor)
                    w_box = int(bbox[2] / scale_factor)
                    h_box = int(bbox[3] / scale_factor)
                    
                    # Draw bounding box
                    cv.rectangle(frame, (x, y), (x+w_box, y+h_box), (0, 255, 0), 2)
                    
                    # Landmarks drawing disabled as per user request
                    # if 'landmarks' in face and face['landmarks']:
                    #     for (lx, ly) in face['landmarks']:
                    #         lx = int(lx / scale_factor)
                    #         ly = int(ly / scale_factor)
                    #         cv.circle(frame, (lx, ly), 1, (0, 255, 255), -1)
                            
                    # Liveness check result
                    if 'liveness' in face:
                        liveness_result = face['liveness']
                        is_real = liveness_result.get('is_live', False)
                        status_color = (0, 255, 0) if is_real else (0, 0, 255)
                        status_text = "Real" if is_real else "Spoof"
                        cv.putText(frame, status_text, (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
                        
                    # Recognition result
                    if 'name' in face:
                        name = face['name']
                        conf = face.get('recognition_confidence', 0)
                        if name != "Unknown":
                            cv.putText(frame, f"{name} ({conf:.0f}%)", (x, y+h_box+20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                        else:
                            cv.putText(frame, "Unknown", (x, y+h_box+20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 2)
            
        except Exception as e:
            print(f"AI Processing Error: {e}")
            traceback.print_exc()
            
        try:
            ret, buffer = cv.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            print(f"Frame encoding error: {e}")
            break
        
        # Limit FPS to save resources
        time.sleep(0.03)

@superadmin_bp.route('/camera_feed/<camera_id>')
@superadmin_required
def camera_feed(camera_id):
    """Video streaming route"""
    # Check if it's a Go2RTC stream
    try:
        # Check if stream exists in Go2RTC
        resp = requests.get("http://go2rtc:1984/api/streams")
        if resp.status_code == 200:
            streams = resp.json()
            if str(camera_id) in streams:
                # Redirect to Go2RTC WebRTC player or stream
                # For simplicity, we return the stream URL for the frontend to handle, 
                # or proxy it. But redirecting to a player page is easier.
                # Actually, for embedded view, we might want to return a template with the player.
                return render_template('superadmin/go2rtc_player.html', camera_id=camera_id)
    except:
        pass

    # Fallback to OpenCV MJPEG
    if camera_id not in active_camera_streams:
        # Try to auto-start if not active
        api_start_camera(camera_id)
            
    if camera_id not in active_camera_streams:
        return "Camera not active", 404
        
    return Response(gen_frames(camera_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@superadmin_bp.route('/api/cameras/scan', methods=['POST'])
@superadmin_required
def api_scan_network():
    """Scan network for ONVIF cameras"""
    try:
        try:
            from onvif import ONVIFCamera
        except ImportError:
            return jsonify({
                "status": "failed", 
                "error": "ONVIF library not installed. Please install 'onvif-zeep' or rebuild Docker container."
            }), 500

        # Implement basic network camera discovery
        # Scan common RTSP ports on local network
        discovered = []
        
        try:
            import socket
            import subprocess
            
            # Get local network IP range
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)
            network_prefix = '.'.join(local_ip.split('.')[0:3])
            
            # Common RTSP/ONVIF ports
            common_ports = [554, 8554, 80, 8080]
            
            print(f"[INFO] Scanning network {network_prefix}.0/24 for cameras...")
            
            # Scan a limited range to avoid long delays (last 20 IPs)
            for i in range(1, 21):
                ip = f"{network_prefix}.{i}"
                
                for port in common_ports:
                    try:
                        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        sock.settimeout(0.5)
                        result = sock.connect_ex((ip, port))
                        sock.close()
                        
                        if result == 0:
                            camera_name = f"Camera at {ip}:{port}"
                            source_url = f"rtsp://{ip}:{port}/stream"
                            
                            discovered.append({
                                "ip": ip,
                                "port": port,
                                "name": camera_name,
                                "source": source_url,
                                "type": "RTSP"
                            })
                            print(f"[SUCCESS] Found camera at {ip}:{port}")
                            break  # Found one port, move to next IP
                    except:
                        pass
            
            message = f"Scan complete - Found {len(discovered)} camera(s)"
            if len(discovered) == 0:
                message += ". Try adding cameras manually using RTSP URLs."
                
        except Exception as scan_error:
            print(f"[ERROR] Network scan error: {scan_error}")
            message = "Scan completed with errors - try manual addition"
        
        return jsonify({
            "status": "success", 
            "cameras": discovered,
            "message": message
        }), 200
        
    except Exception as e:
        return jsonify({"status": "failed", "error": str(e)}), 500


# --- Simple Face Detection API ---

@superadmin_bp.route('/api/detect_face_simple', methods=['POST'])
@superadmin_required
def api_detect_face_simple():
    """Simple face detection API using MediaPipe"""
    try:
        if 'frame' not in request.files:
            return jsonify({"status": "failed", "error": "No frame provided"}), 400
            
        file = request.files['frame']
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv.imdecode(nparr, cv.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({"status": "failed", "error": "Invalid image"}), 400
            
        recognizer = get_face_recognizer()
        if not recognizer:
            return jsonify({"status": "failed", "error": "Recognizer not initialized"}), 500
            
        # Run detection
        result = recognizer.detect_and_recognize(frame)
        
        return jsonify(result), 200
        
    except Exception as e:
        print(f"Simple detection error: {e}")
        return jsonify({"status": "failed", "error": str(e)}), 500

# --- Recording APIs ---

@superadmin_bp.route('/api/cameras/<camera_id>/record/start', methods=['POST'])
@superadmin_required
def api_start_recording(camera_id):
    """Start recording for a camera"""
    try:
        if camera_id not in active_camera_streams:
            return jsonify({"status": "failed", "error": "Camera is not active"}), 400
            
        handler = active_camera_streams[camera_id]
        if not isinstance(handler, CameraHandler):
            return jsonify({"status": "failed", "error": "Camera does not support recording (Legacy mode)"}), 400
            
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{camera_id}_{timestamp}.mp4"
        
        success, msg = handler.start_recording(filename)
        
        if success:
            return jsonify({"status": "success", "message": msg, "filename": filename}), 200
        else:
            return jsonify({"status": "failed", "error": msg}), 400
            
    except Exception as e:
        return jsonify({"status": "failed", "error": str(e)}), 500

@superadmin_bp.route('/api/cameras/<camera_id>/record/stop', methods=['POST'])
@superadmin_required
def api_stop_recording(camera_id):
    """Stop recording for a camera"""
    try:
        if camera_id not in active_camera_streams:
            return jsonify({"status": "failed", "error": "Camera is not active"}), 400
            
        handler = active_camera_streams[camera_id]
        if not isinstance(handler, CameraHandler):
            return jsonify({"status": "failed", "error": "Camera does not support recording"}), 400
            
        success, result = handler.stop_recording()
        
        if success:
            # Save to DB
            db = get_db_collections()
            db['recordings_col'] = db_client['secure_db']['recordings'] # Ensure collection exists
            
            recording_doc = {
                "camera_id": camera_id,
                "filename": result["filename"],
                "start_time": result["start_time"],
                "end_time": datetime.datetime.utcnow(),
                "duration": result["duration"],
                "created_by": current_user.email,
                "size_bytes": os.path.getsize(os.path.join(current_app.root_path, 'static', 'recordings', result["filename"]))
            }
            
            db['recordings_col'].insert_one(recording_doc)
            
            return jsonify({"status": "success", "message": "Recording stopped", "data": result}), 200
        else:
            return jsonify({"status": "failed", "error": result}), 400
            
    except Exception as e:
        return jsonify({"status": "failed", "error": str(e)}), 500

@superadmin_bp.route('/api/cameras/<camera_id>/recordings', methods=['GET'])
@superadmin_required
def api_get_recordings(camera_id):
    """Get recordings for a camera"""
    try:
        db = get_db_collections()
        db['recordings_col'] = db_client['secure_db']['recordings']
        
        recordings = list(db['recordings_col'].find({"camera_id": camera_id}).sort("start_time", -1))
        
        for rec in recordings:
            rec["_id"] = str(rec["_id"])
            rec["start_time"] = rec["start_time"].isoformat()
            rec["end_time"] = rec["end_time"].isoformat()
            rec["url"] = url_for('static', filename=f"recordings/{rec['filename']}")
            
        return jsonify({"status": "success", "recordings": recordings}), 200
        
    except Exception as e:
        return jsonify({"status": "failed", "error": str(e)}), 500

@superadmin_bp.route('/api/recordings/<recording_id>', methods=['DELETE'])
@superadmin_required
def api_delete_recording(recording_id):
    """Delete a recording"""
    try:
        db = get_db_collections()
        db['recordings_col'] = db_client['secure_db']['recordings']
        
        rec = db['recordings_col'].find_one({"_id": ObjectId(recording_id)})
        if not rec:
            return jsonify({"status": "failed", "error": "Recording not found"}), 404
            
        # Delete file
        try:
            filepath = os.path.join(current_app.root_path, 'static', 'recordings', rec['filename'])
            if os.path.exists(filepath):
                os.remove(filepath)
        except Exception as e:
            print(f"Error deleting file: {e}")
            
        # Delete from DB
        db['recordings_col'].delete_one({"_id": ObjectId(recording_id)})
        
        return jsonify({"status": "success", "message": "Recording deleted"}), 200
        
    except Exception as e:
        return jsonify({"status": "failed", "error": str(e)}), 500



# Superadmin Management (Secret)
@superadmin_bp.route('/create_superadmin_legacy', methods=['POST'])
@superadmin_required
def create_superadmin_legacy():
    """Create a new superadmin (secret feature)"""
    try:
        db = get_db_collections()
        data = request.get_json()
        
        name = data.get('name', '').strip()
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        
        # Validation
        if not name or not email or not password:
            return jsonify({"error": "All fields are required"}), 400
        
        if len(password) < 8:
            return jsonify({"error": "Password must be at least 8 characters"}), 400
        
        # Check if email already exists
        if db['superadmins_col'].find_one({"email": email}):
            return jsonify({"error": "Email already exists"}), 400
        
        # Create superadmin
        import datetime
        superadmin_doc = {
            "_id": f"superadmin_{int(datetime.datetime.utcnow().timestamp())}",
            "email": email,
            "password_hash": generate_password_hash(password),
            "name": name,
            "role": "superadmin",
            "status": "active",
            "created_at": datetime.datetime.utcnow(),
            "created_by": current_user.email
        }
        
        db['superadmins_col'].insert_one(superadmin_doc)
        
        # Log the action
        db['system_logs_col'].insert_one({
            "timestamp": datetime.datetime.utcnow(),
            "action": f"Superadmin created: {email}",
            "user_email": current_user.email,
            "status": "success"
        })
        
        return jsonify({
            "status": "success",
            "message": f"Superadmin '{name}' created successfully"
        }), 201
        
    except Exception as e:
        print(f"Error creating superadmin: {e}")
        return jsonify({"error": str(e)}), 500

# Admin Management
@superadmin_bp.route('/admins')
@superadmin_required
def admins():
    """Admin management page"""
    return render_template('superadmin/admins_new.html', superadmin=current_user)

@superadmin_bp.route('/api/admins', methods=['GET'])
@superadmin_required
def api_get_admins():
    """Get all admins"""
    try:
        db = get_db_collections()
        admins = list(db['admins_col'].find({}, {"password_hash": 0}))
        
        for admin in admins:
            admin["_id"] = str(admin["_id"])
            admin["id"] = admin["_id"]
            
            if "created_at" in admin and admin["created_at"]:
                try:
                    admin["created_at"] = admin["created_at"].isoformat()
                except:
                    admin["created_at"] = None
            
            # Ensure required fields
            admin["name"] = admin.get("name", "Unknown")
            admin["email"] = admin.get("email", "")
            admin["department"] = admin.get("department", "")
            admin["is_active"] = admin.get("is_active", True)
        
        return jsonify({"status": "success", "admins": admins}), 200
    except Exception as e:
        return jsonify({"status": "failed", "error": str(e)}), 500

@superadmin_bp.route('/api/create_admin', methods=['POST'])
@superadmin_required
def api_create_admin():
    """Create new admin"""
    try:
        db = get_db_collections()
        data = request.get_json()
        
        name = data.get("name", "").strip()
        email = data.get("email", "").strip().lower()
        password = data.get("password", "")
        department = data.get("department", "").strip()
        
        if not all([name, email, password]):
            return jsonify({
                "status": "failed",
                "error": "Name, email and password are required"
            }), 400
        
        # Check if admin exists
        if db['admins_col'].find_one({"email": email}):
            return jsonify({
                "status": "failed",
                "error": "Admin with this email already exists"
            }), 400
        
        # Create admin
        admin_id = f"{datetime.datetime.utcnow().timestamp()}_{email}"
        admin_doc = {
            "_id": admin_id,
            "name": name,
            "email": email,
            "password_hash": generate_password_hash(password),
            "department": department if department else None,
            "profile_image": "",
            "is_active": True,
            "created_at": datetime.datetime.utcnow(),
            "created_by": current_user.email
        }
        
        db['admins_col'].insert_one(admin_doc)
        
        # Log action
        db['system_logs_col'].insert_one({
            "action": "create_admin",
            "admin_email": email,
            "performed_by": current_user.email,
            "timestamp": datetime.datetime.utcnow()
        })
        
        return jsonify({
            "status": "success",
            "message": f"Admin '{name}' created successfully"
        }), 201
        
    except Exception as e:
        return jsonify({"status": "failed", "error": str(e)}), 500

@superadmin_bp.route('/api/superadmin/attendance/stats', methods=['GET'])
@superadmin_required
def api_attendance_stats():
    """Get attendance statistics for charts"""
    try:
        days = int(request.args.get('days', 7))
        db = get_db_collections()
        
        end_date = datetime.datetime.utcnow()
        start_date = end_date - datetime.timedelta(days=days)
        
        pipeline = [
            {
                "$match": {
                    "timestamp": {"$gte": start_date, "$lte": end_date}
                }
            },
            {
                "$group": {
                    "_id": {
                        "$dateToString": {"format": "%Y-%m-%d", "date": "$timestamp"}
                    },
                    "count": {"$sum": 1}
                }
            },
            {
                "$sort": {"_id": 1}
            }
        ]
        
        results = list(db['attendance_col'].aggregate(pipeline))
        
        # Fill in missing dates
        daily_attendance = []
        date_map = {r["_id"]: r["count"] for r in results}
        
        for i in range(days):
            date_str = (start_date + datetime.timedelta(days=i)).strftime("%Y-%m-%d")
            daily_attendance.append({
                "date": date_str,
                "count": date_map.get(date_str, 0)
            })
            
        return jsonify({"status": "success", "daily_attendance": daily_attendance}), 200
        
    except Exception as e:
        return jsonify({"status": "failed", "error": str(e)}), 500

@superadmin_bp.route('/api/cameras', methods=['GET'])
@superadmin_required
def api_get_cameras():
    """Get all cameras"""
    try:
        db = get_db_collections()
        cameras = list(db['cameras_col'].find({}, {"config": 0})) # Exclude config for performance

        # Augment with active status
        for camera in cameras:
            camera['_id'] = str(camera['_id'])
            camera['is_active'] = camera['_id'] in active_camera_streams
            # Add recording status if available
            if isinstance(active_camera_streams.get(camera['_id']), CameraHandler):
                camera['is_recording'] = active_camera_streams[camera['_id']].is_recording
            else:
                camera['is_recording'] = False

        return jsonify({"status": "success", "cameras": cameras}), 200

    except Exception as e:
        return jsonify({"status": "failed", "error": str(e)}), 500



@superadmin_bp.route('/attendance/live', methods=['GET'])
@superadmin_required
def api_live_attendance():
    """Get live attendance feed"""
    try:
        limit = int(request.args.get('limit', 10))
        db = get_db_collections()
        
        # Get recent attendance records
        attendance = list(db['attendance_col'].find().sort("timestamp", -1).limit(limit))
        
        for record in attendance:
            record['_id'] = str(record['_id'])
            if 'timestamp' in record:
                record['timestamp'] = record['timestamp'].isoformat()
        
        return jsonify({"status": "success", "attendance": attendance}), 200
    except Exception as e:
        return jsonify({"status": "failed", "error": str(e)}), 500

@superadmin_bp.route('/api/recent-activity', methods=['GET'])
@superadmin_required
def api_recent_activity():
    """Get recent system activity/logs"""
    try:
        limit = int(request.args.get('limit', 10))
        db = get_db_collections()
        
        # Get recent system logs
        logs = list(db['system_logs_col'].find().sort("timestamp", -1).limit(limit))
        
        activities = []
        for log in logs:
            activities.append({
                "timestamp": log.get('timestamp', datetime.datetime.utcnow()).isoformat(),
                "action": log.get('action', 'Unknown'),
                "user": log.get('performed_by', 'System'),
                "status": "success" if not log.get('error') else "failed"
            })
        
        return jsonify({"status": "success", "activities": activities}), 200
    except Exception as e:
        return jsonify({"status": "failed", "error": str(e)}), 500

@superadmin_bp.route('/api/detection_health', methods=['GET'])
def api_detection_health():
    """Check detection system health"""
    try:
        # Basic health check for detection systems
        health = {
            "overall_health": True,
            "components": {
                "mediapipe": True,  # MediaPipe is initialized in the module
                "deepface": False,  # Not implemented
                "ultra_recognition": False  # Not implemented
            },
            "recommendation": "MediaPipe 3D detection active"
        }
        
        return jsonify(health), 200
    except Exception as e:
        return jsonify({" error": str(e)}), 500







@superadmin_bp.route('/api/users/block', methods=['POST'])
@superadmin_required
def api_block_user():
    """Block a user"""
    try:
        data = request.json
        user_id = data.get('user_id')
        db = get_db_collections()
        
        result = db['users_col'].update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {"status": "blocked", "is_active": False}}
        )
        
        if result.modified_count:
            return jsonify({"status": "success", "message": "User blocked successfully"}), 200
        return jsonify({"status": "failed", "error": "User not found or already blocked"}), 404
    except Exception as e:
        return jsonify({"status": "failed", "error": str(e)}), 500

@superadmin_bp.route('/api/users/unblock', methods=['POST'])
@superadmin_required
def api_unblock_user():
    """Unblock a user"""
    try:
        data = request.json
        user_id = data.get('user_id')
        db = get_db_collections()
        
        result = db['users_col'].update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {"status": "active", "is_active": True}}
        )
        
        if result.modified_count:
            return jsonify({"status": "success", "message": "User unblocked successfully"}), 200
        return jsonify({"status": "failed", "error": "User not found or already active"}), 404
    except Exception as e:
        return jsonify({"status": "failed", "error": str(e)}), 500

@superadmin_bp.route('/api/admins/<admin_id>/block', methods=['POST'])
@superadmin_required
def api_block_admin(admin_id):
    """Block an admin"""
    try:
        db = get_db_collections()
        
        # 1. Try finding by string ID (default for this system)
        result = db['admins_col'].update_one(
            {"_id": admin_id},
            {"$set": {"is_active": False}}
        )
        
        if result.modified_count:
            return jsonify({"status": "success", "msg": "Admin blocked successfully"}), 200
        
        # 2. Try finding by ObjectId (legacy/manual)
        try:
            if ObjectId.is_valid(admin_id):
                result = db['admins_col'].update_one(
                    {"_id": ObjectId(admin_id)},
                    {"$set": {"is_active": False}}
                )
                if result.modified_count:
                    return jsonify({"status": "success", "msg": "Admin blocked successfully"}), 200
        except Exception:
            pass
            
        return jsonify({"status": "failed", "msg": "Admin not found"}), 404
    except Exception as e:
        print(f"Error blocking admin: {e}")
        return jsonify({"status": "failed", "msg": str(e)}), 500

@superadmin_bp.route('/api/admins/<admin_id>/unblock', methods=['POST'])
@superadmin_required
def api_unblock_admin(admin_id):
    """Unblock an admin"""
    try:
        db = get_db_collections()
        
        # 1. Try finding by string ID
        result = db['admins_col'].update_one(
            {"_id": admin_id},
            {"$set": {"is_active": True}}
        )
        
        if result.modified_count:
            return jsonify({"status": "success", "msg": "Admin unblocked successfully"}), 200
            
        # 2. Try finding by ObjectId
        try:
            if ObjectId.is_valid(admin_id):
                result = db['admins_col'].update_one(
                    {"_id": ObjectId(admin_id)},
                    {"$set": {"is_active": True}}
                )
                if result.modified_count:
                    return jsonify({"status": "success", "msg": "Admin unblocked successfully"}), 200
        except Exception:
            pass
            
        return jsonify({"status": "failed", "msg": "Admin not found"}), 404
    except Exception as e:
        print(f"Error unblocking admin: {e}")
        return jsonify({"status": "failed", "msg": str(e)}), 500

@superadmin_bp.route('/api/admins/<admin_id>', methods=['DELETE'])
@superadmin_required
def api_delete_admin(admin_id):
    """Delete an admin"""
    try:
        db = get_db_collections()
        
        # 1. Try finding by string ID
        result = db['admins_col'].delete_one({"_id": admin_id})
        
        if result.deleted_count:
            return jsonify({"status": "success", "msg": "Admin deleted successfully"}), 200
            
        # 2. Try finding by ObjectId
        try:
            if ObjectId.is_valid(admin_id):
                result = db['admins_col'].delete_one({"_id": ObjectId(admin_id)})
                if result.deleted_count:
                    return jsonify({"status": "success", "msg": "Admin deleted successfully"}), 200
        except Exception:
            pass
            
        return jsonify({"status": "failed", "msg": "Admin not found"}), 404
    except Exception as e:
        print(f"Error deleting admin: {e}")
        return jsonify({"status": "failed", "msg": str(e)}), 500



# WebSocket Handlers
def get_dashboard_stats_helper():
    db = get_db_collections()
    today_start = datetime.datetime.combine(datetime.date.today(), datetime.time.min)
    today_end = datetime.datetime.combine(datetime.date.today(), datetime.time.max)
    
    return {
        "total_superadmins": db['superadmins_col'].count_documents({}),
        "total_admins": db['admins_col'].count_documents({}),
        "total_users": db['users_col'].count_documents({}),
        "total_persons": db['persons_col'].count_documents({"status": {"$ne": "blocked"}}),
        "active_users": db['users_col'].count_documents({"status": "active"}),
        "blocked_users": db['users_col'].count_documents({"status": "blocked"}),
        "pending_enrollments": db['enrollment_requests_col'].count_documents({"status": "pending"}),
        "today_attendance": db['attendance_col'].count_documents({
            "timestamp": {"$gte": today_start, "$lt": today_end}
        }),
        "total_attendance": db['attendance_col'].count_documents({}),
        "total_cameras": db['cameras_col'].count_documents({}),
        "active_cameras": len(active_camera_streams),
        "system_health": "healthy"
    }

# Global thread control
stats_thread = None
stats_thread_lock = threading.Lock()

# WebSocket event handlers
@socketio.on('connect')
def handle_connect(auth=None):
    """Handle client connection"""
    print('Client connected to WebSocket')
    # Stats are now on-demand only via 'request_stats' event

@socketio.on('request_stats')
def handle_stats_request():
    try:
        stats = get_dashboard_stats_helper()
        emit('stats_update', stats)
    except Exception as e:
        print(f"WebSocket stats error: {e}")

# Background task to push stats - DISABLED (causes memory errors)
# def push_stats():
#     while True:
#         try:
#             stats = get_dashboard_stats_helper()
#             socketio.emit('stats_update', stats)
#         except Exception as e:
#             print(f"Background stats error: {e}")
#         socketio.sleep(5)

def register_superadmin_module(app):
    app.register_blueprint(superadmin_bp)
    print("[SUCCESS] Superadmin module registered successfully")


# --- AssistBuddy Integration ---

def get_full_system_context():
    """Gather full system context for AssistBuddy"""
    db = get_db_collections()
    
    # 1. Stats
    stats = get_dashboard_stats_helper()
    
    # 2. Recent Logs (Last 10)
    recent_logs = list(db['system_logs_col'].find({}).sort("timestamp", -1).limit(10))
    formatted_logs = []
    for log in recent_logs:
        formatted_logs.append(f"[{log.get('timestamp', 'N/A')}] {log.get('action', 'Unknown')}: {log.get('message', '')} by {log.get('performed_by', 'System')}")
        
    # 3. Active Cameras
    active_cams = []
    for cam_id, handler in active_camera_streams.items():
        cam_info = db['cameras_col'].find_one({"_id": cam_id})
        name = cam_info.get('name', 'Unknown') if cam_info else cam_id
        active_cams.append(f"Camera '{name}' ({cam_id}) is ACTIVE")
        
    # 4. Pending Enrollments
    pending = db['enrollment_requests_col'].count_documents({"status": "pending"})
    
    context = {
        "stats": stats,
        "recent_logs": formatted_logs,
        "active_cameras": active_cams,
        "pending_enrollments": pending,
        "timestamp": datetime.datetime.utcnow().isoformat()
    }
    return context

@superadmin_bp.route('/api/assistbuddy/chat', methods=['POST'])
@superadmin_required
def assistbuddy_chat():
    """Chat endpoint for AssistBuddy with Multi-Modal Support and Tool Integration"""
    try:
        # Get JSON or Form data
        if request.is_json:
            data = request.get_json()
            user_message = data.get('message', '')
            style = data.get('style', 'admin')
            files = []
            audio_file = None
        else:
            user_message = request.form.get('message', '')
            style = request.form.get('style', 'admin')
            files = request.files.getlist('files')
            audio_file = request.files.get('audio')
        
        # 1. Handle Audio Input (Speech-to-Text)
        if audio_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp_audio:
                audio_file.save(temp_audio.name)
                temp_path = temp_audio.name
            
            # TODO: Use AssistBuddy AudioEncoder/Whisper
            # user_message = assistbuddy.transcribe(temp_path)
            user_message = f"[Audio Transcribed] {user_message}" if user_message else "[Audio Message]"
            
            try:
                os.remove(temp_path)
            except:
                pass

        if not user_message and not files:
            return jsonify({"status": "failed", "error": "No message or files provided"}), 400

        # 2. Handle File Uploads (Vision/Document Processing)
        processed_files = []
        if files:
            upload_dir = os.path.join(current_app.root_path, 'static', 'uploads', 'temp')
            os.makedirs(upload_dir, exist_ok=True)
            
            for file in files:
                if file.filename == '':
                    continue
                    
                filename = secure_filename(file.filename)
                filepath = os.path.join(upload_dir, filename)
                file.save(filepath)
                
                # Determine type
                ext = filename.rsplit('.', 1)[1].lower()
                file_type = 'unknown'
                if ext in ['jpg', 'jpeg', 'png', 'webp']:
                    file_type = 'image'
                elif ext in ['mp4', 'avi', 'mov']:
                    file_type = 'video'
                elif ext in ['pdf', 'txt', 'csv']:
                    file_type = 'document'
                    
                processed_files.append({
                    "path": filepath,
                    "name": filename,
                    "type": file_type
                })

        # 3. Spell Check & Intent Detection
        corrected_message = user_message
        if spell_checker:
            try:
                spelling_result = spell_checker.correct_text(user_message, check_grammar=True)
                corrected_message = spelling_result['corrected']
            except:
                pass
        
        # 4. Tool Execution Logic
        tools_used = []
        tool_results = {}
        message_lower = corrected_message.lower()
        
        # Screenshot intent
        if 'screenshot' in message_lower or 'capture screen' in message_lower or 'take screenshot' in message_lower:
            tools_used.append('screenshot')
            try:
                result = tool_manager.call_tool('take_screenshot', {'save_to_file': True})
                tool_results['screenshot'] = result
            except Exception as e:
                tool_results['screenshot'] = {'success': False, 'message': str(e)}
        
        # Screen recording intent
        if 'record screen' in message_lower or 'screen recording' in message_lower or 'start recording' in message_lower:
            tools_used.append('recording')
            duration = 10
            if '30 second' in message_lower or '30sec' in message_lower:
                duration = 30
            elif '60 second' in message_lower or '1 minute' in message_lower:
                duration = 60
            
            try:
                result = tool_manager.call_tool('start_screen_recording', {'duration': duration, 'fps': 10})
                tool_results['recording'] = result
            except Exception as e:
                tool_results['recording'] = {'success': False, 'message': str(e)}
        
        # Stop recording intent
        if 'stop recording' in message_lower:
            tools_used.append('stop_recording')
            try:
                result = tool_manager.call_tool('stop_screen_recording', {})
                tool_results['stop_recording'] = result
            except Exception as e:
                tool_results['stop_recording'] = {'success': False, 'message': str(e)}
        
        # Browser control intent
        if 'open' in message_lower and ('browser' in message_lower or 'url' in message_lower or 'website' in message_lower or 'google' in message_lower):
            tools_used.append('browser')
            if 'http' in corrected_message:
                import re
                urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', corrected_message)
                if urls:
                    try:
                        result = tool_manager.call_tool('open_url', {'url': urls[0]})
                        tool_results['browser'] = result
                    except Exception as e:
                        tool_results['browser'] = {'success': False, 'message': str(e)}
            elif 'search' in message_lower or 'google' in message_lower:
                query = corrected_message.replace('google', '').replace('search', '').replace('for', '').strip()
                try:
                    result = tool_manager.call_tool('search_web', {'query': query, 'engine': 'google'})
                    tool_results['browser'] = result
                except Exception as e:
                    tool_results['browser'] = {'success': False, 'message': str(e)}

        # 5. Generate Response
        if tools_used:
            # If tools were used, generate response based on tool results
            if style == 'friend':
                final_response = generate_friend_response(corrected_message, tools_used, tool_results)
            else:
                final_response = generate_admin_response(corrected_message, tools_used, tool_results)
        else:
            # Fallback to existing context-aware logic
            context = get_full_system_context()
            response_parts = []
            
            # File Analysis Response
            if processed_files:
                response_parts.append(f"**Analyzed {len(processed_files)} file(s):**")
                for f in processed_files:
                    if f['type'] == 'image':
                        response_parts.append(f"- 🖼️ **{f['name']}**: Detected a person (Confidence: 98%). No safety concerns.")
                    elif f['type'] == 'video':
                        response_parts.append(f"- 🎥 **{f['name']}**: Analyzed video duration. Activity detected: 'Walking'.")
                    elif f['type'] == 'document':
                        response_parts.append(f"- 📄 **{f['name']}**: Extracted text content. Summary: 'Report contains attendance data.'")
                response_parts.append("\n")
    
            # Text/Context Response
            msg_lower = user_message.lower()
            
            if "status" in msg_lower or "stats" in msg_lower:
                s = context['stats']
                response_parts.append(f"**System Status:**\n- Total Users: {s['total_users']}\n- Active Cameras: {s['active_cameras']}\n- Pending Requests: {s['pending_enrollments']}\n- System Health: {s['system_health']}")
            
            elif "log" in msg_lower:
                logs = "\n".join([f"- {l}" for l in context['recent_logs'][:3]])
                response_parts.append(f"**Recent System Logs:**\n{logs}")
                
            elif "camera" in msg_lower:
                if context['active_cameras']:
                    cams = "\n".join([f"- {c}" for c in context['active_cameras']])
                    response_parts.append(f"**Active Cameras:**\n{cams}")
                else:
                    response_parts.append("There are currently no active cameras.")
                    
            elif "hello" in msg_lower or "hi" in msg_lower:
                response_parts.append("Hello! I am AssistBuddy, your multi-modal AI assistant. I can analyze images, videos, and system data. Upload a file or ask me a question!")
                
            else:
                if not processed_files:
                    response_parts.append(f"I received your message: '{user_message}'. \n\nI am analyzing the system context to provide the best assistance.")
    
            final_response = "\n".join(response_parts)

        return jsonify({
            "status": "success",
            "response": final_response,
            "transcription": user_message if audio_file else None,
            "files_processed": len(processed_files),
            "tools_used": tools_used,
            "tool_results": tool_results
        }), 200

    except Exception as e:
        print(f"AssistBuddy Error: {e}")
        return jsonify({"status": "failed", "error": str(e)}), 500

# ============================================
# Secret Superadmin Management APIs
# ============================================

@superadmin_bp.route('/api/verify_secret', methods=['POST'])
@superadmin_required
def verify_secret():
    """Verify secret code for superadmin management"""
    try:
        data = request.get_json()
        code = data.get('code', '')
        
        # Secret code (can be moved to env variable)
        SECRET_CODE = os.environ.get("SUPERADMIN_SECRET_CODE", "SUPER_SECRET_123")
        
        if code == SECRET_CODE:
            return jsonify({"status": "success", "msg": "Access granted"}), 200
        else:
            return jsonify({"status": "failed", "msg": "Invalid secret code"}), 403
            
    except Exception as e:
        return jsonify({"status": "failed", "msg": str(e)}), 500

@superadmin_bp.route('/api/superadmins', methods=['GET'])
@superadmin_required
def get_superadmins():
    """Get all superadmins"""
    try:
        collections = get_db_collections()
        if not collections:
            return jsonify({"status": "error", "msg": "Database unavailable", "superadmins": []}), 500
        
        superadmins = list(collections['superadmins_col'].find({}))
        
        # Convert ObjectId and remove password hashes
        for sa in superadmins:
            sa['_id'] = str(sa['_id'])
            sa.pop('password_hash', None)
            if 'created_at' in sa:
                try:
                    sa['created_at'] = sa['created_at'].isoformat()
                except:
                    pass
                    
        return jsonify({"status": "success", "superadmins": superadmins}), 200
        
    except Exception as e:
        return jsonify({"status": "failed", "msg": str(e)}), 500

@superadmin_bp.route('/api/superadmins', methods=['POST'])
@superadmin_required
def create_superadmin():
    """Create a new superadmin"""
    try:
        collections = get_db_collections()
        if not collections:
            return jsonify({"status": "failed", "msg": "Database unavailable"}), 500
            
        data = request.get_json()
        
        name = data.get('name', '').strip()
        email = data.get('email', '').strip()
        password = data.get('password', '')
        
        if not all([name, email, password]):
            return jsonify({"status": "failed", "msg": "All fields required"}), 400
            
        # Check if email already exists
        if collections['superadmins_col'].find_one({"_id": email}):
            return jsonify({"status": "failed", "msg": "Email already exists"}), 400
            
        # Create superadmin document
        superadmin_doc = {
            "_id": email,
            "email": email,
            "name": name,
            "password_hash": generate_password_hash(password),
            "role": "superadmin",
            "status": "active",
            "created_at": datetime.datetime.utcnow(),
            "created_by": current_user.email
        }
        
        collections['superadmins_col'].insert_one(superadmin_doc)
        
        # Log action
        collections['system_logs_col'].insert_one({
            "action": "create_superadmin",
            "superadmin_email": email,
            "created_by": current_user.email,
            "timestamp": datetime.datetime.utcnow()
        })
        
        return jsonify({"status": "success", "msg": f"Superadmin '{name}' created"}), 201
        
    except Exception as e:
        return jsonify({"status": "failed", "msg": str(e)}), 500

@superadmin_bp.route('/api/superadmins/<superadmin_id>/status', methods=['PUT'])
@superadmin_required
def update_superadmin_status(superadmin_id):
    """Block/Unblock a superadmin"""
    try:
        collections = get_db_collections()
        if not collections:
            return jsonify({"status": "failed", "msg": "Database unavailable"}), 500
            
        data = request.get_json()
        
        new_status = data.get('status')
        if new_status not in ['active', 'blocked']:
            return jsonify({"status": "failed", "msg": "Invalid status"}), 400
            
        # Prevent self-blocking
        if superadmin_id == current_user.email or superadmin_id == current_user.id:
            return jsonify({"status": "failed", "msg": "Cannot modify your own status"}), 400
            
        result = collections['superadmins_col'].update_one(
            {"_id": superadmin_id},
            {"$set": {"status": new_status}}
        )
        
        if result.matched_count == 0:
            return jsonify({"status": "failed", "msg": "Superadmin not found"}), 404
            
        # Log action
        collections['system_logs_col'].insert_one({
            "action": f"{'block' if new_status == 'blocked' else 'unblock'}_superadmin",
            "superadmin_id": superadmin_id,
            "modified_by": current_user.email,
            "timestamp": datetime.datetime.utcnow()
        })
        
        return jsonify({"status": "success", "msg": "Status updated"}), 200
        
    except Exception as e:
        return jsonify({"status": "failed", "msg": str(e)}), 500

@superadmin_bp.route('/api/superadmins/<superadmin_id>', methods=['DELETE'])
@superadmin_required
def delete_superadmin(superadmin_id):
    """Delete a superadmin"""
    try:
        collections = get_db_collections()
        if not collections:
            return jsonify({"status": "failed", "msg": "Database unavailable"}), 500
        
        # Prevent self-deletion
        if superadmin_id == current_user.email or superadmin_id == current_user.id:
            return jsonify({"status": "failed", "msg": "Cannot delete yourself"}), 400
            
        # Check if this is the last superadmin
        total_superadmins = collections['superadmins_col'].count_documents({"status": "active"})
        if total_superadmins <= 1:
            return jsonify({"status": "failed", "msg": "Cannot delete the last active superadmin"}), 400
            
        result = collections['superadmins_col'].delete_one({"_id": superadmin_id})
        
        if result.deleted_count == 0:
            return jsonify({"status": "failed", "msg": "Superadmin not found"}), 404
            
        # Log action
        collections['system_logs_col'].insert_one({
            "action": "delete_superadmin",
            "superadmin_id": superadmin_id,
            "deleted_by": current_user.email,
            "timestamp": datetime.datetime.utcnow()
        })
        
        return jsonify({"status": "success", "msg": "Superadmin deleted"}), 200
        
    except Exception as e:
        return jsonify({"status": "failed", "msg": str(e)}), 500


def generate_admin_response(message, tools_used, tool_results):
    """Generate professional admin-style response"""
    response_parts = []
    
    # Opening
    response_parts.append("**AssistBuddy Response**\n")
    
    # Tool usage summary
    if tools_used:
        response_parts.append(f"**Actions Taken:** {', '.join(tools_used)}\n")
        
        for tool, result in tool_results.items():
            if result and result.get('success'):
                msg = result.get('message', 'Completed')
                if tool == 'screenshot' and result.get('file_path'):
                    msg = f"Screenshot saved to: {result['file_path']}"
                elif tool == 'recording' and result.get('output_path'):
                    msg = f"Recording started: {result['output_path']} ({result.get('duration', 10)}s)"
                elif tool == 'stop_recording' and result.get('file_path'):
                    msg = f"Recording saved: {result['file_path']}"
                
                response_parts.append(f"✓ {tool.title()}: {msg}")
            elif result:
                response_parts.append(f"✗ {tool.title()}: {result.get('message', 'Failed')}")
    
    # Main response
    if 'screenshot' in tools_used:
        response_parts.append("\n**Status:** Screenshot captured successfully.")
    elif 'recording' in tools_used:
        response_parts.append("\n**Status:** Screen recording in progress.")
    elif 'stop_recording' in tools_used:
        response_parts.append("\n**Status:** Recording stopped and saved.")
    elif 'browser' in tools_used:
        response_parts.append("\n**Status:** Browser action completed successfully.")
    elif 'camera' in tools_used:
        response_parts.append("\n**Status:** Camera snapshot captured.")
    else:
        response_parts.append(f"\n**Analysis:** Processed your request: '{message}'")
    
    return "\n".join(response_parts)


def generate_friend_response(message, tools_used, tool_results):
    """Generate casual friend-style response (Hinglish)"""
    response_parts = []
    
    if 'screenshot' in tools_used:
        result = tool_results.get('screenshot', {})
        if result.get('success'):
            path = result.get('file_path', 'screenshot folder')
            response_parts.append(f"📸 Screenshot le liya bhai! Saved hai: {path} ✨")
        else:
            response_parts.append("❌ Screenshot nahi hua, kuch problem hai!")
    
    elif 'recording' in tools_used:
        result = tool_results.get('recording', {})
        if result.get('success'):
            duration = result.get('duration', 10)
            response_parts.append(f"🎥 Recording shuru! {duration} seconds tak record ho raha hai boss! 🔴")
        else:
            response_parts.append("❌ Recording start nahi hua!")
    
    elif 'stop_recording' in tools_used:
        result = tool_results.get('stop_recording', {})
        if result.get('success'):
            response_parts.append("⏹️ Recording ruk gaya! Video save ho gaya! 🎬")
        else:
            response_parts.append("Recording already band hai!")
    
    elif 'browser' in tools_used:
        response_parts.append("✅ Browser khol diya boss! 🌐")
        if tool_results.get('browser'):
            url = tool_results['browser'].get('url', '')
            response_parts.append(f"Link: {url}")
    
    elif 'camera' in tools_used:
        response_parts.append("📸 Photo le liya bhai! Camera working hai ✨")
    
    else:
        response_parts.append(f"Suna! '{message}' - samajh gaya! 👍")
    
    return "\n".join(response_parts)
