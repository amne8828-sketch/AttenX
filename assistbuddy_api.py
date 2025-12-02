"""
AssistBuddy Chat API for SuperAdmin Dashboard
Provides REST API endpoints for chat interaction with AssistBuddy AI
Supports file uploads, memory, and intelligent intent detection
"""

from flask import Blueprint, request, jsonify, current_app
from assistbuddy.tool_manager import get_tool_manager
from assistbuddy.utils.spell_checker import SpellingCorrector
from assistbuddy.tools.user_management import get_user_management_tool
from werkzeug.utils import secure_filename
import json
import os
import tempfile
import datetime

# Create Blueprint for AssistBuddy API under SuperAdmin
assistbuddy_bp = Blueprint('assistbuddy', __name__, url_prefix='/superadmin/api/assistbuddy')

# Initialize components
spell_checker = SpellingCorrector()
tool_manager = get_tool_manager()
user_tool = get_user_management_tool()

# Import memory manager (ChromaDB-based)
try:
    from assistbuddy.mcp_servers.memory_manager import MemoryManager
    memory_manager = MemoryManager()
    MEMORY_AVAILABLE = True
except Exception as e:
    print(f"Memory manager not available: {e}")
    MEMORY_AVAILABLE = False
    memory_manager = None

# Store chat history and sessions
chat_sessions = {}  # session_id -> list of messages


def get_session_id():
    """Get or create session ID for the current user"""
    # For now, use a simple session ID. In production, use Flask sessions
    return "superadmin_session"


@assistbuddy_bp.route('/chat', methods=['POST'])
def chat():
    """
    Main chat endpoint for AssistBuddy
    
    Accepts both JSON and multipart/form-data
    
    Form Fields:
        - message: User message text
        - files: Optional file uploads (images, videos, etc.)
    
    Returns:
        {
            "status": "success",
            "response": "AssistBuddy response",
            "tool_results": {},
            "form_data": {} // If user creation is triggered
        }
    """
    try:
        session_id = get_session_id()
        
        # Initialize session if needed
        if session_id not in chat_sessions:
            chat_sessions[session_id] = []
        
        # Parse request (support both JSON and form-data)
        if request.is_json:
            data = request.json
            message = data.get('message', '')
            uploaded_files = []
        else:
            # Form data with file upload
            message = request.form.get('message', '')
            uploaded_files = request.files.getlist('files')
        
        check_spelling = request.form.get('check_spelling', 'true').lower() == 'true'
        
        # Step 1: Spell check
        corrected_message = message
        spelling_result = None
        
        if check_spelling and message:
            try:
                spelling_result = spell_checker.correct_text(message, check_grammar=True)
                corrected_message = spelling_result['corrected']
            except Exception as e:
                print(f"Spell check error: {e}")
        
        # Step 2: Save uploaded files
        saved_files = []
        if uploaded_files:
            upload_folder = os.path.join(current_app.root_path, 'static', 'uploads')
            os.makedirs(upload_folder, exist_ok=True)
            
            for file in uploaded_files:
                if file and file.filename:
                    filename = secure_filename(file.filename)
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    unique_filename = f"{timestamp}_{filename}"
                    filepath = os.path.join(upload_folder, unique_filename)
                    file.save(filepath)
                    saved_files.append({
                        'filename': unique_filename,
                        'original_name': file.filename,
                        'path': filepath
                    })
        
        # Step 3: Retrieve relevant memory context
        context = []
        if MEMORY_AVAILABLE and corrected_message:
            try:
                memories = memory_manager.query_memory(corrected_message, top_k=3)
                context = [m['content'] for m in memories] if memories else []
            except Exception as e:
                print(f"Memory retrieval error: {e}")
        
        # Step 4: Intent detection (improved)
        intent, confidence = detect_intent(corrected_message, saved_files, context)
        
        # Step 5: Handle intent
        tools_used = []
        tool_results = {}
        form_data = None
        response_text = ""
        
        if intent == "create_user":
            # User creation workflow
            response_text, form_data = handle_user_creation(corrected_message, saved_files)
            
        elif intent == "screenshot":
            tools_used.append('screenshot')
            try:
                result = tool_manager.call_tool('take_screenshot', {'save_to_file': True})
                tool_results['screenshot'] = result
                if result.get('success'):
                    response_text = f"📸 Screenshot captured successfully!\n\nSaved to: `{result.get('file_path')}`"
                else:
                    response_text = f"❌ Screenshot failed: {result.get('message')}"
            except Exception as e:
                tool_results['screenshot'] = {'success': False, 'message': str(e)}
                response_text = f"❌ Screenshot error: {str(e)}"
                
        elif intent == "start_recording":
            tools_used.append('recording')
            # Extract duration
            duration = 10
            if '30' in corrected_message:
                duration = 30
            elif '60' in corrected_message or 'minute' in corrected_message.lower():
                duration = 60
                
            try:
                result = tool_manager.call_tool('start_screen_recording', {'duration': duration, 'fps': 10})
                tool_results['recording'] = result
                if result.get('success'):
                    response_text = f"🎥 Screen recording started!\n\nDuration: {duration} seconds\nFile: `{result.get('output_path')}`"
                else:
                    response_text = f"❌ Recording failed: {result.get('message')}"
            except Exception as e:
                tool_results['recording'] = {'success': False, 'message': str(e)}
                response_text = f"❌ Recording error: {str(e)}"
                
        elif intent == "stop_recording":
            tools_used.append('stop_recording')
            try:
                result = tool_manager.call_tool('stop_screen_recording', {})
                tool_results['stop_recording'] = result
                if result.get('success'):
                    response_text = f"⏹️ Recording stopped!\n\nSaved to: `{result.get('file_path')}`"
                else:
                    response_text = f"⚠️ {result.get('message')}"
            except Exception as e:
                tool_results['stop_recording'] = {'success': False, 'message': str(e)}
                response_text = f"❌ Error: {str(e)}"
                
        elif intent == "browser":
            tools_used.append('browser')
            # Check if URL is provided
            if 'http' in corrected_message:
                import re
                urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', corrected_message)
                if urls:
                    try:
                        result = tool_manager.call_tool('open_url', {'url': urls[0]})
                        tool_results['browser'] = result
                        response_text = f"🌐 Opened {urls[0]} in your browser"
                    except Exception as e:
                        response_text = f"❌ Browser error: {str(e)}"
            else:
                # Search query
                query = corrected_message.replace('google', '').replace('search', '').replace('for', '').strip()
                try:
                    result = tool_manager.call_tool('search_web', {'query': query, 'engine': 'google'})
                    tool_results['browser'] = result
                    response_text = f"🔍 Searching Google for: {query}"
                except Exception as e:
                    response_text = f"❌ Search error: {str(e)}"
                    
        else:
            # Conversational response
            response_text = generate_conversational_response(corrected_message, saved_files, context)
        
        # Step 6: Store in memory
        if MEMORY_AVAILABLE:
            try:
                # Store user message
                memory_manager.add_memory(message, {
                    "role": "user",
                    "session_id": session_id,
                    "timestamp": datetime.datetime.utcnow().isoformat()
                })
                
                # Store assistant response
                memory_manager.add_memory(response_text, {
                    "role": "assistant",
                    "session_id": session_id,
                    "timestamp": datetime.datetime.utcnow().isoformat(),
                    "intent": intent
                })
            except Exception as e:
                print(f"Memory storage error: {e}")
        
        # Step 7: Update session history
        chat_sessions[session_id].append({
            'user': message,
            'corrected': corrected_message,
            'assistant': response_text,
            'intent': intent,
            'tools_used': tools_used,
            'files': [f['original_name'] for f in saved_files],
            'timestamp': datetime.datetime.now().isoformat()
        })
        
        # Keep only last 50 messages
        if len(chat_sessions[session_id]) > 50:
            chat_sessions[session_id] = chat_sessions[session_id][-50:]
        
        result = {
            'status': 'success',
            'response': response_text,
            'intent': intent,
            'confidence': confidence,
            'tools_used': tools_used,
            'tool_results': tool_results,
            'corrected_input': corrected_message if check_spelling and message != corrected_message else None,
            'files_uploaded': len(saved_files)
        }
        
        # Add form data if user creation
        if form_data:
            result['form_data'] = form_data
        
        return jsonify(result), 200
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


def detect_intent(message: str, files: list, context: list) -> tuple:
    """
    Improved intent detection with keyword weighting and context
    
    Returns:
        (intent, confidence_score)
    """
    if not message:
        return ("none", 0.0)
    
    msg_lower = message.lower().strip()
    
    # Define intent keywords with weights and negative keywords
    intents = {
        "create_user": {
            "required_keywords": [
                ["create", "add", "new", "register", "enroll"],  # One of these must be present
                ["user"]  # AND "user" must be present
            ],
            "boost_keywords": ["email", "name", "password", "photo", "profile"],
            "weight": 1.5,
            "file_boost": 0.5  # Strong boost if image file is attached
        },
        "screenshot": {
            "required_keywords": [["screenshot", "screen capture", "capture screen", "print screen", "take screenshot"]],
            "boost_keywords": [],
            "weight": 2.0,  # High weight for exact matches
            "file_boost": 0.0
        },
        "start_recording": {
            "required_keywords": [["record", "recording"], ["screen", "start"]],
            "boost_keywords": ["video", "capture"],
            "negative_keywords": ["stop", "end", "finish"],  # Don't match if these are present
            "weight": 2.0,
            "file_boost": 0.0
        },
        "stop_recording": {
            "required_keywords": [["stop", "end", "finish"], ["record", "recording"]],
            "boost_keywords": [],
            "weight": 2.0,
            "file_boost": 0.0
        },
        "browser": {
            "required_keywords": [["open", "browse", "go to", "navigate", "visit"]],
            "boost_keywords": ["http", "www", "url", "website", "link"],
            "weight": 1.5,
            "file_boost": 0.0
        },
        "search_web": {
            "required_keywords": [["search", "google", "find", "look up"]],
            "boost_keywords": ["for", "about"],
            "weight": 1.3,
            "file_boost": 0.0
        }
    }
    
    scores = {}
    
    for intent_name, config in intents.items():
        score = 0.0
        
        # Check required keywords (all groups must match)
        required_groups = config.get("required_keywords", [])
        all_groups_matched = True
        
        for group in required_groups:
            group_matched = False
            for keyword in group:
                if keyword in msg_lower:
                    group_matched = True
                    break
            if not group_matched:
                all_groups_matched = False
                break
        
        if not all_groups_matched:
            scores[intent_name] = 0.0
            continue
        
        # Check negative keywords (if present, penalize heavily)
        negative_keywords = config.get("negative_keywords", [])
        if any(neg in msg_lower for neg in negative_keywords):
            scores[intent_name] = 0.0
            continue
        
        # Base score for required match
        score += config["weight"]
        
        # Boost keywords
        for keyword in config.get("boost_keywords", []):
            if keyword in msg_lower:
                score += 0.3
        
        # File boost for create_user
        if files and config["file_boost"] > 0:
            # Check if any file is an image
            has_image = any(f['original_name'].lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.avif', '.jfif')) for f in files)
            if has_image:
                score += config["file_boost"]
        
        scores[intent_name] = score
    
    # Get best match
    if scores:
        best_intent = max(scores, key=scores.get)
        best_score = scores[best_intent]
        
        # Threshold: require at least 1.0 confidence for action intents
        if best_score >= 1.0:
            # Normalize score to 0-1 range
            confidence = min(best_score / 3.0, 1.0)
            return (best_intent, confidence)
    
    # Default: conversational
    return ("conversation", 0.0)


def handle_user_creation(message: str, files: list) -> tuple:
    """
    Handle user creation intent
    
    If all details are provided, create the user.
    Otherwise, return a form for the user to fill.
    
    Returns:
        (response_text, form_data)
    """
    # Parse details from message
    parsed = user_tool.parse_user_details(message)
    
    # Check if photo is uploaded
    photo_path = None
    if files:
        for f in files:
            if f['original_name'].lower().endswith(('.png', '.jpg', '.jpeg')):
                photo_path = f['path']
                break
    
    # Check if we have all required fields
    has_all = all([parsed.get('name'), parsed.get('email'), parsed.get('password')])
    
    if has_all and photo_path:
        # Create user immediately
        result = user_tool.create_user(
            name=parsed['name'],
            email=parsed['email'],
            password=parsed['password'],
            department=parsed.get('department', 'General'),
            photo_path=photo_path
        )
        
        if result['success']:
            response = f"✅ **User Created Successfully!**\n\n" \
                      f"**Name:** {parsed['name']}\n" \
                      f"**Email:** {parsed['email']}\n" \
                      f"**Department:** {parsed.get('department', 'General')}\n" \
                      f"**Face Enrolled:** {'Yes ✓' if result.get('enrolled') else 'No'}\n\n" \
                      f"The user can now log in and access the system."
        else:
            response = f"❌ **User Creation Failed**\n\n{result['message']}"
        
        return (response, None)
    else:
        # Return form data for frontend to render
        form_data = {
            "name": parsed.get('name', ''),
            "email": parsed.get('email', ''),
            "password": parsed.get('password', ''),
            "department": parsed.get('department', 'General'),
            "has_photo": photo_path is not None,
            "photo_files": [f['original_name'] for f in files if f['original_name'].lower().endswith(('.png', '.jpg', '.jpeg'))] if files else []
        }
        
        response = "📝 **Please complete the user registration form:**\n\n" \
                   "I've detected you want to create a new user. Please provide the following details:\n" \
                   "- Full Name\n" \
                   "- Email Address\n" \
                   "- Password\n" \
                   "- Department (optional)\n" \
                   "- Profile Photo (optional)"
        
        return (response, form_data)


def generate_conversational_response(message: str, files: list, context: list) -> str:
    """
    Generate a conversational response for non-command messages
    """
    msg_lower = message.lower().strip()
    
    # Greetings
    if any(word in msg_lower for word in ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening']):
        return "👋 **Hello!** I'm AssistBuddy, your AI assistant.\n\n" \
               "I can help you with:\n" \
               "• **User Management** - Create users with face enrollment\n" \
               "• **Screen Capture** - Take screenshots and record your screen\n" \
               "• **Web Control** - Open websites and perform searches\n\n" \
               "What would you like me to do?"
    
    # Help
    if any(word in msg_lower for word in ['help', 'what can you do', 'capabilities', 'commands', 'features']):
        return "🤖 **AssistBuddy Features:**\n\n" \
               "**📸 Screen Capture**\n" \
               "• `\"Take a screenshot\"` - Capture current screen\n" \
               "• `\"Record screen for 30 seconds\"` - Start screen recording\n" \
               "• `\"Stop recording\"` - Stop current recording\n\n" \
               "**👤 User Management**\n" \
               "• `\"Create user [name] email [email]\"` - Create new user\n" \
               "• Upload a photo to enable face enrollment\n\n" \
               "**🌐 Web Control**\n" \
               "• `\"Open https://example.com\"` - Open a website\n" \
               "• `\"Search for Python tutorials\"` - Google search\n\n" \
               "Just tell me what you need in plain English!"
    
    # Thanks
    if any(word in msg_lower for word in ['thank', 'thanks', 'appreciate']):
        return "😊 You're very welcome! Let me know if you need anything else."
    
    # Status/How are you
    if any(phrase in msg_lower for phrase in ['how are you', 'how do you do', 'hows it going']):
        return "I'm functioning perfectly and ready to assist! 🚀\n\nWhat can I help you with today?"
    
    # File upload acknowledgment
    if files:
        file_count = len(files)
        file_types = {}
        for f in files:
            ext = f['original_name'].split('.')[-1].upper()
            file_types[ext] = file_types.get(ext, 0) + 1
        
        type_str = ', '.join([f"{count} {ext}" for ext, count in file_types.items()])
        
        is_image = all(f['original_name'].lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.avif', '.jfif')) for f in files)
        
        if is_image and file_count >= 1:
            return f"📎 **I received {file_count} image(s)** ({type_str})\n\n" \
                   "**What would you like to do?**\n" \
                   "• Create a new user with these photos\n" \
                   "• Analyze the images\n\n" \
                   "Just let me know!"
        else:
            return f"📎 **I received {file_count} file(s)** ({type_str})\n\n" \
                   "How can I help you with " + ("these files?" if file_count > 1 else "this file?")
    
    # Unclear intent - provide gentle guidance
    short_message = len(msg_lower.split()) <= 3
    
    if short_message:
        return f"I heard you say: *\"{message}\"*\n\n" \
               "I'm here to help! Try:\n" \
               "• **\"Take a screenshot\"** to capture your screen\n" \
               "• **\"Create user\"** to add a new user\n" \
               "• **\"Help\"** to see all my features"
    else:
        return f"I understand you said: *\"{message}\"*\n\n" \
               "I'm not quite sure what action you'd like me to take. Here's what I can do:\n\n" \
               "**Quick Actions:**\n" \
               "• Screenshot or screen recording\n" \
               "• Create new users\n" \
               "• Open websites or search\n\n" \
               "Type **\"help\"** for detailed examples!"


@assistbuddy_bp.route('/tools', methods=['GET'])
def list_tools():
    """List available tools"""
    try:
        tools = tool_manager.list_tools()
        # Add user management tool
        tools.append({
            "name": "create_user",
            "description": "Create a new user with face enrollment",
            "category": "user_management",
            "available": True
        })
        return jsonify({
            'success': True,
            'tools': tools,
            'count': len(tools)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'tools': []
        })


@assistbuddy_bp.route('/history', methods=['GET'])
def get_history():
    """Get chat history"""
    session_id = get_session_id()
    limit = request.args.get('limit', 50, type=int)
    
    history = chat_sessions.get(session_id, [])
    
    return jsonify({
        'success': True,
        'history': history[-limit:],
        'total': len(history)
    })


@assistbuddy_bp.route('/clear-history', methods=['POST'])
def clear_history():
    """Clear chat history"""
    session_id = get_session_id()
    if session_id in chat_sessions:
        chat_sessions[session_id] = []
    
    return jsonify({
        'success': True,
        'message': 'Chat history cleared'
    })


@assistbuddy_bp.route('/spell-check', methods=['POST'])
def spell_check():
    """Standalone spell check endpoint"""
    try:
        data = request.json
        text = data.get('text', '')
        
        result = spell_checker.correct_text(text, check_grammar=True)
        
        return jsonify({
            'success': True,
            **result
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
