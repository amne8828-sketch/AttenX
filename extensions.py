from flask_socketio import SocketIO

# Initialize SocketIO with CORS enabled
socketio = SocketIO(cors_allowed_origins="*")
