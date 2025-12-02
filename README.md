# AttendX - AI Attendance System (Web App)

This directory contains the Flask web application for the AI Attendance System, designed for deployment on Render.

## Features

- **User Management**: Registration, authentication, and profile management
- **Super Admin Dashboard**: Comprehensive system management and monitoring
- **AssistBuddy AI Chat**: Integrated AI assistant with tool management
- **Camera Management**: Support for webcams and IP cameras
- **Real-time Attendance**: Face recognition powered by separate microservice
- **Gallery**: Screenshots and video recordings management

## Architecture

This web app communicates with a separate **Face Recognition Engine** deployed on Hugging Face Spaces via HTTP API for all ML operations (face detection, recognition, enrollment).

## Deployment on Render

### Prerequisites

1. MongoDB instance (Docker container or MongoDB Atlas)
2. Face Recognition Engine deployed on Hugging Face Spaces
3. Render account

### Deployment Steps

1. **Create New Web Service** on Render
   - Connect your GitHub repository
   - Select this directory (`attendX`) as the root directory
   - Choose "Docker" as the environment

2. **Configure Environment Variables**:
   ```
   FLASK_SECRET=<generate-random-secret>
   MONGODB_URI=mongodb://mongodb:27017/...
   FACE_ENGINE_URL=https://your-hf-space.hf.space
   PORT=10000
   ```

3. **Deploy**: Render will build and deploy automatically

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your values

# Run the app
python main.py
```

## File Structure

```
attendX/
├── main.py                 # Main Flask application
├── superadmin_module.py    # Super admin routes and logic
├── assistbuddy_api.py      # AssistBuddy API endpoints
├── database_utils.py       # Database connection utilities
├── extensions.py           # Flask extensions (SocketIO)
├── simple_face_recognition.py  # Simple face recognizer (fallback)
├── templates/              # HTML templates
├── static/                 # CSS, JS, images
├── assistbuddy/            # AssistBuddy package
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker configuration
└── render.yaml            # Render deployment config
```

## API Integration

The web app connects to the Face Engine via HTTP:

- `POST /api/embed` - Generate face embeddings
- `POST /api/recognize` - Recognize faces
- `POST /api/enroll` - Process enrollment
- `POST /api/liveness` - Liveness detection

## Dependencies

See `requirements.txt` for complete list. Notable dependencies:
- Flask 3.0.0
- Flask-SocketIO 5.3.6
- pymongo 4.6.1
- Flask-Login 0.6.3
- requests 2.31.0 (for Face Engine API calls)

## Environment Variables

| Variable | Description |
|----------|-------------|
| `FLASK_SECRET` | Flask secret key for sessions |
| `MONGODB_URI` | MongoDB connection string |
| `FACE_ENGINE_URL` | URL of Face Recognition Engine |
| `PORT` | Port to run the app (default: 10000) |
| `ADMIN_EMAIL` | Default admin email |
| `ADMIN_PASSWORD` | Default admin password |

## License

MIT License
