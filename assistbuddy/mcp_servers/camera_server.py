"""
MCP Camera Server - Webcam Snapshot Tool
Provides camera access via Model Context Protocol
"""

import asyncio
import base64
import io
from typing import Optional

import cv2
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, ImageContent


app = Server("camera-server")


def capture_snapshot(camera_index: int = 0, max_width: int = 1024) -> Optional[str]:
    """
    Capture a single frame from the webcam
    
    Args:
        camera_index: Camera device index (default 0)
        max_width: Maximum width for resizing (default 1024px)
    
    Returns:
        Base64-encoded JPEG string or None if error
    """
    cap = None
    try:
        # Open camera
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open camera {camera_index}. Camera may be in use.")
        
        # Capture frame
        ret, frame = cap.read()
        
        if not ret or frame is None:
            raise RuntimeError("Failed to capture frame from camera")
        
        # Resize if needed (maintain aspect ratio)
        height, width = frame.shape[:2]
        if width > max_width:
            scale = max_width / width
            new_width = max_width
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Encode as JPEG
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
        _, buffer = cv2.imencode('.jpg', frame, encode_param)
        
        # Convert to Base64
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        
        return jpg_as_text
    
    except cv2.error as e:
        raise RuntimeError(f"OpenCV error: {str(e)}")
    
    except Exception as e:
        raise RuntimeError(f"Camera capture failed: {str(e)}")
    
    finally:
        # Always release camera
        if cap is not None:
            cap.release()


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available camera tools"""
    return [
        Tool(
            name="get_snapshot",
            description=(
                "Capture a single frame from the computer's webcam. "
                "Returns a Base64-encoded JPEG image (starts with '/9j/'). "
                "Image is automatically resized to max 1024px width to save tokens. "
                "Use this when the user asks 'What do you see?', 'Look at the camera', "
                "or needs visual context from their environment."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "camera_index": {
                        "type": "integer",
                        "description": "Camera device index (0 for default camera)",
                        "default": 0
                    },
                    "max_width": {
                        "type": "integer",
                        "description": "Maximum image width in pixels (default 1024)",
                        "default": 1024
                    }
                },
                "required": []
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Execute camera tool"""
    
    if name != "get_snapshot":
        raise ValueError(f"Unknown tool: {name}")
    
    # Extract parameters
    camera_index = arguments.get("camera_index", 0)
    max_width = arguments.get("max_width", 1024)
    
    # Validate inputs
    if not isinstance(camera_index, int) or camera_index < 0:
        return [TextContent(
            type="text",
            text=f"Error: Invalid camera_index {camera_index}. Must be non-negative integer."
        )]
    
    if not isinstance(max_width, int) or max_width < 100 or max_width > 4096:
        return [TextContent(
            type="text",
            text=f"Error: Invalid max_width {max_width}. Must be between 100-4096."
        )]
    
    try:
        # Capture snapshot
        base64_image = capture_snapshot(camera_index, max_width)
        
        if base64_image is None:
            return [TextContent(
                type="text",
                text="Error: Failed to capture snapshot. Camera may be unavailable."
            )]
        
        # Return Base64 image with description
        response_text = (
            f"Camera snapshot captured successfully from device {camera_index}.\n"
            f"Image encoded as Base64 JPEG (starts with '/9j/').\n"
            f"Image size: {len(base64_image)} characters.\n\n"
            f"Base64 Data:\n{base64_image}"
        )
        
        return [TextContent(
            type="text",
            text=response_text
        )]
    
    except RuntimeError as e:
        # Handle specific camera errors
        error_msg = str(e)
        
        if "in use" in error_msg.lower():
            return [TextContent(
                type="text",
                text=(
                    f"Error: Camera {camera_index} is currently in use by another application. "
                    "Please close other camera apps and try again."
                )
            )]
        else:
            return [TextContent(
                type="text",
                text=f"Error capturing snapshot: {error_msg}"
            )]
    
    except Exception as e:
        return [TextContent(
            type="text",
            text=f"Unexpected error: {str(e)}"
        )]


async def main():
    """Run the MCP camera server"""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
