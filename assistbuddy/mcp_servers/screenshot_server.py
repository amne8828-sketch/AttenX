"""
MCP Screenshot & Screen Recording Server
Provides screen capture capabilities via Model Context Protocol
"""

import asyncio
import base64
import io
from datetime import datetime
from pathlib import Path
from typing import Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

try:
    from PIL import ImageGrab
    import pyautogui
except ImportError:
    raise ImportError("Install: pip install pillow pyautogui")


app = Server("screenshot-server")

# Storage for screenshots
SCREENSHOT_DIR = Path(__file__).parent / "screenshots"
SCREENSHOT_DIR.mkdir(exist_ok=True)


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available screenshot tools"""
    return [
        Tool(
            name="take_screenshot",
            description=(
                "Capture a screenshot of the current screen. "
                "Returns Base64-encoded PNG image. "
                "Optionally save to file for later reference."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "save_to_file": {
                        "type": "boolean",
                        "description": "Save screenshot to file (default: False)",
                        "default": False
                    },
                    "max_width": {
                        "type": "integer",
                        "description": "Maximum width for resizing (default: 1920)",
                        "default": 1920
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="take_region_screenshot",
            description=(
                "Capture a specific region of the screen. "
                "Requires x, y coordinates and width, height."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "x": {"type": "integer", "description": "Left edge X coordinate"},
                    "y": {"type": "integer", "description": "Top edge Y coordinate"},
                    "width": {"type": "integer", "description": "Region width"},
                    "height": {"type": "integer", "description": "Region height"},
                    "save_to_file": {"type": "boolean", "default": False}
                },
                "required": ["x", "y", "width", "height"]
            }
        ),
        Tool(
            name="get_screen_size",
            description="Get current screen resolution (width x height)",
            inputSchema={"type": "object", "properties": {}, "required": []}
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Execute screenshot tools"""
    
    try:
        # Tool: take_screenshot
        if name == "take_screenshot":
            save_to_file = arguments.get("save_to_file", False)
            max_width = arguments.get("max_width", 1920)
            
            # Capture full screen
            screenshot = ImageGrab.grab()
            
            # Resize if needed
            if screenshot.width > max_width:
                scale = max_width / screenshot.width
                new_width = max_width
                new_height = int(screenshot.height * scale)
                screenshot = screenshot.resize((new_width, new_height))
            
            # Convert to Base64
            buffer = io.BytesIO()
            screenshot.save(buffer, format='PNG')
            base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Optionally save to file
            file_path = None
            if save_to_file:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_path = SCREENSHOT_DIR / f"screenshot_{timestamp}.png"
                screenshot.save(file_path)
            
            response = (
                f"Screenshot captured: {screenshot.width}x{screenshot.height} px\n"
                f"Base64 size: {len(base64_image)} characters\n"
            )
            if file_path:
                response += f"Saved to: {file_path}\n"
            response += f"\nBase64 Data:\n{base64_image}"
            
            return [TextContent(type="text", text=response)]
        
        # Tool: get_screen_size
        elif name == "get_screen_size":
            screen_size = pyautogui.size()
            
            return [TextContent(
                type="text",
                text=f"Screen resolution: {screen_size.width}x{screen_size.height} px"
            )]
        
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
    
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def main():
    """Run the MCP screenshot server"""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
