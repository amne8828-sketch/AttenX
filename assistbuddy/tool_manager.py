
import base64
import webbrowser
import os
import datetime
import threading
import time
from typing import Dict, Any, Optional, List
from io import BytesIO

# Import optional dependencies
try:
    from PIL import ImageGrab
    import pyautogui
    SCREENSHOT_AVAILABLE = True
except ImportError:
    SCREENSHOT_AVAILABLE = False
    print("[WARNING] Screenshot tools not available. Install: pip install pillow pyautogui")

try:
    import cv2
    import numpy as np
    RECORDING_AVAILABLE = True
except ImportError:
    RECORDING_AVAILABLE = False
    print("[WARNING] Recording not available. Install: pip install opencv-python")


class ToolManager:
    """Manages all AssistBuddy tools without external MCP dependencies"""
    
    def __init__(self):
        self.recording_active = False
        self.recording_thread = None
        self.recording_path = None
        
        # Create output directories in static folder for web access
        os.makedirs("static/screenshots", exist_ok=True)
        os.makedirs("static/recordings", exist_ok=True)
    
    def list_tools(self) -> List[Dict[str, str]]:
        """List all available tools"""
        tools = [
            {
                "name": "take_screenshot",
                "description": "Capture a screenshot of the screen",
                "category": "screen",
                "available": SCREENSHOT_AVAILABLE
            },
            {
                "name": "start_screen_recording",
                "description": "Start recording the screen",
                "category": "screen",
                "available": SCREENSHOT_AVAILABLE and RECORDING_AVAILABLE
            },
            {
                "name": "stop_screen_recording",
                "description": "Stop screen recording",
                "category": "screen",
                "available": True
            },
            {
                "name": "open_url",
                "description": "Open a URL in browser",
                "category": "browser",
                "available": True
            },
            {
                "name": "search_web",
                "description": "Search the web",
                "category": "browser",
                "available": True
            },
            {
                "name": "get_screen_size",
                "description": "Get screen resolution",
                "category": "screen",
                "available": SCREENSHOT_AVAILABLE
            }
        ]
        return [t for t in tools if t['available']]
    
    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool by name"""
        
        # Map tool names to methods
        tool_map = {
            'take_screenshot': self.take_screenshot,
            'start_screen_recording': self.start_screen_recording,
            'stop_screen_recording': self.stop_screen_recording,
            'open_url': self.open_url,
            'search_web': self.search_web,
            'get_screen_size': self.get_screen_size
        }
        
        if tool_name not in tool_map:
            return {
                "success": False,
                "message": f"Unknown tool: {tool_name}"
            }
        
        try:
            return tool_map[tool_name](**arguments)
        except Exception as e:
            return {
                "success": False,
                "message": f"Tool execution error: {str(e)}"
            }
    
    # ========== Screenshot Tools ==========
    
    def take_screenshot(self, save_to_file: bool = True, filename: Optional[str] = None) -> Dict[str, Any]:
        """Take a screenshot"""
        if not SCREENSHOT_AVAILABLE:
            return {
                "success": False,
                "message": "Screenshot library not available. Install: pip install pillow pyautogui"
            }
        
        try:
            screenshot = ImageGrab.grab()
            
            # Generate filename
            if filename is None:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"screenshot_{timestamp}"
            
            file_path = None
            if save_to_file:
                file_path = os.path.join("static", "screenshots", f"{filename}.png")
                screenshot.save(file_path)
            
            # Convert to base64 (optional)
            buffered = BytesIO()
            screenshot.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            return {
                "success": True,
                "message": "Screenshot captured successfully",
                "file_path": file_path,
                "width": screenshot.width,
                "height": screenshot.height,
                "timestamp": datetime.datetime.now().isoformat(),
                "image_base64": img_base64[:100] + "..." if len(img_base64) > 100 else img_base64  # Truncate for response
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Screenshot failed: {str(e)}"
            }
    
    def start_screen_recording(self, duration: int = 10, fps: int = 10, filename: Optional[str] = None) -> Dict[str, Any]:
        """Start screen recording"""
        if not SCREENSHOT_AVAILABLE or not RECORDING_AVAILABLE:
            return {
                "success": False,
                "message": "Recording not available. Install: pip install pillow opencv-python"
            }
        
        if self.recording_active:
            return {
                "success": False,
                "message": "Recording already in progress"
            }
        
        try:
            # Limit duration
            duration = min(duration, 60)
            
            # Generate filename
            if filename is None:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"recording_{timestamp}"
            
            self.recording_path = os.path.join("static", "recordings", f"{filename}.mp4")
            
            # Start recording thread
            self.recording_thread = threading.Thread(
                target=self._record_screen,
                args=(self.recording_path, duration, fps),
                daemon=True
            )
            self.recording_thread.start()
            self.recording_active = True
            
            return {
                "success": True,
                "message": f"Recording started for {duration} seconds",
                "duration": duration,
                "fps": fps,
                "output_path": self.recording_path
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Recording failed: {str(e)}"
            }
    
    def stop_screen_recording(self) -> Dict[str, Any]:
        """Stop screen recording"""
        if not self.recording_active:
            return {
                "success": False,
                "message": "No recording in progress"
            }
        
        self.recording_active = False
        
        # Wait for thread to finish
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=2)
        
        return {
            "success": True,
            "message": "Recording stopped",
            "file_path": self.recording_path
        }
    
    def get_screen_size(self) -> Dict[str, Any]:
        """Get screen resolution"""
        if not SCREENSHOT_AVAILABLE:
            return {
                "success": False,
                "message": "Screenshot library not available"
            }
        
        try:
            width, height = pyautogui.size()
            return {
                "success": True,
                "width": width,
                "height": height,
                "message": f"Screen size: {width}x{height}"
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed: {str(e)}"
            }
    
    def _record_screen(self, output_path: str, duration: int, fps: int):
        """Background recording function"""
        try:
            screenshot = ImageGrab.grab()
            width, height = screenshot.size
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            start_time = time.time()
            frame_count = 0
            
            while self.recording_active and (time.time() - start_time) < duration:
                screenshot = ImageGrab.grab()
                frame = np.array(screenshot)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video_writer.write(frame)
                frame_count += 1
                time.sleep(1.0 / fps)
            
            video_writer.release()
            self.recording_active = False
            print(f"✅ Recording saved: {output_path} ({frame_count} frames)")
            
        except Exception as e:
            print(f"Recording error: {e}")
            self.recording_active = False
    
    # ========== Browser Tools ==========
    
    def open_url(self, url: str, new_window: bool = False) -> Dict[str, Any]:
        """Open URL in browser"""
        try:
            if new_window:
                webbrowser.open_new(url)
            else:
                webbrowser.open(url)
            
            return {
                "success": True,
                "message": f"Opened {url}",
                "url": url
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to open URL: {str(e)}",
                "url": url
            }
    
    def search_web(self, query: str, engine: str = "google") -> Dict[str, Any]:
        """Search the web"""
        try:
            engines = {
                "google": f"https://www.google.com/search?q={query}",
                "bing": f"https://www.bing.com/search?q={query}",
                "duckduckgo": f"https://duckduckgo.com/?q={query}"
            }
            
            search_url = engines.get(engine.lower(), engines["google"])
            webbrowser.open(search_url)
            
            return {
                "success": True,
                "message": f"Searching for '{query}' on {engine}",
                "query": query,
                "engine": engine,
                "url": search_url
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Search failed: {str(e)}",
                "query": query,
                "engine": engine
            }


# Global instance
_tool_manager = None

def get_tool_manager() -> ToolManager:
    """Get global tool manager instance"""
    global _tool_manager
    if _tool_manager is None:
        _tool_manager = ToolManager()
    return _tool_manager


# Convenience functions
def take_screenshot(**kwargs):
    return get_tool_manager().take_screenshot(**kwargs)

def start_recording(**kwargs):
    return get_tool_manager().start_screen_recording(**kwargs)

def stop_recording():
    return get_tool_manager().stop_screen_recording()

def open_url(**kwargs):
    return get_tool_manager().open_url(**kwargs)

def search_web(**kwargs):
    return get_tool_manager().search_web(**kwargs)


if __name__ == "__main__":
    # Test the tool manager
    manager = get_tool_manager()
    
    print("🛠️  Available tools:")
    for tool in manager.list_tools():
        print(f"  - {tool['name']}: {tool['description']}")
    
    print("\n📸 Testing screenshot...")
    result = manager.call_tool('take_screenshot', {'save_to_file': True})
    print(f"  Result: {result['message']}")
    if result.get('file_path'):
        print(f"  Saved to: {result['file_path']}")
