
from fastmcp import FastMCP
import webbrowser
import subprocess
import platform
from typing import Optional, List

# Create MCP server
mcp = FastMCP("Browser Control")

# Store opened browser instances
browser_instances = []


@mcp.tool()
def open_url(url: str, new_window: bool = False) -> dict:
    
    try:
        if new_window:
            webbrowser.open_new(url)
        else:
            webbrowser.open(url)
        
        return {
            "success": True,
            "message": f"Opened {url} successfully",
            "url": url
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Failed to open URL: {str(e)}",
            "url": url
        }


@mcp.tool()
def open_browser(url: str = "https://google.com", browser_type: str = "chrome") -> dict:
   
    try:
        system = platform.system()
        
        # Browser executables by platform
        browsers = {
            'Windows': {
                'chrome': 'chrome',
                'firefox': 'firefox',
                'edge': 'msedge'
            },
            'Darwin': {  # macOS
                'chrome': 'open -a "Google Chrome"',
                'firefox': 'open -a Firefox',
                'safari': 'open -a Safari'
            },
            'Linux': {
                'chrome': 'google-chrome',
                'firefox': 'firefox',
                'edge': 'microsoft-edge'
            }
        }
        
        if browser_type == "chrome":
            webbrowser.get('chrome').open(url)
        elif browser_type == "firefox":
            webbrowser.get('firefox').open(url)
        else:
            webbrowser.open(url)
        
        return {
            "success": True,
            "message": f"Opened {url} in {browser_type}",
            "browser": browser_type,
            "url": url
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Failed to open browser: {str(e)}",
            "browser": browser_type,
            "url": url
        }


@mcp.tool()
def search_web(query: str, engine: str = "google") -> dict:
    
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


@mcp.tool()
def get_browser_info() -> dict:
    
    browsers = []
    
    # Try to detect available browsers
    try:
        webbrowser.get('chrome')
        browsers.append("chrome")
    except:
        pass
    
    try:
        webbrowser.get('firefox')
        browsers.append("firefox")
    except:
        pass
    
    return {
        "success": True,
        "available_browsers": browsers,
        "default_browser": webbrowser.get().name,
        "platform": platform.system()
    }


@mcp.tool()
def open_multiple_tabs(urls: List[str]) -> dict:
    
    try:
        opened = 0
        failed = []
        
        for url in urls:
            try:
                webbrowser.open_new_tab(url)
                opened += 1
            except Exception as e:
                failed.append({"url": url, "error": str(e)})
        
        return {
            "success": True,
            "message": f"Opened {opened} tabs",
            "opened_count": opened,
            "failed": failed,
            "total_requested": len(urls)
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Failed to open tabs: {str(e)}"
        }


if __name__ == "__main__":
    # Run the MCP server
    print("🌐 Browser Control MCP Server Starting...")
    print("Available tools:")
    print("  - open_url: Open a URL in default browser")
    print("  - open_browser: Open specific browser with URL")
    print("  - search_web: Search using Google/Bing/DuckDuckGo")
    print("  - get_browser_info: Get available browsers")
    print("  - open_multiple_tabs: Open multiple URLs")
    print("\nServer running on stdio (MCP protocol)")
    
    mcp.run()
