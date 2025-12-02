import requests
import json
from typing import List, Dict, Any, Optional

class MCPClient:
    def __init__(self, servers: List[str] = None):
        
        self.servers = servers or []
        self.tools = {}
        self.discover_tools()
        
    def discover_tools(self):
        """Query all servers for available tools"""
        for server in self.servers:
            try:
                response = requests.get(f"{server}/tools")
                if response.status_code == 200:
                    server_tools = response.json()
                    for tool in server_tools:
                        tool['server'] = server
                        self.tools[tool['name']] = tool
            except Exception as e:
                print(f"Failed to discover tools from {server}: {e}")
                
    def list_tools(self) -> List[Dict]:
        """Return list of available tools"""
        return list(self.tools.values())
        
    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Optional[Any]:
        
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not found")
            
        tool = self.tools[tool_name]
        server = tool['server']
        
        try:
            response = requests.post(
                f"{server}/tools/{tool_name}/call",
                json=arguments,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"Tool execution failed: {response.text}")
                
        except Exception as e:
            print(f"Error calling tool {tool_name}: {e}")
            return None

# Example usage
if __name__ == "__main__":
    # Mock server for testing
    client = MCPClient(servers=["http://localhost:8000"])
    print("Available tools:", client.list_tools())
