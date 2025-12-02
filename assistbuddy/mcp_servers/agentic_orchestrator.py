"""
Agentic MCP Orchestrator - "Nexus" (APGA Edition)
Autonomous Production-Grade Agent operating via Model Context Protocol
Implements: PLANNING -> EXECUTION -> REFLECTION -> SYNTHESIS
"""

import asyncio
import json
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, List, Dict, Optional

from contextlib import AsyncExitStack
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession
from mcp.types import TextContent

# Import Memory Manager
try:
    from memory_manager import MemoryManager
except ImportError:
    # Fallback if not in same dir (for testing)
    MemoryManager = None

class AgentMode(Enum):
    PLANNING = "PLANNING"
    EXECUTION = "EXECUTION"
    REFLECTION = "REFLECTION"
    SYNTHESIS = "SYNTHESIS"

class Intent(Enum):
    INFO_RETRIEVAL = "info_retrieval"
    TASK_EXECUTION = "task_execution"
    ANALYSIS = "analysis"
    CREATION = "creation"

class ContextManager:
    """Manages Short-Term and Long-Term Memory"""
    def __init__(self):
        self.short_term_memory: List[Dict] = []
        self.long_term_memory = MemoryManager() if MemoryManager else None
        self.session_id = str(uuid.uuid4())

    def add_short_term(self, role: str, content: str, metadata: Dict = None):
        entry = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        }
        self.short_term_memory.append(entry)
        
        # Also save to long-term if significant
        if self.long_term_memory and role in ["assistant", "user"]:
            self.long_term_memory.add_memory(content, {"role": role, "session_id": self.session_id})

    def get_context(self) -> str:
        return json.dumps(self.short_term_memory, indent=2)

    def retrieve_relevant_history(self, query: str) -> List[str]:
        # Simple caching could be added here if needed
        if not self.long_term_memory:
            return []
        try:
            memories = self.long_term_memory.query_memory(query)
            return [m["content"] for m in memories]
        except Exception as e:
            print(f"Memory retrieval failed: {e}")
            return []

class NexusOrchestrator:
    """
    Autonomous Production-Grade Agent (APGA)
    """
    
    def __init__(self):
        self.sessions = {}
        self.exit_stack = AsyncExitStack()
        self.tool_registry = {}
        self.context = ContextManager()
        self.hitl_enabled = True  # Human-in-the-Loop
        
        # Sensitive tools requiring approval
        self.sensitive_tools = [
            "bash.run_command", 
            "excel.add_entry", 
            "excel.run_pandas_script"
        ]

    async def _init_session(self, name: str, session: ClientSession):
        """Helper to initialize a session and list tools"""
        try:
            await session.initialize()
            tools_list = await session.list_tools()
            
            for tool in tools_list.tools:
                tool_key = f"{name}.{tool.name}"
                self.tool_registry[tool_key] = {
                    "session": session,
                    "tool": tool,
                    "server": name
                }
            
            self.sessions[name] = session
            print(f"Connected to {name} ({len(tools_list.tools)} tools)")
        except Exception as e:
            print(f"Failed to connect to {name}: {e}")

    async def initialize(self, server_configs: list[dict]):
        """Connect to MCP servers in parallel"""
        print(f"[{AgentMode.PLANNING.value}] Initializing Nexus APGA...")
        
        init_tasks = []
        
        # 1. Create connections sequentially (Context Entry must be in main task)
        for config in server_configs:
            name = config["name"]
            try:
                params = StdioServerParameters(
                    command=config["command"],
                    args=config["args"],
                    env=config.get("env")
                )
                
                # Create connection using ExitStack to keep it open
                read, write = await self.exit_stack.enter_async_context(stdio_client(params))
                session = await self.exit_stack.enter_async_context(ClientSession(read, write))
                
                # Schedule initialization
                init_tasks.append(self._init_session(name, session))
                
            except Exception as e:
                print(f"Failed to create connection for {name}: {e}")

        # 2. Run handshakes in parallel
        if init_tasks:
            await asyncio.gather(*init_tasks)

    async def _request_approval(self, tool_name: str, args: dict) -> bool:
        """HITL: Request user approval for sensitive actions"""
        if not self.hitl_enabled:
            return True
            
        print(f"\nAPPROVAL REQUIRED: Tool '{tool_name}' is sensitive.")
        print(f"Arguments: {json.dumps(args, indent=2)}")
        response = input("Proceed? (y/n): ").strip().lower()
        return response == 'y'

    async def process_request(self, user_request: str) -> str:
        """Main Execution Protocol"""
        
        # 1. PLANNING & INTENT
        print(f"\n{AgentMode.PLANNING.value}")
        self.context.add_short_term("user", user_request)
        
        # Retrieve relevant history
        history = self.context.retrieve_relevant_history(user_request)
        if history:
            print(f"Retrieved {len(history)} relevant past memories")

        # Intent Mapping (Simplified Rule-Based for now, would be LLM in full prod)
        intent = self._map_intent(user_request)
        print(f"Intent Detected: {intent.name}")
        
        # Goal Decomposition
        plan = self._create_plan(user_request, intent)
        print(f"Execution Plan: {len(plan)} steps")
        
        results = []
        
        # 2. EXECUTION LOOP
        print(f"\n{AgentMode.EXECUTION.value}")
        
        for step in plan:
            tool_key = step["tool"]
            args = step.get("args", {})
            
            # HITL Check
            if any(s in tool_key for s in self.sensitive_tools):
                if not await self._request_approval(tool_key, args):
                    print(f"Action denied by user: {tool_key}")
                    results.append({"step": step["id"], "status": "denied", "output": "User denied action"})
                    continue

            # Execute
            try:
                print(f"Executing: {tool_key}...")
                tool_info = self.tool_registry.get(tool_key)
                if not tool_info:
                    raise ValueError(f"Tool {tool_key} not found")
                
                session = tool_info["session"]
                result = await session.call_tool(tool_info["tool"].name, args)
                
                # Extract text
                output = result.content[0].text if result and result.content and hasattr(result.content[0], 'text') else str(result)
                
                # 3. REFLECTION & STATE UPDATE
                print(f"{AgentMode.REFLECTION.value} Step {step['id']} Success")
                self.context.add_short_term("tool", output, {"tool": tool_key})
                results.append({"step": step["id"], "status": "success", "output": output})
                
            except Exception as e:
                print(f"Step {step['id']} Failed: {e}")
                # Self-Correction Logic could go here (e.g., retry with different args)
                results.append({"step": step["id"], "status": "error", "error": str(e)})
        
        # 4. FINAL SYNTHESIS
        print(f"\n{AgentMode.SYNTHESIS.value}")
        final_response = self._synthesize_response(user_request, results)
        
        self.context.add_short_term("assistant", final_response)
        return final_response

    def _map_intent(self, request: str) -> Intent:
        request = request.lower()
        if "search" in request or "find" in request:
            return Intent.INFO_RETRIEVAL
        elif "analyze" in request or "predict" in request or "correlation" in request:
            return Intent.ANALYSIS
        elif "create" in request or "generate" in request:
            return Intent.CREATION
        return Intent.TASK_EXECUTION

    def _create_plan(self, request: str, intent: Intent) -> List[Dict]:
        """
        Dynamic Planning based on Intent and available tools.
        In a real LLM-backed agent, this would be generated by the model.
        Here we use heuristic mapping for the demo.
        """
        plan = []
        req = request.lower()
        
        # Example: "Research X"
        if "search" in req or "research" in req:
            plan.append({
                "id": 1,
                "tool": "search.search_google",
                "args": {"query": request.replace("search for", "").strip()}
            })
            
        # Example: "Check camera"
        if "camera" in req or "look" in req:
            plan.append({
                "id": 1, 
                "tool": "camera.get_snapshot",
                "args": {}
            })
            
        # Example: "Check files" or "Excel"
        if "excel" in req or "file" in req:
            plan.append({
                "id": 1,
                "tool": "excel.inspect_file",
                "args": {"file_path": "data/budget.xlsx"} # Mock default
            })
            
        # Example: "Run command"
        if "run" in req and "command" in req:
            # Extract command (naive)
            cmd = request.split("command")[-1].strip()
            plan.append({
                "id": 1,
                "tool": "bash.run_command",
                "args": {"command": cmd}
            })

        # Default fallback if no specific keywords
        if not plan:
             plan.append({
                "id": 1,
                "tool": "bash.list_allowed_commands",
                "args": {}
            })
            
        return plan

    def _synthesize_response(self, request: str, results: List[Dict]) -> str:
        """Generate final answer from results"""
        response = f"## Results for: {request}\n\n"
        
        for res in results:
            if res["status"] == "success":
                response += f"### Step {res['step']} Output:\n{res['output']}\n\n"
            else:
                response += f"### Step {res['step']} Failed:\n{res.get('error', 'Unknown error')}\n\n"
                
        response += "\n---\n*Processed via APGA Protocol*"
        return response

    async def shutdown(self):
        await self.exit_stack.aclose()

class LinguisticAdaptor:
    """
    Indian English Linguistic Adaptor
    Ensures output conforms to Indian Standard English (ISE) protocols.
    """
    import re
    
    # Pre-compile patterns for performance
    _SUBSTITUTIONS = [
        (re.compile(r"(?i)\bget back to me\b"), "revert to me"),
        (re.compile(r"(?i)\bdo what's necessary\b"), "do the needful"),
        (re.compile(r"(?i)\bplease adjust\b"), "kindly adjust"),
        (re.compile(r"(?i)\bplease\b"), "kindly"),
        (re.compile(r"(?i)\bmove the meeting earlier\b"), "prepone the meeting"),
        (re.compile(r"(?i)\bwarehouse\b"), "godown"),
        (re.compile(r"(?i)\bstorage facility\b"), "godown"),
        (re.compile(r"(?i)\bcheck\b"), "verify"),
    ]

    @classmethod
    def adapt(cls, text: str) -> str:
        for pattern, replacement in cls._SUBSTITUTIONS:
            text = pattern.sub(replacement, text)
        return text

async def main():
    nexus = NexusOrchestrator()
    
    # Config
    servers = [
        {"name": "camera", "command": "python", "args": ["mcp_servers/camera_server.py"]},
        {"name": "excel", "command": "python", "args": ["mcp_servers/excel_server.py"]},
        {"name": "bash", "command": "python", "args": ["mcp_servers/bash_server.py"]},
        {"name": "search", "command": "python", "args": ["mcp_servers/search_server.py"]}
    ]
    
    await nexus.initialize(servers)
    
    # Interactive Loop
    while True:
        try:
            user_input = input("\nUSER (or 'exit'): ")
            if user_input.lower() in ['exit', 'quit']:
                break
                
            response = await nexus.process_request(user_input)
            
            # Apply Linguistic Adaptor (Double check for interactive mode)
            response = LinguisticAdaptor.adapt(response)
            
            print(f"\nAGENT:\n{response}")
            
        except KeyboardInterrupt:
            break
            
    await nexus.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
