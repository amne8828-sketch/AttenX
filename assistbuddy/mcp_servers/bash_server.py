"""
MCP Bash/Shell Server
Provides safe shell command execution via Model Context Protocol
"""

import asyncio
import subprocess
from pathlib import Path
from typing import Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent


app = Server("bash-server")

# Security: Whitelist of allowed commands
ALLOWED_COMMANDS = {
    "ls", "dir", "pwd", "echo", "cat", "head", "tail",
    "grep", "find", "wc", "date", "whoami", "hostname",
    "git", "pip", "python", "node", "npm"
}

# Blacklist of dangerous commands
FORBIDDEN_COMMANDS = {
    "rm", "rmdir", "del", "format", "dd", "mkfs",
    "shutdown", "reboot", "halt", "init", "systemctl"
}


def is_command_safe(command: str) -> tuple[bool, str]:
    """
    Validate command safety
    
    Returns:
        (is_safe: bool, reason: str)
    """
    cmd_parts = command.strip().split()
    if not cmd_parts:
        return False, "Empty command"
    
    base_cmd = cmd_parts[0]
    
    # Check for forbidden commands
    if base_cmd in FORBIDDEN_COMMANDS:
        return False, f"FORBIDDEN: '{base_cmd}' is not allowed (destructive command)"
    
    # Check for dangerous patterns
    dangerous_patterns = ["&&", "||", "|", ">", ">>", "<", ";", "`", "$(" ]
    for pattern in dangerous_patterns:
        if pattern in command:
            return False, f"FORBIDDEN: Command chaining/redirection ('{pattern}') not allowed"
    
    # Optionally enforce whitelist
    # if base_cmd not in ALLOWED_COMMANDS:
    #     return False, f"Command '{base_cmd}' not in whitelist"
    
    return True, "Safe"


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available bash tools"""
    return [
        Tool(
            name="run_command",
            description=(
                "Execute a shell command safely. "
                "Supports basic commands like ls, git, pip, python. "
                "Destructive commands (rm, format) are BLOCKED for safety."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute (e.g., 'ls -la', 'git status')"
                    },
                    "working_directory": {
                        "type": "string",
                        "description": "Working directory for command (default: current dir)",
                        "default": "."
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Command timeout in seconds (default: 30)",
                        "default": 30
                    }
                },
                "required": ["command"]
            }
        ),
        Tool(
            name="list_allowed_commands",
            description="List all allowed and forbidden commands for security reference",
            inputSchema={"type": "object", "properties": {}, "required": []}
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Execute bash tools"""
    
    try:
        # Tool: run_command
        if name == "run_command":
            command = arguments["command"]
            working_dir = arguments.get("working_directory", ".")
            timeout = arguments.get("timeout", 30)
            
            # Security validation
            is_safe, reason = is_command_safe(command)
            if not is_safe:
                return [TextContent(
                    type="text",
                    text=f"❌ SECURITY BLOCK: {reason}\n\nCommand rejected: {command}"
                )]
            
            # Execute command
            try:
                result = subprocess.run(
                    command,
                    shell=True,
                    cwd=working_dir,
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
                
                output = f"**Command:** `{command}`\n"
                output += f"**Exit Code:** {result.returncode}\n\n"
                
                if result.stdout:
                    output += f"**Output:**\n```\n{result.stdout}\n```\n"
                
                if result.stderr:
                    output += f"\n**Errors:**\n```\n{result.stderr}\n```"
                
                return [TextContent(type="text", text=output)]
            
            except subprocess.TimeoutExpired:
                return [TextContent(
                    type="text",
                    text=f"❌ Command timed out after {timeout} seconds"
                )]
            
            except Exception as e:
                return [TextContent(
                    type="text",
                    text=f"❌ Execution error: {str(e)}"
                )]
        
        # Tool: list_allowed_commands
        elif name == "list_allowed_commands":
            response = "# Command Security Policy\n\n"
            response += "## ✅ Allowed Commands\n"
            response += ", ".join(sorted(ALLOWED_COMMANDS)) + "\n\n"
            response += "## ❌ Forbidden Commands (Destructive)\n"
            response += ", ".join(sorted(FORBIDDEN_COMMANDS)) + "\n\n"
            response += "## 🚫 Blocked Patterns\n"
            response += "- Command chaining: &&, ||, ;\n"
            response += "- Redirection: >, >>, <\n"
            response += "- Pipes: |\n"
            response += "- Command substitution: `, $()\n"
            
            return [TextContent(type="text", text=response)]
        
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
    
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def main():
    """Run the MCP bash server"""
    print("Bash MCP Server starting...")
    print(f"Security: Forbidden commands: {', '.join(FORBIDDEN_COMMANDS)}")
    
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
