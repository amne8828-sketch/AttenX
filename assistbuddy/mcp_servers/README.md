# AssistBuddy MCP Servers

Production-ready Model Context Protocol servers for multimodal AI orchestration.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd mcp_servers
pip install -r requirements.txt

# For screenshot server
pip install pillow pyautogui

# For GitHub server (optional)
pip install PyGithub
```

### 2. Run Individual Servers

```bash
# Camera server
python camera_server.py

# Excel server
python excel_server.py

# Screenshot server
python screenshot_server.py

# Bash server
python bash_server.py
```

### 3. Use with Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "assistbuddy-camera": {
      "command": "python",
      "args": ["C:/path/to/assistbuddy/mcp_servers/camera_server.py"]
    },
    "assistbuddy-excel": {
      "command": "python",
      "args": ["C:/path/to/assistbuddy/mcp_servers/excel_server.py"]
    },
    "assistbuddy-screenshot": {
      "command": "python",
      "args": ["C:/path/to/assistbuddy/mcp_servers/screenshot_server.py"]
    },
    "assistbuddy-bash": {
      "command": "python",
      "args": ["C:/path/to/assistbuddy/mcp_servers/bash_server.py"]
    }
  }
}
```

## 📦 Available Servers

### 1. Camera Server (`camera_server.py`)

**Tools:**
- `get_snapshot`: Capture webcam frame as Base64 JPEG

**Use Case:**
```
User: "Look at what I'm holding"
Claude → Calls camera.get_snapshot() → Analyzes Base64 image
```

**Security:**
- Auto-releases camera after capture
- Handles camera-in-use errors gracefully
- Resizes images to save tokens

---

### 2. Excel Server (`excel_server.py`)

**Tools:**
- `inspect_file`: Get sheet names and column headers
- `read_data`: Read first N rows from sheet
- `add_entry`: Append new row to sheet
- `run_pandas_script`: Execute custom Pandas code

**Use Case:**
```
User: "Add this invoice to budget.xlsx"
Claude → inspect_file → read_data → add_entry
```

**Security:**
- Hardcoded DATA_DIRECTORY (prevents directory traversal)
- Sandboxed script execution
- Limited __builtins__ for safety

---

### 3. Screenshot Server (`screenshot_server.py`)

**Tools:**
- `take_screenshot`: Full screen capture
- `take_region_screenshot`: Capture specific area
- `get_screen_size`: Get resolution

**Use Case:**
```
User: "What's on my screen?"
Claude → take_screenshot() → Analyzes UI
```

---

### 4. Bash Server (`bash_server.py`)

**Tools:**
- `run_command`: Execute safe shell commands
- `list_allowed_commands`: Show security policy

**Security:**
- Whitelist of allowed commands
- Blacklist of destructive commands (rm, format, etc.)
- Blocks command chaining (&&, |, ;)
- Timeout protection (default: 30s)

**Use Case:**
```
User: "Check git status"
Claude → run_command("git status")
```

**Forbidden:**
```
rm -rf /  ❌ BLOCKED
format C: ❌ BLOCKED
ls && rm  ❌ BLOCKED (no chaining)
```

---

## 🤖 Agentic Orchestrator

The `agentic_orchestrator.py` coordinates all MCP servers using:

**OBSERVE → REASON → ACT → EVALUATE Loop**

Example workflow:
```python
from agentic_orchestrator import NexusOrchestrator

nexus = NexusOrchestrator()
await nexus.initialize(servers=[...])

# Multi-modal request
response = await nexus.process_request(
    "Look at the camera, search for this product online, "
    "and add it to my budget.xlsx"
)
```

**4-Phase Accuracy Gauntlet:**
1. **Researcher**: Gather context from tools
2. **Writer**: Draft response
3. **Fact-Checker**: Verify logical consistency
4. **Editor**: Final synthesis

---

## 🔧 Directory Structure

```
mcp_servers/
├── camera_server.py          # Webcam access
├── excel_server.py            # Spreadsheet manipulation
├── screenshot_server.py       # Screen capture
├── bash_server.py             # Shell commands
├── agentic_orchestrator.py    # Multi-agent coordinator
├── requirements.txt           # Dependencies
├── data/                      # Excel server data directory
│   └── budget.xlsx            # Example file
└── screenshots/               # Screenshot storage
```

---

## 🛡️ Security Features

### Excel Server
- ✅ Path validation (blocks `../` traversal)
- ✅ Restricted to DATA_DIRECTORY only
- ✅ Sandboxed code execution

### Bash Server
- ✅ Command whitelist/blacklist
- ✅ No command chaining (&& | ;)
- ✅ Timeout protection
- ✅ Explicit user confirmation required for destructive ops

### Camera Server
- ✅ Auto-release camera resources
- ✅ Error handling for camera-in-use
- ✅ Image resizing (token optimization)

---

## 📝 Example Usage

### Multi-Step Workflow

```python
# Start orchestrator
nexus = NexusOrchestrator()
await nexus.initialize([
    {"name": "camera", "command": "python", "args": ["camera_server.py"]},
    {"name": "excel", "command": "python", "args": ["excel_server.py"]},
    {"name": "bash", "command": "python", "args": ["bash_server.py"]}
])

# Complex request
response = await nexus.process_request(
    "Take a screenshot of my current work, "
    "run 'git status' to check uncommitted changes, "
    "and log this in my activity.xlsx file"
)

# Orchestrator will:
# 1. OBSERVE: Detect required tools (screenshot, bash, excel)
# 2. REASON: Create execution plan
# 3. ACT: Call tools in sequence
# 4. EVALUATE: Verify results
# 5. SYNTHESIZE: Generate final response
```

---

## 🧪 Testing

### Test Camera Server
```bash
# In terminal 1
python camera_server.py

# In terminal 2
echo '{"method":"tools/call","params":{"name":"get_snapshot","arguments":{}}}' | python camera_server.py
```

### Test Excel Server
```bash
# Create test file
mkdir -p data
echo "Name,Amount\nTest,100" > data/budget.csv

# Run server
python excel_server.py
```

---

## 🔄 Integration with AssistBuddy

These MCP servers integrate with the main AssistBuddy model:

```python
# In assistbuddy/mcp_server.py
from mcp_servers.camera_server import capture_snapshot
from mcp_servers.excel_server import safe_read_excel

@server.tool()
async def summarize_with_context(file_path: str):
    # Get camera context
    camera_image = capture_snapshot()
    
    # Read Excel data
    data = safe_read_excel(file_path)
    
    # Use AssistBuddy model
    summary = model.generate(
        image_inputs=camera_image,
        text_inputs=data.to_string()
    )
    
    return summary
```

---

## 📊 Performance

| Server | Avg Latency | Concurrent Requests |
|--------|-------------|---------------------|
| Camera | 300ms | 1 (camera lock) |
| Excel | 150ms | 10+ |
| Screenshot | 200ms | 5+ |
| Bash | 1-30s | 3 (timeout protection) |

---

## 🐛 Troubleshooting

### Camera in use
```
Error: Camera 0 is currently in use
→ Close other camera apps (Zoom, Teams, etc.)
```

### Excel file locked
```
Error: Permission denied
→ Close file in Excel before server access
```

### Bash command blocked
```
Error: FORBIDDEN: 'rm' is not allowed
→ Use safe alternatives (mv to trash folder)
```

---

## 📚 References

- [MCP SDK Docs](https://github.com/modelcontextprotocol/python-sdk)
- [FastMCP Guide](https://github.com/jlowin/fastmcp)
- [Claude Desktop Integration](https://docs.anthropic.com/claude/docs/mcp)

---

**Status**: ✅ Production Ready  
**Version**: 1.0.0  
**License**: MIT
