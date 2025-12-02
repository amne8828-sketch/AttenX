import asyncio
import os
import sys

# Add current dir to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agentic_orchestrator import NexusOrchestrator

async def test_main():
    print("Starting APGA Verification Test...")
    
    nexus = NexusOrchestrator()
    nexus.hitl_enabled = False # Disable HITL for automated test
    
    # Config - Use absolute paths or relative to where we run it
    # We will run from mcp_servers dir
    servers = [
        {"name": "search", "command": "python", "args": ["search_server.py"]},
        {"name": "bash", "command": "python", "args": ["bash_server.py"]}
    ]
    
    try:
        await nexus.initialize(servers)
        
        # Test 1: Search Intent
        # print("\n[Test 1] Testing Search Intent...")
        # response = await nexus.process_request("Search for 'Model Context Protocol'")
        # print(f"Response length: {len(response)}")
        # assert "Results for:" in response
        
        # Test 2: Bash Intent (Safe)
        print("\n[Test 2] Testing Bash Intent...")
        response = await nexus.process_request("Run command echo 'Hello APGA'")
        assert "Hello APGA" in response
        
        print("\nVerification Passed!")
        
    except Exception as e:
        print(f"\nVerification Failed: {e}")
        raise
    finally:
        await nexus.shutdown()

if __name__ == "__main__":
    asyncio.run(test_main())
