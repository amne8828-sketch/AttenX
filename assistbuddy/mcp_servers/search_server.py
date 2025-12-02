"""
MCP Search Server
Provides web search capabilities via Model Context Protocol
"""

import asyncio
from typing import List
from googlesearch import search
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

app = Server("search-server")

@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available search tools"""
    return [
        Tool(
            name="search_google",
            description="Perform a Google web search to find information, facts, or latest news.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Execute search tools"""
    if name != "search_google":
        raise ValueError(f"Unknown tool: {name}")

    query = arguments.get("query")
    num_results = arguments.get("num_results", 5)

    if not query:
        return [TextContent(type="text", text="Error: Query is required")]

    try:
        results = []
        # googlesearch-python's search function yields URLs. 
        # To get snippets, we might need 'advanced=True' if supported or just return URLs.
        # The basic 'search' yields URLs.
        # Let's try to get a bit more info if possible, but for now URLs are better than nothing.
        # We will format it as a list.
        
        search_results = list(search(query, num_results=num_results, advanced=True))
        
        output = f"## Search Results for '{query}'\n\n"
        for i, res in enumerate(search_results, 1):
            # res is a SearchResult object with .title, .url, .description
            output += f"**{i}. {res.title}**\n"
            output += f"   {res.description}\n"
            output += f"   [Link]({res.url})\n\n"
            
        return [TextContent(type="text", text=output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Search failed: {str(e)}")]

async def main():
    """Run the MCP search server"""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())
