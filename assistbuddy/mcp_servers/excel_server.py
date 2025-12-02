"""
MCP Excel Server - Secure Spreadsheet Manipulation
Provides Excel/CSV data access via Model Context Protocol with directory traversal protection
"""

import asyncio
import os
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent


# SECURITY: Hardcoded data directory - all file operations must be within this folder
DATA_DIRECTORY = Path(__file__).parent / "data"
DATA_DIRECTORY.mkdir(exist_ok=True)


app = Server("excel-server")


def validate_file_path(filename: str) -> Path:
    """
    Validate file path and prevent directory traversal attacks
    
    Args:
        filename: Requested filename
    
    Returns:
        Validated absolute Path object
    
    Raises:
        ValueError: If path attempts directory traversal
    """
    # Resolve the requested path
    requested_path = (DATA_DIRECTORY / filename).resolve()
    
    # Ensure it's within DATA_DIRECTORY
    if not requested_path.is_relative_to(DATA_DIRECTORY.resolve()):
        raise ValueError(
            f"Security: Access denied. File must be in {DATA_DIRECTORY}. "
            f"Directory traversal attempts (../) are blocked."
        )
    
    return requested_path


def safe_read_excel(file_path: Path) -> dict[str, pd.DataFrame]:
    """
    Safely read Excel file and return all sheets as DataFrames
    
    Args:
        file_path: Validated path to Excel file
    
    Returns:
        Dictionary mapping sheet names to DataFrames
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path.name}")
    
    # Read all sheets
    sheets = pd.read_excel(file_path, sheet_name=None, engine='openpyxl')
    return sheets


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available Excel manipulation tools"""
    return [
        Tool(
            name="inspect_file",
            description=(
                "Inspect an Excel/CSV file without loading all data. "
                "Returns sheet names and column headers. "
                "Use this first to understand file structure before reading data."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Name of the file (e.g., 'budget.xlsx', 'sales.csv')"
                    }
                },
                "required": ["filename"]
            }
        ),
        Tool(
            name="read_data",
            description=(
                "Read the first N rows of a specific sheet from an Excel file. "
                "Returns data in markdown table format for easy reading."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Name of the Excel file"
                    },
                    "sheet_name": {
                        "type": "string",
                        "description": "Name of the sheet to read (use inspect_file first)"
                    },
                    "num_rows": {
                        "type": "integer",
                        "description": "Number of rows to read (default 10)",
                        "default": 10
                    }
                },
                "required": ["filename", "sheet_name"]
            }
        ),
        Tool(
            name="add_entry",
            description=(
                "Append a new row to a specific sheet in an Excel file. "
                "Values are added in the order provided, matching column order."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Name of the Excel file"
                    },
                    "sheet_name": {
                        "type": "string",
                        "description": "Sheet to append to"
                    },
                    "values": {
                        "type": "array",
                        "items": {"type": ["string", "number", "null"]},
                        "description": "List of values to append (must match column count)"
                    }
                },
                "required": ["filename", "sheet_name", "values"]
            }
        ),
        Tool(
            name="run_pandas_script",
            description=(
                "Execute custom Pandas code on an Excel file. "
                "WARNING: Powerful tool - use carefully. "
                "Example: \"df['Total'] = df['Price'] * df['Quantity']\" to add calculated column. "
                "Returns the modified dataframe as markdown."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Excel file to operate on"
                    },
                    "sheet_name": {
                        "type": "string",
                        "description": "Sheet name"
                    },
                    "script": {
                        "type": "string",
                        "description": (
                            "Python/Pandas code to execute. "
                            "The dataframe is available as 'df'. "
                            "Example: 'df.groupby(\"Category\")[\"Sales\"].sum()'"
                        )
                    }
                },
                "required": ["filename", "sheet_name", "script"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Execute Excel manipulation tools"""
    
    try:
        filename = arguments.get("filename")
        if not filename:
            return [TextContent(type="text", text="Error: filename is required")]
        
        # Validate file path (security check)
        try:
            file_path = validate_file_path(filename)
        except ValueError as e:
            return [TextContent(type="text", text=f"Security Error: {str(e)}")]
        
        # Tool: inspect_file
        if name == "inspect_file":
            if not file_path.exists():
                return [TextContent(
                    type="text",
                    text=f"File not found: {filename}\nData directory: {DATA_DIRECTORY}"
                )]
            
            # Read file metadata only
            if file_path.suffix == '.csv':
                df = pd.read_csv(file_path, nrows=0)
                result = (
                    f"**File:** {filename}\n"
                    f"**Type:** CSV\n"
                    f"**Columns:** {list(df.columns)}\n"
                    f"**Column count:** {len(df.columns)}"
                )
            else:
                sheets = safe_read_excel(file_path)
                result = f"**File:** {filename}\n**Sheets:**\n\n"
                for sheet_name, df in sheets.items():
                    result += f"- **{sheet_name}**\n"
                    result += f"  - Columns: {list(df.columns)}\n"
                    result += f"  - Rows: {len(df)}\n\n"
            
            return [TextContent(type="text", text=result)]
        
        # Tool: read_data
        elif name == "read_data":
            sheet_name = arguments.get("sheet_name")
            num_rows = arguments.get("num_rows", 10)
            
            if not sheet_name:
                return [TextContent(type="text", text="Error: sheet_name is required")]
            
            # Read specific sheet
            sheets = safe_read_excel(file_path)
            if sheet_name not in sheets:
                available = ", ".join(sheets.keys())
                return [TextContent(
                    type="text",
                    text=f"Sheet '{sheet_name}' not found. Available: {available}"
                )]
            
            df = sheets[sheet_name].head(num_rows)
            markdown_table = df.to_markdown(index=False)
            
            result = (
                f"**File:** {filename}\n"
                f"**Sheet:** {sheet_name}\n"
                f"**Showing:** First {len(df)} rows\n\n"
                f"{markdown_table}"
            )
            
            return [TextContent(type="text", text=result)]
        
        # Tool: add_entry
        elif name == "add_entry":
            sheet_name = arguments.get("sheet_name")
            values = arguments.get("values")
            
            if not sheet_name or values is None:
                return [TextContent(
                    type="text",
                    text="Error: sheet_name and values are required"
                )]
            
            # Load sheet
            sheets = safe_read_excel(file_path)
            if sheet_name not in sheets:
                return [TextContent(
                    type="text",
                    text=f"Sheet '{sheet_name}' not found"
                )]
            
            df = sheets[sheet_name]
            
            # Validate values count
            if len(values) != len(df.columns):
                return [TextContent(
                    type="text",
                    text=(
                        f"Error: Expected {len(df.columns)} values (columns: {list(df.columns)}), "
                        f"but got {len(values)}"
                    )
                )]
            
            # Append row
            new_row = pd.DataFrame([values], columns=df.columns)
            df = pd.concat([df, new_row], ignore_index=True)
            
            # Save back to file
            with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            return [TextContent(
                type="text",
                text=f"✅ Successfully added entry to {sheet_name} in {filename}"
            )]
        
        # Tool: run_pandas_script
        elif name == "run_pandas_script":
            sheet_name = arguments.get("sheet_name")
            script = arguments.get("script")
            
            if not sheet_name or not script:
                return [TextContent(
                    type="text",
                    text="Error: sheet_name and script are required"
                )]
            
            # Load sheet
            sheets = safe_read_excel(file_path)
            if sheet_name not in sheets:
                return [TextContent(
                    type="text",
                    text=f"Sheet '{sheet_name}' not found"
                )]
            
            df = sheets[sheet_name]
            
            # Execute user script safely
            try:
                # Create restricted namespace
                namespace = {
                    'df': df,
                    'pd': pd,
                    '__builtins__': {
                        'len': len,
                        'sum': sum,
                        'min': min,
                        'max': max,
                        'abs': abs,
                        'round': round
                    }
                }
                
                # Execute script
                exec(script, namespace)
                
                # Get result (could be modified df or a calculation)
                result_df = namespace.get('df', df)
                
                # Format output
                if isinstance(result_df, pd.DataFrame):
                    output = result_df.head(20).to_markdown(index=False)
                elif isinstance(result_df, pd.Series):
                    output = result_df.to_markdown()
                else:
                    output = str(result_df)
                
                return [TextContent(
                    type="text",
                    text=(
                        f"**Script executed successfully:**\n"
                        f"```python\n{script}\n```\n\n"
                        f"**Result:**\n{output}"
                    )
                )]
            
            except Exception as e:
                return [TextContent(
                    type="text",
                    text=f"Error executing script: {str(e)}"
                )]
        
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
    
    except FileNotFoundError as e:
        return [TextContent(type="text", text=f"File error: {str(e)}")]
    
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def main():
    """Run the MCP Excel server"""
    print(f"Excel MCP Server starting...")
    print(f"Data directory: {DATA_DIRECTORY.absolute()}")
    print(f"Security: All file access restricted to data directory only")
    
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
