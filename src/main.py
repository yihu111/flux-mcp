"""
Flux MCP Server
A Model Context Protocol server for AI image generation using Black Forest Labs' Flux models
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv

from mcp.server.fastmcp import FastMCP
from flux_adapter import FluxAdapter

# Load environment variables - try multiple locations
env_path = Path('.') / '.env.local'
if env_path.exists():
    load_dotenv(env_path)
load_dotenv()  # Also load .env if exists

# Try config/.env for local development
try:
    config_dir = Path(__file__).parent.parent / "config"
    if (config_dir / ".env").exists():
        load_dotenv(config_dir / ".env")
except Exception:
    # Ignore errors in deployment environments
    pass

# Get port from environment or command line args
port = int(os.environ.get('PORT', '8080'))
for i, arg in enumerate(sys.argv):
    if arg == '--port' and i + 1 < len(sys.argv):
        port = int(sys.argv[i + 1])
        break

# Get host from environment or command line args
host = os.environ.get('HOST', '0.0.0.0')
for i, arg in enumerate(sys.argv):
    if arg == '--host' and i + 1 < len(sys.argv):
        host = sys.argv[i + 1]
        break

mcp = FastMCP(
    name='Flux Image Generator',
    host=host,
    port=port,
    instructions="""This MCP server provides AI image generation using Black Forest Labs' Flux models.
    
Available tools:
- flux_generate(prompt): Generate images using Flux diffusion models

This server requires a BFL_API_KEY environment variable to function.""",
)


@mcp.tool()
async def flux_generate(prompt: str) -> Dict[str, Any]:
    """
    Generate a new image using Black Forest Labs FLUX diffusion model.

    Args:
        prompt: The prompt to be given to diffusion model

    Returns:
        Dictionary containing success status and URL of the generated image
    """
    safety_tolerance = 6
    prompt_upsampling = False
    raw = False
    model = "flux-pro-1.1"
    width: int = 1024
    height: int = 1024
    aspect_ratio: str | None = "16:9"

    api_key = os.getenv("BFL_API_KEY")
    if not api_key:
        return {"status": "error", "message": "BFL_API_KEY not set"}
    try:
        adapter = FluxAdapter(
            model=model,
            use_raw_mode=raw,
            api_key=api_key,
            aspect_ratio=aspect_ratio,
            width=width,
            height=height,
            safety_tolerance=safety_tolerance,
            prompt_upsampling=prompt_upsampling,
        )
        image_url, meta = await adapter.generate(prompt)
        return {"status": "success", "image": image_url, "meta": meta}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def main():
    """Main entry point for the MCP server"""
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Flux Image Generation MCP Server')
    parser.add_argument('--port', type=int, help='Port for HTTP transport')
    parser.add_argument(
        '--host', type=str, default='0.0.0.0', help='Host for HTTP transport'
    )
    parser.add_argument('--stdio', action='store_true', help='Force STDIO transport')
    parser.add_argument('--test', action='store_true', help='Test mode')
    args = parser.parse_args()

    # Check if running in test mode
    if args.test:
        # Test mode - just verify everything loads
        print('Flux MCP Server loaded successfully')
        print('Tools available: flux_generate')
        api_key = os.getenv("BFL_API_KEY")
        print(f'API Key configured: {"Yes" if api_key else "No"}')
        return 0

    # Determine transport mode
    # Note: port and host are already configured in the FastMCP instance above
    if (args.port or os.environ.get('PORT')) and not args.stdio:
        # HTTP transport mode
        actual_host = host if not args.host else args.host
        actual_port = port if not args.port else args.port
        print(f'Starting HTTP server on {actual_host}:{actual_port}')
        print(f'MCP endpoint: http://{actual_host}:{actual_port}/mcp')
        mcp.run(transport='streamable-http')
    else:
        # STDIO transport (default for MCP)
        mcp.run('stdio')
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())