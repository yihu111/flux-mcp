from mcp.server.fastmcp import FastMCP
# from fastmcp import FastMCP
from dotenv import load_dotenv
from flux_adapter import FluxAdapter
import os
from typing import Any, Dict, Optional
from pathlib import Path


# Load environment variables from config/.env file (for local development)
# In deployment, environment variables are provided by the platform
try:
    config_dir = Path(__file__).parent.parent / "config"
    if (config_dir / ".env").exists():
        load_dotenv(config_dir / ".env")
except Exception:
    # Ignore errors in deployment environments
    pass

# Create an MCP server
mcp = FastMCP("ImageEditor")


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
    

if __name__ == "__main__":
    mcp.run()