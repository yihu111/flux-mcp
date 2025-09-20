import asyncio
import os
import sys
from pathlib import Path

# Add src to path and set up environment
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

# Load environment variables manually
env_file = project_root / "config" / ".env"
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            if line.strip() and not line.startswith('#') and '=' in line:
                key, value = line.strip().split('=', 1)
                os.environ[key] = value

# Import after setting up environment
from flux_adapter import FluxAdapter

async def test():
    print("Testing Flux MCP Server")
    print("=" * 30)
    
    # Check API key
    api_key = os.getenv("BFL_API_KEY")
    if not api_key or api_key == "key":
        print("Please set your BFL_API_KEY in config/.env")
        return
    
    print("API key found")
    print("Testing image generation...")
    
    try:
        # Create adapter
        adapter = FluxAdapter(
            model="flux-dev",  # Faster model for testing
            use_raw_mode=False,
            api_key=api_key,
            aspect_ratio="1:1"
        )
        
        # Generate test image
        image_url, meta = await adapter.generate("A simple red circle")
        
        print("SUCCESS!")
        print(f"Image URL: {image_url}")
        print(f"Request ID: {meta['request_id']}")
        print(f"Model: {meta['model']}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test())
