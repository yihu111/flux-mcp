import asyncio
from dedalus_labs import AsyncDedalus, DedalusRunner
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent / "config" / ".env")

async def main():
    client = AsyncDedalus()
    runner = DedalusRunner(client)

    # Try passing your prompt as 'input' and specify your MCP server
    result = await runner.run(
        messages=[{"role": "user", "content": "A beautiful sunset over mountains"}],
        model="openai/gpt-4.1",
        mcp_servers=["yihu/flux-mcp"],  # Your MCP server name
        stream=False
    )

    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())