import asyncio
import os
from pathlib import Path

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

try:
    # Preferred adapter package for MCP client-tools in LangGraph
    from langchain_mcp_adapters.client import MultiServerMCPClient
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Missing dependency 'langchain-mcp-adapters'. Install with: pip install langchain-mcp-adapters langgraph langchain-openai"
    ) from e


async def main() -> None:
    # Ensure API keys/env are loaded (mirror tests/image_generation.py behavior)
    project_root = Path(__file__).parent.parent
    env_file = project_root / "config" / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                if line.strip() and not line.startswith('#') and '=' in line:
                    key, value = line.strip().split('=', 1)
                    os.environ.setdefault(key, value)

    # LLM backend for the agent. You can change the model via OPENAI_API_KEY env var.
    # For basic tool-calling planning, a small model is fine.
    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

    # Use stdio transport which is the most compatible for local dev.
    async with MultiServerMCPClient(
        {
            "flux_mcp": {
                "command": "python",
                # Run as a module so relative imports in src/main.py work
                "args": ["-m", "src.main"],
                "transport": "stdio",
            }
        }
    ) as client:
        tools = await client.get_tools()

        agent = create_react_agent(llm, tools)

        # Ask the agent to call the image generation tool and surface the URL.
        user_prompt = (
            "Use the flux_generate tool to create an image of 'a sunrise over a mountain lake' "
            "and return the image URL."
        )

        # LangGraph agents expect messages or input depending on version.
        # Using ainvoke with input mapping is broadly compatible.
        result = await agent.ainvoke({"messages": [
            {"role": "user", "content": user_prompt}
        ]})

        # Print the final output/messages for inspection.
        print(result)


if __name__ == "__main__":
    asyncio.run(main())


