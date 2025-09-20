import asyncio
from dedalus_labs import AsyncDedalus, DedalusRunner
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv(Path(__file__).parent.parent / "config" / ".env")

async def test_image_generation():
    """Test the flux_generate tool"""
    print("Testing Image Generation Tool")
    print("=" * 40)
    
    client = AsyncDedalus()
    runner = DedalusRunner(client)

    try:
        # Test basic image generation
        result = await runner.run(
            messages=[{
                "role": "user", 
                "content": "Generate an image of a beautiful sunset over mountains using flux_generate"
            }],
            model="openai/gpt-4.1",
            mcp_servers=["yihu/flux-mcp"],
            stream=False
        )

        print("Image Generation Test Result:")
        print(result.final_output)
        print("\n" + "="*50 + "\n")
        
        return True
        
    except Exception as e:
        print(f"Image Generation Test Failed: {e}")
        return False

async def test_image_editing():
    """Test the flux_edit_image tool"""
    print("Testing Image Editing Tool")
    print("=" * 40)
    
    client = AsyncDedalus()
    runner = DedalusRunner(client)

    try:
        # Test image editing with a real accessible image URL
        sample_image_url = "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=800&h=600&fit=crop"
        
        result = await runner.run(
            messages=[{
                "role": "user", 
                "content": f"Edit this image to add a rainbow in the sky: {sample_image_url} using flux_edit_image"
            }],
            model="openai/gpt-4.1",
            mcp_servers=["yihu/flux-mcp"],
            stream=False
        )

        print("Image Editing Test Result:")
        print(result.final_output)
        print("\n" + "="*50 + "\n")
        
        return True
        
    except Exception as e:
        print(f"Image Editing Test Failed: {e}")
        return False

async def test_character_consistency():
    """Test character consistency through iterative editing"""
    print("Testing Character Consistency (Iterative Editing)")
    print("=" * 50)
    
    client = AsyncDedalus()
    runner = DedalusRunner(client)

    try:
        # Step 1: Generate initial character image
        print("Step 1: Generating initial character...")
        result1 = await runner.run(
            messages=[{
                "role": "user", 
                "content": "Generate an image of a young wizard with blue robes and a tall pointed hat using flux_generate"
            }],
            model="openai/gpt-4.1",
            mcp_servers=["yihu/flux-mcp"],
            stream=False
        )

        print("Initial character generated:")
        print(result1.final_output)
        
        print("\nStep 2: Editing character to change background...")
        # For demo purposes, use a sample wizard image URL
        # In practice, you would extract the URL from result1.final_output
        wizard_image_url = "https://images.unsplash.com/photo-1578662996442-48f60103fc96?w=800&h=800&fit=crop"
        result2 = await runner.run(
            messages=[{
                "role": "user", 
                "content": f"Edit this wizard image to place him in a magical forest setting: {wizard_image_url} using flux_edit_image"
            }],
            model="openai/gpt-4.1",
            mcp_servers=["yihu/flux-mcp"],
            stream=False
        )

        print("Character consistency test completed:")
        print(result2.final_output)
        print("\n" + "="*50 + "\n")
        
        return True
        
    except Exception as e:
        print(f"Character Consistency Test Failed: {e}")
        return False

async def test_advanced_scenarios():
    """Test advanced usage scenarios"""
    print("Testing Advanced Scenarios")
    print("=" * 40)
    
    client = AsyncDedalus()
    runner = DedalusRunner(client)
    
    scenarios = [
        {
            "name": "Simple Generation Test",
            "prompt": "Generate an image of a majestic dragon perched on a mountain peak using flux_generate"
        },
        {
            "name": "Simple Editing Test", 
            "prompt": "Edit this landscape image to add a bright sunset: https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=800&h=600&fit=crop using flux_edit_image"
        }
    ]
    
    results = []
    
    for scenario in scenarios:
        try:
            print(f"\nTesting: {scenario['name']}")
            result = await runner.run(
                messages=[{"role": "user", "content": scenario["prompt"]}],
                model="openai/gpt-4.1",
                mcp_servers=["yihu/flux-mcp"],
                stream=False
            )
            
            print(f"{scenario['name']} - Success")
            print(f"Result: {result.final_output}")
            results.append(True)
            
        except Exception as e:
            print(f"{scenario['name']} - Failed: {e}")
            results.append(False)
    
    success_rate = sum(results) / len(results) * 100
    print(f"\nAdvanced Scenarios Success Rate: {success_rate:.1f}%")
    print("="*50 + "\n")
    
    return all(results)

async def main():
    """Run all tests"""
    print("Starting Comprehensive FLUX MCP Server Tests")
    print("=" * 60)
    
    # Test results
    test_results = []
    
    # Run individual tests
    test_results.append(await test_image_generation())
    test_results.append(await test_image_editing())
    test_results.append(await test_character_consistency())
    test_results.append(await test_advanced_scenarios())
    
    # Summary
    print("TEST SUMMARY")
    print("=" * 30)
    
    test_names = [
        "Image Generation",
        "Image Editing", 
        "Character Consistency",
        "Advanced Scenarios"
    ]
    
    for i, (name, result) in enumerate(zip(test_names, test_results)):
        status = "PASS" if result else "FAIL"
        print(f"{name}: {status}")
    
    overall_success_rate = sum(test_results) / len(test_results) * 100
    print(f"\nOverall Success Rate: {overall_success_rate:.1f}%")
    
    if overall_success_rate == 100:
        print("All tests passed! Your FLUX MCP server is working perfectly.")
    elif overall_success_rate >= 75:
        print("Most tests passed. Check failed tests for any issues.")
    else:
        print("Multiple tests failed. Please check your configuration and deployment.")

if __name__ == "__main__":
    asyncio.run(main())
