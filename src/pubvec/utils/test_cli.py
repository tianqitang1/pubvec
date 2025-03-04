import asyncio
import logging
import argparse
import json
from pubvec.utils.cli import process_query, display_results, read_api_key

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("test-cli")

async def test_query(query: str, cheap: bool = False, debug: bool = False):
    """Test the process_query function with a sample query."""
    # Read API key
    api_key = read_api_key()
    if not api_key:
        logger.error("No API key found. Please create config/deepseek_api_key.txt with your DeepSeek API key.")
        return
    
    # Choose model
    model = "deepseek-chat" if cheap else "deepseek-reasoner"
    logger.info(f"Testing query using {model}: {query}")
    
    # Process query
    results = await process_query(
        query=query,
        api_key=api_key,
        base_url="https://api.deepseek.com",
        model=model,
        debug=debug
    )
    
    # Display results
    display_results(results)
    
    # Save debug info if requested
    if debug and "debug_info" in results:
        with open("test_results_debug.json", "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Debug info saved to test_results_debug.json")

def main():
    parser = argparse.ArgumentParser(description="Test CLI functionality")
    parser.add_argument("query", nargs="?", default="What's the efficacy of BRAF, MEK, and PD-1 inhibitors in treating melanoma in skin tissue?", 
                       help="The test query")
    parser.add_argument("--cheap", action="store_true", help="Use cheaper deepseek-chat model")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Run the test
    asyncio.run(test_query(args.query, args.cheap, args.debug))

if __name__ == "__main__":
    main() 