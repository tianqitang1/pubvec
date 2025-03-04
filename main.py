import os
import argparse
import json
import asyncio
from dotenv import load_dotenv

from pubvec.core.pubmed_fetcher import PubMedFetcher
from pubvec.core.vector_store import VectorStore
from pubvec.utils.cli import (
    read_api_key,
    process_query,
    display_results
)

def main():
    parser = argparse.ArgumentParser(description="PubVec - Biomedical Entity Ranker")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Fetch command
    fetch_parser = subparsers.add_parser("fetch", help="Fetch PubMed articles and create vector database")
    fetch_parser.add_argument("--email", required=True, help="Email for PubMed API")
    fetch_parser.add_argument("--query", default="cancer immunotherapy", help="PubMed search query")
    fetch_parser.add_argument("--max_results", type=int, default=1000, help="Maximum number of articles to fetch")
    
    # Rank command
    rank_parser = subparsers.add_parser("rank", help="Rank entities from a query based on efficacy from PubMed")
    rank_parser.add_argument("query", help="The natural language query containing entities, disease, and type")
    rank_parser.add_argument("--cheap", action="store_true", help="Use cheaper deepseek-chat model instead of deepseek-reasoner")
    rank_parser.add_argument("--debug", action="store_true", help="Enable debug mode with detailed logs")
    rank_parser.add_argument("--output", help="Output file to save results (JSON format)")
    rank_parser.add_argument("--base-url", default="https://api.deepseek.com", help="API base URL (default: DeepSeek)")
    
    # Web command
    web_parser = subparsers.add_parser("web", help="Run the web application")
    web_parser.add_argument("--host", default="0.0.0.0", help="Host to run the web server on")
    web_parser.add_argument("--port", type=int, default=8001, help="Port to run the web server on")
    web_parser.add_argument("--prefill-api-key", action="store_true", help="Prefill API key from config/deepseek_api_key.txt")
    
    args = parser.parse_args()
    
    if args.command == "fetch":
        # Initialize fetcher and vector store
        fetcher = PubMedFetcher(email=args.email)
        store = VectorStore()

        # Fetch articles
        print(f"Searching PubMed for: {args.query}")
        pmids = fetcher.search_pubmed(args.query, max_results=args.max_results)
        print(f"Found {len(pmids)} articles. Fetching details...")
        
        articles = fetcher.fetch_articles(pmids)
        print(f"Adding {len(articles)} articles to vector store...")
        
        # Add to vector store
        store.add_articles(articles)
        print("Done! You can now use the API to search the articles.")
    
    elif args.command == "rank":
        # Get API key
        api_key = read_api_key()
        if not api_key:
            print("Error: No API key found. Please create config/deepseek_api_key.txt with your DeepSeek API key.")
            return
        
        # Choose model based on the --cheap flag
        model = "deepseek-chat" if args.cheap else "deepseek-reasoner"
        print(f"Processing query using {model}...")
        
        # Process query
        results = asyncio.run(process_query(
            query=args.query,
            api_key=api_key,
            base_url=args.base_url,
            model=model,
            debug=args.debug
        ))
        
        # Display results
        display_results(results)
        
        # Save to output file if specified
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.output}")
        
    elif args.command == "web":
        # Run the web application
        api_key = None
        if args.prefill_api_key:
            api_key = read_api_key()
            if api_key:
                print(f"Using API key from config/deepseek_api_key.txt")
            else:
                print(f"Warning: Could not load API key from config/deepseek_api_key.txt")
        
        print(f"Starting web application on {args.host}:{args.port}")
        from pubvec.web.app import app, initialize_api_key
        if api_key:
            initialize_api_key(api_key)
            
        import uvicorn
        uvicorn.run(app, host=args.host, port=args.port)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    load_dotenv()
    main() 