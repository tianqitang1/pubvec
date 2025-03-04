import os
import argparse
from dotenv import load_dotenv

from pubvec.core.pubmed_fetcher import PubMedFetcher
from pubvec.core.vector_store import VectorStore

def main():
    parser = argparse.ArgumentParser(description="PubVec - Biomedical Entity Ranker")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Fetch command
    fetch_parser = subparsers.add_parser("fetch", help="Fetch PubMed articles and create vector database")
    fetch_parser.add_argument("--email", required=True, help="Email for PubMed API")
    fetch_parser.add_argument("--query", default="cancer immunotherapy", help="PubMed search query")
    fetch_parser.add_argument("--max_results", type=int, default=1000, help="Maximum number of articles to fetch")
    
    # Web command
    web_parser = subparsers.add_parser("web", help="Run the web application")
    web_parser.add_argument("--host", default="0.0.0.0", help="Host to run the web server on")
    web_parser.add_argument("--port", type=int, default=8001, help="Port to run the web server on")
    
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
    
    elif args.command == "web":
        # Run the web application
        print(f"Starting web application on {args.host}:{args.port}")
        from pubvec.web.app import app
        import uvicorn
        uvicorn.run(app, host=args.host, port=args.port)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    load_dotenv()
    main() 