from pubmed_fetcher import PubMedFetcher
from vector_store import VectorStore
import argparse
import os
from dotenv import load_dotenv

def main():
    parser = argparse.ArgumentParser(description="Fetch PubMed articles and create vector database")
    parser.add_argument("--email", required=True, help="Email for PubMed API")
    parser.add_argument("--query", default="cancer immunotherapy", help="PubMed search query")
    parser.add_argument("--max_results", type=int, default=1000, help="Maximum number of articles to fetch")
    args = parser.parse_args()

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

if __name__ == "__main__":
    load_dotenv()
    main() 