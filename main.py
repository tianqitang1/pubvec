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
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="PubVec - Biomedical Entity Ranker")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Download command
    download_parser = subparsers.add_parser("download", help="Download PubMed articles and store in SQLite database")
    download_parser.add_argument("--db-path", default="data/db/pubmed_data.db", help="Path to SQLite database")
    download_parser.add_argument("--download-dir", default="data/downloads", help="Directory to store downloads")
    download_parser.add_argument("--max-workers", type=int, default=4, help="Maximum number of concurrent downloads")
    download_parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for database operations")
    
    # Import to Chroma command
    import_parser = subparsers.add_parser("import-to-chroma", help="Import downloaded PubMed articles into ChromaDB")
    import_parser.add_argument("--sqlite-path", default="data/db/pubmed_data.db", help="Path to SQLite database")
    import_parser.add_argument("--persist-dir", default="chroma_db", help="Directory to store ChromaDB")
    import_parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for import operations")
    import_parser.add_argument("--use-gpu", action="store_true", help="Use GPU for embedding generation")
    
    # Combined command for download and import
    download_import_parser = subparsers.add_parser("process-pubmed", help="Download PubMed articles and import to ChromaDB in one step")
    download_import_parser.add_argument("--db-path", default="data/db/pubmed_data.db", help="Path to SQLite database")
    download_import_parser.add_argument("--download-dir", default="data/downloads", help="Directory to store downloads")
    download_import_parser.add_argument("--max-workers", type=int, default=4, help="Maximum number of concurrent downloads")
    download_import_parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for database operations")
    download_import_parser.add_argument("--persist-dir", default="chroma_db", help="Directory to store ChromaDB")
    download_import_parser.add_argument("--use-gpu", action="store_true", help="Use GPU for embedding generation")
    
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
    
    if args.command == "download":
        from pubvec.scripts.download_pubmed import PubMedDownloader
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(args.db_path), exist_ok=True)
        os.makedirs(args.download_dir, exist_ok=True)
        
        # Initialize downloader
        downloader = PubMedDownloader(
            db_path=args.db_path,
            download_dir=args.download_dir,
            max_workers=args.max_workers,
            batch_size=args.batch_size
        )
        
        # Download and process PubMed articles
        print("Starting download and processing of PubMed articles...")
        downloader.download_and_process()
        print("Done! PubMed articles have been downloaded and stored in SQLite database.")
    
    elif args.command == "import-to-chroma":
        from pubvec.scripts.import_to_chroma import PubMedChromaImporter
        
        # Initialize importer
        importer = PubMedChromaImporter(
            sqlite_path=Path(args.sqlite_path),
            persist_dir=Path(args.persist_dir),
            batch_size=args.batch_size,
            use_gpu=args.use_gpu
        )
        
        # Import articles to ChromaDB
        print("Starting import of PubMed articles to ChromaDB...")
        importer.import_articles()
        print("Done! PubMed articles have been imported to ChromaDB.")
    
    elif args.command == "process-pubmed":
        from pubvec.scripts.download_pubmed import PubMedDownloader
        from pubvec.scripts.import_to_chroma import PubMedChromaImporter
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(args.db_path), exist_ok=True)
        os.makedirs(args.download_dir, exist_ok=True)
        
        # Initialize downloader
        downloader = PubMedDownloader(
            db_path=args.db_path,
            download_dir=args.download_dir,
            max_workers=args.max_workers,
            batch_size=args.batch_size
        )
        
        # Download and process PubMed articles
        print("Starting download and processing of PubMed articles...")
        downloader.download_and_process()
        print("Done! PubMed articles have been downloaded and stored in SQLite database.")
        
        # Initialize importer
        importer = PubMedChromaImporter(
            sqlite_path=Path(args.db_path),
            persist_dir=Path(args.persist_dir),
            batch_size=args.batch_size,
            use_gpu=args.use_gpu
        )
        
        # Import articles to ChromaDB
        print("Starting import of PubMed articles to ChromaDB...")
        importer.import_articles()
        print("Done! PubMed articles have been imported to ChromaDB.")
    
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