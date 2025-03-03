import sqlite3
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm
import logging
import sys
import json
import torch
import time
from typing import List, Dict, Generator
import argparse
from pathlib import Path
from chromadb.config import Settings
from langchain.embeddings import HuggingFaceEmbeddings

# Get project root directory (2 levels up from this script)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(PROJECT_ROOT / 'logs' / 'chroma_import.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

class PubMedChromaImporter:
    def __init__(self, 
                 sqlite_path: Path = PROJECT_ROOT / "data" / "db" / "pubmed_data.db",
                 persist_dir: Path = PROJECT_ROOT / "chroma_db",
                 batch_size: int = 1000,
                 checkpoint_file: Path = PROJECT_ROOT / "data" / "import_checkpoint.json",
                 use_gpu: bool = True):
        """Initialize the importer."""
        self.sqlite_path = sqlite_path
        self.persist_dir = persist_dir
        self.batch_size = batch_size
        self.checkpoint_file = checkpoint_file
        self.use_gpu = use_gpu
        self.last_processed_offset = 0
        self.total_articles = 0
        
        # Load checkpoint if exists
        self._load_checkpoint()
        
        # Initialize ChromaDB client
        logging.info(f"Initializing ChromaDB with persist_dir: {persist_dir}")
        settings = Settings(anonymized_telemetry=False)
        
        embedding_function = None
        if use_gpu:
            try:
                # Try to use GPU-accelerated embedding
                logging.info("Attempting to use GPU for embeddings")
                embedding_function = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={"device": "cuda"},
                    encode_kwargs={"device": "cuda", "batch_size": 32}
                )
            except Exception as e:
                logging.warning(f"Failed to initialize GPU embeddings: {str(e)}")
                logging.info("Falling back to CPU embeddings")
                embedding_function = None
                
        if embedding_function is None:
            # Fall back to CPU
            embedding_function = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        
        self.client = chromadb.Client(settings=settings, path=str(persist_dir))
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="pubmed_articles",
            embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Initialize total article count
        self._calculate_total_articles()
    
    def _calculate_total_articles(self):
        """Calculate the total number of articles in the database."""
        try:
            conn = sqlite3.connect(str(self.sqlite_path))
            cur = conn.cursor()
            
            # Get total count for progress tracking
            self.total_articles = cur.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
            logging.info(f"Total articles in database: {self.total_articles}")
            
            conn.close()
        except Exception as e:
            logging.error(f"Error calculating total articles: {str(e)}")
            raise
    
    def _load_checkpoint(self):
        """Load current processing offset as a checkpoint."""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    checkpoint_data = json.load(f)
                    self.last_processed_offset = checkpoint_data.get('last_processed_offset', 0)
                    logging.info(f"Resuming from offset: {self.last_processed_offset}")
            except Exception as e:
                logging.warning(f"Failed to load checkpoint file: {str(e)}")
        
        # Check if database file exists
        if not self.sqlite_path.exists():
            logging.error(f"Database file not found: {self.sqlite_path}")
            logging.error("Please run the download_pubmed.py script first to create the database.")
            raise FileNotFoundError(f"Database file not found: {self.sqlite_path}")
    
    def _save_checkpoint(self, offset: int) -> None:
        """Save current processing offset as a checkpoint."""
        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump({
                    'last_processed_offset': offset,
                    'timestamp': str(time.time()),
                }, f)
            logging.debug(f"Checkpoint saved at offset: {offset}")
        except Exception as e:
            logging.warning(f"Failed to save checkpoint: {str(e)}")
    
    def _get_article_batches(self) -> Generator[List[Dict], None, None]:
        """Get articles from SQLite in batches."""
        try:
            conn = sqlite3.connect(str(self.sqlite_path))
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            
            # Check if the articles table exists
            table_check = cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='articles'").fetchone()
            if not table_check:
                logging.error("The 'articles' table does not exist in the database.")
                logging.error("Please make sure you've run the download_pubmed.py script to populate the database.")
                raise sqlite3.OperationalError("Table 'articles' not found in the database")
            
            # Use the pre-calculated total count instead of querying again
            remaining = self.total_articles - self.last_processed_offset
            logging.info(f"Total articles: {self.total_articles}, Already processed: {self.last_processed_offset}, Remaining: {remaining}")
            
            # Process in batches, starting from the last processed offset
            offset = self.last_processed_offset
            with tqdm(total=remaining, desc="Processing articles") as pbar:
                while True:
                    cur.execute("""
                        SELECT pmid, title, abstract, authors, publication_date, 
                               journal, publication_types, mesh_terms, keywords
                        FROM articles 
                        WHERE title IS NOT NULL 
                          AND abstract IS NOT NULL
                        LIMIT ? OFFSET ?
                    """, (self.batch_size, offset))
                    
                    batch = [dict(row) for row in cur.fetchall()]
                    if not batch:
                        break
                        
                    pbar.update(len(batch))
                    yield batch
                    offset += len(batch)
                    self._save_checkpoint(offset)
            
            conn.close()
        except sqlite3.OperationalError as e:
            logging.error(f"SQLite error: {str(e)}")
            logging.error(f"Please check that the database at {self.sqlite_path} is properly initialized.")
            raise
    
    def import_articles(self):
        """Import articles from SQLite to ChromaDB."""
        try:
            # Calculate total articles count once before processing batches
            self._calculate_total_articles()
            
            total_imported = 0
            total_skipped = 0
            
            for batch in self._get_article_batches():
                # Prepare data for ChromaDB
                ids = []
                documents = []
                metadatas = []
                
                for article in batch:
                    # Create document text combining title and abstract
                    doc_text = f"Title: {article['title']}\nAbstract: {article['abstract']}"
                    
                    # Prepare metadata
                    metadata = {
                        "pmid": article["pmid"],
                        "authors": article["authors"][:1000] if article["authors"] else "",  # Limit length
                        "publication_date": article["publication_date"],
                        "journal": article["journal"],
                        "publication_types": article["publication_types"],
                        "mesh_terms": article["mesh_terms"][:1000] if article["mesh_terms"] else "",
                        "keywords": article["keywords"][:1000] if article["keywords"] else ""
                    }
                    
                    ids.append(article["pmid"])
                    documents.append(doc_text)
                    metadatas.append(metadata)
                
                # Add to ChromaDB
                try:
                    self.collection.add(
                        ids=ids,
                        documents=documents,
                        metadatas=metadatas
                    )
                    total_imported += len(ids)
                except Exception as e:
                    logging.error(f"Error adding batch to ChromaDB: {str(e)}")
                    total_skipped += len(ids)
            
            # Clear checkpoint file after successful completion
            if self.checkpoint_file.exists():
                self.checkpoint_file.unlink()
                logging.info("Checkpoint file removed after successful completion")
                
            logging.info(f"Import completed. Imported: {total_imported}, Skipped: {total_skipped}")
            
        except Exception as e:
            logging.error(f"Error during import: {str(e)}")
            raise

def main():
    parser = argparse.ArgumentParser(description="Import PubMed articles from SQLite to ChromaDB")
    parser.add_argument("--sqlite-path", 
                       type=Path,
                       default=PROJECT_ROOT / "data" / "db" / "pubmed_data.db", 
                       help="Path to SQLite database")
    parser.add_argument("--persist-dir", 
                       type=Path,
                       default=PROJECT_ROOT / "data" / "chroma_db", 
                       help="ChromaDB persistence directory")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for processing")
    parser.add_argument("--checkpoint-file",
                       type=Path,
                       default=PROJECT_ROOT / "data" / "import_checkpoint.json",
                       help="File to store checkpoint information for resuming")
    parser.add_argument("--use-gpu", action="store_true", default=True, 
                       help="Use GPU for embeddings if available")
    parser.add_argument("--resume", action="store_true", 
                       help="Resume from last checkpoint if available")
    parser.add_argument("--reset", action="store_true", 
                       help="Reset checkpoint and start from beginning")
    
    args = parser.parse_args()
    
    try:
        # Make sure the data directory exists
        data_dir = PROJECT_ROOT / "data"
        if not data_dir.exists():
            logging.warning(f"Creating data directory: {data_dir}")
            data_dir.mkdir(parents=True, exist_ok=True)
        
        # Make sure the ChromaDB directory exists
        chroma_dir = args.persist_dir
        if not chroma_dir.exists():
            logging.warning(f"Creating ChromaDB directory: {chroma_dir}")
            chroma_dir.mkdir(parents=True, exist_ok=True)
        
        # Handle checkpoint reset if requested
        if args.reset and args.checkpoint_file.exists():
            args.checkpoint_file.unlink()
            logging.info("Checkpoint reset, starting from beginning")
        
        importer = PubMedChromaImporter(
            sqlite_path=args.sqlite_path,
            persist_dir=args.persist_dir,
            batch_size=args.batch_size,
            checkpoint_file=args.checkpoint_file,
            use_gpu=args.use_gpu
        )
        
        importer.import_articles()
    except FileNotFoundError as e:
        logging.error(str(e))
        logging.error("\nTo download PubMed data and create the database, run:")
        logging.error(f"python -m pubvec.scripts.download_pubmed --output-dir {PROJECT_ROOT / 'data'}")
        sys.exit(1)
    except sqlite3.OperationalError as e:
        logging.error(f"Database error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 