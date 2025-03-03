import sqlite3
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm
import logging
import sys
import json
import torch
import time
import cProfile
import pstats
import io
from typing import List, Dict, Generator
import argparse
from pathlib import Path

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
        
        # Check if database file exists
        if not self.sqlite_path.exists():
            logging.error(f"Database file not found: {self.sqlite_path}")
            logging.error("Please run the download_pubmed.py script first to create the database.")
            raise FileNotFoundError(f"Database file not found: {self.sqlite_path}")
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=str(persist_dir))
        
        # Device selection for the embedding model
        device = "cuda" if self.use_gpu else "cpu"
        if self.use_gpu:
            logging.info(f"Using GPU for embeddings: {torch.cuda.get_device_name(0)}")
        else:
            if torch.cuda.is_available():
                logging.warning("GPU is available but not being used. Set use_gpu=True to enable.")
            else:
                logging.info("No GPU detected, using CPU for embeddings.")
        
        # Use BGE-M3 for embeddings (better for biomedical text)
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="BAAI/bge-m3",
            device=device
        )
        
        # Create or get collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="pubmed_articles",
            embedding_function=self.embedding_function,
            metadata={"description": "PubMed articles with titles and abstracts"}
        )
        
        # Initialize total article count
        self._calculate_total_articles()
        
        # Profiling data
        self.profiling_data = {
            "db_fetch_time": 0,
            "embedding_time": 0,
            "chroma_add_time": 0,
            "batch_count": 0,
            "total_docs": 0,
        }
    
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
                batch_start_time = time.time()
                
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
                
                data_prep_time = time.time() - batch_start_time
                
                # Add to ChromaDB
                try:
                    # Track memory usage before embedding
                    if self.use_gpu and torch.cuda.is_available():
                        before_gpu_mem = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
                    
                    # Time the actual embedding and adding to ChromaDB
                    embedding_start = time.time()
                    self.collection.add(
                        ids=ids,
                        documents=documents,
                        metadatas=metadatas
                    )
                    embedding_time = time.time() - embedding_start
                    
                    # Track memory after embedding
                    if self.use_gpu and torch.cuda.is_available():
                        after_gpu_mem = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
                        gpu_mem_diff = after_gpu_mem - before_gpu_mem
                        logging.info(f"Batch size: {len(ids)}, GPU memory change: {gpu_mem_diff:.2f} MB")
                    
                    # Update profiling data
                    self.profiling_data["embedding_time"] += embedding_time
                    self.profiling_data["batch_count"] += 1
                    self.profiling_data["total_docs"] += len(ids)
                    
                    # Log performance metrics
                    docs_per_second = len(ids) / embedding_time if embedding_time > 0 else 0
                    logging.info(f"Batch processing: {len(ids)} docs in {embedding_time:.2f}s ({docs_per_second:.2f} docs/s)")
                    logging.info(f"Data prep time: {data_prep_time:.2f}s, Embedding time: {embedding_time:.2f}s")
                    
                    total_imported += len(ids)
                except Exception as e:
                    logging.error(f"Error adding batch to ChromaDB: {str(e)}")
                    total_skipped += len(ids)
            
            # Print profiling summary
            if self.profiling_data["batch_count"] > 0:
                avg_embedding_time = self.profiling_data["embedding_time"] / self.profiling_data["batch_count"]
                avg_docs_per_batch = self.profiling_data["total_docs"] / self.profiling_data["batch_count"]
                logging.info("=== Profiling Summary ===")
                logging.info(f"Total batches processed: {self.profiling_data['batch_count']}")
                logging.info(f"Total documents processed: {self.profiling_data['total_docs']}")
                logging.info(f"Average embedding time per batch: {avg_embedding_time:.2f}s")
                logging.info(f"Average docs per batch: {avg_docs_per_batch:.2f}")
                if avg_embedding_time > 0:
                    logging.info(f"Average throughput: {avg_docs_per_batch / avg_embedding_time:.2f} docs/s")
            
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
    parser.add_argument("--profile", action="store_true",
                       help="Run profiling on the script")
    
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
        
        if args.profile:
            # Run with profiling
            pr = cProfile.Profile()
            pr.enable()
            
            importer = PubMedChromaImporter(
                sqlite_path=args.sqlite_path,
                persist_dir=args.persist_dir,
                batch_size=args.batch_size,
                checkpoint_file=args.checkpoint_file,
                use_gpu=args.use_gpu
            )
            
            importer.import_articles()
            
            pr.disable()
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
            ps.print_stats(30)  # Print top 30 functions by time
            logging.info(f"\n==== Profiling Results ====\n{s.getvalue()}")
        else:
            # Run normally
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