import sqlite3
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm
import logging
import sys
from typing import List, Dict, Generator
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chroma_import.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

class PubMedChromaImporter:
    def __init__(self, 
                 sqlite_path: str = "pubmed_data.db",
                 persist_dir: str = "./chroma_db",
                 batch_size: int = 1000):
        """Initialize the importer."""
        self.sqlite_path = sqlite_path
        self.persist_dir = persist_dir
        self.batch_size = batch_size
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=persist_dir)
        
        # Use BGE-M3 for embeddings (better for biomedical text)
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="BAAI/bge-m3"
        )
        
        # Create or get collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="pubmed_articles",
            embedding_function=self.embedding_function,
            metadata={"description": "PubMed articles with titles and abstracts"}
        )
    
    def _get_article_batches(self) -> Generator[List[Dict], None, None]:
        """Get articles from SQLite in batches."""
        conn = sqlite3.connect(self.sqlite_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        
        # Get total count for progress bar
        total = cur.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        logging.info(f"Total articles to process: {total}")
        
        # Process in batches
        offset = 0
        with tqdm(total=total, desc="Processing articles") as pbar:
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
                offset += self.batch_size
        
        conn.close()
    
    def import_articles(self):
        """Import articles from SQLite to ChromaDB."""
        try:
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
            
            logging.info(f"Import completed. Imported: {total_imported}, Skipped: {total_skipped}")
            
        except Exception as e:
            logging.error(f"Error during import: {str(e)}")
            raise

def main():
    parser = argparse.ArgumentParser(description="Import PubMed articles from SQLite to ChromaDB")
    parser.add_argument("--sqlite-path", default="pubmed_data.db", help="Path to SQLite database")
    parser.add_argument("--persist-dir", default="./chroma_db", help="ChromaDB persistence directory")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for processing")
    
    args = parser.parse_args()
    
    importer = PubMedChromaImporter(
        sqlite_path=args.sqlite_path,
        persist_dir=args.persist_dir,
        batch_size=args.batch_size
    )
    
    importer.import_articles()

if __name__ == "__main__":
    main() 