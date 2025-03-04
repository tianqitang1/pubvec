import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Optional
import os
import numpy as np

# Update the ImageDType in chromadb's types.py
import chromadb.api.types
chromadb.api.types.ImageDType = np.float64

class VectorStore:
    def __init__(self, persist_dir: str = "./data/chroma_db"):
        """Initialize vector store with BGE-M3 embeddings."""
        self.persist_dir = persist_dir
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="BAAI/bge-m3"
        )
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="pubmed_articles",
            embedding_function=self.embedding_function
        )
        
    def add_articles(self, articles: List[Dict]):
        """Add articles to the vector store."""
        ids = [str(article["pmid"]) for article in articles]
        documents = [f"Title: {article['title']}\nAbstract: {article['abstract']}" 
                    for article in articles]
        metadata = [{"year": article["date"], "title": article["title"]} 
                   for article in articles]
        # TODO: fix metadata
        
        # Add in batches of 100
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            self.collection.add(
                ids=ids[i:i + batch_size],
                documents=documents[i:i + batch_size],
                metadatas=metadata[i:i + batch_size]
            )
            
    def search(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search for similar articles."""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        return [{
            "id": id,
            "document": doc,
            "metadata": meta,
            "distance": distance
        } for id, doc, meta, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )]

if __name__ == "__main__":
    # Example usage
    store = VectorStore()
    # Example search
    results = store.search("Latest developments in cancer immunotherapy") 
