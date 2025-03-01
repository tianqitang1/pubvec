from Bio import Entrez
from typing import List, Dict, Optional
import time
from tqdm import tqdm

class PubMedFetcher:
    def __init__(self, email: str):
        """Initialize PubMed fetcher with user email (required by NCBI)."""
        Entrez.email = email
        
    def search_pubmed(self, query: str, max_results: int = 1000) -> List[str]:
        """Search PubMed and return list of PMIDs."""
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
        record = Entrez.read(handle)
        handle.close()
        return record["IdList"]
    
    def _get_date(self, medline_citation: Dict) -> str:
        """Extract date from various possible date fields."""
        date_fields = [
            "DateCompleted",
            "DateRevised",
            "Article/Journal/JournalIssue/PubDate",
            "DateCreated"
        ]
        
        for field in date_fields:
            if "/" in field:
                # Handle nested fields
                current = medline_citation
                for subfield in field.split("/"):
                    if subfield in current:
                        current = current[subfield]
                    else:
                        current = None
                        break
                if current and "Year" in current:
                    return current["Year"]
            else:
                # Handle direct fields
                if field in medline_citation and "Year" in medline_citation[field]:
                    return medline_citation[field]["Year"]
        return ""
    
    def fetch_articles(self, pmids: List[str], batch_size: int = 100) -> List[Dict]:
        """Fetch article details for a list of PMIDs."""
        articles = []
        for i in tqdm(range(0, len(pmids), batch_size)):
            batch = pmids[i:i + batch_size]
            handle = Entrez.efetch(db="pubmed", id=batch, rettype="medline", retmode="xml")
            records = Entrez.read(handle)
            handle.close()
            
            for record in records["PubmedArticle"]:
                article = {
                    "pmid": record["MedlineCitation"]["PMID"],
                    "title": record["MedlineCitation"]["Article"]["ArticleTitle"],
                    "abstract": record["MedlineCitation"]["Article"].get("Abstract", {}).get("AbstractText", [""])[0],
                    "date": self._get_date(record["MedlineCitation"]),
                }
                articles.append(article)
            time.sleep(0.5)  # Be nice to NCBI servers
            
        return articles

if __name__ == "__main__":
    # Example usage
    fetcher = PubMedFetcher(email="your.email@example.com")
    pmids = fetcher.search_pubmed("cancer immunotherapy", max_results=100)
    articles = fetcher.fetch_articles(pmids)
    print(f"Retrieved {len(articles)} articles") 