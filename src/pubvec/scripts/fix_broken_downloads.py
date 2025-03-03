#!/usr/bin/env python3
import os
import re
import logging
import sys
from datetime import datetime
import ftplib
import gzip
import xml.etree.ElementTree as ET
import sqlite3
from typing import List, Dict, Optional, Tuple

# Configure logging
log_dir = "data/logs"
os.makedirs(log_dir, exist_ok=True)
log_filename = os.path.join(log_dir, f"pubmed_fix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# Configure logging handlers
file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(message)s'))

# Configure root logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(file_handler)
logger.addHandler(console_handler)


class PubMedFixer:
    def __init__(self, 
                 db_path: str = "data/db/pubmed_data.db",
                 download_dir: str = "data/downloads",
                 log_path: str = None):
        """Initialize PubMed fixer with configuration."""
        self.db_path = db_path
        self.download_dir = download_dir
        self.log_path = log_path
        
        # FTP settings
        self.ftp_host = "130.14.250.7"
        self.ftp_dirs = ["/pubmed/baseline/", "/pubmed/updatefiles/"]
        self.ftp_user = "anonymous"
        self.ftp_pass = "Tianqi.Tang@ucsf.edu"
        
        logger.info(f"Initializing PubMedFixer")
        logger.info(f"Log file to analyze: {self.log_path}")
        
        # Ensure directories exist
        if not os.path.exists(os.path.dirname(db_path)):
            logger.error(f"Database directory does not exist: {os.path.dirname(db_path)}")
        if not os.path.exists(download_dir):
            logger.error(f"Download directory does not exist: {download_dir}")
    
    def _get_db_connection(self):
        """Create a new database connection."""
        conn = sqlite3.connect(self.db_path, timeout=60.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=60000")
        conn.row_factory = sqlite3.Row
        return conn
    
    def find_broken_files(self) -> List[str]:
        """Parse the log file and identify files that had errors."""
        if not self.log_path:
            logger.error("No log file specified to analyze")
            return []
            
        if not os.path.exists(self.log_path):
            logger.error(f"Log file does not exist: {self.log_path}")
            return []
            
        # Pattern to match error lines in the log
        error_pattern = re.compile(r'ERROR - Error processing file ([^:]+): (.+)')
        broken_files = []
        
        logger.info("Analyzing log file for errors...")
        with open(self.log_path, 'r') as f:
            for line in f:
                match = error_pattern.search(line)
                if match:
                    file_path = match.group(1)
                    error_msg = match.group(2)
                    
                    # Extract just the filename from the path
                    file_name = os.path.basename(file_path)
                    
                    logger.info(f"Found error for {file_name}: {error_msg}")
                    broken_files.append(file_name)
        
        logger.info(f"Found {len(broken_files)} broken files")
        return broken_files
    
    def clean_broken_files(self, broken_files: List[str]) -> int:
        """Remove any broken files so they can be re-downloaded."""
        count = 0
        for file_name in broken_files:
            file_path = os.path.join(self.download_dir, file_name)
            if os.path.exists(file_path):
                try:
                    logger.info(f"Removing corrupted file: {file_path}")
                    os.remove(file_path)
                    count += 1
                except Exception as e:
                    logger.error(f"Failed to remove {file_path}: {str(e)}")
            else:
                logger.warning(f"File does not exist: {file_path}")
                
        logger.info(f"Removed {count} corrupted files")
        return count
    
    def download_file(self, file_name: str) -> bool:
        """Download a specific file from FTP server."""
        file_path = os.path.join(self.download_dir, file_name)
        
        # Determine which FTP directory contains the file
        ftp_dir = None
        
        for attempt in range(3):  # Try up to 3 times
            ftp = None
            try:
                logger.info(f"Attempting to download {file_name} (attempt {attempt+1})")
                
                ftp = ftplib.FTP(self.ftp_host)
                ftp.login(user=self.ftp_user, passwd=self.ftp_pass)
                ftp.voidcmd('TYPE I')
                
                # First determine which directory contains the file
                if ftp_dir is None:
                    for dir_path in self.ftp_dirs:
                        try:
                            ftp.cwd(dir_path)
                            files = ftp.nlst()
                            if file_name in files:
                                ftp_dir = dir_path
                                logger.info(f"Found {file_name} in directory {ftp_dir}")
                                break
                        except Exception:
                            continue
                
                if ftp_dir is None:
                    logger.error(f"Could not find {file_name} in any FTP directory")
                    return False
                
                # Now download the file
                ftp.cwd(ftp_dir)
                with open(file_path, 'wb') as f:
                    ftp.retrbinary(f"RETR {file_name}", f.write)
                
                logger.info(f"Successfully downloaded {file_name}")
                return True
                
            except Exception as e:
                logger.error(f"Error downloading {file_name}, attempt {attempt + 1}: {str(e)}")
                if os.path.exists(file_path):
                    os.remove(file_path)  # Remove partially downloaded file
            finally:
                if ftp:
                    try:
                        ftp.quit()
                    except:
                        try:
                            ftp.close()
                        except:
                            pass
        
        return False
    
    def _extract_article_data(self, xml_article: ET.Element) -> Optional[Dict]:
        """Extract comprehensive article data from XML element."""
        try:
            medline_citation = xml_article.find(".//MedlineCitation")
            if medline_citation is None:
                return None
                
            article = medline_citation.find("Article")
            if article is None:
                return None
            
            # Extract basic metadata
            data = {
                'pmid': medline_citation.findtext("PMID"),
                'title': article.findtext("ArticleTitle", ""),
                'abstract': " ".join(text.text or "" for text in article.findall(".//AbstractText")),
                'journal': article.findtext(".//Title", ""),
                'doi': article.findtext(".//ELocationID[@EIdType='doi']", ""),
                'publication_types': ",".join(ptype.text or "" for ptype in article.findall(".//PublicationType")),
                'mesh_terms': ",".join(term.findtext("DescriptorName", "") or "" for term in medline_citation.findall(".//MeshHeading")),
                'keywords': ",".join(kw.text or "" for kw in medline_citation.findall(".//Keyword")),
            }
            
            # Extract authors
            authors = []
            for author in article.findall(".//Author"):
                last_name = author.findtext("LastName", "")
                fore_name = author.findtext("ForeName", "")
                if last_name or fore_name:
                    authors.append(f"{fore_name} {last_name}".strip())
            data['authors'] = "; ".join(authors)
            
            # Extract publication date
            pub_date = article.find(".//PubDate")
            if pub_date is not None:
                year = pub_date.findtext("Year", "")
                month = pub_date.findtext("Month", "")
                day = pub_date.findtext("Day", "")
                data['publication_date'] = f"{year}-{month}-{day}".strip("-")
            else:
                data['publication_date'] = ""
            
            return data
            
        except Exception as e:
            logger.error(f"Error extracting article data: {str(e)}")
            return None
    
    def process_file(self, file_name: str) -> Tuple[int, int]:
        """Process a single XML file and insert articles into database."""
        file_path = os.path.join(self.download_dir, file_name)
        processed = 0
        errors = 0
        batch_size = 1000
        
        # Create a new database connection
        conn = self._get_db_connection()
        cur = conn.cursor()
        
        try:
            logger.info(f"Processing file {file_name}")
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                tree = ET.iterparse(f, events=('end',))
                batch = []
                
                for event, elem in tree:
                    if elem.tag == 'PubmedArticle':
                        article_data = self._extract_article_data(elem)
                        if article_data:
                            article_data['last_updated'] = datetime.now().isoformat()
                            article_data['source_file'] = file_name
                            batch.append(article_data)
                            processed += 1
                        else:
                            errors += 1
                            
                        # Process in batches
                        if len(batch) >= batch_size:
                            try:
                                self._insert_articles_batch(conn, cur, batch)
                                batch = []
                            except Exception:
                                errors += len(batch)
                                batch = []
                        
                        elem.clear()
                
                # Insert remaining articles
                if batch:
                    try:
                        self._insert_articles_batch(conn, cur, batch)
                    except Exception:
                        errors += len(batch)
                    
            logger.info(f"Successfully processed file: {file_name}")
            logger.info(f"Articles processed: {processed}, Errors: {errors}")
                    
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            errors += 1
        finally:
            conn.close()
            
        return processed, errors
    
    def _insert_articles_batch(self, conn: sqlite3.Connection, cur: sqlite3.Cursor, articles: List[Dict]):
        """Insert a batch of articles into the database."""
        try:
            cur.executemany('''
                INSERT OR REPLACE INTO articles 
                (pmid, title, abstract, authors, publication_date, journal, doi, 
                 publication_types, mesh_terms, keywords, last_updated, source_file)
                VALUES 
                (:pmid, :title, :abstract, :authors, :publication_date, :journal, :doi,
                 :publication_types, :mesh_terms, :keywords, :last_updated, :source_file)
            ''', articles)
            conn.commit()
        except Exception as e:
            logger.error(f"Error inserting batch: {str(e)}")
            conn.rollback()
            raise
    
    def fix_all(self):
        """Main method to fix all broken files with confirmation."""
        # Find broken files from the log
        broken_files = self.find_broken_files()
        if not broken_files:
            logger.info("No broken files found.")
            return
        
        # Display broken files and ask for confirmation
        print("\n=== Files with errors that need fixing ===")
        for i, file_name in enumerate(broken_files, 1):
            print(f"{i}. {file_name}")
        
        confirmation = input("\nDo you want to proceed with fixing these files? (y/n): ").strip().lower()
        if confirmation != 'y':
            logger.info("Operation cancelled by user.")
            return
        
        # Clean broken files
        self.clean_broken_files(broken_files)
        
        # Process each broken file
        total_processed = 0
        total_errors = 0
        
        for file_name in broken_files:
            print(f"\nProcessing {file_name}...")
            
            # Ask for confirmation per file
            file_confirm = input(f"Download and process {file_name}? (y/n/all): ").strip().lower()
            
            # Skip this file if not confirmed
            if file_confirm == 'n':
                logger.info(f"Skipping {file_name} as requested by user")
                continue
            
            # If user selects 'all', process all remaining without asking again
            all_remaining = (file_confirm == 'all')
            if not (file_confirm == 'y' or all_remaining):
                logger.info(f"Skipping {file_name} (unrecognized input)")
                continue
                
            # Download the file
            success = self.download_file(file_name)
            if not success:
                logger.error(f"Failed to download {file_name}, skipping processing")
                continue
                
            # Process the file
            processed, errors = self.process_file(file_name)
            total_processed += processed
            total_errors += errors
            
            # Skip confirmation for remaining files if user selected 'all'
            if all_remaining:
                break  # Exit the confirmation loop and process all remaining files
        
        # If user selected 'all', process remaining files without asking
        if all_remaining:
            for file_name in broken_files[broken_files.index(file_name) + 1:]:
                print(f"\nProcessing {file_name}...")
                success = self.download_file(file_name)
                if success:
                    processed, errors = self.process_file(file_name)
                    total_processed += processed
                    total_errors += errors
        
        logger.info(f"Fix completed. Total articles processed: {total_processed}, Errors: {total_errors}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fix corrupted PubMed downloads")
    parser.add_argument("--log", required=True, help="Path to the log file to analyze")
    parser.add_argument("--db", default="data/db/pubmed_data.db", help="Path to the SQLite database")
    parser.add_argument("--dir", default="data/downloads", help="Path to the downloads directory")
    
    args = parser.parse_args()
    
    fixer = PubMedFixer(db_path=args.db, download_dir=args.dir, log_path=args.log)
    fixer.fix_all()