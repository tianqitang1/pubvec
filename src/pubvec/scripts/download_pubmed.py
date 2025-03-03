import socket
import ftplib
import os
import gzip
import xml.etree.ElementTree as ET
import sqlite3
from tqdm import tqdm
import concurrent.futures
import logging
from datetime import datetime
import time
import requests
from typing import List, Dict, Optional, Generator, Tuple
import sys

# Ensure logs directory exists
log_dir = "data/logs"
os.makedirs(log_dir, exist_ok=True)

# Create log filename with timestamp
log_filename = os.path.join(log_dir, f"pubmed_download_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

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

class PubMedDownloader:
    def __init__(self, 
                 db_path: str = "data/db/pubmed_data.db",
                 download_dir: str = "data/downloads",
                 max_workers: int = 4,
                 batch_size: int = 1000):
        """Initialize PubMed downloader with configuration."""
        self.db_path = db_path
        self.download_dir = download_dir
        self.max_workers = max_workers
        self.batch_size = batch_size
        
        # FTP settings - using direct IP
        self.ftp_host = "130.14.250.7"  # Using a specific IP from the DNS lookup
        self.ftp_dirs = ["/pubmed/baseline/", "/pubmed/updatefiles/"]
        self.ftp_user = "anonymous"
        self.ftp_pass = "Tianqi.Tang@ucsf.edu"
        
        logger.info(f"Initializing with FTP host: {self.ftp_host}")
        
        # Test FTP connection
        try:
            ftp = ftplib.FTP(self.ftp_host)
            logger.info("Attempting anonymous login...")
            ftp.login(user=self.ftp_user, passwd=self.ftp_pass)
            logger.info("FTP connection test successful")
            ftp.quit()
        except Exception as e:
            logger.error(f"FTP connection test failed: {str(e)}")
            logger.error(f"Exception type: {type(e)}")
            if hasattr(e, 'args'):
                logger.error(f"Exception args: {e.args}")
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        os.makedirs(download_dir, exist_ok=True)
        
        # Initialize database schema
        self._init_database_schema()
        
    def _get_db_connection(self):
        """Create a new database connection for the current thread."""
        conn = sqlite3.connect(self.db_path, timeout=60.0)  # 60 second timeout
        conn.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging for better concurrency
        conn.execute("PRAGMA busy_timeout=60000")  # 60 second busy timeout
        conn.row_factory = sqlite3.Row
        return conn
        
    def _init_database_schema(self):
        """Initialize SQLite database schema."""
        conn = self._get_db_connection()
        try:
            cur = conn.cursor()
            # Create tables with more metadata
            cur.executescript('''
                CREATE TABLE IF NOT EXISTS articles (
                    pmid TEXT PRIMARY KEY,
                    title TEXT,
                    abstract TEXT,
                    authors TEXT,
                    publication_date TEXT,
                    journal TEXT,
                    doi TEXT,
                    publication_types TEXT,
                    mesh_terms TEXT,
                    keywords TEXT,
                    last_updated TIMESTAMP,
                    source_file TEXT
                );
                
                CREATE TABLE IF NOT EXISTS download_history (
                    file_name TEXT PRIMARY KEY,
                    download_date TIMESTAMP,
                    status TEXT,
                    error_message TEXT
                );
                
                CREATE INDEX IF NOT EXISTS idx_pub_date ON articles(publication_date);
                CREATE INDEX IF NOT EXISTS idx_journal ON articles(journal);
            ''')
            conn.commit()
        finally:
            conn.close()
    
    def _get_ftp_file_list(self) -> List[Dict]:
        """Get list of files from FTP server with metadata."""
        files = []
        ftp = ftplib.FTP(self.ftp_host)
        ftp.login(user=self.ftp_user, passwd=self.ftp_pass)
        
        try:
            for ftp_dir in self.ftp_dirs:
                logger.info(f"Accessing directory {ftp_dir}")
                try:
                    ftp.cwd(ftp_dir)
                    file_list = ftp.nlst()
                    logger.info(f"Found {len(file_list)} files in {ftp_dir}")
                    
                    for file in file_list:
                        if file.endswith('.xml.gz'):
                            files.append({
                                'name': file,
                                'path': ftp_dir,
                                'local_path': os.path.join(self.download_dir, file)
                            })
                            logger.debug(f"Found file: {file}")
                except ftplib.error_perm as e:
                    logger.error(f"Permission error accessing directory {ftp_dir}: {str(e)}")
                except Exception as e:
                    logger.error(f"Error accessing directory {ftp_dir}: {str(e)}")
                    
        finally:
            try:
                ftp.quit()
            except:
                pass
            
        return files
    
    def _download_file(self, file_info: Dict) -> Optional[str]:
        """Download a single file from FTP with retry logic."""
        max_retries = 3
        for attempt in range(max_retries):
            ftp = None
            try:
                if os.path.exists(file_info['local_path']):
                    logger.debug(f"File {file_info['name']} already exists, skipping download")
                    return file_info['local_path']
                
                logger.debug(f"Attempting to download {file_info['name']} (attempt {attempt + 1}/{max_retries})")
                
                ftp = ftplib.FTP(self.ftp_host)
                ftp.login(user=self.ftp_user, passwd=self.ftp_pass)
                
                # Set binary mode and keep it
                ftp.voidcmd('TYPE I')
                
                ftp.cwd(file_info['path'])
                
                with open(file_info['local_path'], 'wb') as f:
                    ftp.retrbinary(f"RETR {file_info['name']}", f.write)
                
                logger.debug(f"Successfully downloaded {file_info['name']}")
                return file_info['local_path']
                
            except Exception as e:
                logger.error(f"Error downloading {file_info['name']}, attempt {attempt + 1}: {str(e)}")
                if attempt == max_retries - 1:
                    self._record_download_error(file_info['name'], str(e))
                    return None
                time.sleep(5 * (attempt + 1))
            finally:
                if ftp:
                    try:
                        ftp.quit()
                    except:
                        try:
                            ftp.close()
                        except:
                            pass
    
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
    
    def _process_file(self, file_path: str) -> Tuple[int, int]:
        """Process a single XML file and insert articles into database."""
        processed = 0
        errors = 0
        
        # Create a new database connection for this thread
        conn = self._get_db_connection()
        cur = conn.cursor()
        
        try:
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                tree = ET.iterparse(f, events=('end',))
                batch = []
                
                for event, elem in tree:
                    if elem.tag == 'PubmedArticle':
                        article_data = self._extract_article_data(elem)
                        if article_data:
                            article_data['last_updated'] = datetime.now().isoformat()
                            article_data['source_file'] = os.path.basename(file_path)
                            batch.append(article_data)
                            processed += 1
                        else:
                            errors += 1
                            
                        # Process in batches
                        if len(batch) >= self.batch_size:
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
                    
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            errors += 1
        finally:
            conn.close()
            
        return processed, errors
    
    def _insert_articles_batch(self, conn: sqlite3.Connection, cur: sqlite3.Cursor, articles: List[Dict]):
        """Insert a batch of articles into the database with retry logic."""
        max_retries = 5
        retry_delay = 1  # Start with 1 second delay
        
        for attempt in range(max_retries):
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
                return  # Success, exit the retry loop
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and attempt < max_retries - 1:
                    logger.warning(f"Database locked, retrying in {retry_delay} seconds (attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"Error inserting batch after {attempt + 1} attempts: {str(e)}")
                    conn.rollback()
                    raise
            except Exception as e:
                logger.error(f"Error inserting batch: {str(e)}")
                conn.rollback()
                raise
    
    def _record_download_error(self, file_name: str, error_message: str):
        """Record download errors in the database with retry logic."""
        max_retries = 5
        retry_delay = 1
        
        for attempt in range(max_retries):
            conn = None
            try:
                conn = self._get_db_connection()
                cur = conn.cursor()
                cur.execute('''
                    INSERT OR REPLACE INTO download_history 
                    (file_name, download_date, status, error_message)
                    VALUES (?, ?, ?, ?)
                ''', (file_name, datetime.now().isoformat(), 'ERROR', error_message))
                conn.commit()
                return  # Success, exit the retry loop
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and attempt < max_retries - 1:
                    logger.warning(f"Database locked, retrying in {retry_delay} seconds (attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"Error recording download history after {attempt + 1} attempts: {str(e)}")
                    if conn:
                        conn.rollback()
            except Exception as e:
                logger.error(f"Error recording download history: {str(e)}")
                if conn:
                    conn.rollback()
            finally:
                if conn:
                    conn.close()
    
    def download_and_process(self):
        """Main method to download and process all PubMed files."""
        try:
            # Get list of files to process
            files = self._get_ftp_file_list()
            logger.info(f"Found {len(files)} files on server")
            
            # Check which files are already downloaded
            missing_files = []
            existing_files = []
            for file_info in files:
                if os.path.exists(file_info['local_path']):
                    existing_files.append(file_info)
                    logger.debug(f"File exists locally: {file_info['name']}")
                else:
                    missing_files.append(file_info)
                    logger.debug(f"File needs download: {file_info['name']}")
            
            logger.info(f"Found {len(existing_files)} existing files and {len(missing_files)} files to download")
            
            # Download missing files in parallel
            if missing_files:
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    future_to_file = {
                        executor.submit(self._download_file, file_info): file_info
                        for file_info in missing_files
                    }
                    
                    for future in tqdm(concurrent.futures.as_completed(future_to_file), 
                                     total=len(missing_files),
                                     desc="Downloading missing files"):
                        file_info = future_to_file[future]
                        try:
                            local_path = future.result()
                            if local_path:
                                logger.info(f"Successfully downloaded {file_info['name']}")
                                existing_files.append(file_info)
                        except Exception as e:
                            logger.error(f"Error downloading {file_info['name']}: {str(e)}")
            
            # Process all downloaded files
            logger.info(f"Processing {len(existing_files)} files")
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_file = {
                    executor.submit(self._process_file, file_info['local_path']): file_info
                    for file_info in existing_files
                }
                
                total_processed = 0
                total_errors = 0
                
                for future in tqdm(concurrent.futures.as_completed(future_to_file),
                                 total=len(existing_files),
                                 desc="Processing files"):
                    file_info = future_to_file[future]
                    try:
                        processed, errors = future.result()
                        total_processed += processed
                        total_errors += errors
                        
                        # Log successful processing
                        logger.debug(f"Successfully processed file: {file_info['name']}")
                        
                    except Exception as e:
                        logger.error(f"Error processing {file_info['name']}: {str(e)}")
                        
            logger.info(f"Completed processing. Total articles: {total_processed}, Errors: {total_errors}")
            logger.info(f"All downloaded files are kept in: {self.download_dir}")
            
        finally:
            self._init_database_schema()

if __name__ == "__main__":
    downloader = PubMedDownloader()
    downloader.download_and_process() 