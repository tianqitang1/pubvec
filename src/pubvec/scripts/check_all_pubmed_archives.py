#!/usr/bin/env python3
import gzip
import xml.etree.ElementTree as ET
import os
import sqlite3
import re

def extract_pmids_from_xml(xml_file_path):
    """Extract PMIDs from a gzipped PubMed XML file."""
    pmids = set()
    
    try:
        with gzip.open(xml_file_path, 'rt', encoding='utf-8') as f:
            # Use iterparse to avoid loading the entire file into memory
            context = ET.iterparse(f, events=('end',))
            
            for event, elem in context:
                if elem.tag == 'PubmedArticle':
                    # Find the PMID element
                    pmid_elem = elem.find('.//PMID')
                    
                    if pmid_elem is not None:
                        pmid = pmid_elem.text
                        pmids.add(pmid)
                    
                    # Clear the element to free memory
                    elem.clear()
        
        return pmids
    except Exception as e:
        print(f"Error processing {os.path.basename(xml_file_path)}: {str(e)}")
        return set()

def check_pmids_in_db(pmids, db_path):
    """Check how many PMIDs already exist in the database and identify missing ones."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get total count in database
    cursor.execute("SELECT COUNT(*) FROM articles")
    total_in_db = cursor.fetchone()[0]
    print(f"Total articles in database: {total_in_db}")
    
    # Check which PMIDs exist in the database
    existing_pmids = set()
    missing_pmids = set()
    batch_size = 1000  # Process in batches to avoid too large SQL queries
    
    pmids_list = list(pmids)
    for i in range(0, len(pmids_list), batch_size):
        batch = pmids_list[i:i+batch_size]
        placeholders = ','.join(['?'] * len(batch))
        query = f"SELECT pmid FROM articles WHERE pmid IN ({placeholders})"
        cursor.execute(query, batch)
        
        # Add found PMIDs to existing set
        for row in cursor.fetchall():
            existing_pmids.add(row[0])
    
    # Determine missing PMIDs
    missing_pmids = pmids - existing_pmids
    
    conn.close()
    
    return existing_pmids, missing_pmids

def main():
    download_dir = "data/downloads"
    db_path = "data/db/pubmed_data.db"
    
    # Find all PubMed XML files in the download directory
    pubmed_files = []
    for filename in os.listdir(download_dir):
        if re.match(r'^pubmed\d+n\d+\.xml\.gz$', filename):
            pubmed_files.append(filename)
    
    pubmed_files.sort()  # Sort files for consistent reporting
    total_pubmed_files = len(pubmed_files)
    
    print(f"Found {total_pubmed_files} PubMed XML files in {download_dir}")
    
    # Set to track all PMIDs across files
    all_pmids = set()
    files_processed = 0
    
    # Process files in batches to manage memory usage
    batch_size = 20
    for i in range(0, len(pubmed_files), batch_size):
        batch_files = pubmed_files[i:i+batch_size]
        
        print(f"\nProcessing files {i+1} to {min(i+batch_size, total_pubmed_files)} of {total_pubmed_files}...")
        
        # Extract PMIDs from each file in the batch
        for file_name in batch_files:
            file_path = os.path.join(download_dir, file_name)
            pmids = extract_pmids_from_xml(file_path)
            all_pmids.update(pmids)
            files_processed += 1
            
            print(f"  {file_name}: Found {len(pmids)} articles (Total unique PMIDs so far: {len(all_pmids)})")
    
    print(f"\nProcessed {files_processed} files. Total unique PMIDs found: {len(all_pmids)}")
    
    # Check which PMIDs exist in the database
    print("\nChecking database overlap...")
    existing_pmids, missing_pmids = check_pmids_in_db(all_pmids, db_path)
    
    # Calculate statistics
    existing_count = len(existing_pmids)
    missing_count = len(missing_pmids)
    
    print(f"PMIDs found in database: {existing_count} of {len(all_pmids)} ({existing_count/len(all_pmids)*100:.2f}%)")
    
    if missing_count > 0:
        print(f"PMIDs missing from database: {missing_count} ({missing_count/len(all_pmids)*100:.2f}%)")
        print("\nSample of missing PMIDs:")
        for pmid in list(missing_pmids)[:10]:  # Show up to 10 examples
            print(f"  {pmid}")
    else:
        print("All PMIDs from downloaded archives exist in the database! ✓")

if __name__ == "__main__":
    main() 