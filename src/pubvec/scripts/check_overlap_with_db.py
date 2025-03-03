#!/usr/bin/env python3
import gzip
import xml.etree.ElementTree as ET
import os
import sqlite3

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
    """Check how many PMIDs already exist in the database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get total count in database
    cursor.execute("SELECT COUNT(*) FROM articles")
    total_in_db = cursor.fetchone()[0]
    print(f"Total articles in database: {total_in_db}")
    
    # Check how many PMIDs from the XML files already exist in the database
    existing_pmids = 0
    batch_size = 1000  # Process in batches to avoid too large SQL queries
    
    pmids_list = list(pmids)
    for i in range(0, len(pmids_list), batch_size):
        batch = pmids_list[i:i+batch_size]
        placeholders = ','.join(['?'] * len(batch))
        query = f"SELECT COUNT(*) FROM articles WHERE pmid IN ({placeholders})"
        cursor.execute(query, batch)
        existing_pmids += cursor.fetchone()[0]
    
    conn.close()
    
    return existing_pmids

def main():
    download_dir = "data/downloads"
    db_path = "data/db/pubmed_data.db"
    
    files_to_analyze = [
        "pubmed25n0189.xml.gz",
        "pubmed25n0190.xml.gz",
        "pubmed25n0191.xml.gz"
    ]
    
    # Extract PMIDs from all files
    all_pmids = set()
    
    print("Extracting PMIDs from files...")
    for file_name in files_to_analyze:
        file_path = os.path.join(download_dir, file_name)
        print(f"Processing {file_name}...")
        pmids = extract_pmids_from_xml(file_path)
        all_pmids.update(pmids)
        print(f"  Found {len(pmids)} articles in {file_name}")
    
    print(f"\nTotal unique PMIDs across all files: {len(all_pmids)}")
    
    # Check how many already exist in the database
    print("\nChecking database overlap...")
    existing_pmids = check_pmids_in_db(all_pmids, db_path)
    
    print(f"PMIDs already in database: {existing_pmids} ({existing_pmids/len(all_pmids)*100:.2f}%)")
    print(f"PMIDs new to database: {len(all_pmids) - existing_pmids} ({(len(all_pmids) - existing_pmids)/len(all_pmids)*100:.2f}%)")

if __name__ == "__main__":
    main() 