#!/usr/bin/env python3
import gzip
import xml.etree.ElementTree as ET
import os
from collections import defaultdict

def extract_pmids_from_xml(xml_file_path):
    """Extract PMIDs and titles from a gzipped PubMed XML file."""
    pmids = set()
    titles = {}
    
    try:
        with gzip.open(xml_file_path, 'rt', encoding='utf-8') as f:
            # Use iterparse to avoid loading the entire file into memory
            context = ET.iterparse(f, events=('end',))
            
            for event, elem in context:
                if elem.tag == 'PubmedArticle':
                    # Find the PMID element
                    pmid_elem = elem.find('.//PMID')
                    title_elem = elem.find('.//ArticleTitle')
                    
                    if pmid_elem is not None:
                        pmid = pmid_elem.text
                        title = title_elem.text if title_elem is not None else "No title"
                        pmids.add(pmid)
                        titles[pmid] = title
                    
                    # Clear the element to free memory
                    elem.clear()
        
        return pmids, titles
    except Exception as e:
        print(f"Error processing {os.path.basename(xml_file_path)}: {str(e)}")
        return set(), {}

def main():
    download_dir = "data/downloads"
    files_to_analyze = [
        "pubmed25n0189.xml.gz",
        "pubmed25n0190.xml.gz",
        "pubmed25n0191.xml.gz"
    ]
    
    file_pmids = {}
    file_titles = {}
    
    print("Extracting PMIDs from files...")
    
    # Extract PMIDs from each file
    for file_name in files_to_analyze:
        file_path = os.path.join(download_dir, file_name)
        print(f"Processing {file_name}...")
        pmids, titles = extract_pmids_from_xml(file_path)
        file_pmids[file_name] = pmids
        file_titles[file_name] = titles
        print(f"  Found {len(pmids)} articles in {file_name}")
    
    # Analyze overlap between files
    print("\nAnalyzing overlap between files:")
    
    # Pairwise overlaps
    for i, file1 in enumerate(files_to_analyze):
        for file2 in files_to_analyze[i+1:]:
            overlap = file_pmids[file1].intersection(file_pmids[file2])
            print(f"Overlap between {file1} and {file2}: {len(overlap)} articles")
            
            # Show a few examples of overlapping articles
            if overlap:
                print("Example overlapping articles:")
                for pmid in list(overlap)[:3]:  # Show up to 3 examples
                    print(f"  PMID: {pmid}")
                    print(f"  Title: {file_titles[file1].get(pmid, 'No title')}")
                    print()
    
    # Overlap among all three files
    all_overlap = file_pmids[files_to_analyze[0]].intersection(
        file_pmids[files_to_analyze[1]], 
        file_pmids[files_to_analyze[2]]
    )
    print(f"\nOverlap among all three files: {len(all_overlap)} articles")
    
    # Unique PMIDs across all files
    all_pmids = set().union(*file_pmids.values())
    print(f"Total unique PMIDs across all files: {len(all_pmids)}")
    
    # Calculate how many PMIDs appear in only one file
    pmids_in_one_file = 0
    for pmid in all_pmids:
        count = sum(1 for file_name in files_to_analyze if pmid in file_pmids[file_name])
        if count == 1:
            pmids_in_one_file += 1
    
    print(f"PMIDs appearing in only one file: {pmids_in_one_file} ({pmids_in_one_file/len(all_pmids)*100:.2f}%)")
    print(f"PMIDs appearing in multiple files: {len(all_pmids) - pmids_in_one_file} ({(len(all_pmids) - pmids_in_one_file)/len(all_pmids)*100:.2f}%)")

if __name__ == "__main__":
    main() 