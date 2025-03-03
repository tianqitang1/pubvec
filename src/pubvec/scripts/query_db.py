import sqlite3
import sys
import os

def query_random_article():
    # Check if database file exists
    db_path = 'data/db/pubmed_data.db'
    print(f"Checking for database at: {os.path.abspath(db_path)}")
    
    if not os.path.exists(db_path):
        print(f"Database file not found at {db_path}")
        return
    
    print(f"Database file exists, size: {os.path.getsize(db_path) / (1024*1024):.2f} MB")
    
    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Check tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"Tables in database: {', '.join([t['name'] for t in tables])}")
        
        # Get article count
        cursor.execute("SELECT COUNT(*) as count FROM articles")
        count = cursor.fetchone()['count']
        print(f"Total articles in database: {count:,}")
        
        if count == 0:
            print("Database exists but contains no articles.")
            return
        
        # Query a random article
        cursor.execute("""
            SELECT 
                pmid, 
                title, 
                substr(abstract, 1, 200) || '...' as abstract_preview,
                substr(authors, 1, 100) || CASE WHEN length(authors) > 100 THEN '...' ELSE '' END as authors_preview,
                publication_date, 
                journal 
            FROM articles 
            ORDER BY RANDOM() 
            LIMIT 1
        """)
        
        article = cursor.fetchone()
        
        if article:
            print("\n=== Random Article from PubMed Database ===")
            print(f"PMID: {article['pmid']}")
            print(f"Title: {article['title']}")
            print(f"Abstract: {article['abstract_preview']}")
            print(f"Authors: {article['authors_preview']}")
            print(f"Publication Date: {article['publication_date']}")
            print(f"Journal: {article['journal']}")
            print("===========================================\n")
        else:
            print("No articles found in the database.")
            
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'conn' in locals() and conn:
            conn.close()

if __name__ == "__main__":
    query_random_article() 