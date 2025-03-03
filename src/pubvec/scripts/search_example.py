import requests
import json
import os

def get_deepseek_api_key():
    """Read DeepSeek API key from file."""
    try:
        with open("deepseek_api_key.txt", "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        raise ValueError(
            "DeepSeek API key file (deepseek_api_key.txt) not found. "
            "Please create this file with your API key."
        )

def search_articles(query: str, model: str = "direct", api_key: str = None, n_results: int = 5):
    """
    Search articles through the API.
    
    Args:
        query: Search query
        model: 'direct' for vector search only, 'deepseek' for AI-generated response
        api_key: Required if model is 'deepseek'
        n_results: Number of results to return
    """
    url = "http://localhost:8000/search"
    
    payload = {
        "query": query,
        "model": model,
        "n_results": n_results
    }
    
    if model == "deepseek":
        if not api_key:
            api_key = get_deepseek_api_key()
        payload["api_key"] = api_key
    
    response = requests.post(url, json=payload)
    response.raise_for_status()  # Raise exception for error status codes
    
    results = response.json()
    
    # Print results
    print(f"\nSearch Results for: {query}\n")
    
    for i, article in enumerate(results["articles"], 1):
        print(f"Article {i}:")
        print(f"Title: {article['metadata']['title']}")
        print(f"Year: {article['metadata']['year']}")
        print(f"Distance: {article['distance']:.3f}")
        print(f"Document: {article['document'][:300]}...")  # Show first 300 chars
        print()
    
    if results.get("reasoning_process"):
        print("\nChain of Thought:")
        print("-" * 80)
        print(results["reasoning_process"])
        print("-" * 80)
        print()
    
    if results.get("generated_response"):
        print("\nAI-Generated Response:")
        print("-" * 80)
        print(results["generated_response"])
        print("-" * 80)

if __name__ == "__main__":
    # Example 1: Direct vector search
    print("Example 1: Direct vector search")
    search_articles(
        query="What are the latest developments in CAR-T cell therapy?",
        model="direct",
        n_results=3
    )
    
    # Example 2: Using DeepSeek for AI-generated response
    print("\nExample 2: Search with DeepSeek response")
    search_articles(
        query="What are the main challenges in CAR-T cell therapy?",
        model="deepseek",
        n_results=3
    ) 