from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import requests
from functools import lru_cache
import os
from dotenv import load_dotenv
from vector_store import VectorStore

load_dotenv()

app = FastAPI(title="PubMed Vector Search API")

class SearchQuery(BaseModel):
    query: str = Field(..., description="The search query")
    model: str = Field("deepseek", description="Model to use for search: 'deepseek' or 'direct'")
    api_key: Optional[str] = Field(None, description="API key for the model (required for DeepSeek)")
    n_results: int = Field(5, description="Number of results to return")

class SearchResult(BaseModel):
    articles: List[Dict]
    generated_response: Optional[str]

@lru_cache()
def get_vector_store():
    return VectorStore()

async def query_deepseek(query: str, api_key: str, context: str) -> str:
    """Query DeepSeek API with context."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    prompt = f"""Based on the following PubMed articles, please provide a comprehensive answer to this query: {query}

Context from relevant articles:
{context}

Please synthesize the information and provide a detailed response."""

    response = requests.post(
        "https://api.deepseek.com/v1/chat/completions",
        headers=headers,
        json={
            # "model": "deepseek-chat",
            "model": "deepseek-reasoner",
            "messages": [{"role": "user", "content": prompt}]
        }
    )
    
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Error querying DeepSeek API")
        
    return response.json()["choices"][0]["message"]["content"]

@app.post("/search", response_model=SearchResult)
async def search(query: SearchQuery, vector_store: VectorStore = Depends(get_vector_store)):
    """Search PubMed articles and optionally generate a response."""
    articles = vector_store.search(query.query, query.n_results)
    
    if query.model == "deepseek":
        if not query.api_key:
            raise HTTPException(status_code=400, detail="API key required for DeepSeek model")
            
        # Create context from articles
        context = "\n\n".join([f"Article {i+1}:\n{article['document']}" 
                              for i, article in enumerate(articles)])
        
        generated_response = await query_deepseek(query.query, query.api_key, context)
    else:
        generated_response = None
        
    return SearchResult(
        articles=articles,
        generated_response=generated_response
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 