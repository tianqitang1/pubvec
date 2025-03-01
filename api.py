from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import requests
from functools import lru_cache
import os
from dotenv import load_dotenv
from vector_store import VectorStore
from datetime import datetime

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
    reasoning_process: Optional[str] = None

@lru_cache()
def get_vector_store():
    return VectorStore()

async def query_deepseek(query: str, api_key: str, context: str) -> Dict[str, str]:
    """Query DeepSeek API with context."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    prompt = f"""# The following contents are PubMed articles related to your query:
{context}

In the search results provided, each article is formatted as [Article X begin]...[Article X end], where X represents the numerical index of each PubMed article. Please cite the articles at the end of the relevant sentence when appropriate. Use the citation format [PMID:X] in the corresponding part of your answer. If a statement is derived from multiple articles, list all relevant PMIDs, such as [PMID:3][PMID:5].

When responding, please keep the following points in mind:
- Today is {datetime.now().strftime('%Y-%m-%d')}.
- Focus on synthesizing information from multiple articles to provide a comprehensive answer.
- Evaluate the scientific relevance and recency of each article to the query.
- For systematic reviews or meta-analyses, prioritize these as they provide higher-level evidence.
- When discussing research findings:
  * Note the study design (e.g., clinical trial, observational study)
  * Mention sample sizes when relevant
  * Highlight statistical significance if reported
  * Note any limitations mentioned in the articles
- For treatment-related queries, include information about:
  * Efficacy data
  * Safety profiles
  * Patient selection criteria
  * Comparison with standard of care
- Structure your response with clear sections:
  * Overview/Introduction
  * Main Findings
  * Clinical Implications (if applicable)
  * Future Directions/Challenges
- If multiple treatment approaches or methodologies are discussed, compare and contrast them.
- When citing statistics or specific findings, always include the PMID citation.
- If the articles present conflicting findings, acknowledge this and explain the different perspectives.
- For technical or methodological details, provide sufficient context for clarity.

# The query is:
{query}"""

    response = requests.post(
        "https://api.deepseek.com/v1/chat/completions",
        headers=headers,
        json={
            "model": "deepseek-reasoner",
            "messages": [{"role": "user", "content": prompt}]
        }
    )
    
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Error querying DeepSeek API")
    
    response_data = response.json()
    content = response_data["choices"][0]["message"]["content"]
    reasoning = response_data["choices"][0]["message"].get("reasoning_content")
    
    return {
        "reasoning": reasoning,
        "response": content
    }

@app.post("/search", response_model=SearchResult)
async def search(query: SearchQuery, vector_store: VectorStore = Depends(get_vector_store)):
    """Search PubMed articles and optionally generate a response."""
    articles = vector_store.search(query.query, query.n_results)
    
    if query.model == "deepseek":
        if not query.api_key:
            raise HTTPException(status_code=400, detail="API key required for DeepSeek model")
            
        # Create context from articles with proper formatting
        context = "\n\n".join([
            f"[Article {i+1} begin]\nTitle: {article['metadata']['title']}\n"
            f"Year: {article['metadata']['year']}\n"
            f"PMID: {article['id']}\n"
            f"Abstract: {article['document'].split('Abstract: ')[-1]}\n"
            f"[Article {i+1} end]"
            for i, article in enumerate(articles)
        ])
        
        deepseek_response = await query_deepseek(query.query, query.api_key, context)
        generated_response = deepseek_response["response"]
        reasoning_process = deepseek_response["reasoning"]
    else:
        generated_response = None
        reasoning_process = None
        
    return SearchResult(
        articles=articles,
        generated_response=generated_response,
        reasoning_process=reasoning_process
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 