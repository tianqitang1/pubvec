import json
import asyncio
import logging
import traceback
from typing import List, Dict, Any, Optional
import requests

from pubvec.core.vector_store import VectorStore
from pubvec.core.api import query_deepseek

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("pubvec-cli")

def read_api_key(filepath="config/deepseek_api_key.txt"):
    """Read API key from file."""
    try:
        with open(filepath, 'r') as f:
            return f.read().strip()
    except Exception as e:
        logger.warning(f"Could not read API key from {filepath}: {e}")
        return None

async def call_llm_api(prompt: str, base_url: str, api_key: str, model: str = "deepseek-chat") -> str:
    """Call LLM API with the given prompt."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            headers=headers,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}]
            }
        )
        
        if response.status_code != 200:
            logger.error(f"Error calling LLM API: {response.text}")
            raise Exception(f"Error calling LLM API: {response.status_code}, {response.text}")
        
        response_data = response.json()
        return response_data["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"Error in call_llm_api: {str(e)}")
        raise

def extract_json_from_llm_response(content: str) -> str:
    """Extract JSON from LLM response content."""
    try:
        if "```json" in content:
            # Extract JSON from code blocks
            start = content.find("```json") + 7
            end = content.find("```", start)
            return content[start:end].strip()
        elif "[" in content and "]" in content:
            # Try to extract array notation
            start = content.find("[")
            end = content.rfind("]") + 1
            return content[start:end].strip()
        elif "{" in content and "}" in content:
            # Try to extract object notation
            start = content.find("{")
            end = content.rfind("}") + 1
            return content[start:end].strip()
        else:
            return content
    except Exception as e:
        logger.error(f"Error extracting JSON from response: {str(e)}")
        logger.error(f"Original content: {content}")
        raise

async def extract_entities(query: str, base_url: str, api_key: str) -> List[str]:
    """Extract entities (genes/drugs/alleles) from the query."""
    prompt = f"""
You are a biomedical entity extractor. 
Extract all mentioned alleles, genes, and drugs from the following query.
Return ONLY a JSON array of strings containing the extracted entities. 
If no entities are found, return an empty array.

Query: {query}

Output format:
```json
["entity1", "entity2", ...]
```
"""
    
    try:
        content = await call_llm_api(prompt, base_url, api_key)
        json_content = extract_json_from_llm_response(content)
        
        entities = json.loads(json_content)
        logger.info(f"Extracted entities: {entities}")
        return entities
    except Exception as e:
        logger.error(f"Error extracting entities: {str(e)}")
        logger.error(traceback.format_exc())
        raise

async def extract_disease_tissue(query: str, base_url: str, api_key: str) -> Dict[str, str]:
    """Extract disease and tissue information from the query."""
    prompt = f"""
You are a biomedical entity extractor specialized in diseases and tissues.
From the following query, extract:
1. The main disease mentioned
2. The specific tissue or cell type context (if mentioned)

Return ONLY a JSON object with two keys: "disease" and "tissue".
If a tissue is not specified, use a reasonable default based on the disease or general term like "various tissues".

Query: {query}

Output format:
```json
{{
  "disease": "extracted disease",
  "tissue": "extracted tissue or cell type"
}}
```
"""
    
    try:
        content = await call_llm_api(prompt, base_url, api_key)
        json_content = extract_json_from_llm_response(content)
        
        disease_tissue = json.loads(json_content)
        logger.info(f"Extracted disease and tissue: {disease_tissue}")
        return disease_tissue
    except Exception as e:
        logger.error(f"Error extracting disease and tissue: {str(e)}")
        logger.error(traceback.format_exc())
        raise

async def get_entity_summary(entity: str, query: str, base_url: str, api_key: str, 
                           disease_tissue: Dict[str, str], store: VectorStore, 
                           model: str, n_results: int = 5) -> str:
    """Get summary about an entity's efficacy from PubMed articles."""
    disease = disease_tissue.get("disease", "")
    tissue = disease_tissue.get("tissue", "")
    
    try:
        # Construct a specific query for this entity
        entity_query = f"{entity} {disease} {tissue} efficacy"
        logger.info(f"Searching PubMed for: {entity_query}")
        
        # Search for relevant articles
        articles = store.search(entity_query, n_results)
        
        if not articles:
            logger.warning(f"No articles found for {entity} in {tissue} {disease}")
            return f"No information found about {entity} in the context of {disease} in {tissue}."
        
        # Create context from articles using safe access with .get()
        context = "\n\n".join([
            f"[Article {i+1} begin]\n"
            f"PMID: {article['id']}\n"
            f"Title: {article['metadata'].get('title', 'Unknown')}\n"
            f"Year: {article['metadata'].get('year', article['metadata'].get('publication_date', 'Unknown'))}\n"
            f"Document: {article['document']}\n"
            f"[Article {i+1} end]"
            for i, article in enumerate(articles)
        ])
        
        # Create prompt for entity efficacy
        prompt = f"""
Based on the provided PubMed articles, summarize the efficacy of {entity} 
for {disease} in {tissue} tissue/type.
Focus specifically on:
1. Evidence of effectiveness
2. Mechanism of action
3. Any limitations or side effects
4. Comparison with other treatments (if mentioned)

Provide a concise summary (200-300 words).
"""
        
        # Use the query_deepseek function from api.py
        logger.info(f"Generating summary for {entity}")
        deepseek_response = await query_deepseek(prompt, api_key, context)
        summary = deepseek_response["response"]
        
        return summary
    except Exception as e:
        logger.error(f"Error generating summary for {entity}: {str(e)}")
        logger.error(traceback.format_exc())
        raise

async def rank_entities(entity_summaries: List[Dict], query: str, base_url: str, api_key: str) -> List[Dict]:
    """Rank entities based on their efficacy summaries."""
    entities_info = "\n\n".join([
        f"ENTITY: {item['entity']}\nSUMMARY: {item['summary']}"
        for item in entity_summaries
    ])
    
    prompt = f"""
# Task: Rank biomedical entities based on efficacy evidence

## Original Query:
{query}

## Entity Summaries:
{entities_info}

## Instructions:
1. Based ONLY on the provided summaries, rank the entities from most promising to least promising in terms of efficacy.
2. For each entity, assign an efficacy score between 0.0 and 1.0, where:
   - 0.0 = No evidence of efficacy
   - 0.5 = Mixed or moderate evidence of efficacy
   - 1.0 = Strong evidence of efficacy
3. Return ONLY a JSON array of objects with the following structure:
```json
[
  {{
    "name": "entity name",
    "efficacy_score": float between 0.0-1.0,
    "summary": "one-sentence explanation of ranking"
  }}
]
```
4. The array should be sorted by efficacy_score in descending order.
5. Do not include any other text in your response, only the JSON array.
"""
    
    try:
        content = await call_llm_api(prompt, base_url, api_key)
        json_content = extract_json_from_llm_response(content)
        
        try:
            ranked_entities = json.loads(json_content)
            
            # Ensure proper typing
            for entity in ranked_entities:
                entity["efficacy_score"] = float(entity["efficacy_score"])
                logger.info(f"Entity {entity['name']} score: {entity['efficacy_score']}")
        except Exception as e:
            logger.error(f"Error parsing LLM response for ranking: {str(e)}")
            logger.error(f"Response content: {content}")
            logger.error(f"Extracted JSON content: {json_content}")
            raise Exception(f"Error parsing LLM response: {str(e)}\nResponse: {content}")
        
        return ranked_entities
    except Exception as e:
        logger.error(f"Error in rank_entities: {str(e)}")
        logger.error(traceback.format_exc())
        raise

async def process_query(query: str, api_key: str, base_url: str, model: str, debug: bool = False):
    """Process a query to rank entities based on efficacy."""
    try:
        logger.info(f"Processing query: {query}")

        # Initialize vector store
        store = VectorStore()

        # Extract entities from query
        entities = await extract_entities(query, base_url, api_key)
        if not entities:
            logger.warning("No entities found in query")
            return {
                "original_query": query,
                "entities": [],
                "debug_info": {"error": "No entities (genes/drugs/alleles) found in query"} if debug else None
            }

        # Extract disease and tissue information
        disease_tissue = await extract_disease_tissue(query, base_url, api_key)

        # Get summaries for each entity
        entity_summaries = []
        for entity in entities:
            summary = await get_entity_summary(
                entity, query, base_url, api_key, disease_tissue, store, model
            )
            entity_summaries.append({"entity": entity, "summary": summary})

        # Rank entities based on summaries
        ranked_entities = await rank_entities(entity_summaries, query, base_url, api_key)

        result = {
            "original_query": query,
            "entities": ranked_entities,
        }

        if debug:
            result["debug_info"] = {
                "extracted_entities": entities,
                "disease_tissue": disease_tissue,
                "entity_summaries": entity_summaries
            }

        return result

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "original_query": query,
            "entities": [],
            "debug_info": {"error": str(e)} if debug else None
        }

def display_results(results):
    """Display the ranked entities in a nice format."""
    print("\n" + "="*80)
    print(f"QUERY: {results['original_query']}")
    print("="*80)
    
    entities = results.get("entities", [])
    if not entities:
        print("\nNo entities found or no data available for ranking.")
        return
    
    print("\nRANKED ENTITIES BY EFFICACY:\n")
    for i, entity in enumerate(entities):
        score = entity.get("efficacy_score", 0)
        score_bar = "█" * int(score * 10)
        print(f"{i+1}. {entity['name']} - Efficacy Score: {score:.2f} {score_bar}")
        print(f"   {entity.get('summary', 'No summary available')}")
        print()
    
    print("="*80) 