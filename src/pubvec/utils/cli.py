import json
import asyncio
import logging
import traceback
import os
import datetime
from typing import List, Dict, Any, Optional
import requests

from pubvec.core.vector_store import VectorStore
from pubvec.core.api import query_deepseek
from pubvec.utils.logging_utils import log_llm_request, log_llm_response

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Get current datetime for log filename
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"logs/{current_time}_cli.log"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_filename)
    ]
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
    
    # Log the full prompt being sent to the LLM
    log_llm_request(logger, prompt, model=model)
    
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
        content = response_data["choices"][0]["message"]["content"]
        
        # Log the full response from the LLM
        log_llm_response(logger, content, model=model)
        
        return content
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

Example output format:
```json
["BRCA1", "TP53", "Tamoxifen"]
```

Example:
Query: "Rank olaparib, niraparib, rucaparib, and talazoparib for triple-negative breast cancer"
Output: 
```json
["olaparib", "niraparib", "rucaparib", "talazoparib"]
```

Query: "Rank the efficacy of BRCA1, TP53, and Tamoxifen for HER2-positive breast cancer"
Output: 
```json
["BRCA1", "TP53", "Tamoxifen"]
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

async def extract_disease_subtype(query: str, base_url: str, api_key: str) -> Dict[str, str]:
    """Extract the disease and its subtype from the query."""
    logger.info("Extracting disease and subtype information")
    
    prompt = f"""
Extract the disease and its subtype (organ, subtype, stage, etc.) from the query.
Return the result as a JSON object with two fields: 'disease' and 'disease_subtype'.

Query:{query}

Example output format:
```json
{{"disease": "disease name", "disease_subtype": "disease subtype"}}
```

Example:
Query: "Rank olaparib, niraparib, and talazoparib for triple-negative breast cancer"
Output: 
```json
{{"disease": "breast cancer", "disease_subtype": "triple-negative"}}
```

Query: "Rank the efficacy of BRCA1, TP53, and Tamoxifen for HER2-positive breast cancer"
Output: 
```json
{{"disease": "breast cancer", "disease_subtype": "HER2-positive"}}
```
    """
    
    try:
        content = await call_llm_api(prompt, base_url, api_key)
        
        try:
            json_content = extract_json_from_llm_response(content)
            result = json.loads(json_content)
            
            # Ensure all required keys are present
            if "disease" not in result:
                result["disease"] = ""
            if "disease_subtype" not in result:
                # Handle both new and old format responses
                result["disease_subtype"] = result.get("type", "")
                if "type" in result:
                    del result["type"]
                
        except Exception as e:
            logger.warning(f"Failed to parse disease JSON: {str(e)}")
            # If parsing fails, use default values
            result = {"disease": "", "disease_subtype": ""}
        
        return result
    except Exception as e:
        logger.error(f"Error extracting disease information: {str(e)}")
        return {"disease": "", "disease_subtype": ""}

async def get_entity_summary(entity: str, query: str, base_url: str, api_key: str, 
                           disease_info: Dict[str, str], store: VectorStore, 
                           model: str, n_results: int = 10) -> str:
    """Generate a summary of an entity's efficacy from PubMed articles."""
    try:
        # Construct a specific query for this entity
        disease = disease_info.get("disease", "")
        subtype = disease_info.get("disease_subtype", "")
        entity_query = f"{entity} {disease} {subtype}"
        
        # Search for relevant articles
        logger.info(f"Searching for articles about: {entity_query}")
        articles = store.search(entity_query, n_results=n_results)
        
        if not articles:
            logger.warning(f"No articles found for {entity}")
            return f"No information found about {entity} for {disease} {subtype}."
        
        # Create context from articles
        context = "\n\n".join([
            f"[Article {i+1} begin]\n"
            f"PMID: {article['id']}\n"
            f"Journal: {article['metadata'].get('journal', 'Unknown')}\n"
            f"Year: {article['metadata'].get('publication_date', 'Unknown')}\n"
            f"Authors: {article['metadata'].get('authors', 'Unknown')}\n"
            f"Document: {article['document']}\n"
            f"[Article {i+1} end]"
            for i, article in enumerate(articles)
        ])
        
        # Create complete prompt including context and instructions
        prompt = f"""# The following contents are PubMed articles related to {entity} for {disease} {subtype}:
{context}

In the search results provided, each article is formatted as [Article X begin]...[Article X end], where X represents the numerical index of each PubMed article. Please cite the articles at the end of the relevant sentence when appropriate. Use the citation format [PMID:X] in the corresponding part of your answer. If a statement is derived from multiple articles, list all relevant PMIDs, such as [PMID:3][PMID:5].

Based on the provided PubMed articles, summarize the efficacy of {entity} 
for {disease} in {subtype} type.

When responding, please keep the following points in mind:
- Today is {datetime.datetime.now().strftime('%Y-%m-%d')}.
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
- If multiple treatment approaches or methodologies are discussed, compare and contrast them.
- When citing statistics or specific findings, always include the PMID citation.
- If the articles present conflicting findings, acknowledge this and explain the different perspectives.
- For technical or methodological details, provide sufficient context for clarity.
Provide a concise summary (200-300 words).
"""
        
        # Use call_llm_api directly instead of query_deepseek
        logger.info(f"Generating summary for {entity}")
        summary = await call_llm_api(prompt, base_url, api_key, model)
        
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
    "summary": "one-sentence explanation of ranking",
    "references": "PMIDs of the articles used to generate the summary"
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
    """Process a natural language query and return ranked entities."""
    logger.info(f"Processing query: {query}")
    results = {
        "original_query": query,
        "entities": [],
        "debug_info": {} if debug else None
    }
    
    try:
        # Step 1: Extract entities from query
        logger.info("Extracting entities from query")
        entities = await extract_entities(query, base_url, api_key)
        logger.info(f"Extracted entities: {entities}")
        
        if debug:
            results["debug_info"]["extracted_entities"] = entities
        
        # Step 2: Extract disease and subtype information
        logger.info("Extracting disease and subtype information")
        disease_info = await extract_disease_subtype(query, base_url, api_key)
        logger.info(f"Extracted disease info: {disease_info}")
        
        if debug:
            results["debug_info"]["disease_info"] = disease_info
        
        # Initialize vector store
        store = VectorStore()

        # Get summaries for each entity
        entity_summaries = []
        for entity in entities:
            summary = await get_entity_summary(
                entity, query, base_url, api_key, disease_info, store, model, n_results=10
            )
            entity_summaries.append({"entity": entity, "summary": summary})

        # Rank entities based on summaries
        ranked_entities = await rank_entities(entity_summaries, query, base_url, api_key)

        results["entities"] = ranked_entities

        if debug:
            results["debug_info"]["entity_summaries"] = entity_summaries

        return results

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
        print(f"   References: {entity.get('references', 'No references available')}")
        print()
    
    print("="*80) 