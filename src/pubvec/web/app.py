from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import sys
import json
import logging
import re
from typing import List, Dict, Optional, Any
import requests
from pydantic import BaseModel, Field
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pubvec_web.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("pubvec.web")

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pubvec.core.api import query_deepseek

# Global variable to store the prefilled API key
prefilled_api_key = None

def initialize_api_key(api_key):
    """Initialize the prefilled API key."""
    global prefilled_api_key
    prefilled_api_key = api_key
    logger.info("API key prefilled from config file")

app = FastAPI(title="PubVec - Biomedical Entity Ranker")

# Mount static files
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "static"), exist_ok=True)
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")), name="static")

# Set up templates
templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
os.makedirs(templates_dir, exist_ok=True)
templates = Jinja2Templates(directory=templates_dir)

class RankRequest(BaseModel):
    query: str
    base_url: str
    api_key: str
    debug_mode: bool = Field(False, description="Enable debug mode with detailed logs")

class Entity(BaseModel):
    name: str
    efficacy_score: float
    summary: str

class DiseaseInfo(BaseModel):
    disease: str
    disease_subtype: str

class RankResponse(BaseModel):
    entities: List[Entity]
    original_query: str
    debug_info: Optional[Dict[str, Any]] = None

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "prefilled_api_key": prefilled_api_key or ""
    })

@app.post("/process_query", response_model=RankResponse)
async def process_query(request: RankRequest):
    """
    Process a user query to extract entities and rank them based on efficacy.
    """
    debug_info = {} if request.debug_mode else None
    
    try:
        logger.info(f"Processing query: {request.query}")
        
        # Step 1: Extract entities (alleles/genes/drugs) using LLM
        logger.info("Step 1: Extracting entities")
        entities = await extract_entities(request.query, request.base_url, request.api_key)
        logger.info(f"Extracted entities: {entities}")
        
        if request.debug_mode:
            debug_info["extracted_entities"] = entities
        
        # Step 2: Extract disease and type information
        logger.info("Step 2: Extracting disease and subtype information")
        disease_info = await extract_disease_subtype(request.query, request.base_url, request.api_key)
        logger.info(f"Extracted disease info: {disease_info}")
        
        if request.debug_mode:
            debug_info["disease_info"] = disease_info
        
        # Step 3: For each entity, query PubMed and get efficacy summary
        logger.info("Step 3: Getting entity summaries")
        entity_summaries = []
        for entity in entities:
            logger.info(f"Getting summary for entity: {entity}")
            summary = await get_entity_summary(entity, request.query, request.base_url, request.api_key, disease_info)
            entity_summaries.append({"name": entity, "summary": summary})
            logger.info(f"Summary for {entity} retrieved (length: {len(summary)})")
        
        if request.debug_mode:
            debug_info["entity_summaries"] = entity_summaries
        
        # Step 4: Rank entities based on efficacy
        logger.info("Step 4: Ranking entities")
        ranked_entities = await rank_entities(entity_summaries, request.query, request.base_url, request.api_key)
        logger.info(f"Entities ranked: {[e['name'] for e in ranked_entities]}")
        
        return RankResponse(
            entities=ranked_entities,
            original_query=request.query,
            debug_info=debug_info
        )
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error processing query: {error_msg}")
        logger.error(traceback.format_exc())
        
        if request.debug_mode:
            error_details = {
                "error_message": error_msg,
                "traceback": traceback.format_exc(),
                "state": debug_info
            }
            raise HTTPException(status_code=500, detail=json.dumps(error_details))
        else:
            raise HTTPException(status_code=500, detail=error_msg)

async def extract_entities(query: str, base_url: str, api_key: str) -> List[str]:
    """
    Use LLM to extract entities (alleles/genes/drugs) from the user query.
    """
    logger.info("Extracting entities from query")
    
    prompt = f"""
    Extract all alleles, genes, or drugs mentioned in the following query. 
    Return only a JSON array of strings with no additional text.
    
    Query: {query}
    
    Example output format:
    ["BRCA1", "TP53", "Tamoxifen"]
    """
    
    try:
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "deepseek-chat",  # Using deepseek-chat as requested
                "messages": [{"role": "user", "content": prompt}]
            }
        )
        
        if response.status_code != 200:
            logger.error(f"API error: {response.status_code} - {response.text}")
            raise HTTPException(status_code=500, detail=f"Error querying LLM API: {response.text}")
        
        content = response.json()["choices"][0]["message"]["content"]
        logger.info(f"LLM response for entity extraction: {content}")
        
        # Process the content to extract JSON if it's wrapped in markdown
        json_content = extract_json_from_llm_response(content)
        
        # Parse the JSON array from the response
        try:
            entities = json.loads(json_content)
            if not isinstance(entities, list):
                logger.warning(f"Unexpected format in entity extraction, not a list: {json_content}")
                entities = []
        except Exception as e:
            logger.warning(f"Failed to parse JSON from LLM response: {str(e)}")
            # If parsing fails, try to extract entities using simple heuristics
            logger.info("Attempting heuristic extraction")
            entities = re.findall(r'\[(.*?)\]', content)
            if entities:
                try:
                    entities = [e.strip('"\'') for e in entities[0].split(',')]
                except Exception as e2:
                    logger.warning(f"Heuristic extraction failed: {str(e2)}")
                    entities = []
            else:
                entities = []
            
            logger.info(f"Entities after heuristic extraction: {entities}")
        
        return entities
    except Exception as e:
        logger.error(f"Error in extract_entities: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def extract_json_from_llm_response(content: str) -> str:
    """
    Extract JSON content from LLM responses that might be formatted with markdown.
    Handles both raw JSON and JSON wrapped in markdown code blocks.
    """
    # Check if content appears to be wrapped in markdown code blocks
    code_block_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    match = re.search(code_block_pattern, content)
    
    if match:
        logger.info("Detected markdown code block in LLM response, extracting JSON")
        return match.group(1).strip()
    else:
        # No code block found, return original content
        return content

async def extract_disease_subtype(query: str, base_url: str, api_key: str) -> Dict[str, str]:
    """
    Extract the disease and its subtype from the query using LLM.
    """
    try:
        prompt = f"""
Extract the disease and its subtype (organ, subtype, stage, etc.) from the query.
Return the result as a JSON object with two fields: 'disease' and 'disease_subtype'.

Query: {query}

Example:
Query: "Rank olaparib, niraparib, and talazoparib for triple-negative breast cancer"
Output: {{"disease": "breast cancer", "disease_subtype": "triple-negative"}}

Query: "Rank the efficacy of BRCA1, TP53, and Tamoxifen for HER2-positive breast cancer"
Output: {{"disease": "breast cancer", "disease_subtype": "HER2-positive"}}
"""
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "deepseek-chat",  # Using deepseek-chat as requested
                "messages": [{"role": "user", "content": prompt}]
            }
        )
        
        if response.status_code != 200:
            logger.error(f"API error: {response.status_code} - {response.text}")
            raise HTTPException(status_code=500, detail=f"Error querying LLM API: {response.text}")
        
        content = response.json()["choices"][0]["message"]["content"]
        logger.info(f"LLM response for disease extraction: {content}")
        
        # Process the content to extract JSON if it's wrapped in markdown
        json_content = extract_json_from_llm_response(content)
        
        # Parse the JSON object from the response
        try:
            result = json.loads(json_content)
            logger.info(f"Parsed disease info: {result}")
            
            # Ensure all required keys are present
            if "disease" not in result:
                result["disease"] = ""
            if "disease_subtype" not in result or "type" in result:
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
        logger.error(f"Error in extract_disease_subtype: {str(e)}")
        logger.error(traceback.format_exc())
        raise

async def get_entity_summary(entity: str, query: str, base_url: str, api_key: str, disease_info: Dict[str, str] = None) -> str:
    """
    Query PubMed for articles about the entity in the context of the user's query,
    and generate a summary about its efficacy.
    """
    logger.info(f"Getting summary for entity: {entity}")
    
    try:
        from pubvec.core.vector_store import VectorStore
        
        # Use provided disease_info or extract it if not provided
        if disease_info is None:
            disease_info = await extract_disease_subtype(query, base_url, api_key)
        
        # Construct a specific query for this entity
        entity_query = f"{entity} {disease_info['disease']} {disease_info['disease_subtype']}"
        logger.info(f"Constructed entity query: {entity_query}")
        
        # Search for relevant articles
        logger.info("Searching vector store for articles")
        vector_store = VectorStore()
        articles = vector_store.search(entity_query, n_results=5)
        logger.info(f"Found {len(articles)} articles")
        
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
        
        # Generate summary using DeepSeek
        prompt = f"""
        Based on the provided PubMed articles, summarize the efficacy of {entity} 
        for {disease_info['disease']} in {disease_info['disease_subtype']} type.
        Focus specifically on:
        1. Evidence of effectiveness
        2. Mechanism of action
        3. Any limitations or side effects
        4. Comparison with other treatments (if mentioned)
        
        Provide a concise summary (200-300 words).
        """
        
        logger.info(f"Querying DeepSeek for summary of {entity}")
        deepseek_response = await query_deepseek(prompt, api_key, context)
        summary = deepseek_response["response"]
        logger.info(f"Generated summary for {entity} (length: {len(summary)})")
        
        return summary
    except Exception as e:
        logger.error(f"Error in get_entity_summary for {entity}: {str(e)}")
        logger.error(traceback.format_exc())
        raise

async def rank_entities(entity_summaries: List[Dict], query: str, base_url: str, api_key: str) -> List[Entity]:
    """
    Rank entities based on their efficacy summaries.
    """
    logger.info("Ranking entities based on summaries")
    
    try:
        # Prepare the summaries for the LLM
        all_summaries = "\n\n".join([
            f"Entity: {entity['name']}\nSummary: {entity['summary']}"
            for entity in entity_summaries
        ])
        
        prompt = f"""
        Based on the following summaries, rank the entities (alleles/genes/drugs) 
        by their efficacy for the given query: "{query}"
        
        Assign each entity an efficacy score from 0.0 to 10.0, where:
        - 0.0 means no evidence of efficacy or harmful
        - 5.0 means moderate efficacy with some limitations
        - 10.0 means excellent efficacy with strong evidence
        
        Return only a JSON array of objects with no additional text, where each object has:
        - "name": the entity name
        - "efficacy_score": a float between 0.0 and 10.0
        - "summary": a brief 1-2 sentence justification for the score
        
        The array should be sorted by efficacy_score in descending order.
        
        Summaries:
        {all_summaries}
        
        Example output format:
        [
          {{"name": "Drug A", "efficacy_score": 8.5, "summary": "Strong evidence of efficacy with minimal side effects."}},
          {{"name": "Gene B", "efficacy_score": 6.2, "summary": "Moderate efficacy but significant variability between patients."}}
        ]
        """
        
        logger.info("Querying LLM for entity ranking")
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "deepseek-reasoner",
                "messages": [{"role": "user", "content": prompt}]
            }
        )
        
        if response.status_code != 200:
            logger.error(f"API error in ranking: {response.status_code} - {response.text}")
            raise HTTPException(status_code=500, detail=f"Error querying LLM API: {response.text}")
        
        content = response.json()["choices"][0]["message"]["content"]
        logger.info(f"LLM response for ranking: {content}")
        
        # Process the content to extract JSON if it's wrapped in markdown
        json_content = extract_json_from_llm_response(content)
        
        # Parse the JSON array from the response
        try:
            ranked_entities = json.loads(json_content)
            logger.info(f"Successfully parsed ranking JSON: {len(ranked_entities)} entities")
            
            # Ensure proper typing
            for entity in ranked_entities:
                entity["efficacy_score"] = float(entity["efficacy_score"])
                logger.info(f"Entity {entity['name']} score: {entity['efficacy_score']}")
        except Exception as e:
            logger.error(f"Error parsing LLM response for ranking: {str(e)}")
            logger.error(f"Response content: {content}")
            logger.error(f"Extracted JSON content: {json_content}")
            raise HTTPException(status_code=500, detail=f"Error parsing LLM response: {str(e)}\nResponse: {content}")
        
        return ranked_entities
    except Exception as e:
        logger.error(f"Error in rank_entities: {str(e)}")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 