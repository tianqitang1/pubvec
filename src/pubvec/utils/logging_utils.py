import logging
import os
import datetime
from typing import Optional, Dict, Any

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Get current datetime for log filename
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"logs/{current_time}_pubvec.log"

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_filename)
    ]
)

def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name and configured handlers."""
    return logging.getLogger(name)

def log_llm_request(logger: logging.Logger, prompt: str, 
                   model: Optional[str] = None, 
                   system_prompt: Optional[str] = None,
                   additional_info: Optional[Dict[str, Any]] = None) -> None:
    """Log a request to an LLM API."""
    log_data = {
        "prompt": prompt,
        "model": model or "unspecified",
    }
    
    if system_prompt:
        log_data["system_prompt"] = system_prompt
        
    if additional_info:
        log_data.update(additional_info)
    
    logger.info(f"LLM REQUEST: {log_data}")

def log_llm_response(logger: logging.Logger, response: str, 
                    model: Optional[str] = None,
                    reasoning: Optional[str] = None,
                    additional_info: Optional[Dict[str, Any]] = None) -> None:
    """Log a response from an LLM API."""
    log_data = {
        "response": response,
        "model": model or "unspecified",
    }
    
    if reasoning:
        log_data["reasoning"] = reasoning
        
    if additional_info:
        log_data.update(additional_info)
    
    logger.info(f"LLM RESPONSE: {log_data}") 