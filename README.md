# PubMed Vector Search

A vector search engine for PubMed articles using BGE-M3 embeddings and optional DeepSeek API integration for enhanced responses.

## Features

- Fetch articles from PubMed using their API
- Create vector embeddings using BGE-M3
- Search similar articles using vector similarity
- Optional integration with DeepSeek API for enhanced responses
- FastAPI interface for easy integration

## Project Structure

```
pubvec/
├── src/pubvec/
│   ├── core/           # Core functionality
│   │   ├── api.py         # FastAPI server implementation
│   │   ├── pubmed_fetcher.py  # PubMed API interaction
│   │   └── vector_store.py    # Vector database management
│   ├── scripts/        # Utility scripts
│   │   ├── download_pubmed.py     # PubMed data download
│   │   ├── import_to_chroma.py    # Database import
│   │   └── search_example.py      # Search examples
│   └── utils/          # Helper utilities
├── config/            # Configuration files
├── logs/             # Log files
├── data/             # Data storage
├── chroma_db/        # Vector database storage
├── pyproject.toml    # Project configuration and dependencies
└── README.md
```

## Setup

1. Clone the repository and install the package in development mode:
```bash
pip install -e .
```

2. Create a `config/deepseek_api_key.txt` file for DeepSeek API configuration (optional):
```bash
echo "your_api_key_here" > config/deepseek_api_key.txt
```

## Usage

### 1. Populate the Vector Database

Run the import script to fetch articles and create the vector database:

```bash
python -m pubvec.scripts.import_to_chroma --email your.email@example.com --query "your search query" --max_results 1000
```

### 2. Start the API Server

```bash
python -m pubvec.core.api
```

The API will be available at `http://localhost:8000`

### 3. API Endpoints

#### POST /search
Search for articles and optionally get an AI-generated response.

Example request:
```json
{
    "query": "What are the latest developments in CAR-T therapy?",
    "model": "deepseek",
    "api_key": "your_deepseek_api_key",
    "n_results": 5
}
```

For direct vector search without AI response, set `"model": "direct"` and omit the `api_key`.

## Core Components

- `core/pubmed_fetcher.py`: Handles PubMed API interaction
- `core/vector_store.py`: Manages the vector database using ChromaDB and BGE-M3
- `core/api.py`: FastAPI server with search endpoints

## Scripts

- `scripts/download_pubmed.py`: Download PubMed data
- `scripts/import_to_chroma.py`: Import articles into vector database
- `scripts/search_example.py`: Example script for searching articles

## Notes

- The PubMed API requires an email address for identification
- DeepSeek API integration requires an API key
- Vector database is persisted locally in `./chroma_db`
- Logs are stored in the `logs/` directory
- Configuration files are stored in `config/` 