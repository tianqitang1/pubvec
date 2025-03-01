# PubMed Vector Search

A vector search engine for PubMed articles using BGE-M3 embeddings and optional DeepSeek API integration for enhanced responses.

## Features

- Fetch articles from PubMed using their API
- Create vector embeddings using BGE-M3
- Search similar articles using vector similarity
- Optional integration with DeepSeek API for enhanced responses
- FastAPI interface for easy integration

## Setup

1. Clone the repository and install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file for configuration (optional):
```bash
DEEPSEEK_API_KEY=your_api_key_here
```

## Usage

### 1. Populate the Vector Database

Run the main script to fetch articles and create the vector database:

```bash
python main.py --email your.email@example.com --query "your search query" --max_results 1000
```

### 2. Start the API Server

```bash
python api.py
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

## Components

- `pubmed_fetcher.py`: Handles PubMed API interaction
- `vector_store.py`: Manages the vector database using ChromaDB and BGE-M3
- `api.py`: FastAPI server with search endpoints
- `main.py`: Script to populate the vector database

## Notes

- The PubMed API requires an email address for identification
- DeepSeek API integration requires an API key
- Vector database is persisted locally in `./chroma_db` 