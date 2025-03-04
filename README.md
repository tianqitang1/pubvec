# PubMed Vector Search

A vector search engine for PubMed articles using BGE-M3 embeddings and optional DeepSeek API integration for enhanced responses.

## Features

- Fetch articles from PubMed using their API
- Create vector embeddings using BGE-M3
- Search similar articles using vector similarity
- Optional integration with DeepSeek API for enhanced responses
- FastAPI interface for easy integration
- Web interface for biomedical entity ranking

## Project Structure

```
pubvec/
├── src/pubvec/
│   ├── core/           # Core functionality
│   │   ├── api.py         # FastAPI server implementation
│   │   ├── pubmed_fetcher.py  # PubMed API interaction
│   │   └── vector_store.py    # Vector database management
│   ├── web/            # Web interface
│   │   ├── app.py         # Web application backend
│   │   ├── templates/     # HTML templates
│   │   └── static/        # Static assets
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

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `config/deepseek_api_key.txt` file for DeepSeek API configuration (optional):
```bash
echo "your_api_key_here" > config/deepseek_api_key.txt
```

## Usage

### 1. Populate the Vector Database

Run the import script to fetch articles and create the vector database:

```bash
python main.py fetch --email your.email@example.com --query "your search query" --max_results 1000
```

### 2. Start the API Server

```bash
python -m pubvec.core.api
```

The API will be available at `http://localhost:8000`

### 3. Start the Web Application

```bash
python main.py web
```

The web application will be available at `http://localhost:8001`

Alternatively, you can run the web application directly:

```bash
python -m pubvec.web
```

### 4. Using the Web Interface

1. Open your browser and navigate to `http://localhost:8001`
2. Enter your query in the text area, including the entities (alleles/genes/drugs), disease, and tissue/type
3. Enter your DeepSeek API base URL and API key in the API Settings section
4. Click "Process Query" to analyze and rank the entities

Example query:
```
Rank the efficacy of BRCA1, TP53, and Tamoxifen for HER2-positive breast cancer
```

### 5. API Endpoints

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

#### POST /process_query
Process a user query to extract entities and rank them based on efficacy.

Example request:
```json
{
    "query": "Rank the efficacy of BRCA1, TP53, and Tamoxifen for HER2-positive breast cancer",
    "base_url": "https://api.deepseek.com",
    "api_key": "your_deepseek_api_key"
}
```

## Core Components

- `core/pubmed_fetcher.py`: Handles PubMed API interaction
- `core/vector_store.py`: Manages the vector database using ChromaDB and BGE-M3
- `core/api.py`: FastAPI server with search endpoints
- `web/app.py`: Web application for entity ranking

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