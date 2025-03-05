# PubMed Vectorized Ranking (pubvec)

A system that helps biomedical researchers rank the efficacy of alleles, genes, or drugs for specific diseases and tissue/types using PubMed data and LLM technology.

## Overview

This system allows users to:
1. Input a natural language query about the efficacy of multiple biomedical entities
2. Extract entities (alleles/genes/drugs) from the query using LLM
3. Search PubMed articles for each entity's efficacy in the context of the disease and tissue/type
4. Rank the entities based on their efficacy using LLM analysis of the research evidence
5. Visualize the rankings through a web interface

## Features

- Natural language query processing
- Entity extraction using LLM
- PubMed article search via vector similarity
- BGE-M3 embeddings for semantic search
- Entity ranking based on efficacy evidence
- Web-based user interface
- DeepSeek API integration (via OpenAI-compatible API)

## Project Structure

```
pubvec/
├── src/pubvec/
│   ├── core/               # Core functionality
│   │   ├── api.py          # FastAPI server implementation
│   │   ├── pubmed_fetcher.py  # PubMed API interaction
│   │   └── vector_store.py    # Vector database management
│   ├── web/                # Web interface
│   │   ├── app.py          # Web application backend
│   │   ├── templates/      # HTML templates
│   │   └── static/         # Static assets
│   ├── scripts/            # Utility scripts
│   │   ├── download_pubmed.py     # PubMed data download script
│   │   ├── import_to_chroma.py    # ChromaDB import script
│   │   └── search_example.py      # Search examples
│   ├── utils/              # Helper utilities
│   └── prompts/            # LLM prompts
├── config/                 # Configuration files
├── logs/                   # Log files
├── data/                   # Data storage
│   ├── db/                 # SQLite database
│   └── downloads/          # PubMed downloads
├── chroma_db/              # Vector database storage
├── pyproject.toml          # Project configuration and dependencies
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

3. Create a `config/deepseek_api_key.txt` file with your DeepSeek API key:
```bash
echo "your_api_key_here" > config/deepseek_api_key.txt
```

## Usage

### Data Pipeline Setup

The system requires PubMed data to be downloaded and vectorized before use. This is a one-time setup process:

#### 1. Download PubMed Data

Download the entire PubMed dataset and process it into a SQLite database:

```bash
python main.py download
```

Options:
- `--db-path`: Path to SQLite database (default: "data/db/pubmed_data.db")
- `--download-dir`: Directory to store downloads (default: "data/downloads")
- `--max-workers`: Maximum number of concurrent downloads (default: 4)
- `--batch-size`: Batch size for database operations (default: 1000)

#### 2. Import to Vector Database

Import the downloaded PubMed articles into ChromaDB with BGE-M3 embeddings:

```bash
python main.py import-to-chroma
```

Options:
- `--sqlite-path`: Path to SQLite database (default: "data/db/pubmed_data.db")
- `--persist-dir`: Directory to store ChromaDB (default: "chroma_db")
- `--batch-size`: Batch size for import operations (default: 1000)
- `--use-gpu`: Use GPU for embedding generation (flag)

#### 3. Combined Download and Import

To perform both steps in one command:

```bash
python main.py process-pubmed
```

This command accepts all options from both the `download` and `import-to-chroma` commands.

### Entity Ranking

#### Command Line Interface

Rank biomedical entities for a specific disease and tissue/type:

```bash
python main.py rank "Rank the efficacy of BRCA1, TP53, and Tamoxifen for HER2-positive breast cancer"
```

Options:
- `--cheap`: Use cheaper deepseek-chat model instead of deepseek-reasoner
- `--debug`: Enable debug mode with detailed logs
- `--output`: Output file to save results (JSON format)
- `--base-url`: API base URL (default: https://api.deepseek.com)

#### Web Interface

Start the web interface for entity ranking:

```bash
python main.py web
```

Options:
- `--host`: Host to run the web server on (default: 0.0.0.0)
- `--port`: Port to run the web server on (default: 8001)
- `--prefill-api-key`: Prefill API key from config/deepseek_api_key.txt

### Using the Web Interface

1. Open your browser and navigate to `http://localhost:8001`
2. Enter your query in the text area, including the entities (alleles/genes/drugs), disease, and tissue/type
3. Enter your LLM API base URL and API key in the API Settings section
4. Click "Process Query" to analyze and rank the entities

Example query:
```
Rank the efficacy of BRCA1, TP53, and Tamoxifen for HER2-positive breast cancer
```

## How It Works

1. The system parses the user's query using an LLM to extract:
   - The list of biomedical entities (alleles/genes/drugs)
   - The disease context
   - The tissue/type context

2. For each identified entity, the system:
   - Constructs a specific search query
   - Uses vector similarity to find relevant PubMed articles
   - Retrieves information about the entity's efficacy

3. All entity information is aggregated and sent to an LLM to:
   - Analyze the evidence for each entity
   - Rank entities based on their efficacy
   - Provide reasoning for the ranking

4. Results are presented to the user through:
   - A ranked list of entities
   - Supporting evidence from PubMed
   - Explanations for the rankings

## API Endpoints

### POST /search
Search for PubMed articles and optionally get an AI-generated response.

### POST /process_query
Process a user query to extract entities and rank them based on efficacy.

## Advanced Configuration

- Modify the LLM prompts in `src/pubvec/prompts/`
- Configure the vector search parameters in `src/pubvec/core/vector_store.py`
- Adjust the web interface in `src/pubvec/web/templates/`

## Notes

- Vector database (ChromaDB) is persisted locally and not updated for every query
- The PubMed dataset download is a one-time operation that fetches the entire dataset
- The system uses BGE-M3 embeddings for vector similarity search
- DeepSeek API integration uses an OpenAI-compatible API interface

## License

This project is licensed under the MIT License with Commons Clause - see the [LICENSE](LICENSE) file for details.

The Commons Clause restriction prevents the software from being used commercially without permission from the copyright holders. This means you can freely use, modify, and distribute this software for non-commercial and educational purposes, but commercial use requires explicit permission.

## TODO
- Add support for more models
- Periodically update the PubMed dataset