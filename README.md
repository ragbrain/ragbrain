# RAGBrain 🧠

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

A personal knowledge base with semantic search and AI-powered answers. Lightweight RAG pipeline without LangChain - direct API calls to OpenAI/Anthropic/Ollama and Qdrant, with hierarchical namespaces, provider fallback chains, and structure-aware chunking.

---

## Features

- **Capture thoughts** - Quick note-taking, instantly searchable
- **Import documents** - PDF, EPUB, Markdown, DOCX, PPTX, VTT/SRT transcripts
- **Semantic search** - Find things by meaning, not just keywords
- **AI answers** - Ask questions, get synthesized answers from your documents
- **Namespaces** - Organize into nested categories
- **One command** - `docker-compose up` and you're running

---

## Quick Start

### Prerequisites

- Docker & Docker Compose
- API key from [Anthropic](https://console.anthropic.com/) or [OpenAI](https://platform.openai.com/)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/ragbrain/ragbrain.git
cd ragbrain

# 2. Configure environment
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY or OPENAI_API_KEY

# 3. Start RAGBrain
docker-compose up -d

# 4. Open the UI
open http://localhost:8000
```

That's it.

---

## Usage

### Capture a Thought

```bash
# Via Web UI
Open http://localhost:8000 and click "Capture"

# Via API
curl -X POST http://localhost:8000/api/capture \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Kubernetes uses etcd for distributed consensus and stores all cluster state there.",
    "metadata": {
      "tags": ["kubernetes", "architecture"],
      "source": "K8s docs"
    }
  }'
```

### Query Your Knowledge

```bash
# Via Web UI
Open http://localhost:8000 and ask a question

# Via API
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How does Kubernetes handle distributed consensus?",
    "top_k": 5
  }'
```

### Import Documents

```bash
# Via Web UI (supports multiple files)
Open http://localhost:8000 → Upload → Drag & drop files

# Via CLI (for bulk directory imports)
docker exec -it ragbrain-app ragbrain-import ./docs --namespace my-docs
```

---

## CLI

Bulk import directories of documents.

### Basic Import

```bash
# Import all supported files from a directory
ragbrain-import /path/to/docs --namespace my-docs

# Or via Docker
docker exec -it ragbrain-app ragbrain-import /data/docs --namespace my-docs
```

### Import with Pattern Matching

```bash
# Import only markdown files
ragbrain-import ./docs -n notes -p "*.md"

# Import recursively (include subdirectories)
ragbrain-import ./docs -n notes -p "*.md" -r
```

### Import with Custom Metadata

```bash
# Add metadata to all imported documents
ragbrain-import ./sermons -n church/sermons \
  -m speaker="Pastor John" \
  -m year=2024

# Prepend metadata to chunks for better semantic search
ragbrain-import ./lectures -n lectures \
  -m speaker="Dr. Smith" \
  --prepend-metadata speaker
```

### Chunking Strategies

```bash
# Use markdown chunking for .md files
ragbrain-import ./docs -n docs -c markdown

# Use transcript chunking for VTT/SRT files
ragbrain-import ./transcripts -n lectures -c transcript
```

Available strategies: `recursive` (default), `hierarchical`, `markdown`, `semantic`, `character`, `transcript`

### Dry Run & Error Handling

```bash
# Preview what would be imported (no actual import)
ragbrain-import ./docs -n test --dry-run

# Continue on errors instead of stopping
ragbrain-import ./mixed-docs -n docs --skip-errors

# Verbose output
ragbrain-import ./docs -n docs -v
```

### CLI Options Reference

```
ragbrain-import PATH [OPTIONS]

Arguments:
  PATH                    Directory containing documents to import

Options:
  -n, --namespace TEXT    Target namespace (required)
  -p, --pattern TEXT      Glob pattern for files (default: *)
  -r, --recursive         Search subdirectories
  -c, --chunking TEXT     Chunking strategy (default: recursive)
  -m, --metadata TEXT     Add metadata as key=value (repeatable)
  --prepend-metadata TEXT Comma-separated metadata keys to prepend to chunks
  --dry-run               Show what would be imported
  --skip-errors           Continue on errors
  -v, --verbose           Verbose output
  --help                  Show help
```

### Supported File Types

- **Text**: `.txt`, `.md`, `.markdown`
- **Documents**: `.pdf`, `.docx`, `.pptx`
- **Ebooks**: `.epub`
- **Transcripts**: `.vtt`, `.srt`

---

## Architecture

```
┌─────────────────────────────────┐      ┌─────────────┐
│         App Container           │─────▶│   Qdrant    │
│  Vue Frontend + FastAPI Backend │      │  (Vector DB)│
│          Port 8000              │      │  Port 6333  │
└─────────────────────────────────┘      └─────────────┘
                 │
                 ▼
          ┌─────────────┐
          │ LLM Provider│
          │Claude / GPT │
          └─────────────┘
```

**Components:**

- **App**: Single container serving Vue frontend + FastAPI backend on port 8000
- **Vector DB**: Qdrant for semantic search
- **LLM**: Claude, GPT, or Ollama for answer synthesis

---

## Development

### Run locally without Docker

**Backend:**
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Start Qdrant separately (Docker)
docker run -p 6333:6333 qdrant/qdrant

# Run backend
uvicorn ragbrain.api.main:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

### Running Tests

```bash
cd backend
pip install -r requirements.txt

# Run all tests
pytest

# Run with coverage report
pytest --cov=ragbrain --cov-report=html

# Run specific test file
pytest tests/test_chunking.py

# Run tests in verbose mode
pytest -v
```

**Test Categories:**
- `test_chunking.py` - Text chunking strategy tests
- `test_config.py` - Configuration module tests
- `test_providers.py` - Provider factory and base class tests
- `test_pipeline.py` - RAG pipeline integration tests
- `test_api.py` - FastAPI endpoint tests

### Project Structure

```
ragbrain/
├── backend/              # Python FastAPI backend
│   ├── ragbrain/
│   │   ├── api/          # API routes
│   │   ├── rag/          # RAG pipeline
│   │   ├── loaders/      # Document loaders
│   │   ├── cli/          # CLI tools
│   │   └── config.py     # Configuration
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/             # Vue frontend
│   ├── src/
│   │   ├── components/   # UI components
│   │   ├── pages/        # Pages (Home, Capture, Query, Namespaces)
│   │   └── api/          # API client
│   ├── package.json
│   └── Dockerfile
│
├── data/                 # Persistent data (gitignored)
│   ├── qdrant/           # Vector database
│   └── uploads/          # Uploaded files
│
├── docker-compose.yml    # One-command deployment
├── .env.example          # Configuration template
└── README.md
```

---

## Contributing

PRs welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

MIT License - see [LICENSE](LICENSE) for details

---

## Acknowledgments

Built with:
- [FastAPI](https://fastapi.tiangolo.com/) - Backend API
- [Qdrant](https://qdrant.tech/) - Vector database
- [Vue.js](https://vuejs.org/) - Frontend
- [Claude](https://www.anthropic.com/claude) / [GPT](https://openai.com/) / [Ollama](https://ollama.ai/) - LLM providers
