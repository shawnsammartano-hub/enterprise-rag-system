# Enterprise RAG System - Production-Ready Implementation

A production-grade Retrieval-Augmented Generation (RAG) pipeline demonstrating enterprise AI architecture patterns used by Databricks, Snowflake, and leading fintech companies.

![RAG Architecture](https://img.shields.io/badge/RAG-Production%20Ready-green)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🎯 Business Context

Built to demonstrate AI enablement capabilities for financial services enterprise environments. Showcases understanding of:
- Vector search architectures (like Databricks Vector Search)
- Semantic embeddings (like Snowflake Cortex)
- Enterprise RAG patterns for document Q&A
- Production deployment considerations

## 🏗️ Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                   User Interface (Streamlit)                 │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    RAG Pipeline Layer                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Document   │  │   Chunking   │  │  Embedding   │      │
│  │   Ingestion  │→ │   Strategy   │→ │  Generation  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              Vector Database (ChromaDB)                      │
│         Persistent Storage with Semantic Search              │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                  Retrieval & Re-ranking                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Hybrid     │  │  Metadata    │  │   Re-rank    │      │
│  │   Search     │→ │  Filtering   │→ │  Top Results │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              LLM Generation (Ollama/Llama 3.1)              │
│           Context-Aware Answer Synthesis                     │
└─────────────────────────────────────────────────────────────┘
```

## ✨ Features

### Core Capabilities
- ✅ **Multi-format ingestion**: PDF (with OCR), DOCX, TXT
- ✅ **Semantic search**: 768-dimensional embeddings (nomic-embed-text)
- ✅ **Hybrid retrieval**: Vector (70%) + BM25 keyword (30%)
- ✅ **Smart chunking**: 500-word chunks with 10% overlap
- ✅ **Metadata filtering**: SQL-like queries on document attributes
- ✅ **Two-stage re-ranking**: Fast retrieval + precision scoring
- ✅ **Local LLM**: Privacy-first with Ollama (Llama 3.1)
- ✅ **Web interface**: Production-ready Streamlit UI

### Advanced Features
- **OCR Support**: Tesseract integration for scanned documents
- **Persistent Storage**: ChromaDB with disk-based persistence
- **Citation Tracking**: Source attribution for all answers
- **Configurable Parameters**: Chunk size, retrieval count, alpha weighting
- **Performance Monitoring**: Query latency and token throughput metrics

## 🚀 Quick Start

### Prerequisites
```bash
# System requirements
- Python 3.12+
- 8GB RAM minimum
- Ubuntu 24.04 or WSL2

# Ollama with models
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

### Installation
```bash
# Clone repository
git clone https://github.com/shawnsammartano-hub/enterprise-rag-system.git
cd enterprise-rag-system

# Install dependencies
pip install -r requirements.txt

# Install OCR support
sudo apt install tesseract-ocr poppler-utils -y

# Launch web interface
streamlit run app.py
```

### Docker Deployment (Coming Soon)
```bash
docker-compose up -d
```

## 📊 Performance Metrics

| Metric | Value | Context |
|--------|-------|---------|
| Embedding Speed | 787 tokens/sec | nomic-embed-text (768 dims) |
| Query Latency | ~15-20s | CPU inference (AMD Radeon) |
| Chunk Processing | 42 chunks from 116-page PDF | With OCR |
| Storage Efficiency | ~3GB per 1M documents | 768-dimensional vectors |

## 🎓 Enterprise Patterns Demonstrated

### 1. Hybrid Search
```python
# 70% semantic, 30% keyword matching
result = rag.hybrid_query(
    question="What is Snowflake used for?",
    alpha=0.7  # Enterprise default
)
```

**Why**: Pure vector search misses exact terms (SKUs, IDs), pure keyword misses synonyms.

### 2. Chunking Strategy
```python
# 500 words with 10% overlap
chunks = rag.chunk_text(
    text=document,
    chunk_size=500,
    overlap=50
)
```

**Why**: Balances context preservation with retrieval precision. Matches Databricks defaults.

### 3. Metadata Filtering
```python
# Filter by document attributes
result = rag.query_with_filter(
    question="Recent policy updates?",
    metadata_filter={"department": "Legal", "year": {"$gte": 2024}}
)
```

**Why**: Reduces search space, improves relevance. Like Databricks partition pruning.

### 4. Two-Stage Re-ranking
```python
# Retrieve 10, re-rank to top 3
result = rag.query_with_reranking(
    question="What are our Q2 goals?",
    initial_results=10,
    final_results=3
)
```

**Why**: Fast first-pass retrieval, expensive precision scoring only on candidates.

## 🏢 Enterprise Mapping

| This System | Databricks | Snowflake | Azure |
|-------------|-----------|-----------|-------|
| ChromaDB | Vector Search | Cortex Search | AI Search |
| nomic-embed-text | instructor-xl | E5-large | text-embedding-ada |
| Hybrid Search | Delta + Vector | UDF + Vector | Hybrid Retrieval |
| Llama 3.1 | DBRX/Llama | Mistral/Llama | GPT-4 |

## 📁 Project Structure
```
enterprise-rag-system/
├── app.py                      # Streamlit web interface
├── scripts/
│   ├── rag_pipeline.py        # Core RAG implementation
│   ├── hybrid_search.py       # Vector + BM25 hybrid
│   ├── reranking.py           # Two-stage retrieval
│   ├── metadata_filtering.py  # Attribute-based filtering
│   ├── chunking_analysis.py   # Strategy comparison
│   └── embedding_comparison.py # Model benchmarking
├── documents/                  # Sample documents
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🔧 Configuration
```python
# ~/.rag_pipeline/chromadb/
# Persistent vector database storage

# Key parameters
EMBEDDING_MODEL = "nomic-embed-text"  # 768 dimensions
LLM_MODEL = "llama3.1-32k"            # 32K context window
CHUNK_SIZE = 500                       # words
CHUNK_OVERLAP = 50                     # 10% overlap
ALPHA = 0.7                            # Hybrid search weight
```

## 📈 Use Cases

### Financial Services
- Client onboarding documentation
- Compliance policy Q&A
- Product knowledge base
- Risk assessment reports

### Technology Companies
- API documentation search
- Internal wiki/knowledge base
- Technical runbooks
- Incident post-mortems

### Healthcare
- Clinical guidelines
- Research paper analysis
- Patient education materials
- Protocol documentation

## 🛠️ Development
```bash
# Run tests
pytest tests/

# Benchmark performance
python scripts/embedding_comparison.py

# Analyze chunking strategies
python scripts/chunking_analysis.py

# Compare search methods
python scripts/hybrid_search.py
```

## 📝 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- Built with [LangChain](https://langchain.com), [ChromaDB](https://www.trychroma.com), [Ollama](https://ollama.ai)
- Embedding model: [nomic-embed-text](https://huggingface.co/nomic-ai/nomic-embed-text-v1)
- LLM: [Llama 3.1](https://ai.meta.com/llama/) by Meta

## 📧 Contact

Built by: Shawn Sammartano 
Portfolio: [Your Portfolio URL] (https://shawnsammartano-hub.github.io/data-driven-csm/) 
LinkedIn: [Your LinkedIn] (https://www.linkedin.com/in/shawnsammartano/)
Purpose: AI Enablement demonstration

---

**Note**: This is a demonstration system. For production deployment, consider:
- Authentication & authorization
- Rate limiting & quotas
- Horizontal scaling (vector DB sharding)
- Monitoring & observability (Datadog, Prometheus)
- CI/CD pipelines
- Security scanning
- Backup & disaster recovery
