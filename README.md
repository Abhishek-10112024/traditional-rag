---
title: RAG Assistant
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.55.0"
python_version: "3.13"
app_file: streamlit_app.py
pinned: false
---

# RAG Assistant - Production Ready Free RAG System

A complete, production-ready Retrieval-Augmented Generation (RAG) system built entirely with free and open-source tools. Features hybrid search, reranking, local LLMs, and comprehensive evaluation.

## Features

✨ **Production Ready**
- Comprehensive error handling and logging
- Caching for performance optimization
- Batch processing support
- Type hints throughout the codebase

🔍 **Hybrid Search**
- Combines BM25 (sparse retrieval) and vector search (dense retrieval)
- Configurable weights for hybrid aggregation
- Optimized for both precision and recall

📊 **Reranking**
- Cross-encoder based reranking with `ms-marco-MiniLM-L-6-v2`
- Improves relevance of retrieved documents
- Configurable thresholds and batch sizes

🤖 **Free LLMs**
- **Ollama**: Run models locally (mistral, neural-chat, orca-mini, etc.)
- **HuggingFace**: Cloud inference (free tier available)
- No API keys required for Ollama

💾 **Vector Store**
- Chroma: Persistent vector storage
- Automatic persistence to disk
- Metadata support for document tracking

🚀 **Speed Optimization**
- Response caching (LRU-based)
- Batch processing for embeddings
- Efficient hybrid search aggregation

📈 **Evaluation Framework**
- RAGAS metrics (faithfulness, answer relevancy, context precision, recall)
- Additional metrics (retrieval precision/recall, MRR, NDCG)
- Comprehensive evaluation reports

💬 **Streamlit UI**
- Intuitive chat interface
- Document management
- Real-time settings configuration
- Retrieved documents visualization

## Tech Stack

```
Embeddings:     sentence-transformers (all-MiniLM-L6-v2)
Vector Store:   Chroma
Reranker:       cross-encoder (MS MARCO)
Search:         BM25 + Vector Hybrid
LLM:            Ollama (local) / HuggingFace (cloud)
Frontend:       Streamlit
Evaluation:     RAGAS + Custom Metrics
Cloud:          HuggingFace Spaces / Streamlit Cloud
```

## Project Structure

```
vectorless-rag/
├── src/                          # Core RAG modules
│   ├── __init__.py
│   ├── embeddings.py             # Embedding generation
│   ├── vectorstore.py            # Chroma vector store
│   ├── bm25_search.py            # BM25 sparse search
│   ├── hybrid_search.py          # Combined search
│   ├── reranker.py               # Cross-encoder reranking
│   ├── llm.py                    # LLM integration (Ollama/HF)
│   └── rag_pipeline.py           # Main RAG pipeline
├── evaluation/                   # Evaluation framework
│   ├── __init__.py
│   ├── evaluator.py              # RAGAS evaluator
│   └── metrics.py                # Additional metrics
├── scripts/                      # Utility scripts
│   ├── ingest_documents.py       # Batch document ingestion
│   └── evaluate.py               # Evaluation runner
├── data/                         # Data directory
│   └── processed/                # Processed documents
├── cache/                        # Cache directory
├── logs/                         # Log directory
├── streamlit_app.py              # Streamlit frontend
├── config.py                     # Configuration management
├── requirements.txt              # Dependencies
├── .env.example                  # Environment template
├── Dockerfile                    # Docker containerization
└── README.md                     # This file
```

## Installation

### 1. Clone/Setup Repository

```bash
cd vectorless-rag
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Setup Environment Variables

```bash
# Copy example env file
cp .env.example .env

# Edit .env as needed (optional, defaults work fine)
```

### 5. Setup LLM Backend

#### Option A: Ollama (Recommended for local inference)

**Install Ollama:**
- Download from [ollama.ai](https://ollama.ai)
- Install and run

**Start Ollama Server:**

```bash
ollama serve
```

**Download a Model (in another terminal):**

```bash
# Balanced model (recommended)
ollama pull mistral

# Faster, smaller model
ollama pull neural-chat

# Lightweight model for slow systems
ollama pull orca-mini

# Large model (requires 8GB+ RAM)
ollama pull llama2
```

#### Option B: HuggingFace Inference

Set environment variable:

```bash
export HUGGINGFACE_API_TOKEN=your_token_here
```

Then update `config.py`:
```python
LLM_TYPE = "huggingface"
```

## Usage

### Option 1: Streamlit Web UI (Recommended)

```bash
streamlit run streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

Features:
- Upload documents via UI
- Interactive chat interface
- View retrieved documents
- Configure settings in real-time

### Option 2: Python API

```python
from src.rag_pipeline import RAGPipeline

# Initialize
rag = RAGPipeline()

# Add documents
documents = [
    "Document 1 text...",
    "Document 2 text...",
]
rag.add_documents(documents)

# Query
result = rag.query("What is your question?")

print("Answer:", result["answer"])
print("Context:", result["context"])
print("Retrieved Docs:", result["retrieved_docs"])
```

### Option 3: Command Line (Document Ingestion)

```bash
python scripts/ingest_documents.py file1.txt file2.txt file3.md
```

## Configuration

Edit `config.py` or `.env` to customize:

### Search Configuration
```python
HYBRID_SEARCH_ENABLED = True
VECTOR_WEIGHT = 0.7          # Weight for vector search
BM25_WEIGHT = 0.3            # Weight for BM25 search
TOP_K_RETRIEVAL = 10         # Documents to retrieve before reranking
```

### Reranking Configuration
```python
RERANKING_ENABLED = True
TOP_K_RERANK = 4             # Documents to send to LLM
RERANKER_SCORE_THRESHOLD = 0.1
```

### LLM Configuration
```python
OLLAMA_MODEL = "mistral"     # Model to use
TEMPERATURE = 0.7            # 0.0 = deterministic, 1.0+ = creative
TOP_P = 0.9                  # Nucleus sampling
MAX_NEW_TOKENS = 512         # Max response length
```

### Caching
```python
CACHE_ENABLED = True         # Enable response caching
CACHE_TTL = 3600            # Cache timeout in seconds
BATCH_SIZE = 32             # Batch size for embeddings
```

## Deployment

### Option 1: HuggingFace Spaces (Free, Recommended)

1. Create account on [huggingface.co](https://huggingface.co)
2. Create new Space
3. Choose Streamlit runtime
4. Import repository or upload files
5. Add `README_DEPLOYMENT.md` content to Space README
6. Note: Ollama models need to use HuggingFace Inference API instead

### Option 2: Streamlit Cloud (Free)

1. Push repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect GitHub account
4. Select repository and main file

### Option 3: Railway (Free Tier)

1. Push to GitHub
2. Connect GitHub to Railway
3. Deploy from Dockerfile

### Option 4: Docker (Any Cloud)

```bash
# Build image
docker build -t rag-assistant .

# Run locally
docker run -p 8501:8501 rag-assistant

# Deploy to cloud (push image to registry)
```

### ⚠️ Important for Cloud Deployment

Free cloud platforms have limitations:
- **Ollama**: Requires local installation (not suitable for web)
- **Solution**: Use HuggingFace Inference API instead in cloud

Update `config.py` for cloud:
```python
LLM_TYPE = "huggingface"
```

## Evaluation

### Using RAGAS

```python
from evaluation.evaluator import get_evaluator

evaluator = get_evaluator()

# Evaluate single response
scores = evaluator.evaluate_response(
    query="What is it?",
    context="Retrieved context...",
    ground_truth="Expected answer...",
    generated_answer="Generated answer..."
)

# Batch evaluation
results = evaluator.evaluate_batch(
    queries=[...],
    contexts=[...],
    ground_truths=[...],
    generated_answers=[...]
)

# Get report
print(evaluator.create_report(results))
```

### Creating Evaluation Dataset

Create `evaluation_dataset.json`:

```json
[
    {
        "question": "What is machine learning?",
        "ground_truth": "Machine learning is...",
        "relevant_contexts": ["Context 1", "Context 2"]
    }
]
```

```bash
python scripts/evaluate.py --dataset evaluation_dataset.json
```

## Optimization Tips

### 1. Performance Optimization
- Use smaller models (`orca-mini`, `neural-chat`)
- Enable caching
- Use batch processing
- Optimize chunk sizes (CHUNK_SIZE = 256-512)

### 2. Quality Optimization
- Increase TOP_K_RETRIEVAL for better context
- Enable reranking (highest impact)
- Fine-tune vector/BM25 weights
- Use larger embedding model if resources allow

### 3. Memory Optimization
- Use quantized models
- Set smaller batch sizes
- Reduce max tokens
- Clear cache periodically

### 4. Cost Optimization (Cloud)
- Use HuggingFace Spaces (truly free)
- Batch queries offline
- Cache responses
- Use smaller models

## Troubleshooting

### Ollama Connection Error
```
Error: Cannot connect to Ollama server at http://localhost:11434
```

**Solution:**
```bash
# Make sure Ollama is running
ollama serve

# Check port is correct in config.py
# Try accessing http://localhost:11434/api/tags
```

### Out of Memory

**Solutions:**
- Use smaller model: `neural-chat` or `orca-mini`
- Reduce batch size in config
- Reduce embedding model size
- Increase chunk overlap, reduce chunk size

### Slow Performance

**Solutions:**
- Enable caching
- Use smaller embedding model
- Reduce TOP_K_RETRIEVAL
- Disable reranking if not needed
- Use quantized models with Ollama

### RAGAS Import Error
```
ModuleNotFoundError: No module named 'ragas'
```

**Solution:**
```bash
pip install ragas datasets
```

## Performance Benchmarks

Typical performance on consumer hardware (CPU):

| Operation | Time | Notes |
|-----------|------|-------|
| Embedding generation | ~0.5s/doc | Batch of 32 |
| Vector search | ~10-50ms | On ~1000 docs |
| BM25 search | ~5-20ms | On ~1000 docs |
| Hybrid search | ~20-70ms | Combined |
| Reranking | ~100-300ms | On top-k results |
| LLM generation | ~2-10s | Depends on model/length |
| **Full pipeline** | **~3-15s** | End-to-end |

## Architecture Diagram

```
User Query
    ↓
Embedding Generation
    ↓
┌─────────────────────┐
│  Hybrid Search      │
├──────────┬──────────┤
│ Vector   │  BM25    │
│ Search   │  Search  │
└────┬─────┴────┬────┘
     ↓          ↓
   Results (Top-K)
     ↓
Reranking (Cross-Encoder)
     ↓
Context Assembly
     ↓
Prompt Building
     ↓
LLM Generation
     ↓
Response Caching
     ↓
Final Answer
```

## Future Enhancements

- [ ] Multi-modal embeddings (text + images)
- [ ] Query expansion for better retrieval
- [ ] Semantic routing
- [ ] Fine-tuned reranker
- [ ] Agentic RAG with tool use
- [ ] Web search integration
- [ ] Multi-document summarization
- [ ] Streaming responses

## License

MIT

## Contributing

Contributions welcome! Areas for improvement:

1. Additional evaluation metrics
2. Support for more LLM backends
3. Advanced chunking strategies
4. Query optimization
5. Performance improvements

## Support

For issues or questions:

1. Check [troubleshooting section](#troubleshooting)
2. Review configuration options
3. Check logs in `logs/` directory
4. Enable debug logging: `LOG_LEVEL=DEBUG`

## Acknowledgments

- [Sentence-Transformers](https://www.sbert.net/)
- [Chroma](https://www.trychroma.com/)
- [Ollama](https://ollama.ai/)
- [Streamlit](https://streamlit.io/)
- [RAGAS](https://docs.ragas.io/)
- [LangChain](https://python.langchain.com/)

---

**Built with ❤️ using free and open-source tools**

Made for production use with zero cost. No credit cards. No API keys (for local LLMs).
