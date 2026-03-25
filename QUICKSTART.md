# Quick Start Guide

Get your RAG system up and running in 10 minutes!

## Prerequisites

- Python 3.10+ (check with `python --version`)
- 4GB+ RAM recommended
- ~2GB disk space for models

## Step 1: Setup (2 minutes)

### 1.1 Install Dependencies

```bash
# Activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

### 1.2 Setup LLM (Choose One)

**Option A: Ollama (Recommended - Local)**

```bash
# 1. Download and install Ollama from https://ollama.ai

# 2. Start Ollama in a terminal
ollama serve

# 3. In another terminal, pull a model
ollama pull mistral
```

**Option B: HuggingFace (Cloud)**

```bash
# Get free API token from https://huggingface.co/
# Set environment variable:
export HUGGINGFACE_API_TOKEN="hf_xxxxx"

# Update config.py
# Change: LLM_TYPE = "huggingface"
```

## Step 2: Run the App (2 minutes)

### Option A: Streamlit Web UI (Easiest)

```bash
streamlit run streamlit_app.py
```

Then:
1. Open browser to http://localhost:8501
2. Upload your documents in the "Documents" tab
3. Start chatting in the "Chat" tab!

### Option B: Python Script

```bash
python main.py
```

This runs a demo with sample documents.

## Step 3: Add Your Documents (5 minutes)

### Method 1: Through Streamlit UI
1. Go to "Documents" tab
2. Upload .txt or .md files
3. Click "Process and Index Documents"

### Method 2: Command Line
```bash
python scripts/ingest_documents.py your_file1.txt your_file2.txt
```

### Method 3: Python Code
```python
from src.rag_pipeline import RAGPipeline

rag = RAGPipeline()
documents = ["Doc 1", "Doc 2", "Doc 3"]
rag.add_documents(documents)

result = rag.query("Your question here")
print(result["answer"])
```

## Step 4: Configure Settings (Optional)

Edit `config.py` or use Streamlit UI sidebar:

```python
# Search
TOP_K_RETRIEVAL = 10    # Retrieve more documents
VECTOR_WEIGHT = 0.7     # Favor vector search

# Reranking
RERANKING_ENABLED = True    # Better quality
TOP_K_RERANK = 4            # Top 4 from reranker

# LLM
TEMPERATURE = 0.7       # 0 = focused, 1+ = creative
MAX_NEW_TOKENS = 512    # Response length
```

## Troubleshooting

### Issue: "Cannot connect to Ollama"
```
Make sure Ollama is running:
ollama serve

And mistral model is downloaded:
ollama pull mistral
```

### Issue: Out of Memory
```
Use smaller model:
ollama pull neural-chat

Or disable reranking in config:
RERANKING_ENABLED = False
```

### Issue: Slow Performance
```
1. Enable caching (default: on)
2. Reduce TOP_K_RETRIEVAL to 5
3. Use smaller model
4. Reduce MAX_NEW_TOKENS
```

### Issue: RAGAS import error
```bash
pip install ragas datasets
```

## Common Commands

```bash
# Run Streamlit UI
streamlit run streamlit_app.py

# Run Python example
python main.py

# Ingest documents
python scripts/ingest_documents.py docs/*.txt

# Run tests
pytest tests/test_rag.py -v

# View logs
tail -f logs/app.log

# Clear cache
rm -rf cache/

# Restart Ollama
killall ollama
ollama serve
```

## Next Steps

1. **Read the docs**: See [README.md](README.md) for full documentation
2. **Evaluate system**: Use `evaluation/evaluator.py` to benchmark
3. **Deploy**: Use HuggingFace Spaces for free cloud deployment
4. **Customize**: Fine-tune settings in `config.py`

## Architecture at a Glance

```
Your Question
    ↓
Search (Vector + BM25)
    ↓
Rerank (Cross-Encoder)
    ↓
Answer (Local LLM)
```

## Free Resources Used

| Component | Tool | Cost |
|-----------|------|------|
| Embeddings | Sentence-Transformers | Free |
| Vector DB | Chroma | Free |
| Reranker | Cross-Encoder | Free |
| LLM (Local) | Ollama | Free |
| LLM (Cloud) | HuggingFace | Free tier |
| Web UI | Streamlit | Free |
| Deployment | HuggingFace Spaces | Free |

**Total Cost: $0** ✅

## Performance Expectations

| Task | Time |
|------|------|
| Embed documents | ~0.5s per 32 docs |
| Search | 20-70ms |
| Rerank | 100-300ms |
| Generate answer | 2-10s |
| **Total query** | **3-15s** |

## Tips for Best Results

1. **Quality Documents**: Better documents → better answers
2. **Good Questions**: Be specific in queries
3. **Enable Reranking**: Improves quality significantly
4. **Use Hybrid Search**: Combines strengths of both methods
5. **Enable Caching**: Faster repeated queries
6. **Experiment Settings**: Try different temperatures

## Next: Advanced Topics

See [README.md](README.md) for:
- Evaluation framework (RAGAS)
- Cloud deployment
- Custom configurations
- Performance optimization
- Integration with other systems

---

**Happy RAG-ing!** 🚀

For issues: Check logs in `logs/app.log`
