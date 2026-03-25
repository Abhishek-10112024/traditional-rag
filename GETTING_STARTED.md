# 🎯 Getting Started - Complete Walkthrough

Welcome! This guide walks you through everything from setup to deployment.

## What You'll Build

A **production-ready RAG (Retrieval-Augmented Generation) system** that:
- 🔍 Retrieves relevant documents using hybrid search
- 📊 Reranks results for better quality
- 🤖 Generates answers using free LLMs
- 💬 Provides a web interface (Streamlit)
- 📈 Evaluates system performance (RAGAS)
- 🚀 Deploys free to the cloud

**Cost: $0** (completely free!)

## Prerequisites Check

```bash
# Check Python version (should be 3.10+)
python --version

# Check Git is installed
git --version

# Create project folder
cd ~/Desktop
mkdir vectorless-rag
cd vectorless-rag
```

---

## Phase 1: Local Setup (10 minutes)

### Step 1.1: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

**You should see `(venv)` in your terminal prompt now.**

### Step 1.2: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs all packages needed (takes 3-5 minutes).

**What gets installed:**
- langchain & langchain-community (LLM framework)
- sentence-transformers (embeddings)
- chromadb (vector store)
- rank-bm25 (sparse search)
- streamlit (web UI)
- ragas (evaluation)
- Plus 15+ other packages

### Step 1.3: Setup LLM Backend

**Option A: Ollama (Recommended for Local)**

1. **Install Ollama**
   - Visit https://ollama.ai
   - Download and install
   - Make sure you have 5GB+ free disk space

2. **Start Ollama Server**
   ```bash
   ollama serve
   ```
   Should show: `Listening on 127.0.0.1:11434`

3. **In a NEW terminal**, download a model
   ```bash
   # Activate venv first (if needed)
   ollama pull mistral
   ```
   This takes 2-3 minutes (~5GB download).

   Other options:
   ```bash
   ollama pull neural-chat     # Faster, smaller
   ollama pull orca-mini       # Smallest
   ollama pull llama2          # Larger, slower
   ```

**Option B: HuggingFace (Cloud)**

1. Get free API token:
   - Go to https://huggingface.co/settings/tokens
   - Create new token (type: read)

2. Set environment variable:
   ```bash
   # Windows PowerShell
   $env:HUGGINGFACE_API_TOKEN = "hf_xxxxx"
   
   # macOS/Linux
   export HUGGINGFACE_API_TOKEN="hf_xxxxx"
   ```

3. Update `config.py`:
   ```python
   LLM_TYPE = "huggingface"
   ```

---

## Phase 2: First Run (5 minutes)

### Option A: Streamlit Web UI (Easiest)

```bash
# Make sure venv is activated
streamlit run streamlit_app.py
```

This opens http://localhost:8501 automatically.

**What you'll see:**
- 💬 Chat tab (empty, no documents yet)
- 📄 Documents tab
- ℹ️ About tab
- ⚙️ Settings sidebar

### Option B: Python Example

```bash
python main.py
```

This runs a demo with sample documents and asks 3 sample questions.

**Output will show:**
- Retrieved documents
- Scores for each document
- Final answer from LLM

---

## Phase 3: Adding Your Documents (10 minutes)

### Method 1: Streamlit UI (Easiest)

1. Open http://localhost:8501
2. Go to "Documents" tab
3. Upload .txt or .md files
4. Click "Process and Index Documents"
5. Go back to "Chat" tab
6. Start asking questions!

### Method 2: Command Line

```bash
python scripts/ingest_documents.py file1.txt file2.txt file3.md
```

### Method 3: Python Code

```python
from src.rag_pipeline import RAGPipeline

rag = RAGPipeline()

# Add documents
documents = [
    "Your document text here...",
    "Another document...",
]
rag.add_documents(documents)

# Ask a question
result = rag.query("What is this about?")
print(result["answer"])
```

---

## Phase 4: Using Your RAG System (Keep Running)

### Basic Workflow

```
1. Upload documents (one time)
   ↓
2. Ask a question
   ↓
3. System:
   - Searches documents
   - Reranks results
   - Generates answer
   ↓
4. Get answer + sources
```

### Example Questions to Try

If you uploaded ML documents:
- "What is machine learning?"
- "How do embeddings work?"
- "Explain gradient descent"

### Adjust Settings (Optional)

In Streamlit sidebar ⚙️:

```
Search Configuration:
├─ Enable Hybrid Search       (✓ on)
├─ Vector Weight               (0.7)
└─ Top-K Documents            (10)

Reranking:
├─ Enable Reranking           (✓ on)
├─ Top-K After Reranking      (4)

LLM Settings:
├─ Temperature                (0.7)  ← 0 = focused, 1+ = creative
├─ Top-P                      (0.9)
└─ Max Tokens                 (512)
```

---

## Phase 5: Understanding What Happened

When you ask a question, here's what happens:

```
Question: "What is machine learning?"
    ↓
1. EMBEDDING (src/embeddings.py)
   Convert question to vector
    ↓
2. RETRIEVAL (src/hybrid_search.py)
   ├─ Vector search: Find similar docs
   │  └─ vectorstore.py (Chroma)
   └─ BM25 search: Find keyword matches
      └─ bm25_search.py
   Result: Top 10 documents
    ↓
3. RERANKING (src/reranker.py)
   Score top 10 with cross-encoder
   Result: Top 4 best documents
    ↓
4. CONTEXT BUILDING
   Create prompt with top 4 docs
    ↓
5. GENERATION (src/llm.py)
   Send prompt to LLM
   Result: Answer
    ↓
6. CACHING
   Save result for future queries
    ↓
Answer: "Machine learning is..."
```

---

## Phase 6: Evaluating Your System (Optional)

### Create Evaluation Dataset

Create `eval_data.json`:
```json
[
    {
        "question": "What is machine learning?",
        "ground_truth": "Machine learning is a method of AI...",
        "relevant_contexts": ["..."]
    }
]
```

### Run Evaluation

```bash
python scripts/evaluate.py --dataset eval_data.json
```

This scores your system on:
- Faithfulness (matches ground truth)
- Answer relevancy (answers the question)
- Context precision (no irrelevant info)
- Context recall (includes all relevant info)

---

## Phase 7: Cloud Deployment (FREE!)

### Deploy to HuggingFace Spaces (Recommended)

#### 7.1: Create Space

1. Go to https://huggingface.co/spaces
2. Click "Create new Space"
3. Name: `my-rag-assistant`
4. SDK: Streamlit
5. Space: Public
6. Click "Create Space"

#### 7.2: Upload Files

```bash
# Clone the space
git clone https://huggingface.co/spaces/YOUR_USERNAME/my-rag-assistant
cd my-rag-assistant

# Copy your files
cp -r path/to/vectorless-rag/* .

# Update config for cloud
# In config.py, change:
# LLM_TYPE = "huggingface"

# Push to HF
git add .
git commit -m "RAG system setup"
git push
```

#### 7.3: Add API Token

1. In Space settings → Secrets
2. Add: `HUGGINGFACE_API_TOKEN` = (your token)

#### 7.4: Done!

HuggingFace auto-deploys. 

Your RAG system is live at:
```
https://huggingface.co/spaces/YOUR_USERNAME/my-rag-assistant
```

Share this URL! Anyone can use it.

---

## Troubleshooting

### Ollama Connection Error

```
Error: Cannot connect to Ollama server
```

**Solution:**
```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Download model
ollama pull mistral

# Terminal 3: Run app
streamlit run streamlit_app.py
```

### Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solution:**
```python
# In config.py:
OLLAMA_MODEL = "orca-mini"  # Smaller model
TOP_K_RETRIEVAL = 5         # Fewer docs
BATCH_SIZE = 8              # Smaller batches
RERANKING_ENABLED = False   # Disable expensive re-ranking
```

### Slow Performance

```
Takes too long to generate answer
```

**Solutions:**
1. Enable caching (default on)
2. Use smaller model
3. Reduce TOP_K_RETRIEVAL
4. Disable reranking

### Models Not Downloading

```
Downloading model... (stuck)
```

**Solutions:**
- Check internet connection
- Check disk space (need 5-10GB)
- Try different model: `ollama pull neural-chat`

---

## What's Next?

### Short Term
- [ ] Add your documents
- [ ] Test with various questions
- [ ] Adjust settings for your use case
- [ ] Try different models

### Medium Term
- [ ] Deploy to HuggingFace Spaces
- [ ] Evaluate system performance
- [ ] Gather user feedback
- [ ] Optimize based on results

### Long Term
- [ ] Fine-tune reranker
- [ ] Add custom chunking
- [ ] Integrate with other data sources
- [ ] Add user analytics

---

## Documentation Map

```
START HERE:
  ↓
QUICKSTART.md ────→ 10-min setup
  ↓
README.md ────────→ Complete documentation
  ↓
ARCHITECTURE.md ──→ How it all works
  ↓
DEPLOYMENT_GUIDE.md ──→ Cloud deployment

FOR SPECIFIC TOPICS:
├─ config.py ──────────→ All configuration options
├─ src/ ───────────────→ Core code implementation
├─ evaluation/ ────────→ Evaluation framework
└─ scripts/ ───────────→ Utility scripts
```

---

## Common Questions

**Q: Does this work offline?**
A: Yes! Using Ollama locally = completely offline (after downloading models)

**Q: How much storage needed?**
A: ~2GB for models + 100MB per 1000 documents

**Q: Can I use my own LLM?**
A: Yes! Edit `src/llm.py` to add custom LLM class

**Q: How accurate is the RAG?**
A: Depends on your documents + settings. Use RAGAS evaluation to measure.

**Q: Can I commercial use?**
A: Yes! All components are open source (MIT license)

**Q: How do I add more documents later?**
A: Just upload in Streamlit UI or use ingest script. Old documents are kept.

---

## Performance Expectations

**On a typical consumer machine (CPU, 8GB RAM):**

| Operation | Time |
|-----------|------|
| First query | 30-60s (model loading) |
| Subsequent queries | 3-15s |
| With caching | <1s (if repeated) |
| Index 1000 documents | 2-5 minutes |

**On cloud (HuggingFace Spaces):**

| Operation | Time |
|-----------|------|
| First call | 30-60s (cold start) |
| Warm queries | 5-20s |
| Caching | <1s |

---

## Final Checklist

- [ ] Python 3.10+ installed
- [ ] Virtual environment created & activated
- [ ] Dependencies installed
- [ ] LLM backend running (Ollama or HF token set)
- [ ] Streamlit app starts without errors
- [ ] Can upload documents
- [ ] Can ask questions and get answers
- [ ] Ready to deploy!

---

## Success! 🎉

You now have a production-ready RAG system!

**What makes this special:**
✅ Completely free (no credit card ever)
✅ Production quality (error handling, logging, caching)
✅ Full-featured (hybrid search, reranking, evaluation)
✅ Easy to use (Streamlit UI)
✅ Easy to deploy (HuggingFace Spaces)

**Next step:** Start adding your documents and chatting!

For detailed documentation, see **README.md**

Good luck! 🚀
