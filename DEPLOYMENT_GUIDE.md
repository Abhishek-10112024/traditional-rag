# Deployment Guide - Free Cloud Platforms

Deploy your RAG system to the cloud **completely free** without credit/debit cards!

## Comparison of Free Platforms

| Platform | Cost | Setup | Limitations | Best For |
|----------|------|-------|-------------|----------|
| **HuggingFace Spaces** | Free | Easiest | CPU only, 30GB storage | Production ✅ |
| **Streamlit Cloud** | Free | Easy | Limited compute | Quick demos |
| **Railway** | Free trial | Medium | Limited hours | Testing |
| **Render** | Free tier | Medium | Spins down inactivity | Small projects |
| **Replit** | Free | Easy | Limited RAM | Learning |

## 🏆 Recommended: HuggingFace Spaces

### Why HuggingFace Spaces?
- ✅ Completely free (no credit card)
- ✅ Persistent storage
- ✅ Easy integration with HuggingFace models
- ✅ Pre-installed transformers/PyTorch
- ✅ Docker support
- ✅ Good for production use

### Step-by-Step Setup

#### 1. Create HuggingFace Account
- Go to [huggingface.co](https://huggingface.co)
- Sign up (FREE - no credit card needed!)
- Verify email

#### 2. Create a New Space
1. Click on your profile → "New Space"
2. Choose name: `rag-assistant` (or your choice)
3. Select SDK: **Streamlit**
4. Space type: **Public**
5. Click "Create Space"

#### 3. Upload Project Files
Option A: Git (Recommended)
```bash
# Clone the space
git clone https://huggingface.co/spaces/YOUR_USERNAME/rag-assistant
cd rag-assistant

# Copy your files
cp -r ../vectorless-rag/* .

# Push to HF
git add .
git commit -m "Initial RAG setup"
git push
```

Option B: Web Upload
1. Go to your Space page
2. Click "Files" → "Upload files"
3. Upload all files from `vectorless-rag/`

#### 4. Configure for Cloud

**Important**: Ollama doesn't work in cloud. Use HuggingFace Inference API instead.

Edit `config.py`:
```python
# Change from:
LLM_TYPE = "ollama"

# To:
LLM_TYPE = "huggingface"
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
```

#### 5. Add HuggingFace API Token
1. Get token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   - Click "New token"
   - Name: `rag-assistant`
   - Type: "read" (fine for inference)
2. Copy token
3. In your HF Space → Settings → Secrets
4. Add secret: `HUGGINGFACE_API_TOKEN` = (paste token)

#### 6. Create/Update Files

**Create `README.md` in Space:**
```markdown
---
title: RAG Assistant
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.28.0
app_file: streamlit_app.py
models:
  - all-MiniLM-L6-v2
  - cross-encoder/ms-marco-MiniLM-L-6-v2
  - meta-llama/Llama-2-7b-chat-hf
---

# RAG Assistant

A free, production-ready Retrieval-Augmented Generation system.

## Features
- Hybrid search (BM25 + Vector)
- Reranking with cross-encoders
- Free cloud LLMs
- Streamlit UI
- Built-in evaluation

## Usage
1. Upload documents
2. Ask questions
3. Get AI-powered answers

### Tech Stack
- Embeddings: Sentence-Transformers
- Vector DB: Chroma
- Reranker: Cross-Encoder
- LLM: HuggingFace Inference
- UI: Streamlit
```

#### 7. Deploy!
- HF Spaces auto-deploys when you push code
- Check "App" tab to see status
- First load ~2-3 minutes (model download)
- Subsequent loads fast (~10-30s)

#### 8. Share Your Space
- URL: `huggingface.co/spaces/YOUR_USERNAME/rag-assistant`
- Anyone can use without signing in!

---

## Alternative: Streamlit Cloud

### Setup (Easiest)

#### 1. GitHub Setup
```bash
# Create GitHub repo
git init
git add .
git commit -m "Initial commit"
git push origin main
```

#### 2. Connect to Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "Deploy an app"
3. Select GitHub repo
4. Branch: `main`
5. File: `streamlit_app.py`
6. Click "Deploy"

#### 3. Add Secrets
1. Settings → Secrets
2. Add: `HUGGINGFACE_API_TOKEN`

**Pros:**
- Easiest setup
- GitHub integration
- Live on GitHub changes

**Cons:**
- Limited to 1GB storage
- Slow cold starts
- Shares resources

---

## Docker Deployment (Any Cloud)

Use if you want more control or custom hardware.

### Build Docker Image
```bash
docker build -t rag-assistant .
```

### Test Locally
```bash
docker run -p 8501:8501 rag-assistant
```

Then use on:
- **Railway**: Push Docker image
- **render.com**: Connect GitHub + Docker
- **Fly.io**: Free tier available
- **Heroku**: Not free anymore

---

## Configuration for Cloud

### Update `config.py`
```python
# Cloud LLM (HuggingFace)
LLM_TYPE = "huggingface"

# Smaller models for faster inference
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Already optimized
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"  # Smaller

# Optimize for cloud
TOP_K_RETRIEVAL = 5      # Fewer docs = faster
BATCH_SIZE = 16          # Smaller batches
CACHE_ENABLED = True     # Essential for cloud
CACHE_TTL = 7200        # 2 hours
```

### Update `requirements.txt` for Cloud
```
langchain==0.1.9
# ... keep all existing packages ...

# Cloud-specific
gunicorn==21.2.0  # For Render/Railway
```

---

## Free LLM Options for Cloud

### Option 1: HuggingFace Inference (Recommended)
```python
LLM_TYPE = "huggingface"
# Free tier: Rate limited but works
# Get token: huggingface.co/settings/tokens
```

### Option 2: Replicate API
```python
# Alternative (if HF rate limits hit)
# Model: meta-llama/Llama-2-7b-chat:9b51bbf61c6f694519f42b4a37a45bcd9aaa193b
```

### Option 3: Together.ai (Free Credits)
```python
# 10B free tokens/month
# https://www.together.ai
```

### Option 4: Groq (Very Fast, Free)
```python
# Free tier available
# https://console.groq.com
```

---

## Storage Considerations

### Chroma Vector Store
- **Local Option**: Works on HF Spaces (persistent `/tmp/`)
- **Cloud Option**: Use Chroma Cloud (paid) or save to HF Hub

### Environment Files
- Don't commit `.env` files
- Use Secrets in platform settings
- Config.py handles defaults

### Models
- Downloaded automatically
- Cached after first download
- ~1GB for embeddings + reranker

---

## Monitoring & Troubleshooting

### Check Logs
- HuggingFace Spaces: "App" tab → View logs
- Streamlit Cloud: "Manage app" → View logs

### Common Cloud Issues

**Issue: "Out of memory"**
```python
# In config.py:
TOP_K_RETRIEVAL = 3         # Fewer documents
BATCH_SIZE = 8              # Smaller batches
RERANKING_ENABLED = False   # Disable expensive reranking
```

**Issue: "Not finding models"**
```python
# Models auto-download on first request
# This takes time - be patient!
# Check logs for download progress
```

**Issue: "Timeout"**
```python
# Increase timeout or use smaller models
MAX_NEW_TOKENS = 256  # Shorter responses
OLLAMA_MODEL = "neural-chat"  # Faster model
```

**Issue: "Token rate limit"**
```python
# HuggingFace free tier is rate-limited
# Use caching (default: enabled)
# Or add delays between requests
```

---

## Cost Analysis

### HuggingFace Spaces
| Item | Cost |
|------|------|
| Compute | Free |
| Storage | Free (30GB) |
| API token | Free |
| Models | Free |
| **Total** | **$0** |

### Streamlit Cloud
| Item | Cost |
|------|------|
| Hosting | Free |
| Storage | 1GB free |
| Apps | Unlimited free apps |
| **Total** | **$0** |

### vs Paid Alternatives
| Service | Free Tier | Cost/Month if Paid |
|---------|-----------|-------------------|
| HuggingFace Spaces | ✅ Fully free | N/A (free) |
| Streamlit Cloud | ✅ Fully free | N/A (free) |
| Vercel | Limited | $20+ |
| AWS | 12-month free | $50+ |
| GCP | Limited free | $50+ |

---

## Production Checklist

- [ ] Update `config.py` for cloud (HF API)
- [ ] Add API tokens to Secrets
- [ ] Test locally first
- [ ] Upload to HuggingFace Spaces or Streamlit Cloud
- [ ] Test on cloud (first load takes time)
- [ ] Monitor logs for errors
- [ ] Set up Google Analytics (optional)
- [ ] Add rate limiting if needed
- [ ] Document usage limits

---

## Getting Help

1. **Check logs** in platform UI
2. **Review config.py** for incorrect settings
3. **Test locally first** with same config
4. **Check HuggingFace docs** for model availability
5. **Monitor API usage** to avoid limits

---

## Next: Custom Domain & Scaling

- HuggingFace Spaces: Use Cloudflare for custom domain
- Streamlit Cloud: Works with custom domains
- Railway/Render: Native custom domain support

---

**You're now deployed! 🚀**

Share your space URL and let people use your RAG system!

Example: `huggingface.co/spaces/YOUR_NAME/rag-assistant`
