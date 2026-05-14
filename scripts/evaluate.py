"""End-to-end RAG evaluation script.

Usage
-----
    python scripts/evaluate.py \\
        --documents data/report.pdf \\
        --dataset   data/eval_sample.json \\
        --output    cache/eval_results.json

The script:
  1. Indexes the supplied documents into an isolated 'eval_collection'
     (does NOT affect the Chroma data used by the Streamlit app).
  2. Runs every question through RAGPipeline.query().
  3. Scores results with MetricsCalculator — no RAGAS / OpenAI needed.
  4. Prints a human-readable report and writes JSON to --output.

Dataset JSON format
-------------------
[
  {
    "question": "What was the total FSU budget in 2023?",
    "ground_truth": "The total FSU budget in 2023 was 500 crore.",
    "relevant_contexts": ["...optional expected chunk text..."]
  }
]
'relevant_contexts' is optional; omitting it skips precision/recall/MRR.
"""

import argparse
import json
import sys
import io
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# pyrefly: ignore [missing-import]
from loguru import logger
from evaluation.metrics import MetricsCalculator
from src.rag_pipeline import RAGPipeline
from config import settings


# ── Helpers: document loading ─────────────────────────────────────────────────

def _word_chunk(text: str, chunk_size: int = 900, overlap: int = 150):
    """Simple word-overlap chunker used during evaluation ingestion."""
    words = text.split()
    if not words:
        return []
    step = max(1, chunk_size - overlap)
    chunks = []
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def _load_pdf(path: str):
    """Extract page text from a PDF and return (texts, metadatas)."""
    texts, metas = [], []
    file_bytes = Path(path).read_bytes()
    filename = Path(path).name

    # Try pdfplumber first (better layout)
    try:
        # pyrefly: ignore [missing-import]
        import pdfplumber
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text() or ""
                if not page_text.strip():
                    continue
                for chunk in _word_chunk(f"[Page {page_num}] {page_text}"):
                    texts.append(chunk)
                    metas.append({"source": filename, "page": page_num})
        return texts, metas
    except Exception as e:
        logger.warning(f"pdfplumber failed ({e}), falling back to pypdf")

    # Fallback: pypdf
    try:
        # pyrefly: ignore [missing-import]
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(file_bytes))
        for page_num, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            if not page_text.strip():
                continue
            for chunk in _word_chunk(f"[Page {page_num}] {page_text}"):
                texts.append(chunk)
                metas.append({"source": filename, "page": page_num})
        return texts, metas
    except Exception as e:
        raise RuntimeError(f"Cannot read PDF '{path}': {e}")


def _load_text(path: str):
    """Load a plain .txt / .md file."""
    filename = Path(path).name
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    chunks = _word_chunk(text)
    return chunks, [{"source": filename, "page": 0}] * len(chunks)


def load_document(path: str):
    """Dispatch to the right loader based on file extension."""
    suffix = Path(path).suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path)
    elif suffix in (".txt", ".md"):
        return _load_text(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Use .pdf, .txt, or .md")


# ── Core evaluation runner ────────────────────────────────────────────────────

def run_evaluation(
    documents: list,
    dataset: list,
    top_k_retrieval: int = None,
    top_k_rerank: int = None,
    temperature: float = None,
) -> list:
    """
    Index documents, run all queries, score each with MetricsCalculator.

    Returns a list of per-question result dicts.
    """
    calc = MetricsCalculator()

    # ── Step 1: redirect to a separate eval collection ────────────────────────
    # This keeps the app's Chroma index untouched.
    original_collection = settings.COLLECTION_NAME
    settings.COLLECTION_NAME = "eval_collection"

    logger.info("Initialising RAG pipeline (eval mode)…")
    rag = RAGPipeline(use_reranking=True, use_hybrid_search=True)
    rag.reset_knowledge_base(delete_storage=True)   # clean slate for eval

    # ── Step 2: index all supplied documents ──────────────────────────────────
    all_texts, all_metas = [], []
    for doc_path in documents:
        logger.info(f"Loading: {doc_path}")
        texts, metas = load_document(doc_path)
        all_texts.extend(texts)
        all_metas.extend(metas)
        logger.info(f"  → {len(texts)} chunks from {Path(doc_path).name}")

    logger.info(f"Indexing {len(all_texts)} chunks total…")
    rag.add_documents(all_texts, all_metas)
    logger.info("Indexing complete. Starting evaluation queries…\n")

    # ── Step 3: run each question ─────────────────────────────────────────────
    results = []
    for i, sample in enumerate(dataset, start=1):
        question     = sample["question"]
        ground_truth = sample.get("ground_truth", "")
        expected_ctx = sample.get("relevant_contexts", [])

        logger.info(f"[{i}/{len(dataset)}] {question[:80]}")

        t0 = time.time()
        rag_result = rag.query(
            question,
            top_k_retrieval=top_k_retrieval or settings.TOP_K_RETRIEVAL,
            top_k_rerank=top_k_rerank    or settings.TOP_K_RERANK,
            temperature=temperature      or settings.TEMPERATURE,
            use_cache=False,             # always fresh during eval
        )
        latency = time.time() - t0

        generated      = rag_result["answer"]
        retrieved_texts = [d["text"] for d in rag_result["retrieved_docs"]]

        # Generation quality
        gen_scores = calc.calculate_generation_metrics(ground_truth, generated)

        # Retrieval quality (only when expected contexts are provided)
        ret_scores = (
            calc.calculate_retrieval_metrics(retrieved_texts, expected_ctx)
            if expected_ctx else {}
        )

        # Latency
        latency_score = calc.calculate_latency_score(latency)

        results.append({
            "question":        question,
            "ground_truth":    ground_truth,
            "generated_answer": generated,
            "latency_seconds": round(latency, 2),
            "retrieved_pages": [
                d.get("metadata", {}).get("page") for d in rag_result["retrieved_docs"]
            ],
            "scores": {
                "answer_similarity": round(gen_scores["similarity"], 3),
                "latency_score":     round(latency_score, 3),
                **{k: round(v, 3) for k, v in ret_scores.items()},
            },
        })

    # Restore original collection name
    settings.COLLECTION_NAME = original_collection
    return results


# ── Report builder ────────────────────────────────────────────────────────────

def build_report(results: list) -> str:
    """Generate a human-readable evaluation summary."""
    n = len(results)
    if n == 0:
        return "No results to report."

    def avg(key):
        vals = [r["scores"][key] for r in results if key in r["scores"]]
        return sum(vals) / len(vals) if vals else None

    total_latency = sum(r["latency_seconds"] for r in results)

    lines = [
        "",
        "=" * 62,
        "  RAG System Evaluation Report",
        "=" * 62,
        f"  Questions evaluated : {n}",
        f"  Total time          : {total_latency:.1f}s",
        "",
        "  Generation Quality",
        f"    Answer Similarity  : {avg('answer_similarity'):.3f}  (0–1, vs ground truth)",
        "",
        "  Latency",
        f"    Avg latency        : {total_latency / n:.1f}s per query",
        f"    Latency score      : {avg('latency_score'):.3f}  (1.0 = under 3s threshold)",
    ]

    if avg("precision") is not None:
        lines += [
            "",
            "  Retrieval Quality  (based on relevant_contexts in dataset)",
            f"    Precision          : {avg('precision'):.3f}",
            f"    Recall             : {avg('recall'):.3f}",
            f"    MRR                : {avg('mrr'):.3f}",
        ]

    lines += [
        "",
        "  Per-question breakdown",
        "-" * 62,
    ]
    for i, r in enumerate(results, 1):
        sim = r["scores"].get("answer_similarity", 0)
        lat = r["latency_seconds"]
        q   = r["question"][:55] + ("…" if len(r["question"]) > 55 else "")
        ans = r["generated_answer"][:110] + "…"
        lines += [
            f"  [{i:2d}] {q}",
            f"       similarity={sim:.2f}  latency={lat:.1f}s",
            f"       → {ans}",
            "",
        ]

    lines += ["=" * 62, ""]
    return "\n".join(lines)


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="End-to-end RAG evaluation (no RAGAS / OpenAI required)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Basic run (similarity + latency only)
  python scripts/evaluate.py \\
      --documents data/report.pdf \\
      --dataset   data/eval_sample.json

  # Full run with retrieval metrics + custom output
  python scripts/evaluate.py \\
      --documents data/report.pdf data/annex.pdf \\
      --dataset   data/eval_sample.json \\
      --output    cache/eval_results.json \\
      --top-k-retrieval 20 --top-k-rerank 8
        """,
    )
    parser.add_argument("--documents", nargs="+", required=True,
                        help="PDF / .txt / .md files to index for evaluation")
    parser.add_argument("--dataset",   required=True,
                        help="Evaluation dataset JSON path")
    parser.add_argument("--output",    default="cache/eval_results.json",
                        help="Output JSON path (default: cache/eval_results.json)")
    parser.add_argument("--top-k-retrieval", type=int, default=None,
                        help=f"Retrieval top-k (default: {settings.TOP_K_RETRIEVAL})")
    parser.add_argument("--top-k-rerank", type=int, default=None,
                        help=f"Rerank top-k (default: {settings.TOP_K_RERANK})")
    parser.add_argument("--temperature", type=float, default=None,
                        help=f"LLM temperature (default: {settings.TEMPERATURE})")
    args = parser.parse_args()

    # Validate inputs
    for doc in args.documents:
        if not Path(doc).exists():
            logger.error(f"Document not found: {doc}")
            sys.exit(1)
    if not Path(args.dataset).exists():
        logger.error(f"Dataset not found: {args.dataset}")
        sys.exit(1)

    # Load dataset
    logger.info(f"Loading dataset: {args.dataset}")
    with open(args.dataset, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    logger.info(f"Loaded {len(dataset)} evaluation samples")

    # Run evaluation
    results = run_evaluation(
        documents=args.documents,
        dataset=dataset,
        top_k_retrieval=args.top_k_retrieval,
        top_k_rerank=args.top_k_rerank,
        temperature=args.temperature,
    )

    # Print report
    print(build_report(results))

    # Save JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(results)
    summary = {
        "n_questions":          n,
        "avg_answer_similarity": round(
            sum(r["scores"]["answer_similarity"] for r in results) / n, 3
        ),
        "avg_latency_seconds":  round(
            sum(r["latency_seconds"] for r in results) / n, 1
        ),
    }
    if any("precision" in r["scores"] for r in results):
        summary["avg_precision"] = round(
            sum(r["scores"].get("precision", 0) for r in results) / n, 3
        )
        summary["avg_recall"] = round(
            sum(r["scores"].get("recall", 0) for r in results) / n, 3
        )
        summary["avg_mrr"] = round(
            sum(r["scores"].get("mrr", 0) for r in results) / n, 3
        )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved → {output_path}")


if __name__ == "__main__":
    main()
