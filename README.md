# ARLC Baseline v1

Baseline RAG pipeline for the ARLC legal challenge over DIFC documents.

## What is implemented

- PDF text extraction with `pdfplumber` and optional PaddleOCR fallback
- Structure-aware chunking for legal documents with a 512-token budget measured by the embedding model tokenizer
- Hybrid retrieval: BM25 + FAISS + reciprocal rank fusion
- Gemini Embedding 2 Preview dense retrieval via API
- Optional reranking via local `bge-reranker-v2-m3` or Voyage `rerank-2.5` API (disabled by default)
- OpenAI-compatible answer generation with answer-type-specific prompts
- Null detection, grounding extraction, and submission telemetry
- Local validation for answer format and telemetry completeness

## Repository layout

- `src/` - pipeline code
- `scripts/preprocess.py` - offline extraction, chunking, and indexing
- `scripts/run.py` - answer questions and build `submission.json`
- `scripts/evaluate.py` - local validation of answer format and telemetry
- `starter_kit/` - organizer starter kit kept as reference

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Copy the env template and fill in your keys:

```powershell
Copy-Item .env.example .env
```

## Required environment variables

- `EVAL_API_KEY` for downloading questions/documents and submitting
- `OPENAI_API_KEY` or `OPENROUTER_API_KEY` for generation
- `GEMINI_API_KEY` for embedding-token counting, chunking, indexing, and semantic retrieval
- `VOYAGE_API_KEY` if you enable the hosted Voyage reranker

Optional overrides:

- `EVAL_BASE_URL`
- `OPENAI_API_BASE`
- `OPENROUTER_API_BASE`
- `EMBEDDING_API_BASE`
- `EMBEDDING_MODEL`
- `EMBEDDING_BATCH_SIZE`
- `MAX_CHUNK_TOKENS`
- `ENABLE_RERANKER`
- `RERANKER_PROVIDER`
- `RERANKER_ENABLED_INTENTS`
- `RERANK_TOP_K`
- `RERANK_CANDIDATES`
- `RERANKER_TIMEOUT_SECONDS`
- `VOYAGE_RERANKER_MODEL`
- `GENERATION_MODEL`
- `SUBMISSION_PATH`
- `CODE_ARCHIVE_PATH`
- `DOCS_DIR`

## Optional OCR dependencies

OCR fallback is optional. If you want PaddleOCR-VL support for scanned pages, install:

```powershell
pip install paddlepaddle-gpu==3.2.1
pip install -U "paddleocr[doc-parser]"
```

## Typical workflow

1. Put documents into `data/documents/` and questions into `data/questions.json`, or download them through the run script.
2. Build indices:

```powershell
python -m scripts.preprocess
```

3. Run the pipeline on questions:

```powershell
python -m scripts.run --no-submit
```

To try `gpt-5.4-mini` without changing the repository default, set `GENERATION_MODEL=gpt-5.4-mini` in `.env` first, or override it for one run:

```powershell
$env:GENERATION_MODEL="gpt-5.4-mini"
python -m scripts.run --no-submit
```

To try the hosted Voyage reranker only for `article_ref` questions:

```powershell
$env:ENABLE_RERANKER="1"
$env:RERANKER_PROVIDER="voyage"
$env:RERANKER_ENABLED_INTENTS="article_ref"
$env:RERANK_TOP_K="10"
$env:RERANK_CANDIDATES="10"
python -m scripts.run --no-submit
```

4. Validate submission format and telemetry:

```powershell
python -m scripts.evaluate --submission submission.json --strict
```

5. Compare the fresh run to the committed golden baseline:

```powershell
python -m scripts.compare_answers
python -m scripts.compare_submissions
python -m scripts.regression_report --strict
```

`golden_submission.json` is the current public-set benchmark snapshot. It has already been validated on the warmup platform: `deterministic=1.000`, `grounding=0.954191`, `assistant=0.68`, `telemetry=0.996`, `ttft_ms=753`, `ttft_multiplier=1.0326`, `total_score=0.887146` on March 13, 2026.

Treat `golden_submission.json` as the mandatory pre-submit gate. The compare scripts default to `golden_submission.json` as baseline and `submission.json` as candidate, and a candidate should not be sent to the platform until this regression check has been reviewed.

Pre-submit rule:
- run `python -m scripts.compare_answers`
- run `python -m scripts.compare_submissions`
- run `python -m scripts.regression_report --strict`
- review any deterministic regressions, grounding regressions, and free-text drift against `golden_submission.json` before submitting

6. Run lightweight tests:

```powershell
python -m pytest
```

## Notes

- `index/` and `data/` are intentionally gitignored.
- After changing embedding model or chunk budget, rebuild `chunks.jsonl`, `bm25.pkl`, `faiss.index`, and `faiss_ids.json`.
- The local evaluator does not score answer correctness because public gold answers are not available.
- The validation step is focused on submission shape, answer types, and telemetry quality.
