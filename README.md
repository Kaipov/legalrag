# ARLC Baseline v1

Baseline RAG pipeline for the ARLC legal challenge over DIFC documents.

## What is implemented

- PDF text extraction with `pdfplumber` and optional PaddleOCR fallback
- Structure-aware chunking for legal documents with a 512-token budget measured by the embedding model tokenizer
- Hybrid retrieval: BM25 + FAISS + reciprocal rank fusion
- Gemini Embedding 2 Preview dense retrieval via API
- Optional cross-encoder reranking with `bge-reranker-v2-m3` (disabled by default)
- GPT-4o answer generation with answer-type-specific prompts
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

Optional overrides:

- `EVAL_BASE_URL`
- `OPENAI_API_BASE`
- `OPENROUTER_API_BASE`
- `EMBEDDING_API_BASE`
- `EMBEDDING_MODEL`
- `EMBEDDING_BATCH_SIZE`
- `MAX_CHUNK_TOKENS`
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

`golden_submission.json` is the current v3-aligned public-set reference snapshot. The compare scripts default to `golden_submission.json` as baseline and `submission.json` as candidate.

6. Run lightweight tests:

```powershell
python -m pytest
```

## Notes

- `index/` and `data/` are intentionally gitignored.
- After changing embedding model or chunk budget, rebuild `chunks.jsonl`, `bm25.pkl`, `faiss.index`, and `faiss_ids.json`.
- The local evaluator does not score answer correctness because public gold answers are not available.
- The validation step is focused on submission shape, answer types, and telemetry quality.