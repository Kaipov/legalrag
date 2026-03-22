# ARLC Legal RAG Pipeline

RAG pipeline for the ARLC legal challenge over DIFC laws, regulations, and case documents.

The repository contains:
- offline PDF extraction, OCR fallback, chunking, and indexing
- hybrid retrieval with BM25 plus Gemini dense embeddings
- optional reranking with either a local cross-encoder or Voyage API
- deterministic routing and resolvers for explicit compare and page-local questions
- answer generation, grounding extraction, telemetry, and submission packaging
- local validation and regression tooling for benchmark-gated runs

## Repository Layout

- `src/` - pipeline, retrieval, preprocessing, grounding, and resolver code
- `scripts/preprocess.py` - offline extraction, chunking, and index build steps
- `scripts/run.py` - answer questions, create `submission.json`, and optionally submit
- `scripts/evaluate.py` - validate answer format and telemetry
- `scripts/compare_answers.py` - compare answer drift between two submissions
- `scripts/compare_submissions.py` - compare grounding drift between two submissions
- `scripts/regression_report.py` - benchmark gate against `golden_submission.json`
- `docs/final-runbook.md` - operational runbook for public and final-stage runs
- `docs/competition-spec.md` - cleaned competition reference
- `starter_kit/` - organizer starter kit kept for API and schema reference

## Current Pipeline

At a high level the system works like this:

1. Extract page text from PDFs with `pdfplumber`.
2. Use OCR fallback for weak or empty pages.
3. Build structure-aware chunks and metadata sidecars.
4. Run hybrid retrieval over chunk and page indices.
5. Route explicit question patterns into deterministic resolvers when confidence is high.
6. Generate answers with an OpenAI-compatible chat model.
7. Derive grounding pages from cited source blocks plus answer-aware support selection.
8. Emit telemetry and package the run as `submission.json`.

Notable features in the current codebase:
- Gemini Embedding 2 Preview for dense retrieval
- optional Voyage `rerank-2.5` API reranking
- vision-model fallback for pages where plain text extraction is empty
- deterministic handling for several compare, title-page, first-page, article, and judge-timeline cases
- grounding policies tuned to preserve recall, which matters heavily in ARLC scoring

## Quick Start

### 1. Create the environment

```powershell
cd E:\legalrag
uv venv .venv
uv pip install -r requirements.txt
Copy-Item .env.example .env
```

### 2. Fill in the required keys

At minimum:
- `EVAL_API_KEY` for dataset download and platform submission
- `OPENAI_API_KEY` or `OPENROUTER_API_KEY` for generation
- `GEMINI_API_KEY` for chunking, embeddings, and semantic retrieval

Optional:
- `VOYAGE_API_KEY` if you enable hosted reranking

See [`.env.example`](.env.example) for the full list of knobs.

### 3. Prepare data

The pipeline expects:
- documents in [`data/documents`](data/documents)
- questions in [`data/questions.json`](data/questions.json)

You can either place them there manually or download them from the competition API before running preprocessing.

### 4. Build indices

Full preprocessing:

```powershell
uv run --python .\.venv\Scripts\python.exe python -m scripts.preprocess
```

Stepwise preprocessing:

```powershell
uv run --python .\.venv\Scripts\python.exe python -m scripts.preprocess --extract
uv run --python .\.venv\Scripts\python.exe python -m scripts.preprocess --chunk
uv run --python .\.venv\Scripts\python.exe python -m scripts.preprocess --index
```

When only retrieval or grounding code changes, you often do not need a full rebuild.
When extraction or chunk metadata logic changes, rebuild the affected artifacts before generating a new submission.

### 5. Generate a local submission

```powershell
uv run --python .\.venv\Scripts\python.exe python -m scripts.run --no-download --no-submit --questions data/questions.json
```

This writes [`submission.json`](submission.json).

### 6. Validate the result

```powershell
uv run --python .\.venv\Scripts\python.exe python -m scripts.evaluate --submission submission.json --strict
```

## Benchmark Gate

`golden_submission.json` is the public-set benchmark snapshot for this repository.

It was evaluated on the warmup platform on March 13, 2026 at:
- `deterministic = 1.000`
- `grounding = 0.954191`
- `total_score = 0.887146`

Before sending any new public-set candidate to the platform, compare it against that benchmark:

```powershell
uv run --python .\.venv\Scripts\python.exe python -m scripts.compare_answers
uv run --python .\.venv\Scripts\python.exe python -m scripts.compare_submissions
uv run --python .\.venv\Scripts\python.exe python -m scripts.regression_report --strict
```

The compare scripts default to:
- baseline: [`golden_submission.json`](golden_submission.json)
- candidate: [`submission.json`](submission.json)

For final-stage runs there is no golden dataset, so use:
- `scripts.evaluate` for structural validation
- silver-set or manual audit workflows for answer quality review
- careful tracking of `submission.*` artifacts outside git

## Submission Workflows

Standard end-to-end submit:

```powershell
uv run --python .\.venv\Scripts\python.exe python -m scripts.run --no-download --questions data/questions.json
```

That command:
- loads the local questions
- regenerates `submission.json`
- creates `code_archive.zip`
- submits both to the platform

If you already have a frozen `submission.json` and do not want another rerun before submission, use the direct-submit approach documented in [`docs/final-runbook.md`](docs/final-runbook.md).

## Environment Knobs You Will Actually Use

Most useful runtime toggles:
- `GENERATION_MODEL`
- `GENERATION_TOP_K`
- `ENABLE_RERANKER`
- `RERANKER_PROVIDER`
- `RERANKER_ENABLED_INTENTS`
- `RERANK_TOP_K`
- `RERANK_CANDIDATES`
- `EMBEDDING_BATCH_SIZE`
- `ENABLE_VISION_OCR_FALLBACK`
- `VISION_OCR_MODEL`
- `VISION_OCR_MAX_OUTPUT_TOKENS`

Example: try Voyage reranking only for article-reference questions

```powershell
$env:ENABLE_RERANKER="1"
$env:RERANKER_PROVIDER="voyage"
$env:RERANKER_ENABLED_INTENTS="article_ref"
uv run --python .\.venv\Scripts\python.exe python -m scripts.run --no-download --no-submit --questions data/questions.json
```

## Operational Notes

- [`data`](data) and [`index`](index) are intentionally gitignored.
- Private submission artifacts such as `submission.*.json`, UUID files, and platform status snapshots should stay out of commits.
- The built-in archive path in [`scripts/run.py`](scripts/run.py) excludes large runtime directories and the tracked public benchmark file.
- If you switch corpora, rebuild the relevant preprocessing artifacts before trusting retrieval results.
- If you change OCR, extraction, or page metadata logic, prefer a fresh preprocess before the next serious run.

## Related Docs

- [Final Runbook](docs/final-runbook.md)
- [Competition Spec](docs/competition-spec.md)
- [Starter Kit Flow](docs/starter-kit-flow.md)
- [Baseline v7.1 Historical Plan](docs/baseline-v7.1-implementation-plan.md)
