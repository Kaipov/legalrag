# ARLC Competition — Full Specification

Sources: introductory seminar, Discord Q&A, official starter kit (March 2026).

---

## 1. Overview

ARLC is an engineering competition for building production-grade legal AI systems.
Part of Machines Can See / Dubai AI conference. Award ceremony online in April.
Prize pool: **$32,000**. Finalists get trip to Dubai. **245+ teams registered.**

Key evaluation dimensions:
- Answer quality (accuracy)
- Grounding quality (source citations) — **strongest signal, acts as multiplier**
- Time-to-first-token (TTFT)

### Repository Snapshot (March 21, 2026)
- Branch snapshot: `codex/baseline-v8`
- Generation model under test: `gpt-5.4-mini`
- Benchmark basis: local regression comparison against `golden_submission.json`
- Structured proxy: `score=1.0000`, `mismatches=0`
- Grounding proxy: `macro_fbeta=0.8933`, `macro_jaccard=0.7825`, `exact_page_set_matches=61`, `answer_exact_grounding_bad=2`
- Free-text proxy: `strong=19`, `mid=5`, `weak=2`, `null_risk=0`
- Regression gate: `PASS` via `python -m scripts.regression_report --strict`
- This snapshot is **not** a platform evaluation. The warmup platform score for `golden_submission.json` remains `total_score=0.887146` on March 13, 2026.

---

## 2. Corpus & Dataset

### Corpus
- Real regulatory documents: DIFC (Dubai International Financial Centre) rules and laws
- All in English
- **Heterogeneous PDFs**: digitally-born + scanned documents (may require OCR)

### Phase-specific corpora (IMPORTANT)
- **Warm-up phase**: ~30 documents, 100 questions
- **Final phase**: ~300 documents, 900 questions
- Corpora may partially overlap, but **must be indexed independently**
- Answer each phase's questions using **only its own corpus**

### Dataset Construction (5 steps)
1. Collected DIFC document corpus
2. Generated questions via LLM
3. Filtered by: relevance, grounding correctness, semantic relevance
4. Professional lawyers manually reviewed every Q&A pair
5. Ran baseline system to ensure benchmark differentiates quality

---

## 3. Question Types

### By Answer Format (visible in JSON as `answer_type`)
| Type | Format | Scoring |
|------|--------|---------|
| `number` | Integer or float (both valid) | ±1% tolerance |
| `boolean` | JSON `true`/`false` | Exact match |
| `name` | Single string (no aliases) | Normalized exact match |
| `names` | JSON array of strings (no aliases) | Jaccard index |
| `date` | ISO `YYYY-MM-DD` | Exact match |
| `free_text` | String ≤280 chars, 1-3 paragraphs | LLM judge (5 criteria) |

- 70% structured (deterministic), 30% free-text
- **Any type can have `null` as correct answer** (= not in corpus)
- For unanswerable free_text: return natural-language statement + empty sources

### By Question Category (hidden from participants)
1. **Single-document** — answer in one document
2. **Clause analysis** — find and interpret a specific clause (often free-text)
3. **Multi-document** — compare/synthesize across documents (often free-text)
4. **Negative questions** — no answer in corpus → correct answer is `null`, sources = []
5. **Adversarial questions** — designed to trigger hallucination from pretrained knowledge
6. **Uncertainty questions** — high interpretation ambiguity, need appropriate hedging

---

## 4. Scoring Details

### 4.1 Final Formula (OFFICIAL from starter kit)
```
Total = (0.7 × S_det + 0.3 × S_asst) × G × T × F
```
Where:
- `S_det` = deterministic score (structured answers accuracy, 0–1)
- `S_asst` = assistant score (free-text LLM judge mean, 0–1)
- `G` = grounding score (F-beta retrieval quality, 0–1)
- `T` = telemetry factor (0.9 if malformed, 1.0 if valid)
- `F` = TTFT factor (0.85–1.05)

**Platform reports `total_score` as 0–1 value.**

### 4.2 Structured Answers (S_det)
- `name`, `boolean`, `date`: binary (0 or 1), normalized exact match
- `names`: Jaccard index (|intersection| / |union|)
- `number`: 1 if within ±1% of gold, else 0
- `null`: both null → 1; only one null → 0

### 4.3 Free-text Answers / LLM-as-a-Judge (S_asst)
5 criteria, each 0 or 1:
1. **Correctness** — key info present, no factual errors?
2. **Completeness** — all aspects addressed, key points covered?
3. **Grounding** — every statement supported by retrieved context?
4. **Confidence calibration** — appropriate uncertainty expressed?
5. **Clarity & relevance** — clear, concise, directly addresses question?

Score per question = mean of 5 criteria (0–1).
Submission S_asst = mean across all free-text questions.

Judging: ≥2 LLMs judge; if disagreement → arbitration model.

### 4.4 Grounding Metric (G) — MOST IMPORTANT
```
precision = |P ∩ G| / |P|
recall    = |P ∩ G| / |G|
F_beta    = (1 + β²) × precision × recall / (β² × precision + recall)
```
With β=2.5: **recall ~6× more important than precision**.

Edge cases:
- Both sets empty → **1.0**
- One empty, other not → **0.0**

**Grounding is a MULTIPLIER — even perfect answers collapse if grounding is low.**

Golden set semantics:
- Golden set = specific pages verified as sufficient for the answer
- Must identify THE EXACT source (e.g., latest edition if multiple exist)
- Secondary/related pages NOT in golden set if not required for the answer

### 4.5 TTFT Factor (F) — OFFICIAL from starter kit
| TTFT (ms) | Factor |
|-----------|--------|
| < 1000 | **1.05** |
| < 2000 | **1.02** |
| < 3000 | **1.00** |
| > 3000 | **0.85–0.99** |

TTFT = time to first token of the **final answer** (not intermediate steps).
If not streaming: TTFT = total_time_ms.
Submission F = mean of all per-answer TTFT factors.

### 4.6 Telemetry Factor (T)
Per-answer telemetry validated:
- `timing` present, non-negative, ttft_ms ≤ total_time_ms
- `usage` present, non-negative token counts
- `retrieval.retrieved_chunk_pages` present and non-empty
- `doc_id` exists in corpus mapping

If any fail → **telemetry_factor = 0.9** for that answer.
Submission T = mean of all telemetry factors.

---

## 5. Submission Format (OFFICIAL)

### JSON Structure
```json
{
  "architecture_summary": "Brief description (max 500 chars)",
  "answers": [
    {
      "question_id": "sha256-hash-of-question",
      "answer": "<number|bool|string|string[]|null>",
      "telemetry": {
        "timing": {
          "ttft_ms": 320,
          "tpot_ms": 45,
          "total_time_ms": 1200
        },
        "retrieval": {
          "retrieved_chunk_pages": [
            {
              "doc_id": "443e04bc...de032",
              "page_numbers": [1, 2, 3]
            }
          ]
        },
        "usage": {
          "input_tokens": 512,
          "output_tokens": 128
        },
        "model_name": "gpt-4o-mini"
      }
    }
  ]
}
```

### Key telemetry rules
- `doc_id` = PDF filename (SHA-like hash string), NOT human label
- `page_numbers` = **1-based physical PDF page numbers** (first page = 1)
- Include **only pages actually used** to generate the answer
- Extra pages reduce precision → hurt grounding score
- Use provider-reported token counts when available
- Unknown `doc_id` → treated as malformed telemetry
- For unanswerable questions: `retrieved_chunk_pages` = `[]`

### Submission package
- `submission.json` — answers + telemetry
- `code_archive.zip` — reproducible code (max 25 MB)
- Submitted via `POST /submissions` as multipart form data

### Submission limits
- **Warm-up**: 10 submissions total
- **Final**: 2 submissions total

---

## 6. API Endpoints

Base URL: `https://platform.agentic-challenge.ai/api/v1`
Auth: `X-API-Key` header

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/questions` | Download questions JSON |
| GET | `/documents` | Download document corpus ZIP |
| POST | `/submissions` | Submit JSON + code archive |
| GET | `/submissions/{uuid}/status` | Check evaluation status |

Question JSON fields: `id` (SHA-256 hash), `question` (text), `answer_type`.

---

## 7. Technical Requirements

### Allowed
- Any programming language (Python preferred)
- Any public LLM API (OpenAI, Gemini, Anthropic, Groq, OpenRouter, etc.)
- Vector DBs, retrieval solutions, local storage/indexing
- Open-source models with local inference
- Multi-agent pipelines
- Dev tools (Copilot, Cursor, etc.)

### Required
- Dependencies file (requirements.txt / pyproject.toml / package.json)
- README with setup & run instructions
- `.env.example` listing required API keys (no real secrets in code)
- Code must run without manual rewriting

### Pre-processing
- **Documents**: CAN be pre-processed (indexed, vectorized, chunked, OCR'd, etc.)
- **Questions**: arrive "online" — TTFT measured from question receipt
- Preprocessing time NOT scored, but must be automated (48h window)

---

## 8. Starter Kit Structure

```
starter_kit/
├── arlc/                    # Client + helpers package
│   ├── client.py            # EvaluationClient (download, submit)
│   ├── config.py            # EnvConfig (env vars)
│   ├── submission.py        # SubmissionBuilder, SubmissionAnswer
│   └── telemetry.py         # TelemetryTimer, RetrievalRef, normalize
├── examples/
│   ├── llamaindex/          # Naive RAG with LlamaIndex
│   ├── langchain/           # Naive RAG with LangChain + FAISS
│   ├── submit.py            # Standalone submission script
│   └── telemetry_example.py # Telemetry calculation demo
├── .env.example
├── openapi.yaml
├── submission.json          # Example submission
├── README.md
├── API.md
└── EVALUATION.md
```

### Baseline config (both examples)
- LLM: `gpt-4o-mini` via OpenRouter
- Embeddings: `text-embedding-ada-002`
- Chunk size: 512 tokens, overlap: 50
- Top-k retrieval: 3

---

## 9. Prizes & Special Nominations
- **1st, 2nd, 3rd place** — by final score
- **Fastest solution** — best TTFT with reasonable quality
- **Best grounding** — highest source attribution quality
- **Best publication** — methodology, insights, error analysis, reproducibility

---

## 10. Optimization Priorities (from official EVALUATION.md)

1. **Grounding first** — improve retrieval precision/recall with tight context
2. **Correct answers** — deterministic accuracy is the base signal
3. **LLM quality** — free-text judged on 5 criteria
4. **Telemetry health** — avoid malformed telemetry
5. **Speed** — TTFT boosts only if everything above is solid

---

## 10a. Recommended Pipeline Architecture (from participant_guide)

### Document Ingestion
- Multi-format PDF handling (digitally-born + scanned with OCR)
- **Clause-level segmentation** respecting legal document hierarchy
- Metadata extraction: titles, sections, case numbers, dates

### Indexing Strategy
- **Structure-aware chunking** (not just fixed token windows)
- Dense embeddings + re-ranking
- **Hybrid search**: BM25 + semantic approaches

### Strict Rules
**Prohibited:**
- Hardcoding answers
- Synthetic data leakage
- Manual log/telemetry editing
- Private question sharing

**Allowed:**
- Model ensembles and hybrid pipelines
- Custom re-rankers
- Local document preprocessing

---

## 10b. Private Phase Strategy (organizer comment)

The 48h window for private dataset is **NOT for experimentation**. Organizer intent:
- You won't have time for major pipeline changes on 1000 questions
- Only 2 submissions — no room for tuning to private data
- The 48h is meant for: running your pipeline, fixing practical breakages (e.g. unusual symbols in docs)
- You should **also manually verify/relabel** your answers before submitting
- Pipeline must be **fully ready before Mar 18** — private phase is just execution + minor fixes

**Implication for us:** all R&D, tuning, and architecture work must happen on the warm-up set (100q, 30 docs). By Mar 18 our pipeline must be battle-tested and robust.

---

## 11. Resolved Discrepancies

| Topic | Seminar said | Starter kit says | Resolution |
|-------|-------------|-----------------|------------|
| TTFT max bonus | 1.25× | 1.05× (at <1s) | **Starter kit is authoritative** |
| Final formula | QA × G × T × F | (0.7×S_det + 0.3×S_asst) × G × T × F | **Starter kit is authoritative** |
| Telemetry factor | 0.9 without telemetry | 0.9 per malformed answer, mean across all | **Per-answer, not binary** |
| Page numbering | TBD | **1-based** (physical PDF pages) | **Confirmed** |
| Submission format | Informal from seminar | Exact JSON schema with field names | **Starter kit is authoritative** |
