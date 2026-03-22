# ARLC Competition Specification

Clean reference for the ARLC legal RAG challenge as used by this repository.

## Overview

- Competition: ARLC, hosted on `platform.agentic-challenge.ai`
- Domain: DIFC legal and regulatory documents
- Language: English
- Task: answer online questions over a supplied PDF corpus and cite supporting pages
- Evaluation emphasis: answer quality, grounding quality, telemetry validity, and TTFT

## Phases

- Warm-up phase:
  - about 30 documents
  - 100 questions
  - up to 10 submissions
- Final phase:
  - about 300 documents
  - 900 questions
  - 2 submissions

Warm-up and final corpora should be treated as separate datasets. Rebuild the local index when you switch corpora.

## Answer Types

Visible `answer_type` values:
- `number`
- `boolean`
- `name`
- `names`
- `date`
- `free_text`

General rules:
- `number` accepts integers or floats and is scored with 1 percent tolerance
- `boolean` is exact match
- `name` is a single normalized string
- `names` is a JSON array of strings and is scored with Jaccard overlap
- `date` must be ISO `YYYY-MM-DD`
- `free_text` should stay concise and grounded
- any question may legitimately have `null` as the correct answer

For unanswerable questions:
- answer should be `null` or a null-like answer according to the task schema
- grounding sources must be empty

## Scoring Formula

Official final score:

```text
Total = (0.7 * S_det + 0.3 * S_asst) * G * T * F
```

Where:
- `S_det` is the structured-answer score
- `S_asst` is the free-text assistant score
- `G` is grounding quality
- `T` is telemetry validity
- `F` is the TTFT multiplier

Repository shorthand often refers to:
- base QA score = `0.7 * S_det + 0.3 * S_asst`
- final score = base QA score multiplied by grounding, telemetry, and TTFT factors

## Grounding

Grounding is the strongest multiplier in the competition.

Key properties:
- evaluated as page-level overlap against a golden set
- F-beta uses `beta = 2.5`
- recall matters much more than precision
- both empty page sets score `1.0`
- one empty and one non-empty page set scores `0.0`

Practical implication:
- missing a necessary page is usually worse than citing one extra relevant page
- but unrelated pages still hurt precision, so citations should remain tight

## Telemetry

Each answer is expected to include telemetry with:
- timing:
  - `ttft_ms`
  - `tpot_ms`
  - `total_time_ms`
- retrieval:
  - `retrieved_chunk_pages`
- usage:
  - `input_tokens`
  - `output_tokens`
- `model_name`

Important rules:
- `doc_id` must match the PDF filename used in the corpus
- `page_numbers` must be 1-based physical PDF page numbers
- citations should include only pages actually used for the answer
- malformed telemetry reduces the telemetry factor

## TTFT

TTFT is measured from question receipt to the first token of the final answer.

Official multiplier bands from the starter kit:
- under 1000 ms: `1.05`
- under 2000 ms: `1.02`
- under 3000 ms: `1.00`
- slower runs receive a penalty

If the model does not stream, TTFT effectively becomes total answer time.

## Submission Format

Each platform submission includes:
- `submission.json`
- `code_archive.zip`

The JSON contains:
- `architecture_summary`
- `answers`
- per-answer telemetry

The code archive must be reproducible:
- dependency file
- README
- `.env.example`
- the code needed to reproduce the run

## API Endpoints

Base URL:

```text
https://platform.agentic-challenge.ai/api/v1
```

Authentication:
- `X-API-Key` header

Endpoints:
- `GET /questions`
- `GET /documents`
- `POST /submissions`
- `GET /submissions/{uuid}/status`

## Repository-Specific Rules

This repository uses the following local conventions:
- `golden_submission.json` is the public-set benchmark gate
- no candidate should be sent to the public warm-up platform until it has been compared against that golden snapshot
- `data/` and `index/` stay out of git
- private-stage artifacts such as `submission.*.json`, UUID files, and platform status snapshots should stay local

Warm-up benchmark note:
- `golden_submission.json` was evaluated on March 13, 2026 at `deterministic = 1.000`, `grounding = 0.954191`, and `total_score = 0.887146`

## Practical Guidance

- Grounding is the main multiplier to optimize once structured correctness is decent.
- Deterministic resolvers are worthwhile for explicit page-local and compare-style questions.
- OCR robustness matters because the corpus mixes digitally born and scanned PDFs.
- The final phase is too small in submission count for broad experimentation, so operational reliability matters as much as model quality.
