# ARLC - Agentic RAG Legal Challenge

## Quick Reference
- **Competition**: ARLC (agentic-challenge.ai)
- **Task**: Build a legal AI RAG system over DIFC (Dubai International Financial Centre) regulatory documents
- **Language**: All questions and documents in English
- **Dataset**: 1000 questions (100 public, 900 private)
- **Team size**: 1-3 people
- **Detailed competition spec**: see `docs/competition-spec.md`

## Timeline
- **Mar 11**: Public dataset release (100 questions), up to 10 submissions
- **Mar 18**: Private dataset unlock (900 questions), 48h window
- **Mar 20 23:59**: Submission deadline (only 2 attempts on private set)
- **~Mar 21-26**: Organizer review, final leaderboard
- **April**: Online ceremony + mini-conference

## Scoring Formula
```
FinalScore = base_QA_score * grounding * completeness * TTFT_multiplier
```
- **base_QA_score**: avg across all questions (structured + free-text)
- **grounding**: F-measure (beta=2.5, recall > precision)
- **completeness**: 1.0 with telemetry, 0.9 without
- **TTFT_multiplier**: 0.85 (slow) to 1.25 (fast), baseline ~2-3s = 1.0

## Answer Types (70% structured, 30% free-text)
- `number` (int, 1% tolerance), `boolean`, `name`, `names` (Jaccard), `date` (ISO), `null`
- Free-text: up to 280 chars, judged by LLM on 5 criteria

## Key Rules
- Documents can be pre-processed (indexed, embedded, etc.)
- Questions arrive "online" - TTFT measured from question receipt
- `null` answer = question not covered by corpus; grounding sources must be empty
- Submission = JSON (answers + telemetry) + code archive
- Code must be reproducible: deps, README, .env.example
- Pre-submit benchmark gate: compare every candidate submission against `golden_submission.json` before platform submission

## Benchmark Gate
- `golden_submission.json` is the repository benchmark for the public set.
- The warmup platform evaluated it at `deterministic=1.000`, `grounding=0.954191`, and `total_score=0.887146` on March 13, 2026.
- Because of that benchmark strength, no candidate submission should be sent to the platform until it has been compared against `golden_submission.json` and the regressions have been reviewed.
