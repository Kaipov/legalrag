# Baseline v7.1 Implementation Plan

## Goal

Reduce public-set grounding drift against `golden_submission.json` without introducing public-only rules or fragile heuristics that are likely to regress on the private set released on March 18, 2026.

## Current Grounding Shape

- `generic`: 19 partial, 3 zero
- `article_ref`: 9 partial, 5 zero
- `judge_compare`: 5 partial, 0 zero
- `last_page`: 4 partial, 2 zero
- `title_page`: 3 partial, 0 zero
- `date_of_issue`: 3 partial, 0 zero
- `party_compare`: 1 partial, 0 zero

## Guardrails

- No public-question or public-page hardcoding.
- No per-question overrides keyed by `question_id`.
- No global “cite more pages” or “cite fewer pages” knob turns without structural justification.
- Prefer deterministic evidence selection only when the document geometry is explicit and reusable across corpora.
- If a deterministic path is not high-confidence, fall back to the existing retrieval-plus-generation path.

## Workstreams

### 1. Structural Evidence Overrides for `article_ref`

Objective:
- For structured questions that point to an explicit Article in a uniquely identifiable law, override noisy post-hoc grounding with page metadata evidence.

Implementation:
- Reuse `extract_question_anchors`, `match_target_law_doc`, and `classify_article_page_match`.
- Select only exact definition pages from the uniquely matched law document.
- Keep the model answer unchanged in v7.1; override only telemetry grounding when confidence is high.

Why it is safe:
- The method relies on article numbering and law disambiguation, both of which are stable legal-document structure signals.
- It does not assume anything about public answer wording.

Acceptance:
- New resolver-level tests cover exact-page preference and ambiguity fallback.
- Pipeline-level test confirms structural evidence overrides the noisier grounding output.

Status:
- Implemented in v7.1.

### 2. Narrow Deterministic Routing for Explicit Page-Localized Metadata Questions

Objective:
- Expand deterministic handling only for questions that explicitly constrain the answer to `first page` or already-supported local page hints.

Implementation:
- Extend `QuestionPlan` so single-case questions with explicit `first page` or `page 1` cues route to `page_local_lookup` when the target field is already supported by metadata extraction.
- Keep the expansion narrow to `claim_number`, `judge`, `party`, and `law_number`.

Why it is safe:
- The route only activates when the question itself supplies page locality.
- If metadata is insufficient, the deterministic resolver returns `None` and the pipeline falls back.

Acceptance:
- Add a plan-level test for `first page` party lookup.
- Existing deterministic resolver tests remain green.

Status:
- Implemented in v7.1.

### 3. Tighten `last_page` / `conclusion` Evidence Selection

Objective:
- Minimize page drift in outcome questions by keeping the smallest page set that still covers the operative order clauses and costs language.

Implementation:
- Audit `resolve_last_page_outcome` against cases where gold keeps both the opening order page and a distant concluding page.
- Preserve cross-page order continuity while avoiding unrelated intermediate pages.
- Only change behavior when clause-level evidence is stronger than current fallback behavior.

Why it is safe:
- Outcome sections in judgments and orders have consistent layout cues.
- The change operates on page evidence, not on public answers.

Acceptance:
- Add tests for multi-page order sections where the decisive clause appears on a later page.
- No regression in existing `last_page_outcome` tests.

Status:
- Planned for next iteration.

### 4. Compare Coverage and Generic Minimal-Cover Cleanup

Objective:
- Reduce partial grounding in multi-document compare questions and simple generic structured questions without broad heuristic overreach.

Implementation:
- Preserve one evidence page per case in compare questions when the answer depends on cross-case overlap.
- For generic structured questions, prefer minimal sufficient cover driven by anchors instead of extra appendix or cross-reference pages.

Why it is safe:
- The approach is coverage-based, not benchmark-tuned.
- It aligns with the official F-beta objective: keep recall, trim precision waste.

Acceptance:
- Add bucketed regression checks before any submission regeneration.
- Verify that `zero grounding` does not increase.

Status:
- Planned for next iteration.

## Verification Plan

- Run targeted unit suites for `question_plan`, `resolvers`, and `pipeline`.
- After enough structural changes land, regenerate a candidate submission and compare it against `golden_submission.json`.
- Review:
  - overall relative grounding
  - zero-grounding count
  - bucket-level drift
  - answer regressions, especially deterministic answers

## Exit Criteria for v7.1

- Structural evidence overrides land with tests.
- Explicit `first page` routing lands with tests.
- Existing targeted test suites remain green.
- The branch is ready for the next tranche of grounding cleanup without introducing public-specific logic.
