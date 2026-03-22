# Final Runbook

This runbook is the shortest reliable path for operating the repository during a competition run.

## 1. Environment Setup

```powershell
cd E:\legalrag
uv venv .venv
uv pip install -r requirements.txt
Copy-Item .env.example .env
```

Fill in at least:
- `EVAL_API_KEY`
- `OPENAI_API_KEY` or `OPENROUTER_API_KEY`
- `GEMINI_API_KEY`

Optional:
- `VOYAGE_API_KEY`

## 2. Prepare the Dataset

Expected local paths:
- [`data/documents`](../data/documents)
- [`data/questions.json`](../data/questions.json)

If the corpus changes, replace both documents and questions before preprocessing.

## 3. Preprocessing

Full rebuild:

```powershell
uv run --python .\.venv\Scripts\python.exe python -m scripts.preprocess
```

Stepwise rebuild:

```powershell
uv run --python .\.venv\Scripts\python.exe python -m scripts.preprocess --extract
uv run --python .\.venv\Scripts\python.exe python -m scripts.preprocess --chunk
uv run --python .\.venv\Scripts\python.exe python -m scripts.preprocess --index
```

Typical shortcuts:
- only retrieval or grounding logic changed: usually no rebuild
- page metadata or deterministic extraction changed: rebuild at least `--index`
- chunking or extraction changed: rebuild `--chunk` and `--index`
- PDF extraction or OCR logic changed: rebuild everything

## 4. Generate a Local Candidate

```powershell
uv run --python .\.venv\Scripts\python.exe python -m scripts.run --no-download --no-submit --questions data/questions.json
```

Then freeze a timestamped copy if the run looks promising:

```powershell
Copy-Item submission.json submission.candidate.$(Get-Date -Format 'yyyyMMdd-HHmmss').json
```

## 5. Validate the Candidate

```powershell
uv run --python .\.venv\Scripts\python.exe python -m scripts.evaluate --submission submission.json --strict
```

For public-set regression checks:

```powershell
uv run --python .\.venv\Scripts\python.exe python -m scripts.compare_answers
uv run --python .\.venv\Scripts\python.exe python -m scripts.compare_submissions
uv run --python .\.venv\Scripts\python.exe python -m scripts.regression_report --strict
```

## 6. Submit by Regenerating and Uploading

If you are comfortable doing one more full rerun before platform submission:

```powershell
uv run --python .\.venv\Scripts\python.exe python -m scripts.run --no-download --questions data/questions.json
```

This regenerates `submission.json`, creates `code_archive.zip`, and submits both to the platform.

## 7. Submit a Frozen `submission.json` Without Another Rerun

Use this when you already trust the current [`submission.json`](../submission.json) and want to avoid answer drift.

```powershell
@'
from __future__ import annotations
import json
import sys
from datetime import datetime
from fnmatch import fnmatch
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "starter_kit"))

import src.config  # loads .env defaults
from arlc import EvaluationClient

stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
archive_path = ROOT / f"code_archive.private-submit.{stamp}.zip"
response_path = ROOT / f"submission.platform-response.{stamp}.json"
uuid_path = ROOT / f"submission.uuid.{stamp}.txt"
status_path = ROOT / f"submission.status.{stamp}.json"
submission_path = ROOT / "submission.json"

exclude_dirs = {
    "__pycache__", "data", "index", "storage", ".venv", "venv", "env",
    ".git", "node_modules", "eval", ".pytest_cache", ".claude",
}
exclude_patterns = [
    ".env",
    "golden_submission.json",
    "submission.json",
    "submission*.json",
    "submission*.txt",
    "code_archive*.zip",
]

with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
    for file_path in ROOT.rglob("*"):
        if not file_path.is_file():
            continue
        rel = file_path.relative_to(ROOT)
        if set(rel.parts) & exclude_dirs:
            continue
        if any(fnmatch(file_path.name, pattern) for pattern in exclude_patterns):
            continue
        if file_path.resolve() == archive_path.resolve():
            continue
        if file_path.stat().st_size > 10_000_000:
            continue
        archive.write(file_path, rel)

client = EvaluationClient.from_env()
response = client.submit_submission(submission_path, archive_path)
status = client.get_submission_status(response["uuid"])

response_path.write_text(json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8")
uuid_path.write_text(response["uuid"], encoding="utf-8")
status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")

print(json.dumps(response, ensure_ascii=False, indent=2))
'@ | uv run --python .\.venv\Scripts\python.exe python -
```

This preserves:
- the submitted UUID
- the platform response
- the first status snapshot
- the exact code archive that went with the frozen submission

## 8. Poll Platform Status

```powershell
@'
import json
import sys
from pathlib import Path
ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "starter_kit"))
import src.config
from arlc import EvaluationClient

uuid = Path("submission.uuid.latest.txt").read_text(encoding="utf-8").strip()
client = EvaluationClient.from_env()
print(json.dumps(client.get_submission_status(uuid), ensure_ascii=False, indent=2))
'@ | uv run --python .\.venv\Scripts\python.exe python -
```

If you keep timestamped UUID files instead, replace the file path with the exact saved artifact.

## 9. Safety Notes

- Do not commit private `submission.*` artifacts.
- Do not include `golden_submission.json` in a final-stage code archive.
- Keep the exact code archive and UUID for every platform submission.
- If a last-minute patch changes retrieval, chunking, or OCR behavior, decide whether it requires an index rebuild before trusting the next run.
