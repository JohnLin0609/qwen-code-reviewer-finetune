"""
Steps 2 + 3: Clean + translate v5 → v5_en (English-only).

Step 2 — Remove all entries where metadata.source == 'github_inline_comment'.
         These are raw PR conversation replies.

Step 3 — Translate Chinese outputs to English via the Claude API.
         - Keeps JSON structure intact.
         - Only translates text fields: type, severity, description, suggestion, summary.
         - Code fields (fixed_code, input) are never touched.
         - Resumable: saves progress to translation_progress.json every 50 entries.
           On restart, already-translated entries are skipped.

Defaults to Haiku 4.5 (cheap + fast + good enough for translation).
Use --model claude-sonnet-4-6 for higher quality.

Usage:
    # Set ANTHROPIC_API_KEY in .env first
    python data/v5_translate.py
    python data/v5_translate.py --model claude-sonnet-4-6
    python data/v5_translate.py --dry-run       # report only, no API calls
    python data/v5_translate.py --resume        # continue from last saved progress
"""

import json
import os
import re
import time
import argparse
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = str(BASE_DIR / "claude_cleaned_training_data_v5.json")
DEFAULT_OUTPUT = str(BASE_DIR / "claude_cleaned_training_data_v5_en.json")
PROGRESS_PATH = str(BASE_DIR / "translation_progress.json")

DEFAULT_MODEL = "claude-haiku-4-5-20251001"   # cheap + fast; override with --model
MAX_RETRIES = 3
BASE_BACKOFF = 2.0    # exponential backoff base (seconds)
SAVE_EVERY = 50       # save progress every N entries

CJK_RE = re.compile(r"[\u4e00-\u9fff]")

SYSTEM_PROMPT = """You translate Chinese security code-review JSON into English. Rules:

1. Input is a JSON object with fields like: issues, overall_score, summary.
2. Each issue has: type, severity, description, suggestion, fixed_code.
3. Translate ONLY these text fields: type, severity, description, suggestion, summary.
4. NEVER translate or modify the fixed_code field — return it byte-for-byte identical.
5. NEVER change overall_score (it's a number).
6. Preserve the exact JSON structure. Return ONLY the JSON object, no explanation, no markdown fences.

Translation conventions:
- "安全漏洞" → "Security Vulnerability"
- "可靠性問題" → "Reliability Issue"
- "效能問題" → "Performance Issue"
- "邏輯錯誤" → "Logic Error"
- "代碼品質" / "程式碼品質" → "Code Quality"
- "高" → "High"
- "中" → "Medium"
- "低" → "Low"
- Descriptions and summaries: direct, technical English. No filler like "I think".
"""


def has_chinese(text: str) -> bool:
    return bool(CJK_RE.search(text or ""))


def load_progress(path: str) -> dict:
    """Load resume state. Returns {'translated': {idx: output_str, ...}, 'model': str}."""
    p = Path(path)
    if not p.exists():
        return {"translated": {}, "model": None}
    with open(p, encoding="utf-8") as f:
        state = json.load(f)
    # JSON keys are always strings — convert to int for convenience
    state["translated"] = {int(k): v for k, v in state.get("translated", {}).items()}
    return state


def save_progress(state: dict, path: str) -> None:
    tmp = Path(path).with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(
            {"translated": {str(k): v for k, v in state["translated"].items()},
             "model": state["model"]},
            f, ensure_ascii=False
        )
    os.replace(tmp, path)


def call_claude(client, model: str, chinese_json: str) -> str:
    """Translate one JSON block. Returns translated JSON string.
    Retries on transient errors with exponential backoff."""
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": chinese_json}],
            )
            text = resp.content[0].text.strip()
            # Occasionally the model wraps output in ```json fences — strip them
            if text.startswith("```"):
                text = re.sub(r"^```(?:json)?\s*", "", text)
                text = re.sub(r"\s*```$", "", text)
            # Validate it's parseable JSON
            json.loads(text)
            return text
        except Exception as e:
            last_err = e
            if attempt < MAX_RETRIES - 1:
                wait = BASE_BACKOFF * (2 ** attempt)
                print(f"    ! retry {attempt + 1}/{MAX_RETRIES} in {wait:.1f}s: {type(e).__name__}: {e}")
                time.sleep(wait)
    raise RuntimeError(f"Translation failed after {MAX_RETRIES} attempts: {last_err}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--output", default=DEFAULT_OUTPUT)
    ap.add_argument("--progress", default=PROGRESS_PATH)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="Load progress file and skip already-translated entries")
    args = ap.parse_args()

    # ── Load .env for API key ────────────────────────────────────────────
    env_path = BASE_DIR / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("ANTHROPIC_API_KEY="):
                os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1].strip()

    # ── Load data ────────────────────────────────────────────────────────
    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} entries from {Path(args.input).name}")

    # ── STEP 2: remove github_inline_comment ─────────────────────────────
    before = len(data)
    data = [d for d in data if d.get("metadata", {}).get("source") != "github_inline_comment"]
    print(f"Step 2 — removed {before - len(data)} github_inline_comment entries → {len(data)} remain")

    # ── Identify Chinese-output entries ──────────────────────────────────
    chinese_indices = [i for i, d in enumerate(data) if has_chinese(d.get("output", ""))]
    print(f"Chinese-output entries to translate: {len(chinese_indices)}")

    if args.dry_run:
        by_src = Counter(data[i].get("metadata", {}).get("source", "?") for i in chinese_indices)
        print("\nBy source (dry run):")
        for s, n in by_src.most_common():
            print(f"  {s:<35} {n:>5}")
        print("\n[dry-run] No API calls made. No file saved.")
        return

    # ── STEP 3: translate (with resume) ──────────────────────────────────
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Add it to .env at the project root:\n"
            "  ANTHROPIC_API_KEY=sk-ant-..."
        )

    try:
        from anthropic import Anthropic
    except ImportError:
        raise ImportError("anthropic SDK not installed. Run: pip install anthropic")

    client = Anthropic()
    progress = load_progress(args.progress) if args.resume else {"translated": {}, "model": args.model}

    if args.resume and progress["model"] and progress["model"] != args.model:
        print(f"WARNING: previous progress used model={progress['model']}, current={args.model}")
    progress["model"] = args.model

    already = len(progress["translated"])
    if already:
        print(f"Resuming — {already} entries already translated")

    for n, idx in enumerate(chinese_indices, 1):
        if idx in progress["translated"]:
            continue

        entry = data[idx]
        try:
            translated = call_claude(client, args.model, entry["output"])
            progress["translated"][idx] = translated
        except Exception as e:
            print(f"  [#{n}/{len(chinese_indices)} idx={idx}] FAILED: {e}")
            # Continue — failures can be retried later via --resume
            continue

        if n % SAVE_EVERY == 0:
            save_progress(progress, args.progress)
            done = len(progress["translated"])
            print(f"  [#{n}/{len(chinese_indices)}] translated — progress saved ({done} done)")

    # Final save
    save_progress(progress, args.progress)

    # ── Apply translations back to data ──────────────────────────────────
    applied = 0
    missing = []
    for idx in chinese_indices:
        if idx in progress["translated"]:
            data[idx]["output"] = progress["translated"][idx]
            applied += 1
        else:
            missing.append(idx)

    print(f"\nApplied {applied}/{len(chinese_indices)} translations")
    if missing:
        print(f"WARNING: {len(missing)} entries could not be translated "
              f"(first few indices: {missing[:5]}). Re-run with --resume to retry.")

    # ── Save output ──────────────────────────────────────────────────────
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\nSaved translated dataset → {args.output}")


if __name__ == "__main__":
    main()
