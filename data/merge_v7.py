"""
Steps 4-5: Merge fixed v6 + generated synthetic into training_data_v7_final.jsonl.

1. Combine training_data_v6_fixed.jsonl (391) + synthetic_generated.jsonl (700)
2. Remove duplicates based on the user message content (exact match)
3. Shuffle with random seed 42
4. Validate ALL entries have valid JSON outputs — drop any that don't
5. Save to training_data_v7_final.jsonl
6. Print final statistics
"""

import json
import random
import sys
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
FIXED = BASE_DIR / "training_data_v6_fixed.jsonl"
GENERATED = BASE_DIR / "synthetic_generated.jsonl"
OUTPUT = BASE_DIR / "training_data_v7_final.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def get_user_msg(entry: dict) -> str:
    for m in entry.get("messages", []):
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


def get_assistant_msg(entry: dict) -> str:
    for m in entry.get("messages", []):
        if m.get("role") == "assistant":
            return m.get("content", "")
    return ""


def has_valid_json_output(entry: dict) -> bool:
    ao = get_assistant_msg(entry)
    if not ao or not ao.strip():
        return False
    try:
        obj = json.loads(ao)
    except Exception:
        return False
    if not isinstance(obj, dict):
        return False
    if "issues" not in obj or "overall_score" not in obj:
        return False
    if not isinstance(obj["issues"], list):
        return False
    try:
        score = float(obj["overall_score"])
    except Exception:
        return False
    return 1 <= score <= 10


def extract_severities(entry: dict) -> list[str]:
    try:
        obj = json.loads(get_assistant_msg(entry))
    except Exception:
        return []
    return [
        str(iss.get("severity", "")).strip()
        for iss in obj.get("issues", [])
        if isinstance(iss, dict)
    ]


def is_clean(entry: dict) -> bool:
    try:
        obj = json.loads(get_assistant_msg(entry))
    except Exception:
        return False
    return len(obj.get("issues", [])) == 0


def main():
    if not FIXED.exists():
        print(f"Missing: {FIXED}", file=sys.stderr)
        sys.exit(1)
    if not GENERATED.exists():
        print(f"Missing: {GENERATED}", file=sys.stderr)
        sys.exit(1)

    fixed = load_jsonl(FIXED)
    generated = load_jsonl(GENERATED)
    print(f"Loaded {len(fixed)} fixed entries")
    print(f"Loaded {len(generated)} generated entries")

    combined = fixed + generated
    print(f"Combined: {len(combined)}")

    # ── Dedupe by user message content (exact match) ──────────────────
    seen = set()
    deduped = []
    dup_count = 0
    for e in combined:
        um = get_user_msg(e)
        if um in seen:
            dup_count += 1
            continue
        seen.add(um)
        deduped.append(e)
    print(f"After dedup: {len(deduped)} (removed {dup_count} duplicates)")

    # ── Validate JSON outputs; drop invalid ───────────────────────────
    valid = []
    dropped = 0
    for e in deduped:
        if has_valid_json_output(e):
            valid.append(e)
        else:
            dropped += 1
    print(f"After JSON validation: {len(valid)} (dropped {dropped} invalid)")

    # ── Shuffle with seed 42 ──────────────────────────────────────────
    rng = random.Random(42)
    rng.shuffle(valid)

    # ── Save ──────────────────────────────────────────────────────────
    with open(OUTPUT, "w", encoding="utf-8") as f:
        for e in valid:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(valid)} entries → {OUTPUT.name}")

    # ── STEP 5: statistics ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FINAL STATISTICS")
    print("=" * 60)

    print(f"\nTotal entries: {len(valid)}")

    sources = Counter(e.get("metadata", {}).get("source", "?") for e in valid)
    print("\nBy source:")
    for k, n in sources.most_common():
        print(f"  {k:<35} {n:>5}")

    langs = Counter(e.get("metadata", {}).get("language", "?") for e in valid)
    print("\nBy language:")
    for k, n in langs.most_common():
        print(f"  {k:<12} {n:>5}")

    # Severity distribution across ALL issues
    sev = Counter()
    for e in valid:
        for s in extract_severities(e):
            sev[s] += 1
    total_issues = sum(sev.values())
    print(f"\nSeverity distribution (across {total_issues} total issues):")
    for k in ["High", "Medium", "Low"]:
        n = sev.get(k, 0)
        pct = 100 * n / total_issues if total_issues else 0
        print(f"  {k:<8} {n:>5} ({pct:.1f}%)")
    other = total_issues - sum(sev.get(k, 0) for k in ["High", "Medium", "Low"])
    if other:
        print(f"  Other    {other:>5}")

    # Clean entries
    clean_count = sum(1 for e in valid if is_clean(e))
    clean_pct = 100 * clean_count / len(valid) if valid else 0
    print(f"\nClean code entries: {clean_count} ({clean_pct:.1f}%)")

    # Problem types
    ptypes = Counter(e.get("metadata", {}).get("problem_type", "?") for e in valid)
    print(f"\nUnique problem types: {len(ptypes)}")

    # Sample 3 random entries (deterministic)
    sample_rng = random.Random(7)
    sample = sample_rng.sample(valid, min(3, len(valid)))
    print("\nSample 3 random entries:")
    for i, e in enumerate(sample, 1):
        meta = e.get("metadata", {})
        um = get_user_msg(e)
        print(f"\n  [{i}] metadata: {json.dumps(meta, ensure_ascii=False)}")
        print(f"      user_msg[:100]: {um[:100]!r}")


if __name__ == "__main__":
    main()
