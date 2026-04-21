"""
Step 1: Analyze v5 training dataset.

Prints:
- Total entries
- Count by metadata.source
- Count of entries with Chinese characters in output
- 2 sample entries from each source type

Usage:
    python data/v5_analyze.py
    python data/v5_analyze.py --input claude_cleaned_training_data_v5.json
"""

import json
import re
import argparse
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = str(BASE_DIR / "claude_cleaned_training_data_v5.json")

CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def has_chinese(text: str) -> bool:
    return bool(CJK_RE.search(text or ""))


def _preview(s: str, n: int = 280) -> str:
    """Truncate long strings for readable output."""
    s = (s or "").replace("\n", "\\n")
    return s if len(s) <= n else s[:n] + "…"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=DEFAULT_INPUT)
    args = ap.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    print("=" * 72)
    print(f"  DATASET ANALYSIS — {path.name}")
    print("=" * 72)

    # Totals
    print(f"\nTotal entries: {len(data)}")

    # By source
    sources = Counter(d.get("metadata", {}).get("source", "?") for d in data)
    print(f"\nEntries by metadata.source:")
    for src, n in sources.most_common():
        print(f"  {src:<35} {n:>5}")

    # Chinese detection in output
    zh_entries = [d for d in data if has_chinese(d.get("output", ""))]
    print(f"\nEntries with Chinese characters in output: {len(zh_entries)} "
          f"({100 * len(zh_entries) / len(data):.1f}%)")

    # Also check instructions for bilingual analysis
    zh_instr = sum(1 for d in data if has_chinese(d.get("instruction", "")))
    print(f"Entries with Chinese in instruction:       {zh_instr} "
          f"({100 * zh_instr / len(data):.1f}%)")

    # 2 samples per source
    print("\n" + "=" * 72)
    print("  SAMPLES — 2 per source type")
    print("=" * 72)

    for src, _ in sources.most_common():
        src_entries = [d for d in data if d.get("metadata", {}).get("source") == src]
        print(f"\n── source: {src} ({len(src_entries)} total) ──")
        for i, entry in enumerate(src_entries[:2], 1):
            print(f"\n  [#{i}]")
            print(f"  instruction : {_preview(entry.get('instruction', ''))}")
            print(f"  input       : {_preview(entry.get('input', ''), 200)}")
            print(f"  output      : {_preview(entry.get('output', ''), 400)}")
            meta = entry.get("metadata", {})
            print(f"  metadata    : {meta}")


if __name__ == "__main__":
    main()
