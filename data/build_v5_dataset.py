"""
Build v5 training dataset — bilingual + multi-language + more clean examples.
==============================================================================

Starts from v4 (464 records, all Python, Chinese JSON output, 60 clean examples)
and addresses the three weaknesses surfaced by eval/SecurityEval_Benchmark_Results.md:

  1. **Python-only**: v4 has no C/C++/Java/JavaScript/Go coverage.
     → Adds 23 multi-language vulnerable examples.
  2. **Missing CWE labels**: model previously misclassified CWE-200, CWE-400,
     CWE-730, CWE-943 because those exact CWE tokens never appear in training.
     → Adds 4 Python examples with explicit CWE labels in the output.
  3. **Mono-lingual output**: every output is Chinese JSON; model cannot produce
     English reviews even when prompted in English.
     → Adds 28 examples whose output is English JSON (parallel format).

Output counts:
    v4 records        : 464  (Python, Chinese JSON)
    + multilang vulns : +23  (5 langs, English JSON)
    + clean multilang : +5   (5 langs, English JSON, {"issues": []})
    --------------------------------
    v5 total          : ~492  (~6% multilang, ~14% clean-code negatives)

Usage:
    python data/build_v5_dataset.py
    python data/build_v5_dataset.py --input claude_cleaned_training_data_v4.json
    python data/build_v5_dataset.py --dry-run
"""

import json
import sys
import random
import argparse
from collections import Counter
from pathlib import Path

# Make sibling imports work regardless of where the script is invoked from
sys.path.insert(0, str(Path(__file__).resolve().parent))
from multilang_examples import ALL_EXAMPLES as MULTILANG_EXAMPLES  # type: ignore[import-not-found]

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = str(BASE_DIR / "claude_cleaned_training_data_v4.json")
DEFAULT_OUTPUT = str(BASE_DIR / "claude_cleaned_training_data_v5.json")
RANDOM_SEED = 42


def load_dataset(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_dataset(data: list, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def summarize(data: list, label: str) -> None:
    """Print a compact breakdown of dataset composition."""
    sources = Counter(d.get("metadata", {}).get("source", "?") for d in data)
    langs = Counter(d.get("metadata", {}).get("language", "python") for d in data)

    # Detect output language (heuristic: look for CJK chars in summary/description)
    zh_count = 0
    en_count = 0
    for d in data:
        out = d.get("output", "")
        # Use the first 200 chars to avoid heavy scans
        sample = out[:500]
        if any("\u4e00" <= c <= "\u9fff" for c in sample):
            zh_count += 1
        else:
            en_count += 1

    print(f"\n── {label} ──")
    print(f"  Total: {len(data)}")
    print(f"  Sources:")
    for src, n in sources.most_common():
        print(f"    {src:<35} {n:>5}")
    print(f"  Languages (from metadata):")
    for lang, n in langs.most_common():
        print(f"    {lang:<15} {n:>5}")
    print(f"  Output language (heuristic):")
    print(f"    Chinese:  {zh_count}")
    print(f"    English:  {en_count}")


def main():
    parser = argparse.ArgumentParser(description="Build v5 training dataset")
    parser.add_argument("--input", default=DEFAULT_INPUT,
                        help="Input v4 JSON file")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="Output v5 JSON file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print report but do not save")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # Load v4
    src = Path(args.input)
    if not src.exists():
        raise FileNotFoundError(
            f"Input file not found: {src}\n"
            f"Run `python data/improve_dataset.py` first to build v4."
        )
    v4 = load_dataset(str(src))
    summarize(v4, f"v4 — loaded from {src.name}")

    # Validate multilang examples parse correctly
    for i, ex in enumerate(MULTILANG_EXAMPLES):
        try:
            json.loads(ex["output"])
        except json.JSONDecodeError as e:
            raise ValueError(f"Multilang example {i} has invalid JSON output: {e}")

    # Merge
    v5 = v4 + MULTILANG_EXAMPLES
    rng.shuffle(v5)

    summarize(v5, "v5 — after adding multi-language examples")

    # Output format distribution: vulnerable vs clean
    clean = sum(1 for d in v5 if '"issues": []' in d.get("output", ""))
    vulnerable = len(v5) - clean
    print(f"\n  Positive/negative balance:")
    print(f"    Vulnerable (issues > 0):  {vulnerable}")
    print(f"    Clean     (issues == 0):  {clean}")
    print(f"    Clean ratio:              {100 * clean / len(v5):.1f}%")

    if args.dry_run:
        print("\n[dry-run] No file saved.")
        return

    save_dataset(v5, args.output)
    out = Path(args.output).resolve()
    print(f"\nSaved v5 → {out}")
    print(f"  Update train.py DATA_PATH to point at: {out.name}")


if __name__ == "__main__":
    main()
