"""
Count issues per review across saved model outputs.

Parses compare_v7_out.txt (side-by-side fine-tuned vs base runs) and reports
the average number of issues flagged per test case for each model. Used to
quantitatively back the 'multi-issue detection' claim in the README.

Usage:
    python eval/count_issues.py                         # default inputs
    python eval/count_issues.py --file compare_v7_out.txt
"""

import argparse
import json
import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_FILE = BASE_DIR / "compare_v7_out.txt"


def strip_fences(text: str) -> str:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```\s*$", "", t)
    return t.strip()


def parse_compare_log(path: Path) -> tuple[list[int], list[int]]:
    """Return (fine_tuned_counts, base_counts) — list of issues per test case."""
    text = path.read_text(encoding="utf-8")
    parts = re.split(r"測試 \d+:", text)
    ft, base = [], []
    for p in parts[1:]:
        ft_m = re.search(r"Fine-tuned:\s*\n─+\n(.+?)\n─+", p, re.DOTALL)
        base_m = re.search(r"Original \(.*?\):\s*\n─+\n(.+?)(?=\n=+|\Z)", p, re.DOTALL)
        for m, lst in [(ft_m, ft), (base_m, base)]:
            if not m:
                continue
            raw = strip_fences(m.group(1))
            try:
                obj = json.loads(raw)
                lst.append(len(obj.get("issues", [])))
            except Exception:
                # Fallback: naive count of issue "type" entries
                lst.append(raw.count('"type":'))
    return ft, base


def report(label: str, counts: list[int]) -> None:
    if not counts:
        print(f"{label}: no reviews parsed")
        return
    avg = sum(counts) / len(counts)
    print(f"{label}: n={len(counts)} counts={counts} avg={avg:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(DEFAULT_FILE), help="compare.py output log")
    args = ap.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"Not found: {path}", file=sys.stderr)
        sys.exit(1)

    ft, base = parse_compare_log(path)
    report("Fine-tuned", ft)
    report("Base Qwen2.5-Coder-7B", base)


if __name__ == "__main__":
    main()
