"""
Dataset Post-Processing: fix_dataset.py
========================================
Runs AFTER claude_clean.py to fix three remaining issues:

  Issue 1 — Duplicate inputs (same diff, multiple reviews)
            The 9-pass cleaner deduplicates by OUTPUT text but not by INPUT.
            542 inputs appear more than once → 1,137 extra records.
            Fix: keep only the longest review for each unique input.

  Issue 2 — Mixed output format (JSON vs free text)
            303 handcrafted samples output structured JSON.
            2,380 GitHub samples output free text.
            The model works fine with both (compare-0.txt shows it learned JSON),
            but we add a format hint to the instruction so the model knows
            which format to use for which context.

  Issue 3 — All instructions in Chinese
            2,683 / 2,683 instructions are Chinese.
            Model generalises poorly to English prompts.
            Fix: randomly replace ~45% of instructions with English variants.

Usage:
    python fix_dataset.py
    python fix_dataset.py --input my_other_file.json --output fixed.json
    python fix_dataset.py --no-dedup      # skip deduplication
    python fix_dataset.py --no-en         # skip English instructions
    python fix_dataset.py --dry-run       # print report only, don't save
"""

import json
import random
import argparse
from collections import defaultdict
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_INPUT  = "claude_cleaned_training_data.json"
DEFAULT_OUTPUT = "claude_cleaned_training_data_v3.json"
RANDOM_SEED    = 42
EN_RATIO       = 0.45   # 45% of instructions will be English

# ── Instructions ──────────────────────────────────────────────────────────────
INSTRUCTIONS_ZH_FREETEXT = [
    "請對以下 Python 程式碼做 code review",
    "請審查以下 Python 程式碼並指出問題",
    "請對以下程式碼提供詳細的 code review",
    "請仔細檢查以下程式碼並提供改進建議",
    "請對以下程式碼進行安全性與品質審查",
]

INSTRUCTIONS_ZH_JSON = [
    "請對以下 Python 程式碼做 code review，以 JSON 格式回覆",
    "請審查以下程式碼，以 JSON 格式列出所有問題，包含嚴重程度與修正方式",
    "請對以下程式碼進行結構化 code review，回覆格式為 JSON",
]

INSTRUCTIONS_EN_FREETEXT = [
    "Please review the following Python code and provide feedback.",
    "Perform a code review on the following Python code.",
    "Review the following code diff and provide constructive feedback.",
    "Analyze the following code change and suggest improvements.",
    "Please review this code and identify any issues or improvements.",
    "Conduct a thorough code review of the following Python snippet.",
]

INSTRUCTIONS_EN_JSON = [
    "Please review the following Python code and respond in JSON format with issues, severity, and fixes.",
    "Perform a structured code review of the following code. Return results as JSON.",
    "Review this code and output a JSON object listing all issues with severity ratings.",
]


def is_json_output(text: str) -> bool:
    """Return True if the output is a structured JSON review."""
    try:
        obj = json.loads(text.strip())
        return isinstance(obj, dict) and "issues" in obj
    except (json.JSONDecodeError, TypeError):
        return False


# ── Fix 1: Deduplicate by input, keep longest output ──────────────────────────

def dedup_by_input(data: list) -> tuple[list, int]:
    """
    For each unique input diff, keep only the entry with the longest output.
    Rationale: longer reviews contain more information and train better signal.
    """
    bucket: dict[str, list] = defaultdict(list)
    for entry in data:
        bucket[entry["input"].strip()].append(entry)

    deduped = []
    for entries in bucket.values():
        best = max(entries, key=lambda x: len(x["output"]))
        deduped.append(best)

    removed = len(data) - len(deduped)
    return deduped, removed


# ── Fix 2 & 3: Rewrite instructions with format hints + language diversity ────

def rewrite_instruction(entry: dict, rng: random.Random) -> dict:
    """
    Replace the instruction with an appropriate variant based on:
    - Output format (JSON vs free text)
    - Target language (ZH or EN, weighted by EN_RATIO)
    """
    use_en = rng.random() < EN_RATIO
    json_out = is_json_output(entry["output"])

    if use_en:
        pool = INSTRUCTIONS_EN_JSON if json_out else INSTRUCTIONS_EN_FREETEXT
    else:
        pool = INSTRUCTIONS_ZH_JSON if json_out else INSTRUCTIONS_ZH_FREETEXT

    entry["instruction"] = rng.choice(pool)
    return entry


# ── Main ──────────────────────────────────────────────────────────────────────

def print_report(original: list, final: list, removed_dupes: int) -> None:
    print("\n" + "=" * 58)
    print("  FIX REPORT")
    print("=" * 58)
    print(f"  Original samples         : {len(original):>6}")
    print(f"  Removed (dup inputs)     : -{removed_dupes:>5}")
    print(f"  Final samples            : {len(final):>6}")
    retention = 100 * len(final) / len(original)
    print(f"  Retention rate           : {retention:>5.1f}%")

    # Instruction language split
    zh = sum(1 for d in final if any(c > "\u4e00" for c in d["instruction"]))
    en = len(final) - zh
    print(f"\n  Instruction language split (after fix):")
    print(f"    Chinese : {zh:>5} ({100*zh/len(final):.1f}%)")
    print(f"    English : {en:>5} ({100*en/len(final):.1f}%)")

    # Output format split
    json_count = sum(1 for d in final if is_json_output(d["output"]))
    print(f"\n  Output format split:")
    print(f"    Structured JSON : {json_count:>5} ({100*json_count/len(final):.1f}%)")
    print(f"    Free text       : {len(final)-json_count:>5} ({100*(len(final)-json_count)/len(final):.1f}%)")

    # Output length stats
    out_lens = [len(d["output"]) for d in final]
    print(f"\n  Output length (chars):")
    print(f"    Min    : {min(out_lens)}")
    print(f"    Avg    : {sum(out_lens) // len(out_lens)}")
    print(f"    Median : {sorted(out_lens)[len(out_lens)//2]}")
    print(f"    Max    : {max(out_lens)}")
    print("=" * 58 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Fix remaining dataset issues after claude_clean.py")
    parser.add_argument("--input",    default=DEFAULT_INPUT,  help="Input JSON file")
    parser.add_argument("--output",   default=DEFAULT_OUTPUT, help="Output JSON file")
    parser.add_argument("--no-dedup", action="store_true",    help="Skip deduplication step")
    parser.add_argument("--no-en",    action="store_true",    help="Skip English instruction injection")
    parser.add_argument("--dry-run",  action="store_true",    help="Report only, don't save")
    parser.add_argument("--seed",     type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # Load
    src = Path(args.input)
    if not src.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")
    with open(src, encoding="utf-8") as f:
        data = json.load(f)
    original = list(data)
    print(f"Loaded {len(data)} samples from {args.input}")

    # Fix 1: Deduplication
    removed_dupes = 0
    if not args.no_dedup:
        data, removed_dupes = dedup_by_input(data)
        print(f"Fix 1 — dedup by input: removed {removed_dupes} → {len(data)} remain")
    else:
        print("Fix 1 — dedup skipped")

    # Fix 2 & 3: Instruction rewrite (format hint + language)
    if not args.no_en:
        data = [rewrite_instruction(d, rng) for d in data]
        zh = sum(1 for d in data if any(c > "\u4e00" for c in d["instruction"]))
        en = len(data) - zh
        print(f"Fix 2+3 — instructions rewritten: {zh} ZH / {en} EN")
    else:
        print("Fix 2+3 — instruction rewrite skipped")

    # Report
    print_report(original, data, removed_dupes)

    if args.dry_run:
        print("[dry-run] No file saved.")
        return

    # Save
    out = Path(args.output)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved → {out.resolve()}")


if __name__ == "__main__":
    main()
