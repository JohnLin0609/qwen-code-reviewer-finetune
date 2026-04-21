"""
Apply the offline ZH→EN translation dictionaries (Part 1 + Part 2) to v5.

This replaces matching Chinese strings in the output JSON with their English
translations. Fields that don't have a translation entry are left unchanged
and reported so they can be handled next.

Steps:
  1. Load v5, drop github_inline_comment.
  2. For each Chinese-output entry, parse the JSON, translate:
     - summary (Part 1 SUMMARIES dict)
     - issues[].type  (Part 1 TYPES dict)
     - issues[].severity (Part 1 SEVERITIES dict)
     - issues[].suggestion (Part 2 SUGGESTIONS dict)
     - issues[].description (no dict yet — left as-is, reported)
     Re-serialize and replace.
  3. Write the partially-translated result to v5_en_partial.json.
  4. Print a coverage report and dump untranslated descriptions to
     translations/untranslated_descriptions.txt for the next batch.
"""

import json
import re
import sys
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "translations"))

from translations_part1 import TYPES, SEVERITIES, SUMMARIES                 # type: ignore
from translations_part2_suggestions import SUGGESTIONS                       # type: ignore
try:
    from translations_part3_descriptions import DESCRIPTIONS                 # type: ignore
except ImportError:
    DESCRIPTIONS = {}
try:
    from translations_part4_final import INSTRUCTIONS, CODE_COMMENT_SNIPPETS  # type: ignore
except ImportError:
    INSTRUCTIONS, CODE_COMMENT_SNIPPETS = {}, {}

INPUT = BASE_DIR / "claude_cleaned_training_data_v5.json"
OUTPUT = BASE_DIR / "claude_cleaned_training_data_v5_en_partial.json"
UNTRANSLATED_OUT = BASE_DIR / "translations" / "untranslated_descriptions.txt"

CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def translate_field(value, dict_, stats, field_name):
    """Translate a field using the dictionary. Track coverage."""
    if not isinstance(value, str) or not value:
        return value
    if not CJK_RE.search(value):
        # Already English or contains no Chinese
        stats[f"{field_name}.already_en"] += 1
        return value
    if value in dict_:
        stats[f"{field_name}.translated"] += 1
        return dict_[value]
    stats[f"{field_name}.MISSING"] += 1
    return value  # leave as-is


def translate_code_comments(code: str, stats: Counter) -> str:
    """Find Chinese runs inside code (typically comments) and translate them
    using CODE_COMMENT_SNIPPETS. Longest keys first so prefixes do not shadow
    longer, more-specific translations."""
    if not isinstance(code, str) or not code:
        return code
    out = code
    for zh in sorted(CODE_COMMENT_SNIPPETS, key=len, reverse=True):
        if zh in out:
            out = out.replace(zh, CODE_COMMENT_SNIPPETS[zh])
            stats["fixed_code.snippet_translated"] += 1
    return out


def translate_entry(output_json_str: str, stats: Counter, missing_desc: Counter) -> str:
    try:
        obj = json.loads(output_json_str)
    except json.JSONDecodeError:
        stats["json_parse_error"] += 1
        return output_json_str

    # Translate top-level summary
    if "summary" in obj:
        obj["summary"] = translate_field(obj["summary"], SUMMARIES, stats, "summary")

    # Translate each issue's fields
    for issue in obj.get("issues", []):
        if "type" in issue:
            issue["type"] = translate_field(issue["type"], TYPES, stats, "type")
        if "severity" in issue:
            issue["severity"] = translate_field(issue["severity"], SEVERITIES, stats, "severity")
        if "suggestion" in issue:
            issue["suggestion"] = translate_field(issue["suggestion"], SUGGESTIONS, stats, "suggestion")
        if "description" in issue:
            before = issue["description"]
            issue["description"] = translate_field(before, DESCRIPTIONS, stats, "description")
            if issue["description"] == before and CJK_RE.search(before):
                missing_desc[before] += 1
        # Translate Chinese comments inside fixed_code
        if "fixed_code" in issue:
            issue["fixed_code"] = translate_code_comments(issue["fixed_code"], stats)

    return json.dumps(obj, ensure_ascii=False, indent=2)


def main():
    data = json.loads(INPUT.read_text(encoding="utf-8"))
    print(f"Loaded {len(data)} entries from {INPUT.name}")

    # Step 2: drop conversational/free-form PR sources (they don't fit the target JSON schema)
    before = len(data)
    DROP_SOURCES = {"github_inline_comment", "github_review_body"}
    data = [d for d in data if d.get("metadata", {}).get("source") not in DROP_SOURCES]
    print(f"Dropped {before - len(data)} entries from {sorted(DROP_SOURCES)} → {len(data)} remain")

    stats: Counter = Counter()
    missing_desc: Counter = Counter()

    for d in data:
        # Translate instruction (top-level, not inside output)
        instr = d.get("instruction", "")
        if instr in INSTRUCTIONS:
            d["instruction"] = INSTRUCTIONS[instr]
            stats["instruction.translated"] += 1
        elif CJK_RE.search(instr):
            stats["instruction.MISSING"] += 1

        # Translate Chinese comments in the input code too
        if CJK_RE.search(d.get("input", "")):
            d["input"] = translate_code_comments(d["input"], stats)

        if CJK_RE.search(d.get("output", "")):
            d["output"] = translate_entry(d["output"], stats, missing_desc)

    # ── Coverage report ──
    print("\n" + "=" * 60)
    print("  TRANSLATION COVERAGE (dict hits vs misses)")
    print("=" * 60)
    for key in sorted(stats):
        print(f"  {key:<32} {stats[key]:>6}")

    # Compute per-field coverage %
    print("\n  Per-field translation coverage:")
    for field in ("type", "severity", "summary", "suggestion", "description"):
        hit = stats.get(f"{field}.translated", 0)
        miss = stats.get(f"{field}.MISSING", 0)
        total = hit + miss
        pct = (100 * hit / total) if total else 0.0
        print(f"    {field:<12} {hit:>5}/{total:<5} ({pct:.1f}%)")

    # ── Untranslated descriptions dump ──
    if missing_desc:
        UNTRANSLATED_OUT.parent.mkdir(exist_ok=True)
        with open(UNTRANSLATED_OUT, "w", encoding="utf-8") as f:
            for text, count in missing_desc.most_common():
                f.write(f"{count}\t{text}\n")
        print(f"\n  {len(missing_desc)} unique untranslated descriptions "
              f"→ {UNTRANSLATED_OUT.relative_to(BASE_DIR)}")

    # ── Save partial result ──
    OUTPUT.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved → {OUTPUT.name}")


if __name__ == "__main__":
    main()
