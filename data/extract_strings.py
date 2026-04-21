"""
Extract all unique Chinese strings from v5 into separate files by category,
so we can build a comprehensive translation dictionary.

Output files (under translations/):
  zh_types.txt
  zh_severities.txt
  zh_suggestions.txt    (sorted by frequency)
  zh_summaries.txt      (sorted by frequency)
  zh_descriptions.txt   (sorted by frequency)

Each line is one unique string (tab-separated: count TAB string).
"""

import json
import re
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT = BASE_DIR / "claude_cleaned_training_data_v5.json"
OUT_DIR = BASE_DIR / "translations"
OUT_DIR.mkdir(exist_ok=True)

CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def main():
    data = json.loads(INPUT.read_text(encoding="utf-8"))
    # Step 2: drop github_inline_comment
    data = [d for d in data if d.get("metadata", {}).get("source") != "github_inline_comment"]
    zh = [d for d in data if CJK_RE.search(d.get("output", ""))]

    types = Counter()
    severities = Counter()
    summaries = Counter()
    descriptions = Counter()
    suggestions = Counter()

    for d in zh:
        try:
            o = json.loads(d["output"])
        except json.JSONDecodeError:
            continue
        summaries[o.get("summary", "")] += 1
        for issue in o.get("issues", []):
            types[issue.get("type", "")] += 1
            severities[issue.get("severity", "")] += 1
            descriptions[issue.get("description", "")] += 1
            suggestions[issue.get("suggestion", "")] += 1

    def write(counter: Counter, name: str) -> None:
        path = OUT_DIR / f"zh_{name}.txt"
        with open(path, "w", encoding="utf-8") as f:
            for s, n in counter.most_common():
                if s:
                    f.write(f"{n}\t{s}\n")
        print(f"  {path.name:<24} {len(counter):>4} unique entries")

    print(f"Extracted from {len(zh)} Chinese-output entries:")
    write(types, "types")
    write(severities, "severities")
    write(suggestions, "suggestions")
    write(summaries, "summaries")
    write(descriptions, "descriptions")


if __name__ == "__main__":
    main()
