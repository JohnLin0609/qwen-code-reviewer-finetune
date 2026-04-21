"""
Steps 4, 5, 6: Convert translated v5 → v6 chat format with validation.

Step 4 — Alpaca {instruction, input, output} → chat {messages: [...]} with
         a strict JSON-review system prompt.
Step 5 — Validate every entry:
           - output is parseable JSON with required fields (issues, overall_score, summary)
           - no Chinese characters in any assistant content
           - no @mentions
           - no conversational phrases (I think, thanks, LGTM, etc.)
         Entries that fail are dropped and reported.
Step 6 — Save as JSONL (one JSON object per line) for training.
         Print final stats and 3 sample converted entries.

Usage:
    python data/v5_to_v6_chat.py
    python data/v5_to_v6_chat.py --input claude_cleaned_training_data_v5_en.json
    python data/v5_to_v6_chat.py --strict          # fail the whole run if anything fails validation
"""

import json
import re
import argparse
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = str(BASE_DIR / "claude_cleaned_training_data_v5_en.json")
DEFAULT_OUTPUT = str(BASE_DIR / "training_data_v6_final.jsonl")

CJK_RE = re.compile(r"[\u4e00-\u9fff]")
MENTION_RE = re.compile(r"@[A-Za-z0-9_-]+")
CONVERSATIONAL_PATTERNS = [
    r"\bi think\b",
    r"\bi agree\b",
    r"\bi disagree\b",
    r"\bthanks\b",
    r"\bthank you\b",
    r"\blgtm\b",
    r"\bplease\b",   # catches "please review", "please fix" — too chatty
    r"\bsorry\b",
    r"\bgood point\b",
    r"\bgood catch\b",
    r"\bnice\b",
]
CONVERSATIONAL_RE = re.compile("|".join(CONVERSATIONAL_PATTERNS), re.IGNORECASE)

SYSTEM_PROMPT = (
    "You are a senior software engineer and security expert performing code review. "
    "Analyze the given code for security vulnerabilities, bugs, and reliability issues. "
    "Always respond in valid JSON format with this structure: "
    '{"issues": [{"type": "Security Vulnerability | Reliability Issue | Code Quality", '
    '"severity": "High | Medium | Low", "description": "Clear description of the issue", '
    '"suggestion": "How to fix it", "fixed_code": "The corrected code"}], '
    '"overall_score": <1-10>, "summary": "Brief overall assessment"}. '
    "If no issues found, return empty issues array with high overall_score."
)


def to_chat(entry: dict) -> dict:
    """Convert one Alpaca entry → chat-format entry."""
    instruction = entry.get("instruction", "").strip()
    code = entry.get("input", "").strip()
    assistant = entry.get("output", "").strip()
    user = f"{instruction}\n\n{code}" if instruction and code else (instruction or code)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "metadata": entry.get("metadata", {}),
    }


_INDENT_RE = re.compile(r"(?:^|\n)[ \t]*$")


def _is_python_decorator(text: str, match: re.Match) -> bool:
    """Heuristic: treat @word as a Python decorator (not a GitHub mention) if
    it is the first non-whitespace token on its line. Indented decorators
    (inside class methods) count."""
    start = match.start()
    before = text[:start]
    # Match only whitespace back to the preceding newline (or start-of-string)
    return bool(_INDENT_RE.search(before))


def _cjk_outside_code(assistant_json_str: str) -> bool:
    """Allow Chinese inside fixed_code (code comments are fine); flag it
    in any textual field (type, severity, description, suggestion, summary)."""
    try:
        obj = json.loads(assistant_json_str)
    except json.JSONDecodeError:
        return bool(CJK_RE.search(assistant_json_str))

    for issue in obj.get("issues", []):
        for field in ("type", "severity", "description", "suggestion"):
            if CJK_RE.search(str(issue.get(field, ""))):
                return True
    if CJK_RE.search(str(obj.get("summary", ""))):
        return True
    return False


def validate(entry: dict) -> tuple[bool, list]:
    """Validate a chat-format entry. Returns (ok, list of failure reasons)."""
    reasons = []
    assistant_content = ""
    for msg in entry["messages"]:
        if msg["role"] == "assistant":
            assistant_content = msg["content"]

    # 1. Assistant output must be valid JSON with required fields
    try:
        obj = json.loads(assistant_content)
    except json.JSONDecodeError as e:
        reasons.append(f"invalid JSON: {e}")
        return False, reasons

    for field in ("issues", "overall_score", "summary"):
        if field not in obj:
            reasons.append(f"missing field: {field}")
    if not isinstance(obj.get("issues"), list):
        reasons.append("issues must be a list")

    # 2. No Chinese in assistant *text* fields (fixed_code may contain zh-comments)
    if _cjk_outside_code(assistant_content):
        reasons.append("Chinese characters remain in assistant text fields")

    # 3. No GitHub-style @mentions (Python decorators in code are fine)
    # For the assistant, only check text fields (description, suggestion, summary);
    # decorators inside fixed_code strings are code, not mentions.
    def _scan_for_mentions(text: str) -> str | None:
        for m in MENTION_RE.finditer(text):
            token = m.group(0)
            start = m.start()
            boundary_ok = start == 0 or text[start - 1] in " \n\t("
            if not boundary_ok:
                continue
            if _is_python_decorator(text, m):
                continue
            # Technical reference heuristic: decorator-like names (lowercase with
            # underscore) are typically code references, not GitHub usernames.
            body = token[1:]
            if "_" in body and body == body.lower():
                continue
            return token
        return None

    for msg in entry["messages"]:
        text = msg["content"]
        if msg["role"] == "assistant":
            try:
                obj = json.loads(text)
                # Check only text fields, not fixed_code
                text_fields = [str(obj.get("summary", ""))]
                for issue in obj.get("issues", []):
                    for f in ("type", "severity", "description", "suggestion"):
                        text_fields.append(str(issue.get(f, "")))
                combined = "\n".join(text_fields)
                found = _scan_for_mentions(combined)
            except json.JSONDecodeError:
                found = _scan_for_mentions(text)
        else:
            found = _scan_for_mentions(text)
        if found:
            reasons.append(f"@mention in {msg['role']}: {found}")
            break

    # 4. No conversational phrases in assistant content
    if CONVERSATIONAL_RE.search(assistant_content):
        matches = set(m.group(0).lower() for m in CONVERSATIONAL_RE.finditer(assistant_content))
        reasons.append(f"conversational phrase(s) in assistant: {sorted(matches)}")

    return len(reasons) == 0, reasons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--output", default=DEFAULT_OUTPUT)
    ap.add_argument("--strict", action="store_true",
                    help="Exit with error if any entry fails validation")
    args = ap.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise FileNotFoundError(
            f"Not found: {path}\n"
            f"Run `python data/v5_translate.py` first to produce the translated dataset."
        )

    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} entries from {path.name}")

    # ── Step 4: convert ──────────────────────────────────────────────────
    converted = [to_chat(d) for d in data]

    # ── Step 5: validate ─────────────────────────────────────────────────
    passed, failed = [], []
    failure_reasons = Counter()
    for i, entry in enumerate(converted):
        ok, reasons = validate(entry)
        if ok:
            passed.append(entry)
        else:
            failed.append((i, reasons))
            for r in reasons:
                # Deduplicate by reason category (strip parameters)
                category = r.split(":")[0]
                failure_reasons[category] += 1

    print("\n" + "=" * 60)
    print(f"  VALIDATION REPORT")
    print("=" * 60)
    print(f"  Passed: {len(passed)} / {len(converted)} "
          f"({100 * len(passed) / max(len(converted), 1):.1f}%)")
    print(f"  Failed: {len(failed)}")
    if failure_reasons:
        print(f"\n  Failure reasons:")
        for reason, count in failure_reasons.most_common():
            print(f"    {reason:<45} {count:>5}")
    if failed:
        print(f"\n  First 3 failed entries:")
        for i, (idx, reasons) in enumerate(failed[:3]):
            print(f"    [idx={idx}] {reasons}")

    if args.strict and failed:
        raise SystemExit(f"Strict mode: {len(failed)} entries failed validation")

    # ── Step 6: save JSONL + stats + samples ─────────────────────────────
    out = Path(args.output)
    with open(out, "w", encoding="utf-8") as f:
        for entry in passed:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print("\n" + "=" * 60)
    print(f"  FINAL DATASET")
    print("=" * 60)
    print(f"  Saved to: {out.resolve()}")
    print(f"  Total entries: {len(passed)}")

    by_src = Counter(e.get("metadata", {}).get("source", "?") for e in passed)
    print(f"\n  By source:")
    for src, n in by_src.most_common():
        print(f"    {src:<35} {n:>5}")

    by_lang = Counter(e.get("metadata", {}).get("language", "python") for e in passed)
    print(f"\n  By language:")
    for lang, n in by_lang.most_common():
        print(f"    {lang:<15} {n:>5}")

    print(f"\n  Sample entries (first 3):")
    for i, entry in enumerate(passed[:3], 1):
        print(f"\n  ── Sample {i} ──")
        for msg in entry["messages"]:
            content = msg["content"]
            preview = content if len(content) <= 200 else content[:200] + "…"
            print(f"    {msg['role']:<10}: {preview}")
        print(f"    metadata  : {entry.get('metadata', {})}")


if __name__ == "__main__":
    main()
