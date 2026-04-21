"""
Steps 1-2: Fix training_data_v6_final.jsonl metadata.

Step 1 — Detect programming language for entries where metadata.language == 'unknown'
         (or missing). Writes the detected language into metadata.language.

Step 2 — Translate metadata.problem_type Chinese labels to English using a mapping.
         For compound labels joined by ' + ', translate each segment individually.
         Segments not in the mapping are kept as-is and printed at the end.

Output: training_data_v6_fixed.jsonl (same directory as the input).
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT = BASE_DIR / "training_data_v6_final.jsonl"
OUTPUT = BASE_DIR / "training_data_v6_fixed.jsonl"

CJK_RE = re.compile(r"[\u4e00-\u9fff]")

PROBLEM_TYPE_MAP = {
    "硬編碼密碼": "Hardcoded Credentials",
    "不安全的亂數生成": "Insecure Random Generation",
    "競態條件": "Race Condition",
    "沒有處理 Exception": "Missing Exception Handling",
    "路徑遍歷漏洞": "Path Traversal",
    "敏感資訊寫入 Log": "Sensitive Data in Logs",
    "記憶體洩漏": "Memory Leak",
    "SQL Injection": "SQL Injection",
    "沒有輸入驗證": "Missing Input Validation",
    "N+1 查詢問題": "N+1 Query Problem",
    "檔名未清理": "Unsanitized Filename",
    "Shell Injection": "Shell Injection",
    "無類型驗證": "Missing Type Validation",
    "路徑洩漏": "Path Disclosure",
    "無大小限制": "Missing Size Limit",
    "無 TTL 限制": "Missing TTL Limit",
    "無認證": "Missing Authentication",
    "無錯誤處理": "Missing Error Handling",
    "任意 JSON 寫入": "Arbitrary JSON Write",
    "過寬權限": "Overly Permissive File Permissions",
}


def has_chinese(text: str) -> bool:
    return bool(CJK_RE.search(text or ""))


def detect_language(code: str) -> str:
    """Heuristic programming-language detection. Order of checks matters —
    more specific patterns first, fallback to python."""
    c = code or ""
    has = lambda s: s in c

    # PHP — look for the opening tag first (very specific)
    if has("<?php"):
        return "php"

    # Rust — fn + let mut is a strong pair signal
    if re.search(r"\bfn\s+\w+\s*\(", c) and "let mut" in c:
        return "rust"

    # Go — func + package together
    if re.search(r"\bpackage\s+\w+", c) and re.search(r"\bfunc\s+\w+\s*\(", c):
        return "go"

    # Java — very specific class signatures
    if "public class" in c or "public static void" in c:
        return "java"

    # C / C++ — #include or int main()
    if "#include" in c or re.search(r"\bint\s+main\s*\(", c):
        # Prefer cpp if we see C++-only markers
        if re.search(r"\bstd::|<iostream>|class\s+\w+\s*\{|using\s+namespace", c):
            return "cpp"
        return "c"

    # JavaScript — multiple possible markers
    if (re.search(r"\bfunction\s+\w+\s*\(", c)
            or re.search(r"\bconst\s+\w+\s*=", c)
            or re.search(r"\blet\s+\w+\s*=", c)
            or re.search(r"\bvar\s+\w+\s*=", c)
            or "=>" in c):
        # Disambiguate from Python (which also uses `let`/`const`? no). But Python uses `def`.
        # If we see `def ` or `import ` this is more likely python, check first below.
        if not re.search(r"^\s*def\s+\w+\s*\(", c, re.M) and not re.search(r"^\s*import\s+\w", c, re.M):
            return "javascript"

    # Python — def / import / print(
    if (re.search(r"^\s*def\s+\w+\s*\(", c, re.M)
            or re.search(r"^\s*import\s+\w", c, re.M)
            or re.search(r"^\s*from\s+\w+\s+import\b", c, re.M)
            or "print(" in c):
        return "python"

    return "python"  # default


def extract_code_from_user_msg(user_content: str) -> str:
    """User messages look like:
        'Review this code ...\n\n<code>'
    or 'Review this <lang> code ...\n\n<code>'
    We keep the whole content — detection heuristics are robust to leading prose."""
    return user_content or ""


def translate_problem_type(pt: str, unmatched: Counter) -> str:
    """Translate problem_type. Split on ' + ' for compound labels,
    map each segment; unknown segments kept as-is and counted."""
    if not pt or not has_chinese(pt):
        return pt
    segments = [s.strip() for s in pt.split(" + ")]
    out_segments = []
    for seg in segments:
        if seg in PROBLEM_TYPE_MAP:
            out_segments.append(PROBLEM_TYPE_MAP[seg])
        elif has_chinese(seg):
            unmatched[seg] += 1
            out_segments.append(seg)  # keep as-is
        else:
            out_segments.append(seg)  # already English (e.g. "SQL Injection")
    return " + ".join(out_segments)


def main():
    if not INPUT.exists():
        print(f"Input not found: {INPUT}", file=sys.stderr)
        sys.exit(1)

    entries = []
    with open(INPUT, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))

    print(f"Loaded {len(entries)} entries from {INPUT.name}")

    # ── STEP 1: detect language ───────────────────────────────────────
    updated_lang = 0
    lang_dist_before = Counter()
    lang_dist_after = Counter()
    for e in entries:
        meta = e.setdefault("metadata", {})
        cur = meta.get("language", "unknown")
        lang_dist_before[cur] += 1
        if cur == "unknown" or not cur:
            # Find the user message (role == 'user')
            user_msg = next(
                (m["content"] for m in e.get("messages", []) if m.get("role") == "user"),
                ""
            )
            code = extract_code_from_user_msg(user_msg)
            detected = detect_language(code)
            meta["language"] = detected
            updated_lang += 1
        lang_dist_after[meta["language"]] += 1

    print(f"\nStep 1 — Language detection:")
    print(f"  Updated {updated_lang} entries (was 'unknown' or missing)")
    print(f"  Language distribution after:")
    for k, v in lang_dist_after.most_common():
        print(f"    {k}: {v}")

    # ── STEP 2: translate problem_type ────────────────────────────────
    unmatched = Counter()
    translated_count = 0
    for e in entries:
        meta = e.setdefault("metadata", {})
        pt = meta.get("problem_type", "")
        if has_chinese(pt):
            new_pt = translate_problem_type(pt, unmatched)
            if new_pt != pt:
                meta["problem_type"] = new_pt
                translated_count += 1

    print(f"\nStep 2 — Problem type translation:")
    print(f"  Translated {translated_count} entries")
    if unmatched:
        print(f"  Untranslated Chinese segments ({len(unmatched)} unique) — add to mapping if needed:")
        for seg, n in unmatched.most_common():
            print(f"    {seg!r} ({n}x)")
    else:
        print("  All Chinese segments matched the mapping.")

    # ── Save ──────────────────────────────────────────────────────────
    with open(OUTPUT, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(entries)} entries → {OUTPUT.name}")


if __name__ == "__main__":
    main()
