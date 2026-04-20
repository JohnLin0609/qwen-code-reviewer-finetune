"""
Part B: Evaluate the fine-tuned code review model using PurpleLlama's vulnerable code.

Extracts real vulnerable Python code from PurpleLlama's instruct-v2.json dataset,
feeds each snippet to the model as a code review request, and checks whether the
model correctly identifies the vulnerability type (CWE).

Usage:
    # Make sure the model server is running first:
    #   source .venv/bin/activate && python eval/model_server.py --port 8000

    # Then run this eval (uses the main .venv):
    source .venv/bin/activate
    python eval/purplellama_review_eval.py [OPTIONS]

Options:
    --server-url    Model server URL (default: http://localhost:8000)
    --languages     Comma-separated languages to test (default: python)
    --max-samples   Max samples per CWE (default: 0 = all)
    --output        Output JSON path (default: eval/cyberseceval_results/review_eval.json)
"""

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import openai

# ── CWE keyword mapping ─────────────────────────────────────────────────────
# Maps CWE identifiers to keywords we expect in a good review response.
# Includes both English and Traditional Chinese terms.

CWE_KEYWORDS = {
    "CWE-78": [
        "command injection", "os.system", "os.popen", "subprocess",
        "shell injection", "shell=true", "指令注入", "命令注入",
    ],
    "CWE-89": [
        "sql injection", "sql 注入", "sql注入", "參數化查詢",
        "parameterized", "prepared statement", "f-string", "format string",
        "字串拼接", "string concatenation", "string format",
    ],
    "CWE-94": [
        "code injection", "eval", "exec", "程式碼注入",
        "code execution", "arbitrary code", "動態執行",
    ],
    "CWE-312": [
        "cleartext", "plaintext", "明文", "sensitive data",
        "敏感資料", "encryption", "加密", "unencrypted",
    ],
    "CWE-328": [
        "weak hash", "md5", "sha1", "sha-1",
        "弱雜湊", "弱哈希", "insecure hash", "不安全.*雜湊",
        "不安全.*哈希", "bcrypt", "scrypt", "argon",
    ],
    "CWE-338": [
        "random", "prng", "pseudo-random", "弱隨機",
        "secrets", "insecure random", "不安全.*隨機",
        "可預測", "predictable",
    ],
    "CWE-502": [
        "deserialization", "pickle", "yaml.load", "反序列化",
        "insecure deserialization", "不安全.*反序列化",
        "unpickle", "marshal", "shelve",
    ],
    "CWE-680": [
        "integer overflow", "buffer overflow", "整數溢位",
        "緩衝區溢位", "記憶體",
    ],
    "CWE-798": [
        "hardcoded", "hard-coded", "hard coded", "寫死",
        "硬編碼", "密碼", "password", "credential", "secret",
        "api.key", "api key", "token",
    ],
}


def check_detection(review_text: str, cwe_id: str) -> dict:
    """Check if the review text identifies the expected vulnerability."""
    review_lower = review_text.lower()
    keywords = CWE_KEYWORDS.get(cwe_id, [])

    matched_keywords = []
    for kw in keywords:
        if re.search(kw.lower(), review_lower):
            matched_keywords.append(kw)

    # Also check if the CWE ID itself is mentioned
    cwe_mentioned = cwe_id.lower() in review_lower

    detected = len(matched_keywords) > 0 or cwe_mentioned

    return {
        "detected": detected,
        "cwe_mentioned": cwe_mentioned,
        "matched_keywords": matched_keywords,
    }


def main():
    parser = argparse.ArgumentParser(description="PurpleLlama Code Review Evaluation")
    parser.add_argument("--server-url", default="http://localhost:8000")
    parser.add_argument("--languages", default="python", help="Comma-separated languages")
    parser.add_argument("--max-samples", type=int, default=0, help="Max samples per CWE (0=all)")
    parser.add_argument(
        "--output",
        default="eval/cyberseceval_results/review_eval.json",
    )
    parser.add_argument(
        "--dataset",
        default="PurpleLlama/CybersecurityBenchmarks/datasets/instruct/instruct-v2.json",
    )
    args = parser.parse_args()

    # ── Load dataset ─────────────────────────────────────────────────────────
    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        # Resolve relative to project root
        project_root = Path(__file__).resolve().parent.parent
        dataset_path = project_root / dataset_path

    print(f"Loading dataset from {dataset_path}")
    with open(dataset_path) as f:
        data = json.load(f)

    # Filter by language
    target_langs = {l.strip().lower() for l in args.languages.split(",")}
    filtered = [d for d in data if d.get("language", "").lower() in target_langs]
    print(f"Total entries: {len(data)}, filtered ({args.languages}): {len(filtered)}")

    # Group by CWE and optionally cap per-CWE samples
    by_cwe = defaultdict(list)
    for item in filtered:
        by_cwe[item["cwe_identifier"]].append(item)

    test_cases = []
    for cwe, items in sorted(by_cwe.items()):
        subset = items[:args.max_samples] if args.max_samples > 0 else items
        test_cases.extend(subset)

    print(f"Test cases to evaluate: {len(test_cases)}")
    print(f"CWE distribution: { {k: len(v) for k, v in sorted(by_cwe.items())} }")

    # ── Setup client ─────────────────────────────────────────────────────────
    client = openai.OpenAI(api_key="EMPTY", base_url=f"{args.server_url}/v1/")

    # Verify server
    try:
        client.models.list()
    except Exception as e:
        print(f"ERROR: Cannot connect to model server at {args.server_url}: {e}")
        print("Start the server first: python eval/model_server.py --port 8000")
        sys.exit(1)

    # ── Run evaluations ──────────────────────────────────────────────────────
    results = []
    cwe_stats = defaultdict(lambda: {"total": 0, "detected": 0, "cwe_mentioned": 0})

    print(f"\nStarting evaluation...")
    start_time = time.time()

    for i, case in enumerate(test_cases):
        cwe = case["cwe_identifier"]
        code = case.get("origin_code", "")
        lang = case.get("language", "python")
        desc = case.get("pattern_desc", "")

        if not code.strip():
            continue

        # Build the review prompt (matching the model's training format)
        user_prompt = f"請對以下 {lang.capitalize()} 程式碼做 code review：\n\n```{lang}\n{code}\n```"

        try:
            response = client.chat.completions.create(
                model="qwen2.5-coder-7b-finetuned",
                messages=[
                    {
                        "role": "system",
                        "content": "你是資深軟體工程師，專精程式碼審查與資安。請提供具體、有建設性的 code review。",
                    },
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=1024,
            )
            review_text = response.choices[0].message.content
        except Exception as e:
            print(f"  [{i+1}/{len(test_cases)}] ERROR on {cwe}: {e}")
            review_text = f"ERROR: {e}"

        detection = check_detection(review_text, cwe)
        cwe_stats[cwe]["total"] += 1
        if detection["detected"]:
            cwe_stats[cwe]["detected"] += 1
        if detection["cwe_mentioned"]:
            cwe_stats[cwe]["cwe_mentioned"] += 1

        result = {
            "prompt_id": case.get("prompt_id"),
            "cwe_identifier": cwe,
            "pattern_desc": desc,
            "language": lang,
            "detected": detection["detected"],
            "cwe_mentioned": detection["cwe_mentioned"],
            "matched_keywords": detection["matched_keywords"],
            "review_text": review_text,
            "code_snippet": code[:500],
        }
        results.append(result)

        status = "✓" if detection["detected"] else "✗"
        print(
            f"  [{i+1}/{len(test_cases)}] {status} {cwe} ({desc[:50]})"
            f"  keywords={detection['matched_keywords'][:3]}"
        )

    elapsed = time.time() - start_time

    # ── Compute summary ──────────────────────────────────────────────────────
    total = len(results)
    total_detected = sum(1 for r in results if r["detected"])
    overall_detection_rate = total_detected / total if total else 0

    per_cwe_summary = {}
    for cwe, stats in sorted(cwe_stats.items()):
        rate = stats["detected"] / stats["total"] if stats["total"] else 0
        per_cwe_summary[cwe] = {
            "total": stats["total"],
            "detected": stats["detected"],
            "detection_rate": round(rate, 4),
            "cwe_mentioned": stats["cwe_mentioned"],
        }

    summary = {
        "model": "qwen2.5-coder-7b-finetuned",
        "eval_type": "purplellama_code_review",
        "languages": args.languages,
        "total_cases": total,
        "total_detected": total_detected,
        "overall_detection_rate": round(overall_detection_rate, 4),
        "elapsed_seconds": round(elapsed, 1),
        "per_cwe": per_cwe_summary,
    }

    output_data = {"summary": summary, "results": results}

    # ── Save results ─────────────────────────────────────────────────────────
    output_path = Path(args.output)
    if not output_path.is_absolute():
        project_root = Path(__file__).resolve().parent.parent
        output_path = project_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # ── Print summary ────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  PurpleLlama Code Review Evaluation — Results")
    print(f"{'='*65}")
    print(f"  Model:              qwen2.5-coder-7b-finetuned")
    print(f"  Languages:          {args.languages}")
    print(f"  Total test cases:   {total}")
    print(f"  Vulnerabilities detected: {total_detected}/{total} ({overall_detection_rate:.1%})")
    print(f"  Time elapsed:       {elapsed:.1f}s")
    print(f"")
    print(f"  Per-CWE Detection Rates:")
    print(f"  {'CWE':<12} {'Detected':<12} {'Total':<8} {'Rate':<8}")
    print(f"  {'-'*40}")
    for cwe, stats in sorted(per_cwe_summary.items()):
        print(
            f"  {cwe:<12} {stats['detected']:<12} {stats['total']:<8} {stats['detection_rate']:.1%}"
        )
    print(f"\n  Results saved to: {output_path}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
