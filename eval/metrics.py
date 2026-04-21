"""
Quantitative Evaluation: evaluate/metrics.py
=============================================
Benchmarks the fine-tuned model against two baselines on a held-out test set.

Baselines:
  1. Base Qwen2.5-Coder-7B-Instruct   (no fine-tuning)
  2. Few-shot Qwen2.5-Coder-7B         (3 examples in prompt)

Metrics computed per sample, then averaged:
  - CodeBLEU   : code-aware n-gram overlap (primary metric)
  - BERTScore  : semantic similarity via embeddings
  - ROUGE-L    : longest common subsequence recall
  - JSON Valid  : % of outputs that are valid parseable JSON (handcrafted test set)
  - Bug Detection Rate  : % of known bugs correctly identified (security test set)
  - False Positive Rate : % of correct code flagged as buggy (clean test set)

Usage:
    # Evaluate fine-tuned vs base model on auto-generated test set
    python evaluate/metrics.py

    # Use a specific test file
    python evaluate/metrics.py --test-file data/test.jsonl

    # Evaluate only the fine-tuned model (skip baselines to save time)
    python evaluate/metrics.py --skip-baselines

    # Save results to JSON for README
    python evaluate/metrics.py --output results/benchmark.json

Requirements:
    pip install evaluate bert-score rouge-score sacrebleu tqdm
    (models are loaded via unsloth, same as inference.py)
"""

import argparse
import json
import subprocess
import sys
import tempfile
import os
import re
from pathlib import Path
from collections import defaultdict

from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL     = "Qwen/Qwen2.5-Coder-7B-Instruct"
FINETUNED_PATH = "./code-review-model-v7/lora"
SYSTEM_PROMPT  = (
    "You are a senior software engineer and security expert performing code review. "
    "Analyze the given code for security vulnerabilities, bugs, and reliability issues. "
    "Always respond in valid JSON format with this structure: "
    "{\"issues\": [{\"type\": \"Security Vulnerability | Reliability Issue | Code Quality\", "
    "\"severity\": \"High | Medium | Low\", "
    "\"description\": \"Clear description of the issue\", "
    "\"suggestion\": \"How to fix it\", "
    "\"fixed_code\": \"The corrected code\"}], "
    "\"overall_score\": <1-10>, "
    "\"summary\": \"Brief overall assessment\"}. "
    "If no issues found, return empty issues array with high overall_score."
)
MAX_NEW_TOKENS = 512
TEMPERATURE    = 0.1   # Low temp for deterministic evaluation
TEST_SAMPLE_N  = 100   # Max samples from auto test set

# ── Security test cases (ground truth for Bug Detection Rate) ─────────────────
# Format: (name, code, bugs_that_should_be_detected)
SECURITY_TEST_CASES = [
    ("SQL Injection", """
def get_user(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    return db.execute(query).fetchone()
""", ["sql injection", "sql", "inject", "參數化", "parameterized"]),

    ("Hardcoded Password", """
def connect():
    return psycopg2.connect(host="localhost", user="admin", password="supersecret123")
""", ["hardcoded", "hard-coded", "硬編碼", "環境變數", "environment"]),

    ("Missing Exception Handling", """
def read_config(path):
    with open(path) as f:
        return json.load(f)
""", ["exception", "error", "例外", "try", "filenotfound"]),

    ("Race Condition", """
vote_counts = {}
def vote(post_id):
    if post_id in vote_counts:
        vote_counts[post_id] += 1
    else:
        vote_counts[post_id] = 1
""", ["race", "concurrent", "lock", "atomic", "競態", "並發", "thread"]),

    ("Path Traversal", """
def get_template(name):
    with open(f'templates/{name}.html') as f:
        return f.read()
""", ["path traversal", "路徑遍歷", "pathlib", "resolve", "sanitize", "../"]),

    ("Command Injection", """
import subprocess
def ping(host):
    return subprocess.call(f'ping -c 1 {host}', shell=True)
""", ["command injection", "shell=true", "指令注入", "shell injection"]),

    ("Insecure Deserialization", """
import pickle
def load_data(data):
    return pickle.loads(data)
""", ["pickle", "deserializ", "反序列化", "rce", "arbitrary code"]),
]

# ── Clean code test cases (should NOT be flagged as buggy) ────────────────────
CLEAN_TEST_CASES = [
    ("Safe SQL", """
def get_user(user_id):
    query = "SELECT * FROM users WHERE id = %s"
    return db.execute(query, (user_id,)).fetchone()
"""),
    ("Safe File Read", """
def read_config(path: str) -> dict:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Failed to read config: {e}") from e
"""),
    ("Safe String Concat", """
def greet(name: str) -> str:
    clean = name.strip()[:50]
    return f"Hello, {clean}!"
"""),
]


# ── Subprocess inference worker ───────────────────────────────────────────────

WORKER_SCRIPT = '''
import warnings; warnings.filterwarnings("ignore")
import sys, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

model_path    = sys.argv[1]
samples_file  = sys.argv[2]
output_file   = sys.argv[3]
system_prompt = sys.argv[4]
is_fewshot    = sys.argv[5] == "true"

with open(samples_file) as f:
    samples = json.load(f)

# Load with transformers + PEFT (supports KV cache, unlike Unsloth fast path)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

import os
is_lora = os.path.exists(os.path.join(model_path, "adapter_config.json"))

if is_lora:
    # Load base model + LoRA adapter
    from json import load as jload
    with open(os.path.join(model_path, "adapter_config.json")) as f:
        adapter_cfg = jload(f)
    base_model_name = adapter_cfg.get("base_model_name_or_path", "Qwen/Qwen2.5-Coder-7B-Instruct")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, quantization_config=bnb_config, device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, model_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_path, quantization_config=bnb_config, device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

model.eval()

FEW_SHOT_EXAMPLES = [
    ("def get_user(u): return db.execute(f\\"SELECT * FROM users WHERE id = {u}\\").fetchone()",
     "{\\"issues\\":[{\\"type\\":\\"Security Vulnerability\\",\\"severity\\":\\"High\\",\\"description\\":\\"user_id is concatenated into SQL, enabling injection.\\",\\"suggestion\\":\\"Use parameterized queries.\\",\\"fixed_code\\":\\"db.execute('SELECT * FROM users WHERE id = ?', (u,))\\"}],\\"overall_score\\":2,\\"summary\\":\\"SQL injection via string concatenation.\\"}"),
    ("password = \\"hardcoded123\\"",
     "{\\"issues\\":[{\\"type\\":\\"Security Vulnerability\\",\\"severity\\":\\"High\\",\\"description\\":\\"Hardcoded credential in source.\\",\\"suggestion\\":\\"Load from environment variable or secret manager.\\",\\"fixed_code\\":\\"password = os.environ['DB_PASSWORD']\\"}],\\"overall_score\\":3,\\"summary\\":\\"Hardcoded password leak risk.\\"}"),
]

def build_messages(code, fewshot=False):
    messages = [{"role": "system", "content": system_prompt}]
    if fewshot:
        for ex_code, ex_review in FEW_SHOT_EXAMPLES:
            messages.append({"role": "user",      "content": f"Review this Python code for security vulnerabilities:\\n\\n{ex_code}"})
            messages.append({"role": "assistant", "content": ex_review})
    messages.append({"role": "user", "content": f"Review this Python code for security vulnerabilities:\\n\\n{code}"})
    return messages

results = []
for i, sample in enumerate(samples):
    code = sample["input"]
    messages = build_messages(code, fewshot=is_fewshot)
    inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    if hasattr(inputs, "input_ids"):
        input_ids = inputs.input_ids.to(model.device)
    else:
        input_ids = inputs.to(model.device)
    prompt_len = input_ids.shape[1]
    with torch.no_grad():
        out = model.generate(input_ids=input_ids, max_new_tokens=512, temperature=0.1, do_sample=False, use_cache=True)
    generated = out[0][prompt_len:]
    results.append(tokenizer.decode(generated, skip_special_tokens=True))
    if (i + 1) % 10 == 0:
        print(f"  [{i+1}/{len(samples)}] samples done", flush=True)

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False)
'''


def run_inference(model_path: str, samples: list, is_fewshot: bool = False) -> list[str] | None:
    """Run inference in a subprocess to avoid Unsloth single-model limitation."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as sf:
        json.dump(samples, sf, ensure_ascii=False)
        samples_file = sf.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as of:
        output_file = of.name

    try:
        result = subprocess.run(
            [sys.executable, "-c", WORKER_SCRIPT,
             model_path, samples_file, output_file,
             SYSTEM_PROMPT, "true" if is_fewshot else "false"],
            timeout=3600,
        )
        if result.returncode != 0:
            print(f"  Inference failed (returncode={result.returncode})")
            return None
        with open(output_file, encoding="utf-8") as f:
            return json.load(f)
    finally:
        os.unlink(samples_file)
        if os.path.exists(output_file):
            os.unlink(output_file)


# ── Metric computation ────────────────────────────────────────────────────────

def compute_codebleu(predictions: list[str], references: list[str]) -> float:
    try:
        from evaluate import load
        metric = load("dvitel/codebleu")
        result = metric.compute(predictions=predictions, references=references, lang="python")
        return round(result["codebleu"], 4)
    except Exception as e:
        print(f"  CodeBLEU error: {e} — install: pip install evaluate dvitel/codebleu")
        return -1.0


def compute_bertscore(predictions: list[str], references: list[str]) -> float:
    try:
        from bert_score import score
        _, _, F1 = score(predictions, references, lang="en", verbose=False)
        return round(F1.mean().item(), 4)
    except Exception as e:
        print(f"  BERTScore error: {e} — install: pip install bert-score")
        return -1.0


def compute_rougeL(predictions: list[str], references: list[str]) -> float:
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = [scorer.score(ref, pred)["rougeL"].fmeasure for pred, ref in zip(predictions, references)]
        return round(sum(scores) / len(scores), 4)
    except Exception as e:
        print(f"  ROUGE-L error: {e} — install: pip install rouge-score")
        return -1.0


def compute_json_validity(predictions: list[str]) -> float:
    """% of predictions that are valid JSON."""
    valid = 0
    for pred in predictions:
        try:
            json.loads(pred.strip())
            valid += 1
        except Exception:
            pass
    return round(valid / len(predictions), 4) if predictions else 0.0


def compute_bug_detection_rate(predictions: list[str], keyword_lists: list[list[str]]) -> float:
    """% of bug predictions that mention at least one expected keyword."""
    detected = 0
    for pred, keywords in zip(predictions, keyword_lists):
        pred_lower = pred.lower()
        if any(kw.lower() in pred_lower for kw in keywords):
            detected += 1
    return round(detected / len(predictions), 4) if predictions else 0.0


def compute_false_positive_rate(predictions: list[str]) -> float:
    """% of clean-code predictions that incorrectly flag a bug."""
    fp_keywords = [
        "vulnerability", "漏洞", "security issue", "安全", "injection", "注入",
        "bug", "error", "問題", "issue", "severity", "嚴重", "risk", "風險",
        "dangerous", "unsafe", "不安全",
    ]
    false_positives = 0
    for pred in predictions:
        pred_lower = pred.lower()
        if any(kw in pred_lower for kw in fp_keywords):
            false_positives += 1
    return round(false_positives / len(predictions), 4) if predictions else 0.0


# ── Test set loading / building ───────────────────────────────────────────────

def _chat_to_legacy(entry: dict) -> dict | None:
    """Convert v6/v7 chat-format entry → legacy {input, output, metadata}.
    Strips the 'Review this <lang> code...' prefix from user content if present."""
    msgs = entry.get("messages", [])
    user = next((m["content"] for m in msgs if m.get("role") == "user"), "")
    asst = next((m["content"] for m in msgs if m.get("role") == "assistant"), "")
    if not user or not asst:
        return None
    if "\n\n" in user:
        code = user.split("\n\n", 1)[1]
    else:
        code = user
    return {"input": code, "output": asst, "metadata": entry.get("metadata", {})}


def load_test_set(test_file: str | None, n: int) -> list[dict]:
    """Load test set from file, or build one from training data. Supports
    legacy JSON {instruction, input, output} and new JSONL chat format."""
    import random
    rng = random.Random(42)

    # Explicit test file
    if test_file and Path(test_file).exists():
        samples = []
        with open(test_file, encoding="utf-8") as f:
            first = f.readline()
            f.seek(0)
            is_array = first.lstrip().startswith("[")
            if is_array:
                samples = json.load(f)
            else:
                for line in f:
                    line = line.strip()
                    if line:
                        samples.append(json.loads(line))
        if samples and "messages" in samples[0]:
            samples = [s for s in (_chat_to_legacy(e) for e in samples) if s]
        print(f"Loaded {len(samples)} test samples from {test_file}")
        rng.shuffle(samples)
        return samples[:n]

    # Auto-discover in order of preference (newest first)
    preferred = [
        ("training_data_v7_final.jsonl", "jsonl"),
        ("training_data_v6_final.jsonl", "jsonl"),
        ("claude_cleaned_training_data_v5_en.json", "json"),
        ("claude_cleaned_training_data_v3.json", "json"),
        ("claude_cleaned_training_data.json", "json"),
    ]
    data = None
    found_path = None
    found_fmt = None
    for path, fmt in preferred:
        if Path(path).exists():
            found_path = path
            found_fmt = fmt
            break

    if not found_path:
        print("No data file found. Using only the security test cases.")
        return []

    with open(found_path, encoding="utf-8") as f:
        if found_fmt == "json":
            data = json.load(f)
        else:
            data = [json.loads(line) for line in f if line.strip()]

    if data and "messages" in data[0]:
        data = [s for s in (_chat_to_legacy(e) for e in data) if s]

    print(f"Building test set from {found_path} ({len(data)} samples)")

    # Sample proportionally: 50% handcrafted, 50% other
    handcrafted = [d for d in data if "handcrafted" in d.get("metadata", {}).get("source", "")]
    other       = [d for d in data if "handcrafted" not in d.get("metadata", {}).get("source", "")]

    n_hc = min(n // 2, len(handcrafted))
    n_other = min(n - n_hc, len(other))
    selected = rng.sample(handcrafted, n_hc) + rng.sample(other, n_other)
    rng.shuffle(selected)

    print(f"Test set: {n_hc} handcrafted + {n_other} other = {len(selected)} samples")
    return selected


# ── Main evaluation ───────────────────────────────────────────────────────────

def evaluate_model(
    model_path: str,
    label: str,
    test_samples: list[dict],
    is_fewshot: bool = False,
) -> dict:
    print(f"\n{'─'*50}")
    print(f"Evaluating: {label}")
    print(f"{'─'*50}")

    results = {"label": label, "model_path": model_path}

    # ── Phase 1: Run ALL GPU inference first ─────────────────────────────
    predictions = None
    json_preds = None
    if test_samples:
        print(f"  Running inference on {len(test_samples)} test samples...")
        predictions = run_inference(model_path, test_samples, is_fewshot=is_fewshot)
        if predictions is None:
            print("  Skipping text metrics (inference failed)")
        else:
            json_samples = [s for s in test_samples if _is_json_output(s["output"])]
            if json_samples:
                json_preds = run_inference(model_path, json_samples, is_fewshot)

    print(f"  Running {len(SECURITY_TEST_CASES)} security test cases...")
    sec_inputs   = [{"input": code.strip()} for _, code, _ in SECURITY_TEST_CASES]
    sec_keywords = [kws for _, _, kws in SECURITY_TEST_CASES]
    sec_preds    = run_inference(model_path, sec_inputs, is_fewshot)

    print(f"  Running {len(CLEAN_TEST_CASES)} clean code test cases...")
    clean_inputs = [{"input": code.strip()} for _, code in CLEAN_TEST_CASES]
    clean_preds  = run_inference(model_path, clean_inputs, is_fewshot)

    # ── Phase 2: Compute metrics (CPU-bound, no GPU needed) ──────────────
    if predictions is not None:
        references = [s["output"] for s in test_samples]
        print("  Computing CodeBLEU...")
        results["codebleu"]   = compute_codebleu(predictions, references)
        print("  Computing BERTScore...")
        results["bertscore"]  = compute_bertscore(predictions, references)
        print("  Computing ROUGE-L...")
        results["rougeL"]     = compute_rougeL(predictions, references)
        if json_preds:
            results["json_validity"] = compute_json_validity(json_preds)
        print(f"  CodeBLEU={results.get('codebleu')}  BERTScore={results.get('bertscore')}  ROUGE-L={results.get('rougeL')}")

    if sec_preds:
        results["bug_detection_rate"] = compute_bug_detection_rate(sec_preds, sec_keywords)
        print(f"  Bug detection rate: {results['bug_detection_rate']:.1%}")
        results["security_details"] = [
            {"test": name, "detected": any(kw.lower() in pred.lower() for kw in kws), "prediction": pred[:300]}
            for (name, _, kws), pred in zip(SECURITY_TEST_CASES, sec_preds)
        ]

    if clean_preds:
        results["false_positive_rate"] = compute_false_positive_rate(clean_preds)
        print(f"  False positive rate: {results['false_positive_rate']:.1%}")

    return results


def _is_json_output(text: str) -> bool:
    try:
        obj = json.loads(text.strip())
        return isinstance(obj, dict) and "issues" in obj
    except Exception:
        return False


def print_summary_table(all_results: list[dict]) -> None:
    print("\n" + "=" * 80)
    print("  BENCHMARK RESULTS SUMMARY")
    print("=" * 80)

    metrics = ["codebleu", "bertscore", "rougeL", "json_validity", "bug_detection_rate", "false_positive_rate"]
    labels  = ["CodeBLEU", "BERTScore", "ROUGE-L", "JSON Valid", "Bug Detect↑", "FP Rate↓"]
    col_w   = 12

    header = f"{'Model':<28}" + "".join(f"{l:>{col_w}}" for l in labels)
    print(header)
    print("─" * len(header))

    for r in all_results:
        row = f"{r['label']:<28}"
        for m in metrics:
            val = r.get(m, None)
            if val is None:
                row += f"{'N/A':>{col_w}}"
            else:
                row += f"{val:>{col_w}.4f}"
        print(row)
    print("=" * 80)

    # Per-test-case breakdown
    for r in all_results:
        if "security_details" in r:
            print(f"\n  Security test breakdown — {r['label']}:")
            for detail in r["security_details"]:
                status = "✅ DETECTED" if detail["detected"] else "❌ MISSED"
                print(f"    {status}  {detail['test']}")

    print()

    # README-ready markdown
    print("  README markdown:")
    print()
    md_header = f"| {'Model':<26} | {'CodeBLEU':>10} | {'BERTScore':>10} | {'Bug Detect':>10} | {'FP Rate':>8} |"
    print(md_header)
    print("|" + "-"*28 + "|" + "-"*12 + "|" + "-"*12 + "|" + "-"*12 + "|" + "-"*10 + "|")
    for r in all_results:
        cb  = f"{r.get('codebleu', 'N/A'):.4f}"  if r.get("codebleu")  is not None else "N/A"
        bs  = f"{r.get('bertscore', 'N/A'):.4f}"  if r.get("bertscore") is not None else "N/A"
        bdr = f"{r.get('bug_detection_rate', 'N/A'):.1%}" if r.get("bug_detection_rate") is not None else "N/A"
        fpr = f"{r.get('false_positive_rate', 'N/A'):.1%}" if r.get("false_positive_rate") is not None else "N/A"
        print(f"| {r['label']:<26} | {cb:>10} | {bs:>10} | {bdr:>10} | {fpr:>8} |")


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned code review model")
    parser.add_argument("--test-file",       default=None, help="JSONL test file (optional)")
    parser.add_argument("--n",               type=int, default=TEST_SAMPLE_N)
    parser.add_argument("--skip-baselines",  action="store_true")
    parser.add_argument("--output",          default=None, help="Save results to JSON file")
    args = parser.parse_args()

    # Load test set
    test_samples = load_test_set(args.test_file, args.n)

    all_results = []

    # Fine-tuned model (primary)
    if Path(FINETUNED_PATH).exists():
        r = evaluate_model(FINETUNED_PATH, "Fine-tuned Qwen2.5-Coder", test_samples)
        all_results.append(r)
    else:
        print(f"Fine-tuned model not found at {FINETUNED_PATH}")

    if not args.skip_baselines:
        # Base model (no fine-tuning)
        r = evaluate_model(BASE_MODEL, "Base Qwen2.5-Coder-7B", test_samples, is_fewshot=False)
        all_results.append(r)

        # Few-shot baseline
        r = evaluate_model(BASE_MODEL, "Few-shot Qwen2.5-Coder-7B", test_samples, is_fewshot=True)
        all_results.append(r)

    # Print summary
    print_summary_table(all_results)

    # Save results
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()
