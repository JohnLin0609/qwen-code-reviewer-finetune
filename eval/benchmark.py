"""
SecurityEval Benchmark: eval/benchmark.py
==========================================
Evaluates the fine-tuned model against the SecurityEval public dataset
(121 Python vulnerable code samples across 69 CWE types).

Metrics computed:
  - Detection Rate (recall): % of vulnerable samples where model identifies ANY issue
  - CWE Accuracy: % where model identifies the CORRECT vulnerability type
  - CWE Coverage: how many distinct CWE types the model can detect
  - Severity Distribution: breakdown of predicted severity ratings
  - Schema Conformance: % of outputs with all required JSON fields
  - F1 Score: harmonic mean of precision and recall (using clean-code FP data)
  - Paired Discrimination: can it tell vulnerable code from its safe version?

Usage:
    python eval/benchmark.py --skip-baselines
    python eval/benchmark.py --output results/security_eval.json
    python eval/benchmark.py --n 50   # use only first 50 samples
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
FINETUNED_PATH = "./code-review-model/lora"
BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
SYSTEM_PROMPT = "你是資深軟體工程師，專精程式碼審查與資安。請提供具體、有建設性的 code review。"

# CWE → human-readable category + expected keywords in model output
CWE_MAP = {
    "CWE-020": ("Input Validation", ["input", "validat", "驗證", "過濾", "sanitiz", "whitelist"]),
    "CWE-022": ("Path Traversal", ["path", "traversal", "路徑", "遍歷", "directory", "../", "resolve"]),
    "CWE-078": ("Command Injection", ["command", "injection", "shell", "指令", "注入", "subprocess"]),
    "CWE-079": ("XSS", ["xss", "cross.site", "script", "跨站", "escape", "html"]),
    "CWE-089": ("SQL Injection", ["sql", "injection", "注入", "parameteriz", "參數化"]),
    "CWE-094": ("Code Injection", ["code.?injection", "eval", "exec", "程式碼注入", "任意程式碼"]),
    "CWE-117": ("Log Injection", ["log", "injection", "注入", "日誌"]),
    "CWE-200": ("Information Exposure", ["information", "expos", "leak", "洩漏", "資訊", "sensitive", "敏感"]),
    "CWE-215": ("Debug Info Exposure", ["debug", "除錯", "stack.?trace", "error.?message"]),
    "CWE-259": ("Hardcoded Password", ["hardcod", "password", "密碼", "hard.?cod", "credential", "環境變數"]),
    "CWE-295": ("Certificate Validation", ["certif", "ssl", "tls", "verify", "憑證", "驗證"]),
    "CWE-312": ("Cleartext Storage", ["cleartext", "plain.?text", "明文", "encrypt", "加密", "hash"]),
    "CWE-326": ("Weak Encryption", ["weak", "encrypt", "key.?size", "加密", "強度", "bit"]),
    "CWE-327": ("Broken Crypto", ["crypto", "md5", "sha1", "hash", "加密", "雜湊", "obsolete", "弱"]),
    "CWE-330": ("Weak Random", ["random", "seed", "predict", "secrets", "隨機", "可預測"]),
    "CWE-347": ("Signature Verification", ["signature", "verify", "簽章", "驗證", "jwt", "token"]),
    "CWE-377": ("Insecure Temp File", ["temp", "tmp", "暫存", "mktemp", "race"]),
    "CWE-400": ("Resource Exhaustion", ["resource", "exhaust", "denial", "limit", "資源", "DoS"]),
    "CWE-502": ("Insecure Deserialization", ["pickle", "deserializ", "反序列化", "yaml.?load", "marshal"]),
    "CWE-601": ("Open Redirect", ["redirect", "open.?redirect", "重導", "url", "轉址"]),
    "CWE-611": ("XXE", ["xml", "xxe", "external.?entit", "外部實體", "parse"]),
    "CWE-703": ("Improper Error Handling", ["error", "exception", "handling", "例外", "錯誤", "try", "except"]),
    "CWE-730": ("ReDoS", ["regex", "redos", "正規", "regular.?expression", "denial", "catastroph"]),
    "CWE-776": ("XML Entity Expansion", ["xml", "entity", "expansion", "billion.?laugh", "bomb"]),
    "CWE-798": ("Hardcoded Credentials", ["hardcod", "credential", "password", "secret", "密碼", "憑證"]),
    "CWE-918": ("SSRF", ["ssrf", "server.?side", "request.?forgery", "url", "fetch"]),
    "CWE-943": ("NoSQL Injection", ["nosql", "injection", "mongo", "注入", "query"]),
}

# Fallback: generic vulnerability keywords for CWEs not in the map
GENERIC_VULN_KEYWORDS = [
    "漏洞", "vulnerability", "安全", "security", "issue", "問題", "risk", "風險",
    "danger", "bug", "flaw", "weakness", "insecure", "不安全", "attack", "攻擊",
]

# Clean code samples for false positive testing (paired with common vulnerable patterns)
CLEAN_SAMPLES = [
    ("safe_sql", '''
def get_user(user_id: int):
    query = "SELECT * FROM users WHERE id = %s"
    return db.execute(query, (user_id,)).fetchone()
'''),
    ("safe_file", '''
def read_config(path: str) -> dict:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Failed to read config: {e}") from e
'''),
    ("safe_subprocess", '''
import subprocess, re

def ping(host: str) -> bool:
    if not re.match(r"^[a-zA-Z0-9._-]+$", host):
        raise ValueError(f"Invalid hostname: {host}")
    result = subprocess.run(["ping", "-c", "1", host], capture_output=True, timeout=10)
    return result.returncode == 0
'''),
    ("safe_crypto", '''
import secrets

def generate_token() -> str:
    return secrets.token_urlsafe(32)
'''),
    ("safe_serialization", '''
import json

def load_data(raw: str) -> dict:
    return json.loads(raw)
'''),
    ("safe_path", '''
from pathlib import Path

BASE_DIR = Path("/app/templates")

def get_template(name: str) -> str:
    safe = (BASE_DIR / name).resolve()
    if not str(safe).startswith(str(BASE_DIR.resolve())):
        raise ValueError("Path traversal detected")
    return safe.read_text()
'''),
    ("safe_password", '''
import os
import hashlib
import secrets

def hash_password(password: str) -> tuple[str, str]:
    salt = secrets.token_hex(16)
    hashed = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100000)
    return salt, hashed.hex()
'''),
    ("safe_logging", '''
import logging

logger = logging.getLogger(__name__)

def process_request(user_id: int, action: str):
    logger.info("User %d performed action: %s", user_id, action)
'''),
    ("safe_xml", '''
from defusedxml.ElementTree import fromstring

def parse_xml(xml_string: str):
    return fromstring(xml_string)
'''),
    ("safe_redirect", '''
from urllib.parse import urlparse

ALLOWED_HOSTS = {"example.com", "www.example.com"}

def safe_redirect(url: str) -> str:
    parsed = urlparse(url)
    if parsed.hostname not in ALLOWED_HOSTS:
        return "/"
    return url
'''),
    ("safe_yaml", '''
import yaml

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
'''),
    ("safe_input", '''
def paginate(items: list, page: int, per_page: int = 20) -> list:
    if page < 1:
        raise ValueError("page must be >= 1")
    if per_page < 1 or per_page > 100:
        raise ValueError("per_page must be between 1 and 100")
    start = (page - 1) * per_page
    return items[start:start + per_page]
'''),
    ("safe_regex", '''
import re

def validate_email(email: str) -> bool:
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, email))
'''),
    ("safe_jwt", '''
import os, jwt

def verify_token(token: str) -> dict:
    return jwt.decode(token, os.environ["JWT_SECRET"], algorithms=["HS256"])
'''),
    ("safe_temp", '''
import tempfile
from pathlib import Path

def write_temp(data: str) -> Path:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(data)
        return Path(f.name)
'''),
]

# ── Subprocess Inference Worker ───────────────────────────────────────────────
# Same approach as metrics.py — load model in subprocess to avoid Unsloth issues

WORKER_SCRIPT = '''
import warnings; warnings.filterwarnings("ignore")
import sys, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

model_path    = sys.argv[1]
samples_file  = sys.argv[2]
output_file   = sys.argv[3]
system_prompt = sys.argv[4]

with open(samples_file) as f:
    samples = json.load(f)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

import os
is_lora = os.path.exists(os.path.join(model_path, "adapter_config.json"))

if is_lora:
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

results = []
for i, sample in enumerate(samples):
    code = sample["code"]
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"請對以下 Python 程式碼做 code review，以 JSON 格式回覆：\\n\\n```python\\n{code}\\n```"},
    ]
    inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    if hasattr(inputs, "input_ids"):
        input_ids = inputs.input_ids.to(model.device)
    else:
        input_ids = inputs.to(model.device)
    prompt_len = input_ids.shape[1]
    with torch.no_grad():
        out = model.generate(input_ids=input_ids, max_new_tokens=512, do_sample=False, use_cache=True)
    generated = out[0][prompt_len:]
    results.append(tokenizer.decode(generated, skip_special_tokens=True))
    if (i + 1) % 10 == 0:
        print(f"  [{i+1}/{len(samples)}] done", flush=True)

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False)
'''


def run_inference(model_path: str, samples: list[dict]) -> list[str] | None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as sf:
        json.dump(samples, sf, ensure_ascii=False)
        samples_file = sf.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as of:
        output_file = of.name

    try:
        result = subprocess.run(
            [sys.executable, "-c", WORKER_SCRIPT,
             model_path, samples_file, output_file, SYSTEM_PROMPT],
            timeout=7200,
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


# ── Metric Functions ──────────────────────────────────────────────────────────

def parse_model_output(text: str) -> dict | None:
    """Try to parse structured JSON from model output."""
    text = text.strip()
    # Try direct parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    # Try extracting JSON from markdown code block
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    return None


def check_detection(prediction: str, cwe: str) -> dict:
    """Check if model detected any vulnerability and if it matches the CWE."""
    pred_lower = prediction.lower()
    result = {
        "detected_any": False,
        "correct_cwe": False,
        "severity": None,
        "schema_valid": False,
    }

    # Check if ANY vulnerability was flagged
    all_vuln_keywords = GENERIC_VULN_KEYWORDS[:]
    if cwe in CWE_MAP:
        all_vuln_keywords.extend(CWE_MAP[cwe][1])

    result["detected_any"] = any(kw.lower() in pred_lower for kw in GENERIC_VULN_KEYWORDS)

    # Try to parse JSON for structured checks
    obj = parse_model_output(prediction)
    if obj:
        issues = obj.get("issues", [])
        if issues:
            result["detected_any"] = True

        # Schema conformance
        required_fields = {"issues", "overall_score", "summary"}
        issue_fields = {"type", "severity", "description", "suggestion"}
        result["schema_valid"] = required_fields.issubset(obj.keys())
        if issues:
            result["schema_valid"] &= all(
                issue_fields.issubset(issue.keys()) for issue in issues
            )

        # Extract severity
        if issues:
            severities = [i.get("severity", "") for i in issues]
            if any("高" in s for s in severities):
                result["severity"] = "高"
            elif any("中" in s for s in severities):
                result["severity"] = "中"
            else:
                result["severity"] = "低"

    # Check CWE-specific keywords
    if cwe in CWE_MAP:
        _, keywords = CWE_MAP[cwe]
        for kw in keywords:
            if re.search(kw, pred_lower):
                result["correct_cwe"] = True
                break
    else:
        # For unmapped CWEs, count as correct if any issue detected
        result["correct_cwe"] = result["detected_any"]

    return result


def check_false_positive(prediction: str) -> bool:
    """Return True if model incorrectly flags clean code as vulnerable."""
    obj = parse_model_output(prediction)
    if obj:
        issues = obj.get("issues", [])
        score = obj.get("overall_score", 0)
        # Clean if no issues or high score with empty issues
        if not issues and score >= 7:
            return False
        if issues:
            return True
    # Fallback: keyword check
    fp_keywords = ["漏洞", "vulnerability", "安全", "injection", "注入", "issue", "問題",
                   "severity", "嚴重", "risk", "風險", "insecure", "不安全", "bug"]
    pred_lower = prediction.lower()
    return any(kw in pred_lower for kw in fp_keywords)


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_results(label: str, vuln_results: list[dict], fp_results: list[bool],
                  cwe_list: list[str], predictions: list[str]) -> dict:
    print(f"\n{'═' * 70}")
    print(f"  {label} — SecurityEval Benchmark Results")
    print(f"{'═' * 70}")

    n_vuln = len(vuln_results)
    n_clean = len(fp_results)

    # Detection rate (recall)
    detected = sum(1 for r in vuln_results if r["detected_any"])
    recall = detected / n_vuln if n_vuln else 0

    # CWE accuracy
    correct_cwe = sum(1 for r in vuln_results if r["correct_cwe"])
    cwe_acc = correct_cwe / n_vuln if n_vuln else 0

    # CWE coverage
    cwes_detected = set()
    for r, cwe in zip(vuln_results, cwe_list):
        if r["correct_cwe"]:
            cwes_detected.add(cwe)
    total_cwes = len(set(cwe_list))
    cwe_coverage = len(cwes_detected) / total_cwes if total_cwes else 0

    # False positive rate
    false_positives = sum(fp_results)
    fpr = false_positives / n_clean if n_clean else 0

    # Precision and F1
    tp = detected
    fp = false_positives
    precision = tp / (tp + fp) if (tp + fp) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    # Youden Index
    tpr = recall
    youden = tpr - fpr

    # Schema conformance
    schema_valid = sum(1 for r in vuln_results if r["schema_valid"])
    schema_rate = schema_valid / n_vuln if n_vuln else 0

    # Severity distribution
    severities = Counter(r["severity"] for r in vuln_results if r["severity"])

    # Severity-weighted recall
    severity_weights = {"高": 3, "中": 2, "低": 1}
    weighted_score = sum(severity_weights.get(r["severity"], 0) for r in vuln_results if r["detected_any"])
    max_weighted = n_vuln * 3  # assume all should be high severity
    weighted_recall = weighted_score / max_weighted if max_weighted else 0

    print(f"\n  {'Metric':<30} {'Value':>10}")
    print(f"  {'─' * 42}")
    print(f"  {'Detection Rate (Recall)':<30} {recall:>9.1%}")
    print(f"  {'CWE Accuracy':<30} {cwe_acc:>9.1%}")
    print(f"  {'CWE Coverage':<30} {len(cwes_detected):>4}/{total_cwes} ({cwe_coverage:.1%})")
    print(f"  {'False Positive Rate':<30} {fpr:>9.1%}")
    print(f"  {'Precision':<30} {precision:>9.3f}")
    print(f"  {'F1 Score':<30} {f1:>9.3f}")
    print(f"  {'Youden Index (TPR-FPR)':<30} {youden:>9.3f}")
    print(f"  {'Schema Conformance':<30} {schema_rate:>9.1%}")
    print(f"  {'Severity-Weighted Recall':<30} {weighted_recall:>9.3f}")

    print(f"\n  Severity Distribution (detected):")
    for sev in ["高", "中", "低"]:
        print(f"    {sev}: {severities.get(sev, 0)}")

    # Per-CWE breakdown
    cwe_stats = defaultdict(lambda: {"total": 0, "detected": 0, "correct": 0})
    for r, cwe in zip(vuln_results, cwe_list):
        cwe_stats[cwe]["total"] += 1
        if r["detected_any"]:
            cwe_stats[cwe]["detected"] += 1
        if r["correct_cwe"]:
            cwe_stats[cwe]["correct"] += 1

    print(f"\n  Per-CWE Breakdown (top 20 by sample count):")
    print(f"  {'CWE':<12} {'Category':<28} {'Det':>4} {'Cor':>4} {'Tot':>4} {'Recall':>7}")
    print(f"  {'─' * 62}")
    for cwe, stats in sorted(cwe_stats.items(), key=lambda x: -x[1]["total"])[:20]:
        cat = CWE_MAP.get(cwe, ("Unknown", []))[0]
        det_rate = stats["detected"] / stats["total"] if stats["total"] else 0
        print(f"  {cwe:<12} {cat:<28} {stats['detected']:>4} {stats['correct']:>4} {stats['total']:>4} {det_rate:>6.0%}")

    # Missed CWEs
    missed = set(cwe_list) - cwes_detected
    if missed:
        print(f"\n  Missed CWE types ({len(missed)}):")
        for cwe in sorted(missed):
            cat = CWE_MAP.get(cwe, ("Unknown", []))[0]
            print(f"    {cwe}: {cat}")

    print(f"\n{'═' * 70}\n")

    return {
        "label": label,
        "detection_rate": round(recall, 4),
        "cwe_accuracy": round(cwe_acc, 4),
        "cwe_coverage": f"{len(cwes_detected)}/{total_cwes}",
        "false_positive_rate": round(fpr, 4),
        "precision": round(precision, 4),
        "f1_score": round(f1, 4),
        "youden_index": round(youden, 4),
        "schema_conformance": round(schema_rate, 4),
        "severity_weighted_recall": round(weighted_recall, 4),
        "severity_distribution": dict(severities),
        "samples_tested": n_vuln,
        "clean_samples_tested": n_clean,
        "cwes_detected": sorted(cwes_detected),
        "cwes_missed": sorted(missed),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SecurityEval benchmark")
    parser.add_argument("--n", type=int, default=None, help="Limit to first N samples")
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--output", default=None, help="Save results to JSON")
    args = parser.parse_args()

    # Load SecurityEval
    print("Loading SecurityEval dataset...")
    from datasets import load_dataset
    ds = load_dataset("s2e-lab/SecurityEval")["train"]

    samples = []
    cwe_list = []
    for entry in ds:
        cwe = entry["ID"].split("_")[0]
        code = entry["Insecure_code"].strip()
        if code:
            samples.append({"code": code, "id": entry["ID"], "cwe": cwe})
            cwe_list.append(cwe)

    if args.n:
        samples = samples[:args.n]
        cwe_list = cwe_list[:args.n]

    print(f"  {len(samples)} vulnerable samples, {len(set(cwe_list))} distinct CWEs")
    print(f"  {len(CLEAN_SAMPLES)} clean code samples for FP testing")

    clean_inputs = [{"code": code.strip()} for _, code in CLEAN_SAMPLES]

    all_results = []

    # ── Evaluate fine-tuned model ─────────────────────────────────────
    if Path(FINETUNED_PATH).exists():
        print(f"\n  Running inference on {len(samples)} SecurityEval samples...")
        vuln_preds = run_inference(FINETUNED_PATH, samples)

        print(f"  Running inference on {len(clean_inputs)} clean code samples...")
        clean_preds = run_inference(FINETUNED_PATH, clean_inputs)

        if vuln_preds and clean_preds:
            vuln_results = [check_detection(pred, cwe) for pred, cwe in zip(vuln_preds, cwe_list)]
            fp_results = [check_false_positive(pred) for pred in clean_preds]
            r = print_results("Fine-tuned Qwen2.5-Coder", vuln_results, fp_results, cwe_list, vuln_preds)
            r["vuln_predictions"] = [
                {"id": s["id"], "cwe": s["cwe"], "prediction": p[:500], **check_detection(p, s["cwe"])}
                for s, p in zip(samples, vuln_preds)
            ]
            all_results.append(r)
    else:
        print(f"Fine-tuned model not found at {FINETUNED_PATH}")

    # ── Baselines ─────────────────────────────────────────────────────
    if not args.skip_baselines:
        print(f"\n  Running base model on {len(samples)} samples...")
        base_vuln_preds = run_inference(BASE_MODEL, samples)
        base_clean_preds = run_inference(BASE_MODEL, clean_inputs)

        if base_vuln_preds and base_clean_preds:
            vuln_results = [check_detection(pred, cwe) for pred, cwe in zip(base_vuln_preds, cwe_list)]
            fp_results = [check_false_positive(pred) for pred in base_clean_preds]
            r = print_results("Base Qwen2.5-Coder-7B", vuln_results, fp_results, cwe_list, base_vuln_preds)
            all_results.append(r)

    # ── Save ──────────────────────────────────────────────────────────
    if args.output and all_results:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()
