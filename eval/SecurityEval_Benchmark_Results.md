# SecurityEval Benchmark Results

**Model:** Fine-tuned Qwen2.5-Coder-7B-Instruct (QLoRA)  
**Dataset:** [SecurityEval](https://github.com/s2e-lab/SecurityEval) (121 vulnerable Python samples, 69 CWE types)  
**Date:** 2026-04-08  
**Training Data:** `claude_cleaned_training_data.json` (2,683 records, pre-v4)

---

## Summary

| Metric | Value |
|---|---|
| Detection Rate (Recall) | **100.0%** (121/121) |
| CWE Accuracy | **92.6%** (112/121 correct type) |
| CWE Coverage | **65/69** (94.2%) |
| Schema Conformance | **99.2%** |
| Severity-Weighted Recall | **0.964** |
| Precision | **0.890** |
| F1 Score | **0.942** |
| False Positive Rate | 100.0% (15/15 clean samples flagged) |
| Youden Index (TPR - FPR) | 0.000 |

### Severity Distribution

| Severity | Count |
|---|---|
| 高 (High) | 110 |
| 中 (Medium) | 10 |
| 低 (Low) | 0 |

---

## Per-CWE Breakdown

| CWE | Samples | Detected | Correct Type | Detection Rate | CWE Accuracy |
|---|---|---|---|---|---|
| CWE-020 (Input Validation) | 6 | 6 | 4 | 100% | 67% |
| CWE-611 (XXE) | 6 | 6 | 6 | 100% | 100% |
| CWE-601 (Open Redirect) | 5 | 5 | 5 | 100% | 100% |
| CWE-022 (Path Traversal) | 4 | 4 | 4 | 100% | 100% |
| CWE-327 (Broken Crypto) | 4 | 4 | 3 | 100% | 75% |
| CWE-502 (Insecure Deserialization) | 4 | 4 | 4 | 100% | 100% |
| CWE-079 (XSS) | 3 | 3 | 3 | 100% | 100% |
| CWE-094 (Code Injection) | 3 | 3 | 3 | 100% | 100% |
| CWE-117 (Log Injection) | 3 | 3 | 3 | 100% | 100% |
| CWE-295 (Certificate Validation) | 3 | 3 | 3 | 100% | 100% |
| CWE-347 (Signature Verification) | 3 | 3 | 3 | 100% | 100% |
| CWE-703 (Improper Error Handling) | 3 | 3 | 3 | 100% | 100% |
| CWE-730 (ReDoS) | 3 | 3 | 0 | 100% | 0% |
| CWE-078 (Command Injection) | 2 | 2 | 2 | 100% | 100% |
| CWE-089 (SQL Injection) | 2 | 2 | 2 | 100% | 100% |
| CWE-090 (LDAP Injection) | 2 | 2 | 2 | 100% | 100% |
| CWE-113 (HTTP Response Splitting) | 2 | 2 | 2 | 100% | 100% |
| CWE-116 (Improper Encoding) | 2 | 2 | 2 | 100% | 100% |
| CWE-259 (Hardcoded Password) | 2 | 2 | 2 | 100% | 100% |
| CWE-319 (Cleartext Transmission) | 2 | 2 | 2 | 100% | 100% |
| CWE-321 (Hard-coded Crypto Key) | 2 | 2 | 2 | 100% | 100% |
| CWE-326 (Weak Encryption) | 2 | 2 | 2 | 100% | 100% |
| CWE-434 (Unrestricted Upload) | 2 | 2 | 2 | 100% | 100% |
| CWE-521 (Weak Password Requirements) | 2 | 2 | 2 | 100% | 100% |
| CWE-522 (Insufficiently Protected Credentials) | 2 | 2 | 2 | 100% | 100% |
| CWE-643 (XPath Injection) | 2 | 2 | 2 | 100% | 100% |
| CWE-798 (Hardcoded Credentials) | 2 | 2 | 2 | 100% | 100% |
| CWE-918 (SSRF) | 2 | 2 | 2 | 100% | 100% |
| CWE-080 (XSS in Script) | 1 | 1 | 1 | 100% | 100% |
| CWE-095 (Eval Injection) | 1 | 1 | 1 | 100% | 100% |
| CWE-099 (Resource Injection) | 1 | 1 | 1 | 100% | 100% |
| CWE-193 (Off-by-one) | 1 | 1 | 1 | 100% | 100% |
| CWE-200 (Information Exposure) | 1 | 1 | 0 | 100% | 0% |
| CWE-209 (Error Info Exposure) | 1 | 1 | 1 | 100% | 100% |
| CWE-215 (Debug Info Exposure) | 1 | 1 | 1 | 100% | 100% |
| CWE-250 (Unnecessary Privileges) | 1 | 1 | 1 | 100% | 100% |
| CWE-252 (Unchecked Return) | 1 | 1 | 1 | 100% | 100% |
| CWE-269 (Improper Privilege) | 1 | 1 | 1 | 100% | 100% |
| CWE-283 (Unverified Ownership) | 1 | 1 | 1 | 100% | 100% |
| CWE-285 (Improper Authorization) | 1 | 1 | 1 | 100% | 100% |
| CWE-306 (Missing Authentication) | 1 | 1 | 1 | 100% | 100% |
| CWE-329 (Not Using Random IV) | 1 | 1 | 1 | 100% | 100% |
| CWE-330 (Weak Random) | 1 | 1 | 1 | 100% | 100% |
| CWE-331 (Insufficient Entropy) | 1 | 1 | 1 | 100% | 100% |
| CWE-339 (Small PRNG Seed) | 1 | 1 | 1 | 100% | 100% |
| CWE-367 (TOCTOU Race) | 1 | 1 | 1 | 100% | 100% |
| CWE-377 (Insecure Temp File) | 1 | 1 | 1 | 100% | 100% |
| CWE-379 (Insecure Temp Dir) | 1 | 1 | 1 | 100% | 100% |
| CWE-385 (Covert Timing Channel) | 1 | 1 | 1 | 100% | 100% |
| CWE-400 (Resource Exhaustion) | 1 | 1 | 0 | 100% | 0% |
| CWE-406 (Insufficient Net Traffic Control) | 1 | 1 | 1 | 100% | 100% |
| CWE-414 (Missing Lock Check) | 1 | 1 | 1 | 100% | 100% |
| CWE-425 (Direct Request) | 1 | 1 | 1 | 100% | 100% |
| CWE-454 (External Variable Init) | 1 | 1 | 1 | 100% | 100% |
| CWE-462 (Duplicate Key in List) | 1 | 1 | 1 | 100% | 100% |
| CWE-477 (Obsolete Function) | 1 | 1 | 1 | 100% | 100% |
| CWE-595 (Identity vs Equality) | 1 | 1 | 1 | 100% | 100% |
| CWE-605 (Multiple Binds Same Port) | 1 | 1 | 1 | 100% | 100% |
| CWE-641 (Improper File Restriction) | 1 | 1 | 1 | 100% | 100% |
| CWE-732 (Incorrect Permission) | 1 | 1 | 1 | 100% | 100% |
| CWE-759 (No Salt in Hash) | 1 | 1 | 1 | 100% | 100% |
| CWE-760 (Predictable Salt) | 1 | 1 | 1 | 100% | 100% |
| CWE-776 (XML Entity Expansion) | 1 | 1 | 1 | 100% | 100% |
| CWE-827 (Improper Control of Doc Type) | 1 | 1 | 1 | 100% | 100% |
| CWE-835 (Infinite Loop) | 1 | 1 | 1 | 100% | 100% |
| CWE-841 (Improper Behavioral Workflow) | 1 | 1 | 1 | 100% | 100% |
| CWE-941 (Incorrectly Specified Destination) | 1 | 1 | 1 | 100% | 100% |
| CWE-943 (NoSQL Injection) | 1 | 1 | 0 | 100% | 0% |
| CWE-1204 (Generation of Weak Init Vector) | 1 | 1 | 1 | 100% | 100% |

---

## Misclassified CWE Types (4)

The model detected these vulnerabilities but classified them under the wrong category:

| CWE | Expected Type | Notes |
|---|---|---|
| CWE-200 | Information Exposure | Model found the issue but didn't use expected keywords |
| CWE-400 | Resource Exhaustion | Detected as general code quality issue |
| CWE-730 | ReDoS | Detected regex issue but missed denial-of-service aspect |
| CWE-943 | NoSQL Injection | Detected injection but classified generically |

---

## False Positive Analysis

15 clean Python code samples were tested (safe SQL, safe file handling, safe subprocess, safe crypto, safe serialization, safe path, safe password, safe logging, safe XML, safe redirect, safe YAML, safe input validation, safe regex, safe JWT, safe temp file).

**Result: 100% false positive rate** — the model flagged all 15 clean samples as having issues.

**Root cause:** The training data contains zero "clean code" negative examples. The model learned to always find something wrong.

**Expected fix:** The v4 dataset (scheduled for retraining at 17:00 2026-04-08) includes 60 clean-code negative examples with `{"issues": []}` output.

---

## Baseline Comparison

The base Qwen2.5-Coder-7B-Instruct (no fine-tuning) could not be evaluated — **CUDA OOM** on 8GB VRAM (RTX 2000 Ada). The base model without LoRA requires more memory than the fine-tuned LoRA adapter.

To obtain baseline numbers, run on a machine with >= 12GB VRAM:
```bash
python eval/benchmark.py --output results/security_eval_baseline.json
```

---

## Test Configuration

| Parameter | Value |
|---|---|
| Benchmark Dataset | SecurityEval (s2e-lab/SecurityEval) |
| Vulnerable Samples | 121 |
| Clean Samples | 15 |
| CWE Types | 69 |
| Max New Tokens | 512 |
| Temperature | greedy (do_sample=False) |
| Quantization | 4-bit NF4 (BitsAndBytes) |
| GPU | NVIDIA RTX 2000 Ada (8GB VRAM) |

---

## How to Reproduce

```bash
# Full benchmark (fine-tuned only)
python eval/benchmark.py --skip-baselines --output results/security_eval.json

# With base model baseline (needs >= 12GB VRAM)
python eval/benchmark.py --output results/security_eval.json

# Quick test (first 10 samples)
python eval/benchmark.py --skip-baselines --n 10 --output results/security_eval_test.json
```
