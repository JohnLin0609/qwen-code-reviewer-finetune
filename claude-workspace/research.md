# Fine-Tune Project: Detailed Research Report

## Project Overview

This project is an end-to-end pipeline for **fine-tuning Qwen2.5-Coder-7B-Instruct into a specialized Python code review model**. It covers every stage: data collection from GitHub, synthetic data crafting, multi-pass data cleaning, QLoRA fine-tuning with Unsloth, inference, and side-by-side evaluation against the base model.

The entire pipeline is written in Python, uses Traditional Chinese (繁體中文) for all prompts/system messages, and targets a **security-focused code review** use case.

---

## Pipeline Architecture

```
fetch_github_pr.py          code_review_training_data.json (handcrafted)
       |                                  |
       v                                  v
github_pr_training_data.json    combine_dataset.py
       |                                  |
       +----------------------------------+
                      |
                      v
         combined_training_data.json (3,177 records)
                      |
            review_result.py (light cleaning, v1)
                      |
                      v
           cleaned_training_data.json (2,968 records)
                      |
              claude_clean.py (deep 9-pass cleaning)
                      |
                      v
        claude_cleaned_training_data.json (2,580 records)
                      |
                  train.py
                      |
                      v
               code-review-model/
              (lora/ + merged/)
                      |
           +----------+----------+
           |                     |
       inference.py          compare.py
```

---

## Stage 1: Data Collection

### 1a. GitHub PR Scraping (`fetch_github_pr.py`)

- **Target**: 20 major Python open-source repos (Django, Flask, FastAPI, CPython, NumPy, pandas, scikit-learn, PyTorch, Transformers, Pydantic, SQLAlchemy, aiohttp, pytest, Starlette, Poetry, pip, Scrapy, Requests, httpx, Celery)
- **Method**: GitHub REST API v3, authenticating with a personal access token from `.env`
- **Per repo**: Up to 100 closed PRs (sorted by most recently updated)
- **Two data sources per PR**:
  1. **Inline comments** — code-line-level review comments (`GET /pulls/{n}/comments`), filtered to Python files only (`.py`, `.pyi`, `.pyx`, `.pxd`)
  2. **Review bodies** — top-level review text (`GET /pulls/{n}/reviews`), associated with the first Python file's patch as context
- **Filtering at collection time**:
  - Skip non-Python files
  - Skip trivial/empty comments (< 20 chars, exact matches like "lgtm", "+1", "done", etc.)
  - Skip diffs shorter than 10 chars
  - Review bodies require min 50 chars
- **Deduplication**: By comment URL
- **Rate limit handling**: Auto-wait on 403, pre-emptive 30s pause when remaining quota < 50
- **Output**: `github_pr_training_data.json` — **2,977 records** (6.2 MB)
  - 2,388 inline comments + 589 review bodies
  - Top repos: scikit-learn (547), SQLAlchemy (500), Transformers (379), Celery (355)

### 1b. Handcrafted Synthetic Data (`code_review_training_data.json`)

- **200 manually crafted examples** covering specific vulnerability categories
- Each example has a structured JSON output format:
  ```json
  {
    "issues": [{ "type": "安全漏洞", "severity": "高", "description": "...", "suggestion": "...", "fixed_code": "..." }],
    "overall_score": 2,
    "summary": "..."
  }
  ```
- **Problem types covered**: SQL injection, hardcoded credentials, path traversal, race conditions, sensitive data in logs, missing exception handling, SSRF, XSS, insecure deserialization, and more
- These serve as the "gold standard" for the desired output format — structured, actionable, with severity ratings and fix suggestions

---

## Stage 2: Dataset Merging (`combine_dataset.py`)

- Simple concatenation of synthetic (200) + GitHub PR (2,977) data
- Random shuffle
- **Output**: `combined_training_data.json` — **3,177 records** (6.4 MB)

---

## Stage 3: Data Quality Inspection (`checking_dataset.py`)

A diagnostic script (not a pipeline step) that:
- Reports source distribution, input/output length stats
- Flags anomalies: missing fields, outputs < 20 chars, inputs > 2000 chars
- Random-samples 5 entries for manual inspection
- Validates all records have the three required fields (instruction, input, output)

---

## Stage 4: Data Cleaning

### 4a. Light Cleaning v1 (`review_result.py`)

- Removes bot comments (sqla-tester, Codecov, Dependabot, Mergify, etc.)
- Removes empty outputs
- Truncates inputs > 2000 chars
- **Output**: `cleaned_training_data.json` — **2,968 records** (dropped ~209 records)

### 4b. Deep Cleaning v2 (`claude_clean.py`) — The main cleaning pipeline

A **9-pass multi-stage cleaning pipeline**:

| Pass | Name | What it does |
|------|------|-------------|
| 1 | Bot/Automated | Removes sqla-tester, Copilot reviews, Codecov, Dependabot, SonarQube, deploy previews, etc. |
| 2 | Low-quality Output | Removes < 25 char outputs, bare URLs, bare `suggestion` blocks (no explanation), trivial one-word responses ("done", "fixed", "thanks", etc.), commit-hash-only references |
| 3 | Gerrit Boilerplate | Strips Gerrit view links and "Name wrote:" prefixes; removes entries where actual content < 40 chars |
| 4 | Non-review Content | Removes welcome greetings, PR closing messages, unfilled markdown templates, "thank you for your interest" closings |
| 5 | Weak Questions | Removes short questions (< 50 chars ending with "?") that have no review value, while preserving genuine review questions ("shouldn't this...", "why not...", "consider...") |
| 6 | Input Quality | Removes entries with input < 10 chars; truncates inputs > 4000 chars |
| 7 | Output Normalization | Strips Gerrit suffixes, collapses excessive newlines, removes entries that normalize to < 20 chars |
| 8 | Deduplication | Exact dedup on lowercased output text + near-dedup on first 150 chars for short (< 200 char) outputs |
| 9 | Final Validation | Ensures all three fields exist, output >= 20 chars (handcrafted exempted from length check) |

- **Key design decision**: Handcrafted data is exempt from most quality filters (passes 2, 4, 5, 7) since it's already curated and uses a different structured JSON format
- **Output**: `claude_cleaned_training_data.json` — **2,580 records** (4.2 MB)

### Final Dataset Statistics

| Metric | Value |
|--------|-------|
| Total records | 2,580 |
| Sources | github_inline_comment: 2,080 / github_review_body: 300 / handcrafted: 200 |
| Input length | avg: 935, min: 11, max: 4,000 |
| Output length | avg: 272, min: 25, max: 6,701 |

---

## Stage 5: Fine-Tuning (`train.py`)

### Model & Method

| Parameter | Value |
|-----------|-------|
| Base model | `Qwen/Qwen2.5-Coder-7B-Instruct` |
| Quantization | 4-bit (QLoRA via Unsloth) |
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj (all attention + MLP) |
| Trainable params | 40,370,176 / 7,655,986,688 total (0.53%) |
| Max sequence length | 2,048 tokens |

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 3 |
| Batch size | 1 (per device) |
| Gradient accumulation | 16 (effective batch = 16) |
| Learning rate | 2e-4 |
| LR scheduler | Cosine |
| Warmup | 5% of steps |
| Precision | bf16 (auto-detected) |
| Eval strategy | Per epoch |
| Save strategy | Per epoch (keep best 2) |
| Seed | 42 |

### Chat Template Format

```
System: 你是資深軟體工程師，專精程式碼審查與資安。請提供具體、有建設性的 code review。
User: {instruction}：\n\n```python\n{code}\n```
Assistant: {review}
```

### Training Results (from logs)

| Metric | Epoch 1 | Epoch 2 | Epoch 3 |
|--------|---------|---------|---------|
| Eval loss | 0.9048 | 0.7558 | 0.7491 |
| Train loss (avg) | ~0.92 | ~0.56 | ~0.41 |

- **Total steps**: 462
- **Total training time**: 2h 46min (~21.5s/step)
- **Hardware**: NVIDIA RTX 2000 Ada Generation Laptop GPU (8 GB VRAM)
- **Framework**: Unsloth 2026.3.15, Transformers 5.3.0, TRL 0.24.0, PyTorch 2.10.0, CUDA 12.8
- Loss curve shows consistent descent from ~1.98 to ~0.41 with no signs of divergence
- Eval loss improved from 0.9048 -> 0.7558 -> 0.7491, the model at epoch 3 (step 462) was selected as best
- The gap between train and eval loss (0.41 vs 0.75) suggests some overfitting, which is expected with only 2,580 samples

### Model Outputs

- **LoRA adapter**: `code-review-model/lora/` — 155 MB (adapter weights) + 11 MB (tokenizer)
- **Merged full model**: `code-review-model/merged/` — ~15 GB (16-bit merged weights)
- **Checkpoints**: checkpoint-308 (epoch 2), checkpoint-462 (epoch 3)

---

## Stage 6: Inference (`inference.py`)

- Loads the LoRA adapter via Unsloth's `FastLanguageModel`
- 4-bit quantized inference
- Generation params: temperature=0.7, top_p=0.9, max_new_tokens=512
- **Workaround**: `use_cache=False` to avoid an Unsloth KV cache shape mismatch bug
- Tests 3 predefined cases: SQL injection, hardcoded password, missing exception handling

---

## Stage 7: Model Comparison (`compare.py`)

- Runs both the fine-tuned model and the original Qwen2.5-Coder-7B-Instruct on 5 test cases
- Uses **subprocess isolation** to avoid Unsloth's limitation of loading two models in the same process
- Each model runs as a separate Python process that writes results to a temp JSON file
- **5 test cases**: SQL injection, hardcoded password, missing exception handling, race condition, path traversal

### Comparison Results (from `compare-0.txt`)

The fine-tuned model shows clear behavioral changes:

| Aspect | Fine-tuned | Original (base) |
|--------|-----------|-----------------|
| **Output format** | Structured JSON with `issues[]`, `overall_score`, `summary` | Free-form markdown with headers and code blocks |
| **Conciseness** | Compact, focused on the core issue | Verbose, covers many tangential topics |
| **Security focus** | Immediately identifies the primary vulnerability | Identifies security issues but buries them among general suggestions |
| **Actionability** | Includes `fixed_code` in every issue | Provides multiple alternative code snippets, sometimes overwhelming |
| **Language** | Consistent Traditional Chinese | Mix of Traditional Chinese and Simplified Chinese |
| **Scoring** | Provides `overall_score` (1-10) severity rating | No scoring system |

**Notable example — Race condition test**: The fine-tuned model correctly identifies the TOCTOU race condition and suggests threading.Lock, while the original model calls the code "correct" and suggests Counter optimization instead, completely missing the concurrency issue.

**Notable example — Path traversal test**: The fine-tuned model immediately flags the path traversal vulnerability with `pathlib.resolve()` fix, while the original model's first suggestion is generic error handling, mentioning path validation only as a secondary concern.

---

## Key Design Decisions & Specificities

1. **Bilingual Chinese focus**: All system prompts, instructions, and handcrafted data use Traditional Chinese. The model is trained to respond in Traditional Chinese with security terminology.

2. **Structured JSON output for handcrafted data**: The 200 synthetic examples enforce a specific JSON schema (`issues[]` with type/severity/description/suggestion/fixed_code, plus overall_score and summary). This teaches the model a consistent, machine-parseable output format — but only ~7.8% of training data uses this format, while the majority (GitHub data) uses free-form text. The comparison results show the fine-tuned model adopts the JSON format.

3. **Security-first training philosophy**: The handcrafted examples exclusively focus on security vulnerabilities (OWASP-style). The system prompt explicitly mentions "資安" (information security). This biases the model toward security-focused reviews.

4. **QLoRA for memory efficiency**: Using 4-bit quantization + LoRA allows training a 7B parameter model on an 8 GB laptop GPU. The trade-off is slightly lower precision, but this is a practical choice for the hardware available.

5. **Two cleaning pipelines**: The project has both a light cleaner (`review_result.py`) and a deep cleaner (`claude_clean.py`). The deep cleaner was clearly developed later as a more sophisticated replacement. The light cleaner's output (`cleaned_training_data.json`) is not used in the final training — only `claude_cleaned_training_data.json` feeds into `train.py`.

6. **Subprocess-based comparison**: The `compare.py` script works around Unsloth's single-model-per-process limitation by spawning separate Python processes. This is a pragmatic solution to a framework limitation.

7. **Conservative hyperparameters**: The effective batch size of 16, cosine scheduler, and 5% warmup are standard SFT best practices. The 3-epoch training with 2,580 samples is on the light side — the eval loss plateau between epochs 2 and 3 (0.7558 -> 0.7491) suggests diminishing returns.

---

## Data Flow Summary

| File | Records | Size | Role |
|------|---------|------|------|
| `code_review_training_data.json` | 200 | 0.2 MB | Handcrafted structured examples |
| `github_pr_training_data.json` | 2,977 | 6.2 MB | Raw GitHub PR comments |
| `combined_training_data.json` | 3,177 | 6.4 MB | Merged raw data |
| `cleaned_training_data.json` | 2,968 | 4.1 MB | Light-cleaned (v1, unused in final training) |
| `claude_cleaned_training_data.json` | 2,580 | 4.2 MB | Deep-cleaned (v2, used for training) |

Total data reduction: 3,177 -> 2,580 (18.8% removed by deep cleaning)

---

## File Inventory

| File | Purpose |
|------|---------|
| `fetch_github_pr.py` | GitHub API scraper for PR review comments |
| `code_review_training_data.json` | 200 handcrafted security-focused review examples |
| `combine_dataset.py` | Merges synthetic + GitHub data |
| `checking_dataset.py` | Dataset quality inspector / diagnostic tool |
| `review_result.py` | Light data cleaning (v1) |
| `claude_clean.py` | Deep 9-pass data cleaning pipeline (v2) |
| `train.py` | QLoRA fine-tuning with Unsloth + TRL |
| `inference.py` | Single-model inference testing |
| `compare.py` | Side-by-side fine-tuned vs base model evaluation |
| `compare-0.txt` | Saved comparison output |
| `train_logs/1` | Full training console output |
| `.env` | GitHub API token |
| `code-review-model/` | Model output directory (checkpoints, LoRA, merged) |
| `unsloth_compiled_cache/` | Unsloth kernel compilation cache |

---

## Potential Improvements / Observations

1. **Train/eval loss gap**: The ~0.34 gap (0.41 train vs 0.75 eval at epoch 3) suggests mild overfitting. Could benefit from more data, early stopping, or higher dropout.

2. **Format inconsistency in training data**: ~92% of data has free-form text outputs (GitHub comments) while ~8% has structured JSON (handcrafted). Despite this, the fine-tuned model learned to output structured JSON consistently, suggesting the structured format from the handcrafted data dominated the model's learned behavior — likely because those examples are more internally consistent.

3. **Light cleaner is redundant**: `review_result.py` produces `cleaned_training_data.json`, but `claude_clean.py` reads from `combined_training_data.json` directly. The light cleaner's output is unused in the final pipeline.

4. **GitHub token exposed in `.env`**: The `.env` file contains a plaintext GitHub personal access token. This file should be in `.gitignore`.

5. **No quantitative evaluation**: The comparison is qualitative only (human reading `compare-0.txt`). Adding automated metrics (e.g., JSON validity rate, keyword detection accuracy, BLEU/ROUGE against reference reviews) would strengthen evaluation.
