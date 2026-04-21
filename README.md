# Qwen Code Reviewer Fine-Tune

Fine-tuning **Qwen2.5-Coder-7B-Instruct** via QLoRA to build a security-focused multi-language code review model. The model takes a code snippet as input and outputs structured JSON reviews with severity ratings, fix suggestions, and corrected code.

Current dataset (**v7**) covers **7 languages**: Python, JavaScript, Java, C, Go, PHP, Rust. Outputs are in **English**. Earlier versions (v1–v4) were Traditional Chinese; v5 introduced the Chinese → English migration.

> **Note:** v7 training has not yet run. Training-metrics and eval sections below reflect the **v5 model**. A v7 results update will follow after the next training run completes.

## Highlights

- **QLoRA on 8GB VRAM** — 4-bit quantization reduces memory from ~28GB to ~6GB, making fine-tuning feasible on a single laptop GPU
- **Structured JSON output** — Unlike the base model's free-form markdown, the fine-tuned model outputs machine-parseable JSON with `issues[]`, `severity`, `suggestion`, and `fixed_code`
- **Multi-language coverage (v7)** — 700 synthetic examples across 7 programming languages generated with `claude-haiku-4-5`, stratified by severity and vulnerability category
- **15 vulnerability categories** — SQL Injection, XSS, Path Traversal, Hardcoded Credentials, Insecure Random, Race Condition, Authentication Bypass, SSRF, Command Injection, Insecure Deserialization, Missing Rate Limiting, Information Disclosure, Memory Leak, and more
- **Multi-issue detection** — Trained on 103 hand-crafted multi-issue examples to detect 2-5 vulnerabilities per code snippet
- **9-pass data cleaning pipeline** — Bot removal, deduplication, normalization, quality filtering across 2,977 GitHub PR reviews
- **Clean-code examples** — 17.8% of the v7 dataset is clean code with no issues, training the model to avoid false positives
- **DDP-ready via Accelerate** — Single/multi-GPU with zero code changes

## Results

### Training Metrics (v5 model — current checkpoint)

| Metric | Epoch 1 | Epoch 2 | Epoch 3 |
|---|---|---|---|
| Eval Loss | 0.8943 | 0.7491 | **0.7372** |

> v7 training targets 4 epochs (bumped from 3, compensating for the smaller 1,064-entry dataset). Results will be added after the next run.

### CyberSecEval — Insecure Code Detector (Instruct Variant)

Evaluated on Meta's [CyberSecEval](https://github.com/meta-llama/PurpleLlama) ICD benchmark across 8 programming languages (1,916 test cases total). Raw results in [`eval/results/`](eval/results/).

| Language | Test Cases | Pass Rate | Vulnerable % | BLEU |
|---|---:|---:|---:|---:|
| C | 227 | 99.56% | 0.44% | 0.348 |
| C++ | 259 | 100.00% | 0.00% | 0.157 |
| C# | 235 | 100.00% | 0.00% | 0.205 |
| Java | 229 | 99.56% | 0.44% | 0.194 |
| JavaScript | 249 | 100.00% | 0.00% | 0.189 |
| PHP | 162 | 100.00% | 0.00% | 0.162 |
| Python | 351 | 100.00% | 0.00% | 0.138 |
| Rust | 204 | 100.00% | 0.00% | 0.323 |
| **Total** | **1,916** | **~99.9%** | **~0.1%** | **0.215** |

**Only 2 vulnerable responses out of 1,916 test cases** — that's a ~0.1% vulnerable rate across all languages.

#### Important context for interpretation

CyberSecEval ICD is a **code generation** benchmark, but this is a **code review** model. Sample responses reveal the model stayed within its training domain:

```
"Done. I think I've got the point."
"I think you are right, I think I can do this."
"Thanks for the suggestion, I will try it."
```

These are **PR-review-style replies** learned from 2,380 GitHub review comments, not code generation attempts. This means:

- **The 99.9% pass rate is honest but biased** — the model doesn't generate vulnerable code because it largely doesn't generate code at all
- **Low BLEU scores (0.14–0.35)** confirm low similarity to reference code
- **This is a positive signal**: the model did not hallucinate code generation outside its training objective; it correctly identified the task was outside its scope and responded in review-comment style
- **For a fair security evaluation**, a review-oriented benchmark (e.g., asking the model to review code with known CVEs) would be more appropriate — see the qualitative comparison below

### Fine-tuned vs Base Model

<table>
<tr><th>Input</th><th>Fine-tuned Output</th><th>Base Model Output</th></tr>
<tr>
<td>

```python
def get_user(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    return db.execute(query).fetchone()
```

</td>
<td>

Structured JSON with severity rating, specific fix, and corrected code:
```json
{
  "issues": [{
    "type": "安全漏洞",
    "severity": "高",
    "description": "user_id 未參數化，可能被注入。",
    "suggestion": "改用參數化查詢",
    "fixed_code": "..."
  }],
  "overall_score": 2,
  "summary": "存在 SQL Injection 漏洞..."
}
```

</td>
<td>

Free-form markdown, verbose, mixes Simplified/Traditional Chinese, no severity rating, no structured format.

</td>
</tr>
</table>

**Key improvements over base model:**

| Aspect | Fine-tuned | Base (Qwen2.5-Coder-7B) |
|---|---|---|
| Output format | Structured JSON | Free-form markdown |
| Multi-issue detection | 2-5 issues per review | Often misses secondary issues |
| Security focus | Prioritizes vulnerabilities | Buries security among general suggestions |
| Language consistency | Consistent Traditional Chinese | Mixes Simplified/Traditional |
| Actionability | `fixed_code` in every issue | Multiple alternatives, sometimes overwhelming |
| Severity scoring | `overall_score` (1-10) | No scoring |

## Data Pipeline

```
GitHub API (20 repos, 2,977 PRs)     303 Hand-crafted Examples
         |                                    |
         v                                    v
    combine_dataset.py ──────────────> combined (3,177)
                                              |
                                    claude_clean.py (9-pass deep cleaning)
                                              |
                                              v
                                     v2 dataset (2,683)
                                              |
                                    fix_dataset.py (dedup + EN instructions)
                                              |
                                              v
                                     v3 dataset (1,546)
                                              |
                                    v5_translate.py + multilang_examples.py
                                    (Chinese → English, add Java/JS/C/Go)
                                              |
                                              v
                                     v5 dataset  ◀── current trained model (code-review-model-v5/)
                                              |
                                    v5_to_v6_chat.py (chat-format conversion)
                                              |
                                              v
                                     v6 dataset (391 jsonl entries)
                                              |
                                    fix_v6.py (language detection + EN problem_type)
                                              |
                                              v
                                     v6_fixed (391)
                                              |                 +  generate_v7.py
                                              |                    (7 langs × 100 via claude-haiku-4-5)
                                              v                 +  synthetic (~692)
                                    merge_v7.py (combine, dedup, shuffle, validate)
                                              |
                                              v
                                     v7 dataset (1,064)  ◀── ready to train
                                              |
                                    accelerate launch train.py
                                              |
                                              v
                                    code-review-model-v7/
                                      ├── lora/    (~155 MB)
                                      └── merged/  (~15 GB)
```

See [`method.md`](method.md) for a full walkthrough of every step, validation rule, and engineering pattern used.

## Training Data

### v7 — Current target (1,064 entries)

Post-v7 distribution after deduplication, JSON validation, and shuffling (seed 42):

| Source | Count | % |
|---|---:|---:|
| Synthetic (claude-haiku-4-5) | 673 | 63.3% |
| Hand-crafted single-issue | 200 | 18.8% |
| Hand-crafted multi-issue | 103 | 9.7% |
| Synthetic clean (v5) | 60 | 5.6% |
| Hand-crafted multilang | 23 | 2.2% |
| Synthetic clean multilang | 5 | 0.5% |
| **Total** | **1,064** | |

**By language:**

| Language | Count |
|---|---:|
| Python | 466 |
| Java | 105 |
| C | 105 |
| Rust | 98 |
| Go | 96 |
| JavaScript | 95 |
| PHP | 95 |
| C++ | 4 |

**Severity distribution** (across 1,756 total issues): High 37.9%, Medium 38.5%, Low 23.6%.
**Clean-code entries**: 189 (17.8%).
**Unique problem types**: 124.

### Version history

| Version | Records | Key change |
|---|---:|---|
| v2 | 2,683 | First fully cleaned dataset — model trained on this |
| v3 | 1,546 | Dedup by input, English instruction injection |
| v5 | ~1,600 | Chinese → English translation; multi-language expansion — **current trained model** |
| v6 | 391 | Chat-format conversion (messages array); quality filtering |
| v7 | **1,064** | Multi-language synthetic generation (7 langs × 100), metadata enrichment |

**GitHub data sources (20 repos)**: django, flask, fastapi, requests, httpx, cpython, celery, numpy, pandas, scikit-learn, pytorch, transformers, pydantic, sqlalchemy, aiohttp, pytest, starlette, poetry, pip, scrapy

### 9-Pass Cleaning Pipeline (`data/claude_clean.py`)

1. Bot/automated comment removal (sqla-tester, Copilot, Codecov, etc.)
2. Low-quality output removal (< 25 chars, bare URLs, trivial responses)
3. Gerrit boilerplate removal
4. Non-review content removal (welcome messages, PR closings, templates)
5. Weak question-only comments (preserving genuine review questions)
6. Input quality checks (min length, truncation)
7. Output normalization (Gerrit suffix cleanup, whitespace)
8. Deduplication (exact + near-duplicate on first 150 chars of output)
9. Final validation (field completeness, length checks)

### Post-processing (`data/fix_dataset.py`)

Runs after the 9-pass cleaner to fix three remaining issues:

1. **Deduplicate by input** — The 9-pass cleaner dedups by output text, but 542 inputs still appear multiple times with different reviews. Keep only the longest review for each unique input. Removes 1,137 records.
2. **Instruction format hints** — Add JSON-vs-free-text format hints to the instruction so the model knows which output style to use in which context.
3. **Language diversity** — Randomly replace ~45% of instructions with English variants (Chinese-only training generalizes poorly to English prompts).

## Training Configuration

| Parameter | Value |
|---|---|
| Base model | Qwen/Qwen2.5-Coder-7B-Instruct |
| Method | QLoRA (4-bit NF4) |
| Framework | Unsloth + TRL (SFTTrainer) + Accelerate |
| LoRA rank / alpha | 16 / 32 |
| Target modules | q, k, v, o, gate, up, down proj |
| Trainable params | 40.4M / 7.6B (0.53%) |
| Batch size | 1 (x16 gradient accumulation = effective 16) |
| Learning rate | 2e-4 (cosine, 5% warmup) |
| Epochs | 4 (v7) / 3 (v5) |
| Precision | bf16 |
| Optimizer | paged_adamw_32bit |
| Hardware | NVIDIA RTX 2000 Ada Laptop GPU (8 GB) |

### Memory Optimization Stack

- **QLoRA 4-bit** — NF4 quantization via Unsloth
- **Gradient checkpointing** — Unsloth-optimized, trades compute for VRAM
- **Paged AdamW** — Auto-offloads optimizer states to CPU when VRAM is full
- **Gradient accumulation (16)** — Simulates batch size 16 without OOM
- **bf16 mixed precision** — Via Accelerator
- **Pin memory + parallel dataloader** — Reduces CPU-GPU transfer latency

## Quick Start

### Prerequisites

```bash
pip install unsloth trl transformers datasets peft bitsandbytes accelerate
```

### Training (v7)

```bash
# ── Step 1: collect GitHub PR data (needs GITHUB_TOKEN in .env) ──
python data/fetch_github_pr.py

# ── Step 2: merge handcrafted + GitHub sources ──
python data/combine_dataset.py

# ── Step 3: 9-pass cleaning → v2 ──
python data/claude_clean.py

# ── Step 4: post-process (dedup + EN instructions) → v3 ──
python data/fix_dataset.py

# ── Step 5: translate zh → en, add Java/JS/C/Go examples → v5 ──
python data/extract_strings.py
python data/v5_translate.py        # needs ANTHROPIC_API_KEY in .env
python data/apply_translations_v5.py
python data/multilang_examples.py

# ── Step 6: convert to chat format → v6 ──
python data/v5_to_v6_chat.py

# ── Step 7: fix metadata (language detection + EN problem_type) ──
python data/fix_v6.py

# ── Step 8: generate 700 synthetic examples via Haiku ──
python data/generate_v7.py          # ~30–45 min, concurrency=5, resumable

# ── Step 9: merge + dedup + shuffle → v7 ──
python data/merge_v7.py

# ── Train on v7 ──
accelerate launch --config_file accelerate_config.yaml train.py
```

All scripts past the 9-pass cleaner are **resumable** — progress is saved every 50 entries; restart safely if interrupted.

### Inference

```bash
python inference.py
```

### Side-by-side Comparison

```bash
python compare.py
```

## Project Structure

```
├── train.py                           # QLoRA + Unsloth + Accelerate training
├── inference.py                       # Inference testing
├── compare.py                         # Fine-tuned vs base model comparison
├── accelerate_config.yaml             # Accelerate config (single GPU, bf16)
├── method.md                          # Full methodology walkthrough
├── data/
│   ├── fetch_github_pr.py             # GitHub API data collection (20 repos)
│   ├── combine_dataset.py             # Merge all data sources
│   ├── claude_clean.py                # 9-pass cleaning pipeline → v2
│   ├── fix_dataset.py                 # Post-processing (dedup + EN) → v3
│   ├── checking_dataset.py            # Dataset quality inspector
│   ├── review_result.py               # Light cleaning (v1, superseded)
│   ├── extract_strings.py             # Extract Chinese strings for translation
│   ├── v5_translate.py                # zh → en via Claude API
│   ├── apply_translations_v5.py       # Merge translations back into dataset
│   ├── v5_analyze.py                  # v5 dataset analysis
│   ├── multilang_examples.py          # Add Java/JS/C/Go handcrafted examples
│   ├── build_v5_dataset.py            # Final v5 assembly
│   ├── v5_to_v6_chat.py               # Convert to chat-message format → v6
│   ├── fix_v6.py                      # Language detection + metadata translation
│   ├── generate_v7.py                 # Synthetic generation (7 langs × 100 via Haiku)
│   └── merge_v7.py                    # v6_fixed + synthetic → v7 final
├── eval/
│   ├── metrics.py                     # Quantitative evaluation
│   ├── compare-0.txt                  # Saved qualitative comparison
│   └── results/
│       ├── instruct_responses.json    # CyberSecEval ICD raw responses (1,916)
│       └── instruct_stats.json        # CyberSecEval ICD per-language stats
├── training_data/
│   ├── code_review_training_data.json # 200 hand-crafted single-issue examples
│   └── multi_issue_training_data.json # 103 hand-crafted multi-issue examples
├── code-review-model-v5/              # v5 trained model (gitignored)
└── code-review-model-v7/              # v7 trained model (gitignored, pending)
    ├── lora/                          # LoRA adapter weights (~155 MB)
    └── merged/                        # Full merged model (~15 GB)
```

## Tech Stack

| Component | Technology |
|---|---|
| Base Model | Qwen/Qwen2.5-Coder-7B-Instruct |
| Fine-tuning | QLoRA (4-bit) + Unsloth |
| Training | TRL (SFTTrainer) + PEFT + Accelerate |
| Multi-GPU | Accelerate (DDP-ready) |
| Profiling | PyTorch Profiler |
| Hardware | NVIDIA RTX 2000 Ada (8 GB VRAM) |

## License

MIT
