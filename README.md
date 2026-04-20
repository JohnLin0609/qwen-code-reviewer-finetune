# Qwen Code Reviewer Fine-Tune

Fine-tuning **Qwen2.5-Coder-7B-Instruct** via QLoRA to build a security-focused Python code review model. The model takes Python code as input and outputs structured JSON reviews with severity ratings, fix suggestions, and corrected code.

All prompts and outputs are in **Traditional Chinese (繁體中文)**.

## Highlights

- **QLoRA on 8GB VRAM** — 4-bit quantization reduces memory from ~28GB to ~6GB, making fine-tuning feasible on a single laptop GPU
- **Structured JSON output** — Unlike the base model's free-form markdown, the fine-tuned model outputs machine-parseable JSON with `issues[]`, `severity`, `suggestion`, and `fixed_code`
- **Multi-issue detection** — Trained on 103 hand-crafted multi-issue examples to detect 2-5 vulnerabilities per code snippet
- **9-pass data cleaning pipeline** — Bot removal, deduplication, normalization, quality filtering across 2,977 GitHub PR reviews
- **DDP-ready via Accelerate** — Single/multi-GPU with zero code changes

## Results

### Training Metrics

| Metric | Epoch 1 | Epoch 2 | Epoch 3 |
|---|---|---|---|
| Eval Loss | 0.8943 | 0.7491 | **0.7372** |

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
                                     v2 dataset (2,683)  ◀── model trained on this
                                              |
                                    fix_dataset.py (post-processing)
                                              |
                                              v
                                     v3 dataset (1,546)  ◀── ready for next run
                                              |
                                    accelerate launch train.py
                                              |
                                              v
                                    code-review-model/
                                      ├── lora/    (~155 MB)
                                      └── merged/  (~15 GB)
```

## Training Data

### v2 — Current (used for model training)

| Source | Count | % |
|---|---|---|
| GitHub inline comments | 2,080 | 77.5% |
| GitHub review bodies | 300 | 11.2% |
| Hand-crafted single-issue | 200 | 7.5% |
| Hand-crafted multi-issue | 103 | 3.8% |
| **Total** | **2,683** | |

### v3 — Post-processed (ready for retraining)

Same 4 sources but after `fix_dataset.py` applies three additional fixes:

| Metric | v2 | v3 |
|---|---|---|
| Total records | 2,683 | **1,546** |
| GitHub inline comments | 2,080 | 1,052 |
| GitHub review bodies | 300 | 191 |
| Hand-crafted single-issue | 200 | 200 |
| Hand-crafted multi-issue | 103 | 103 |
| Chinese instructions | 100% | ~55% |
| English instructions | 0% | ~45% |

**Data sources (20 repos)**: django, flask, fastapi, requests, httpx, cpython, celery, numpy, pandas, scikit-learn, pytorch, transformers, pydantic, sqlalchemy, aiohttp, pytest, starlette, poetry, pip, scrapy

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
| Epochs | 3 |
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

### Training

```bash
# Collect data (needs GITHUB_TOKEN in .env)
python data/fetch_github_pr.py

# Merge datasets
python data/combine_dataset.py

# Clean (9-pass pipeline) — outputs v2
python data/claude_clean.py

# Post-process (dedup + English instructions) — outputs v3
python data/fix_dataset.py

# Train
accelerate launch --config_file accelerate_config.yaml train.py
```

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
├── data/
│   ├── fetch_github_pr.py             # GitHub API data collection (20 repos)
│   ├── combine_dataset.py             # Merge all data sources
│   ├── claude_clean.py                # 9-pass cleaning pipeline → v2
│   ├── fix_dataset.py                 # Post-processing (dedup + EN) → v3
│   ├── checking_dataset.py            # Dataset quality inspector
│   └── review_result.py               # Light cleaning (v1, superseded)
├── eval/
│   ├── metrics.py                     # Quantitative evaluation
│   ├── compare-0.txt                  # Saved qualitative comparison
│   └── results/
│       ├── instruct_responses.json    # CyberSecEval ICD raw responses (1,916)
│       └── instruct_stats.json        # CyberSecEval ICD per-language stats
├── training_data/
│   ├── code_review_training_data.json # 200 hand-crafted single-issue examples
│   └── multi_issue_training_data.json # 103 hand-crafted multi-issue examples
└── code-review-model/                 # (generated, gitignored)
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
