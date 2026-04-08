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

## Pipeline

```
GitHub API (20 repos, 2,977 PRs)     303 Hand-crafted Examples
         |                                    |
         v                                    v
    combine_dataset.py ──────────────> combined (3,177)
                                              |
                                    claude_clean.py (9-pass)
                                              |
                                              v
                                    Training Data (2,683)
                                              |
                                    accelerate launch train.py
                                              |
                                              v
                                    code-review-model/
                                      ├── lora/    (~155 MB)
                                      └── merged/  (~15 GB)
```

## Training Data

| Source | Count | % |
|---|---|---|
| GitHub inline comments | 2,080 | 77.5% |
| GitHub review bodies | 300 | 11.2% |
| Hand-crafted single-issue | 200 | 7.5% |
| Hand-crafted multi-issue | 103 | 3.8% |
| **Total** | **2,683** | |

**Data sources**: django, flask, fastapi, requests, httpx, cpython, celery, numpy, pandas, scikit-learn, pytorch, transformers, pydantic, sqlalchemy, aiohttp, pytest, starlette, poetry, pip, scrapy

### 9-Pass Cleaning Pipeline (`claude_clean.py`)

1. Bot/automated comment removal (sqla-tester, Copilot, Codecov, etc.)
2. Low-quality output removal (< 25 chars, bare URLs, trivial responses)
3. Gerrit boilerplate removal
4. Non-review content removal (welcome messages, PR closings, templates)
5. Weak question-only comments (preserving genuine review questions)
6. Input quality checks (min length, truncation)
7. Output normalization (Gerrit suffix cleanup, whitespace)
8. Deduplication (exact + near-duplicate on first 150 chars)
9. Final validation (field completeness, length checks)

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
python fetch_github_pr.py

# Merge datasets
python combine_dataset.py

# Clean (9-pass pipeline)
python claude_clean.py

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
├── fetch_github_pr.py                 # GitHub API data collection (20 repos)
├── code_review_training_data.json     # 200 hand-crafted single-issue examples
├── multi_issue_training_data.json     # 103 hand-crafted multi-issue examples
├── combine_dataset.py                 # Merge all data sources
├── claude_clean.py                    # 9-pass cleaning pipeline
├── checking_dataset.py                # Dataset quality inspector
├── train.py                           # QLoRA + Unsloth + Accelerate training
├── accelerate_config.yaml             # Accelerate config (single GPU, bf16)
├── inference.py                       # Inference testing
├── compare.py                         # Fine-tuned vs base model comparison
└── code-review-model/
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
