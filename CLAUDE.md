# qwen-code-reviewer-finetune — CLAUDE.md

## Project Overview

Fine-tunes **Qwen2.5-Coder-7B-Instruct** via QLoRA to build a security-focused Python
code review model. Input: code snippet or diff. Output: structured JSON review with
severity ratings, fix suggestions, and corrected code.

All prompts and outputs are in **Traditional Chinese (繁體中文)**.

---

## Project Status

### COMPLETED ✅
- [x] `data/fetch_github_pr.py` — 2,977 records from 20 repos
- [x] `training_data/code_review_training_data.json` — 200 hand-crafted single-issue examples
- [x] `training_data/multi_issue_training_data.json` — 103 hand-crafted multi-issue examples
- [x] `data/combine_dataset.py` — merges all sources
- [x] `data/review_result.py` — light cleaning v1 (superseded, kept for reference)
- [x] `data/claude_clean.py` — 9-pass deep cleaning pipeline
- [x] `data/fix_dataset.py` — post-processing: dedup by input + EN instruction injection
- [x] `data/checking_dataset.py` — dataset quality inspector
- [x] `train.py` — QLoRA + Unsloth + Accelerate, 3 epochs completed
- [x] `inference.py` — single-model inference testing
- [x] `compare.py` — fine-tuned vs base model (subprocess isolation)
- [x] `eval/compare-0.txt` — saved qualitative comparison output
- [x] `eval/metrics.py` — quantitative benchmark script (written, not yet run)

### PENDING ❌
- [ ] **Run eval/metrics.py** — get CodeBLEU, BERTScore, Bug Detection Rate numbers
  ```bash
  python eval/metrics.py --skip-baselines --output results/benchmark.json
  python eval/metrics.py --output results/benchmark.json   # full, takes hours
  ```
- [ ] **Retrain on v3 dataset** — update `DATA_PATH` in `train.py` to `claude_cleaned_training_data_v3.json`
  - Only do this after seeing eval results
- [ ] **Write profile.py** — PyTorch Profiler to analyze training bottlenecks
  - Standalone script, NOT part of the training loop
  - Output: `trace.json` viewable at `chrome://tracing`
- [ ] **FastAPI inference server** — `serve/api.py`
  - Accepts: `{"diff": "..."}`
  - Returns: `{"review": {...}}`
  - Health check: `GET /health`
- [ ] **Docker + AWS deployment** — `serve/Dockerfile`, push to ECR, EC2 g4dn.xlarge minimum
- [ ] **README benchmark table** — fill in after running eval/metrics.py

---

## Project Structure

```
fine-tune/
├── CLAUDE.md
├── .env                                    # GitHub API token (gitignored)
├── .gitignore
├── LICENSE
├── README.md
├── train.py                                # QLoRA training — Unsloth + SFTTrainer + Accelerate
├── inference.py                            # Single-model inference testing
├── compare.py                              # Fine-tuned vs base model (subprocess isolation)
├── accelerate_config.yaml                  # Single GPU, bf16
│
├── data/
│   ├── fetch_github_pr.py                  # GitHub API scraper (20 repos, 2,977 PRs)
│   ├── combine_dataset.py                  # Merges training_data/ + github_pr JSON
│   ├── claude_clean.py                     # Deep 9-pass cleaning pipeline
│   ├── fix_dataset.py                      # Post-processing: dedup by input + EN instructions
│   ├── checking_dataset.py                 # Dataset quality inspector / diagnostics
│   └── review_result.py                    # Light cleaning v1 (superseded)
│
├── eval/
│   ├── metrics.py                          # Quantitative benchmark: CodeBLEU, BERTScore, etc.
│   └── compare-0.txt                       # Saved qualitative comparison output
│
├── training_data/
│   ├── code_review_training_data.json      # 200 hand-crafted single-issue examples
│   └── multi_issue_training_data.json      # 103 hand-crafted multi-issue examples
│
├── github_pr_training_data.json            # Raw GitHub PR data (gitignored)
├── combined_training_data.json             # 3,177 merged records (gitignored)
├── claude_cleaned_training_data.json       # 2,683 deep-cleaned records (gitignored)
├── claude_cleaned_training_data_v3.json    # 1,546 post-processed records (gitignored)
│
└── code-review-model/                      # Generated model output (gitignored)
    ├── checkpoint-308/                     # Epoch 2 checkpoint
    ├── checkpoint-462/                     # Epoch 3 checkpoint (best)
    ├── lora/                               # LoRA adapter weights (~155 MB)
    └── merged/                             # Full merged model (~15 GB, 16-bit)
```

---

## Key Commands

```bash
# Collect PR data from GitHub (needs GITHUB_TOKEN in .env)
python data/fetch_github_pr.py

# Merge all data sources
python data/combine_dataset.py

# Deep clean (9-pass pipeline)
python data/claude_clean.py

# Post-process: dedup by input + inject English instructions
python data/fix_dataset.py

# Inspect dataset quality
python data/checking_dataset.py

# Train (single GPU)
accelerate launch --config_file accelerate_config.yaml train.py

# Run inference on test cases
python inference.py

# Compare fine-tuned vs base model
python compare.py

# Evaluate (skip baselines to save time)
python eval/metrics.py --skip-baselines --output results/benchmark.json
```

---

## Data Pipeline (in order)

```
training_data/code_review_training_data.json (200 single-issue)
         +
training_data/multi_issue_training_data.json (103 multi-issue)
         +
GitHub API → data/fetch_github_pr.py → github_pr_training_data.json (2,977)
         |
         v
data/combine_dataset.py  →  combined_training_data.json (3,177)
         |
         v
data/claude_clean.py     →  claude_cleaned_training_data.json (2,683)
         |
         v
data/fix_dataset.py      →  claude_cleaned_training_data_v3.json (1,546)
         |
         v
train.py (DATA_PATH currently = claude_cleaned_training_data.json)
          ↑ TODO: change to v3 after evaluating current model
```

---

## Training Configuration

| Parameter | Value |
|---|---|
| Base model | Qwen/Qwen2.5-Coder-7B-Instruct |
| Quantization | 4-bit NF4 (QLoRA via Unsloth) |
| LoRA rank / alpha | 16 / 32 |
| LoRA dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Trainable params | 40.4M / 7.6B (0.53%) |
| Epochs | 3 |
| Effective batch size | 16 (batch=1 × grad_accum=16) |
| Learning rate | 2e-4 cosine, 5% warmup |
| Precision | bf16 |
| Optimizer | paged_adamw_32bit |
| Max sequence length | 2,048 tokens |
| Hardware | NVIDIA RTX 2000 Ada Laptop GPU (8 GB VRAM) |

### Training Results (first run — claude_cleaned_training_data.json, 2,683 records)
| Epoch | Eval Loss | Train Loss |
|---|---|---|
| 1 | 0.8943 | ~0.92 |
| 2 | 0.7491 | ~0.56 |
| 3 | **0.7372** | ~0.41 |

Training time: ~2h 46min on RTX 2000 Ada.

### Memory Optimization Stack (do NOT remove any of these)
- `load_in_4bit=True` — QLoRA 4-bit NF4
- `bnb_4bit_compute_dtype=torch.bfloat16` — faster on Ampere+
- `use_gradient_checkpointing="unsloth"` — trades compute for VRAM
- `optim="paged_adamw_32bit"` — offloads optimizer states to CPU
- `gradient_accumulation_steps=16` — simulates batch=16 without OOM
- `torch_compile=True` — JIT compilation, ~20% speedup on Ada
- `bf16=True` — via Accelerator
- `dataloader_pin_memory=True` — reduces CPU→GPU PCIe latency
- `ddp_find_unused_parameters=False` — DDP optimization

---

## Data Format

### Training sample schema
```json
{
  "instruction": "請對以下 Python 程式碼做 code review",
  "input": "<code snippet or diff>",
  "output": "<review text or structured JSON>",
  "metadata": {
    "source": "github_inline_comment | github_review_body | handcrafted | handcrafted_multi_issue",
    "repo": "owner/repo",
    "pr_number": 123
  }
}
```

### Structured JSON output (handcrafted examples)
```json
{
  "issues": [
    {
      "type": "安全漏洞",
      "severity": "高",
      "description": "...",
      "suggestion": "...",
      "fixed_code": "..."
    }
  ],
  "overall_score": 2,
  "summary": "..."
}
```

---

## Evaluation Plan

### Baselines (in eval/metrics.py)
| Model | Purpose |
|---|---|
| Fine-tuned Qwen2.5-Coder-7B | Primary |
| Base Qwen2.5-Coder-7B | Prove fine-tuning helps |
| Few-shot Qwen2.5-Coder-7B | Prove fine-tune > prompting |

### Security test cases (7 total)
SQL Injection, Hardcoded Password, Missing Exception Handling, Race Condition,
Path Traversal, Command Injection, Insecure Deserialization

### Metrics
- **Bug Detection Rate** — % of known bugs correctly identified (most important)
- **False Positive Rate** — % of clean code incorrectly flagged
- **CodeBLEU** — code-aware n-gram overlap
- **BERTScore F1** — semantic similarity

---

## What NOT to Do
- Do NOT add `.cuda()` or `.to("cuda")` — Accelerate handles device placement
- Do NOT remove Unsloth — handles patching, gradient offloading, 2x speedup
- Do NOT remove `gradient_checkpointing` — needed for 8GB VRAM
- Do NOT change `load_in_4bit` — full precision won't fit
- Do NOT run `eval/metrics.py` during training — standalone only
- Do NOT merge LoRA into base model unless explicitly asked
- Do NOT run `compare.py` without enough VRAM — uses subprocess isolation
- Do NOT commit `.env` — contains GitHub API token

---

## AWS Deployment Notes (pending)
- Docker image in `serve/Dockerfile`
- Push to ECR, deploy on EC2 g4dn.xlarge minimum
- API: accepts `{"diff": "..."}` → returns `{"review": {...}}`
- Health check: `GET /health`
