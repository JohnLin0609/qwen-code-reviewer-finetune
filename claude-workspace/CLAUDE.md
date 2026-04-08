# Code Reviewer Fine-Tuning Project

## Background Context

### Who I Am
- National Chengchi University (NCCU), M.S. in Computer Science
- Thesis: RAG-based chatbot (directly relevant to LLM application development)
- 2 years as Software Engineer at SRAM
  - Built full SPC system: HTML+JS frontend / Flask backend / Docker
  - Deployed YOLO image recognition on Nvidia ORIN NX (Edge AI)
  - Developed motor control software
  - Primary language: Python
- Experienced Claude user with hands-on knowledge of Agent and Skill patterns

### Target Job
This project is a portfolio piece targeting **Micron Technology Senior AI Engineer**
(Smart Manufacturing AI team, Taichung, Taiwan).

Key differentiators to demonstrate:
- LLM fine-tuning with quantifiable results (not just "I trained a model")
- Edge AI deployment experience (YOLO on ORIN NX — rare in Taiwan market)
- End-to-end system delivery: data collection → cleaning → training → evaluation
- DDP/Accelerate knowledge (shows awareness of multi-GPU training concepts)
- CUDA bottleneck analysis via PyTorch Profiler (shows systems-level thinking)
- QLoRA + Unsloth optimization (shows VRAM constraint engineering)

### Key Decisions Made
- **QLoRA over full fine-tuning**: single 8GB GPU constraint, but also demonstrates
  understanding of parameter-efficient fine-tuning tradeoffs
- **Unsloth**: 2x faster QLoRA training with smart gradient offloading
- **Accelerate for DDP**: enables multi-GPU claim without requiring multi-GPU hardware
- **PyTorch Profiler**: demonstrates CUDA memory hierarchy understanding without
  needing to write raw CUDA kernels
- **3-baseline evaluation**: Base model / Few-shot / GPT-4o comparison makes
  results meaningful to interviewers, not just absolute numbers
- **9-pass data cleaning pipeline**: demonstrates data engineering rigor
- **Multi-issue training data**: hand-crafted examples to fix single-issue output limitation
- **Side-by-side evaluation**: Fine-tuned vs base model comparison on 5 security test cases

### Interview Talking Points (preserve these)
When asked about this project, emphasize:
1. VRAM reduction: ~28GB → ~6GB via QLoRA (concrete engineering decision)
2. Structured JSON output with severity ratings vs base model's free-form text
3. Fixed single-issue limitation by crafting 103 multi-issue training examples (data-driven fix)
4. Profiler finding: main bottleneck is attention memory bandwidth
5. DDP support via Accelerate — single/multi-GPU with no code change
6. 9-pass data cleaning pipeline: bot removal, dedup, normalization, quality filtering
7. Full pipeline: GitHub API (20 repos) → cleaning → fine-tune → eval → FastAPI → deployment

---

## Project Status
- [x] Data collection script (`fetch_github_pr.py`) — 2,977 records from 20 repos
- [x] Handcrafted training data (`code_review_training_data.json`) — 200 single-issue
- [x] Multi-issue training data (`multi_issue_training_data.json`) — 103 multi-issue
- [x] Dataset merging (`combine_dataset.py`)
- [x] Data cleaning v1 (`review_result.py`) — light cleaning
- [x] Data cleaning v2 (`claude_clean.py`) — deep 9-pass pipeline
- [x] Dataset inspection (`checking_dataset.py`)
- [x] Training script with Accelerate (`train.py`) — QLoRA + Unsloth + accelerate launch
- [x] Inference script (`inference.py`)
- [x] Side-by-side comparison (`compare.py`) — fine-tuned vs base model
- [ ] PyTorch Profiler analysis
- [ ] Evaluation with quantitative metrics (CodeBLEU, BERTScore)
- [ ] FastAPI inference server
- [ ] Docker + cloud deployment
- [ ] README with benchmark results table

---

## Project Overview
This project fine-tunes Qwen2.5-Coder-7B-Instruct via QLoRA to build a
security-focused Python Code Review model. It takes code as input and outputs
structured JSON reviews with severity ratings, fix suggestions, and fixed code.

All prompts and outputs are in Traditional Chinese (繁體中文).

## Actual Project Structure
```
fine-tune/
├── CLAUDE.md                          # (in claude-workspace/)
├── .env                               # GitHub API token
├── fetch_github_pr.py                 # GitHub API data collection (20 repos, 100 PRs each)
├── code_review_training_data.json     # 200 handcrafted single-issue examples
├── multi_issue_training_data.json     # 103 handcrafted multi-issue examples
├── combine_dataset.py                 # Merges synthetic + GitHub data
├── combined_training_data.json        # 3,177 merged records
├── checking_dataset.py                # Dataset quality inspector
├── review_result.py                   # Light data cleaning (v1, unused in final pipeline)
├── claude_clean.py                    # Deep 9-pass cleaning pipeline
├── cleaned_training_data.json         # Light-cleaned (v1 output, unused)
├── claude_cleaned_training_data.json  # Final training data (2,683 records)
├── train.py                           # QLoRA training with Unsloth + Accelerate
├── accelerate_config.yaml             # Accelerate config (single GPU, bf16)
├── inference.py                       # Single-model inference testing
├── compare.py                         # Side-by-side fine-tuned vs base model
├── compare-0.txt                      # Saved comparison output
├── train_logs/                        # Training console logs
├── code-review-model/                 # Model output
│   ├── checkpoint-308/                # Epoch 2 checkpoint
│   ├── checkpoint-462/                # Epoch 3 checkpoint (best)
│   ├── lora/                          # LoRA adapter weights (~155 MB)
│   └── merged/                        # Full merged model (~15 GB, 16-bit)
└── claude-workspace/
    └── research.md                    # Detailed project analysis
```

## Key Commands
```bash
# Collect PR data from GitHub (needs GITHUB_TOKEN in .env)
python fetch_github_pr.py

# Merge synthetic + GitHub data
python combine_dataset.py

# Deep clean the dataset (9-pass pipeline)
python claude_clean.py

# Train with Accelerate (single GPU)
accelerate launch --config_file accelerate_config.yaml train.py

# Run inference on test cases
python inference.py

# Compare fine-tuned vs base model
python compare.py
```

## Tech Stack
- **Model**: Qwen/Qwen2.5-Coder-7B-Instruct
- **Fine-tuning**: QLoRA (4-bit quantization + LoRA adapters)
- **Training framework**: Unsloth + TRL (SFTTrainer) + PEFT + Accelerate
- **Multi-GPU**: Accelerate (DDP support)
- **Optimization**: torch.compile, adamw_8bit, gradient checkpointing
- **Profiling**: PyTorch Profiler
- **Evaluation**: CodeBLEU, BERTScore, Bug Detection Rate, False Positive Rate
- **Serving**: FastAPI + Docker
- **Cloud**: AWS (EC2 / ECR)
- **Tracking**: Weights & Biases (wandb)
- **Hardware**: NVIDIA RTX 2000 Ada Laptop GPU (8 GB VRAM)
- **Data sources**: GitHub API (20 major Python repos) + 303 handcrafted examples

## Training Configuration

### Hyperparameters
| Parameter | Value |
|---|---|
| Base model | Qwen/Qwen2.5-Coder-7B-Instruct |
| Quantization | 4-bit (QLoRA via Unsloth) |
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Trainable params | 40,370,176 / 7.6B total (0.53%) |
| Epochs | 3 |
| Batch size | 1 (per device) |
| Gradient accumulation | 16 (effective batch = 16) |
| Learning rate | 2e-4 (cosine scheduler, 5% warmup) |
| Precision | bf16 |
| Optimizer | adamw_8bit |
| Max sequence length | 2,048 tokens |

### Training Results (first run, 2,580 records)
| Metric | Epoch 1 | Epoch 2 | Epoch 3 |
|---|---|---|---|
| Eval loss | 0.9048 | 0.7558 | 0.7491 |
| Train loss (avg) | ~0.92 | ~0.56 | ~0.41 |

Training time: ~2h 46min on RTX 2000 Ada.

### Memory Optimization Stack
The following are applied together; preserve all of them:
- `bnb_4bit_compute_dtype=torch.bfloat16` — faster than fp16 on Ampere+
- `load_in_4bit=True` — QLoRA 4-bit NF4 quantization
- `use_gradient_checkpointing="unsloth"` — Unsloth-optimized gradient checkpointing
- `optim="paged_adamw_32bit"` — offloads optimizer states to CPU when VRAM is full
- `gradient_accumulation_steps=16` — simulates larger batch without OOM
- `torch_compile=True` — JIT compilation for Ada GPU (~20% speedup)
- `bf16=True` — bfloat16 mixed precision via Accelerator
- `dataloader_pin_memory=True` — reduces CPU→GPU PCIe transfer latency
- `ddp_find_unused_parameters=False` — DDP optimization, all params are used

## Data Format
Each training sample follows this structure:
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

### Handcrafted output format (structured JSON)
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

### Training Data Composition (current: 2,683 records)
| Source | Count | % |
|---|---|---|
| GitHub inline comments | 2,080 | 77.5% |
| GitHub review bodies | 300 | 11.2% |
| Handcrafted single-issue | 200 | 7.5% |
| Handcrafted multi-issue | 103 | 3.8% |

## Evaluation: How to Measure Results
Always compare against these three baselines:

| Model | Purpose |
|---|---|
| Base Qwen2.5-Coder-7B (no fine-tune) | Prove fine-tuning helps |
| Few-shot Qwen2.5-Coder-7B | Prove fine-tune > prompt engineering |
| GPT-4o | Understand gap to SOTA |

Metrics to report in README:
- **CodeBLEU** — primary code quality metric
- **BERTScore F1** — semantic similarity
- **Bug Detection Rate** — task-specific, most important
- **False Positive Rate** — usability metric

## Accelerate / DDP Integration
- `Accelerator(mixed_precision="bf16")` wraps the training loop
- Enables seamless single-GPU → multi-GPU without code changes
- Launch with `accelerate launch` for multi-GPU; `python train.py` for single-GPU
- Do NOT manually call `.to(device)` — Accelerate handles device placement

## What NOT to Do
- Do not add `.cuda()` or `.to("cuda")` calls manually — Accelerate handles this
- Do not remove Unsloth — it handles model patching, gradient offloading, and 2x speedup
- Do not remove `gradient_checkpointing` — it is needed for 8GB VRAM
- Do not change `load_in_4bit` — full precision won't fit in VRAM
- Do not run `profile.py` during actual training — it is for analysis only
- Do not merge LoRA weights into base model unless explicitly asked
- Do not change `bnb_4bit_quant_type` from `"nf4"` without benchmarking
- Do not run `compare.py` without enough VRAM — it loads models in subprocess isolation
- Do not commit `.env` — it contains the GitHub API token

## AWS Deployment Notes
- Docker image is built from `serve/Dockerfile`
- Push to ECR, deploy on EC2 (g4dn.xlarge minimum for inference)
- Inference API must accept: `{ "diff": "..." }` and return `{ "review": "..." }`
- Health check endpoint: `GET /health`

---

## CUDA & DDP Concepts (for implementation reference)

### Why GPU is Fast
- GPU has thousands of simple cores running in parallel vs CPU's few powerful cores
- Neural network training is fundamentally matrix multiplication — ideal for GPU
- Key bottleneck: data transfer between CPU↔GPU (PCIe ~16 GB/s) vs GPU internal
  compute (~10,000 GB/s). Keep data on GPU; avoid frequent `.cpu()` calls in loops.

### GPU Memory Hierarchy (relevant to profiler output)
```
Global Memory (VRAM) — largest, slowest — model weights live here
Shared Memory       — per SM, small, fast — optimization target
Registers           — per thread, tiny, fastest
```

### DDP / AllReduce (what Accelerate does under the hood)
- Each GPU processes a different batch and computes its own gradients
- AllReduce averages gradients across all GPUs before the optimizer step
- All GPUs end up with identical model weights after each step
- `loss.backward()` triggers AllReduce automatically when using DDP/Accelerate

### QLoRA + DDP relationship
- QLoRA trains only ~0.3% of parameters → fits on single GPU → DDP not required
- Accelerate is added anyway to demonstrate awareness and enable future scaling
- If asked: "DDP would matter if we moved to full fine-tuning of larger models"

### What gradient_checkpointing does
- Instead of storing all intermediate activations in VRAM during forward pass,
  recomputes them during backward pass
- Trades ~30% extra compute for significant memory savings
- Essential for fitting longer sequences in limited VRAM

### PyTorch Profiler
- `profile.py` is a standalone script, not part of the training loop
- Run once to identify CUDA bottlenecks; results go into README.md
- Outputs `trace.json` viewable at chrome://tracing

---

## Session Handoff Instructions

When starting a new Claude Code session, run:
```
Read CLAUDE.md fully before doing anything else.
Then show me the current project structure and status.
```

When returning to this Claude.ai conversation after coding:
- Paste any new files or diffs to review architecture decisions
- Update the Project Status checklist above
- Add any new decisions to "Key Decisions Made" section

Do NOT ask Claude Code to redesign architecture — make those decisions here first,
then instruct Claude Code to implement the agreed design.
