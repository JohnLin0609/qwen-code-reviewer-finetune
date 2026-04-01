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
- End-to-end system delivery: data collection → training → evaluation → API → cloud
- DDP/Accelerate knowledge (shows awareness of multi-GPU training concepts)
- CUDA bottleneck analysis via PyTorch Profiler (shows systems-level thinking)

### Key Decisions Made
- **QLoRA over full fine-tuning**: single GPU constraint, but also demonstrates
  understanding of parameter-efficient fine-tuning tradeoffs
- **Accelerate for DDP**: enables multi-GPU claim without requiring multi-GPU hardware
- **PyTorch Profiler**: demonstrates CUDA memory hierarchy understanding without
  needing to write raw CUDA kernels
- **3-baseline evaluation**: Base model / Few-shot / GPT-4o comparison makes
  results meaningful to interviewers, not just absolute numbers
- **FastAPI + Docker + AWS**: mirrors production MLOps pipeline

### Interview Talking Points (preserve these)
When asked about this project, emphasize:
1. VRAM reduction: ~28GB → ~6GB via QLoRA (concrete engineering decision)
2. Bug detection rate improvement vs baseline (fill in after evaluation)
3. Profiler finding: main bottleneck is attention memory bandwidth
4. DDP support via Accelerate — single/multi-GPU with no code change
5. Full pipeline: GitHub API → cleaning → fine-tune → eval → FastAPI → AWS

---

## Project Status
- [ ] Data collection script (`data/collect_pr.py`)
- [ ] Training script with Accelerate (`train/train.py`)
- [ ] PyTorch Profiler script (`train/profile.py`)
- [ ] Evaluation pipeline (`evaluate/metrics.py`)
- [ ] FastAPI inference server (`serve/api.py`)
- [ ] Docker + AWS deployment
- [ ] README with benchmark results table

Update this checklist as tasks are completed.

---

## Project Overview
This project fine-tunes Qwen2.5-Coder via QLoRA to build a Code Review model.
It takes a GitHub PR diff as input and outputs a structured code review comment.

## Project Structure
```
code-reviewer/
├── CLAUDE.md
├── data/
│   └── collect_pr.py         # GitHub API data collection
├── train/
│   ├── train.py              # Main training script
│   ├── profile.py            # PyTorch Profiler analysis
│   └── config.yaml           # Training hyperparameters
├── evaluate/
│   └── metrics.py            # CodeBLEU / BERTScore / ROUGE-L
├── serve/
│   └── api.py                # FastAPI inference server
└── README.md
```

## Key Commands
```bash
# Install dependencies
pip install -r requirements.txt

# Collect PR data from GitHub
python data/collect_pr.py --repo "django/django" --max-prs 500

# Single GPU training
python train/train.py

# Multi-GPU training (DDP via Accelerate)
accelerate launch --num_processes=2 train/train.py

# Profile training bottlenecks
python train/profile.py

# Run evaluation against baselines
python evaluate/metrics.py --model ./final_model --baseline Qwen2.5-Coder-7B

# Start inference API
uvicorn serve.api:app --host 0.0.0.0 --port 8000
```

## Tech Stack
- **Model**: Qwen/Qwen2.5-Coder-7B-Instruct
- **Fine-tuning**: QLoRA (4-bit quantization + LoRA adapters)
- **Training framework**: HuggingFace Transformers + TRL + PEFT
- **Multi-GPU**: Accelerate (DDP support)
- **Profiling**: PyTorch Profiler
- **Serving**: FastAPI + Docker
- **Cloud**: AWS (EC2 / ECR)
- **Tracking**: Weights & Biases (wandb)

## Training Architecture Decisions

### Why QLoRA
- 4-bit quantization reduces VRAM from ~28GB to ~6GB
- Only LoRA adapter weights (~0.3% of params) are trainable
- Makes fine-tuning feasible on a single consumer GPU

### Memory Optimization Stack
The following are applied together; preserve all of them:
- `bnb_4bit_compute_dtype=torch.bfloat16` — faster than fp16 on Ampere+
- `gradient_checkpointing=True` — trades compute for memory
- `gradient_accumulation_steps=4` — simulates larger batch without OOM
- `optim="paged_adamw_32bit"` — offloads optimizer states to CPU

### Accelerate / DDP Integration
- `Accelerator(mixed_precision="bf16")` wraps the training loop
- Enables seamless single-GPU → multi-GPU without code changes
- Launch with `accelerate launch` for multi-GPU; `python train.py` for single-GPU
- Do NOT manually call `.to(device)` — Accelerate handles device placement

### PyTorch Profiler
- `profile.py` is a standalone script, not part of the training loop
- Run once to identify CUDA bottlenecks; results go into README.md
- Outputs `trace.json` viewable at chrome://tracing

## Data Format
Each training sample must follow this structure:
```json
{
  "diff": "<git diff of the PR>",
  "review": "<the human-written review comment>"
}
```
Stored as `.jsonl` files: `data/train.jsonl`, `data/test.jsonl`

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

## What NOT to Do
- Do not add `.cuda()` or `.to("cuda")` calls manually — Accelerate handles this
- Do not remove `gradient_checkpointing` — it is needed for memory
- Do not run `profile.py` during actual training — it is for analysis only
- Do not merge LoRA weights into base model unless explicitly asked
- Do not change `bnb_4bit_quant_type` from `"nf4"` without benchmarking

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

---

## Session Handoff Instructions

When starting a new Claude Code session, run:
```
Read CLAUDE.md fully before doing anything else.
Then show me the current project structure and status of each file.
```

When returning to this Claude.ai conversation after coding:
- Paste any new files or diffs to review architecture decisions
- Update the Project Status checklist above
- Add any new decisions to "Key Decisions Made" section

Do NOT ask Claude Code to redesign architecture — make those decisions here first,
then instruct Claude Code to implement the agreed design.
