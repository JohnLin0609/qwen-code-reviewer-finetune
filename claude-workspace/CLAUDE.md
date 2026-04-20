# qwen-code-reviewer-finetune — Claude.ai Context File

> This file is for **Claude.ai conversations only** (architecture decisions, career strategy).
> The project-root `CLAUDE.md` is what Claude Code CLI reads.

---

## Who I Am
- National Chengchi University (NCCU), M.S. in Computer Science
- Thesis: RAG-based chatbot
- 2 years Software Engineer at SRAM (Taichung)
  - Built full SPC system: HTML+JS / Flask / Docker
  - Deployed YOLO on Nvidia ORIN NX (Edge AI)
  - Developed motor control software
  - Primary language: Python
- Experienced Claude user, familiar with Agent and Skill patterns

---

## Target Job
**Micron Technology — Senior AI Engineer, Smart Manufacturing AI team**
Taichung, Taiwan — JR92604

### Why this project?
The gap between my background and the JD:
- No CUDA kernel experience → compensate with Profiler + QLoRA memory analysis
- No distributed training → compensate with Accelerate DDP integration
- 2yr exp vs 5yr required → compensate with end-to-end pipeline depth
- NCCU vs NTU/NCTU brand → compensate with concrete, quantified results

### Key differentiators this project demonstrates
- LLM fine-tuning with quantifiable results (not just "I trained a model")
- VRAM constraint engineering: ~28GB → ~6GB via QLoRA
- Edge AI deployment (YOLO on ORIN NX — rare in Taiwan market)
- End-to-end pipeline: data collection → cleaning → training → evaluation → serving
- DDP/Accelerate (shows multi-GPU awareness)
- CUDA bottleneck analysis via PyTorch Profiler (shows systems-level thinking)

### Interview Talking Points
1. VRAM: ~28GB → ~6GB via QLoRA (concrete engineering decision, not default choice)
2. Structured JSON output with severity ratings vs base model's free-form markdown
3. Fixed single-issue limitation with 103 hand-crafted multi-issue examples (data-driven fix)
4. 9-pass + fix_dataset cleaning pipeline: dedup by input, EN/ZH instruction diversity
5. DDP via Accelerate — single/multi-GPU with zero code changes
6. ORIN NX Edge AI deployment (YOLO) — rare combination with LLM work
7. Bug Detection Rate (fill in after running eval/metrics.py)
8. Full pipeline: GitHub API → cleaning → fine-tune → eval → FastAPI → AWS

---

## Architecture Decision Log
All decisions below were made here in Claude.ai — implement in Claude Code, don't redesign there.

| Decision | Rationale |
|---|---|
| QLoRA over full fine-tuning | Single 8GB GPU; demonstrates PEFT tradeoffs |
| Unsloth | 2x faster QLoRA with smart gradient offloading |
| Accelerate for DDP | Multi-GPU claim without multi-GPU hardware |
| Subprocess isolation in compare.py | Unsloth single-model-per-process limitation |
| fix_dataset.py dedup by input | 542 duplicate inputs → 1,137 extra records in cleaned data |
| 45% EN / 55% ZH instruction split | Model was 0% English before, poor English prompt generalization |
| eval/metrics.py 3-baseline design | Makes results meaningful vs just absolute numbers |
| DATA_PATH still = v2 (2,683 records) | v3 is cleaner but smaller — wait for eval before retraining |

---

## Project Status

### COMPLETED ✅
- [x] `data/fetch_github_pr.py` — 2,977 records from 20 repos
- [x] `training_data/code_review_training_data.json` — 200 hand-crafted single-issue
- [x] `training_data/multi_issue_training_data.json` — 103 hand-crafted multi-issue
- [x] `data/combine_dataset.py`
- [x] `data/claude_clean.py` — 9-pass deep cleaning
- [x] `data/fix_dataset.py` — dedup by input + EN instruction injection
- [x] `data/checking_dataset.py`
- [x] `train.py` — 3 epochs complete, eval loss 0.7372
- [x] `inference.py`
- [x] `compare.py`
- [x] `eval/compare-0.txt` — qualitative results saved
- [x] `eval/metrics.py` — written, not yet run

### PENDING ❌
- [ ] Run `eval/metrics.py` — get Bug Detection Rate, CodeBLEU, BERTScore numbers
- [ ] Decide whether to retrain on v3 dataset (after seeing eval results)
- [ ] Write `profile.py` — PyTorch Profiler, document bottleneck finding for interview
- [ ] FastAPI server (`serve/api.py`)
- [ ] Docker + AWS deployment (EC2 g4dn.xlarge)
- [ ] Fill in README benchmark table

---

## Qualitative Results (eval/compare-0.txt)

Fine-tuned vs base Qwen2.5-Coder-7B on 5 security test cases:

| Aspect | Fine-tuned | Base |
|---|---|---|
| Output format | Structured JSON | Free-form markdown |
| Multi-issue detection | Identifies primary + secondary | Often misses secondary |
| Security focus | Vulnerability first | Buries security in general suggestions |
| Language | Consistent Traditional Chinese | Mixes Simplified/Traditional |
| Actionability | `fixed_code` in every issue | Multiple alternatives, overwhelming |
| Severity scoring | `overall_score` 1-10 | No scoring |

**Race condition test (most impressive):**
- Fine-tuned: correctly identified TOCTOU + suggested threading.Lock
- Base: called code "correct", suggested Counter optimization — completely missed it

---

## Session Handoff Rule
Architecture decisions → Claude.ai (here)
Implementation → Claude Code CLI (reads project-root CLAUDE.md)

After each Claude Code session, paste key changes back here to update this file.
