"""
Step 3: Generate 700 synthetic code-review training examples via Anthropic API.

Generates exactly 100 examples per language × 7 languages:
  python, javascript, java, c, go, php, rust

Mix per language:
  - 80 vulnerable (40% High, 45% Medium, 15% Low severity)
  - 20 clean (overall_score 8-10, empty issues array)

Few-shot templates are sampled from training_data_v6_fixed.jsonl entries
matching the target language. If fewer than 3 exist, python examples are
used as fallback.

Output: synthetic_generated.jsonl (one JSON entry per line).
Progress file: generation_progress.json (resumable).
"""

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = BASE_DIR / "training_data_v6_fixed.jsonl"
DEFAULT_OUTPUT = BASE_DIR / "synthetic_generated.jsonl"
PROGRESS_PATH = BASE_DIR / "generation_progress.json"

MODEL = "claude-haiku-4-5-20251001"
MAX_RETRIES = 3
BASE_BACKOFF = 1.0  # 1s, 2s, 4s
SAVE_EVERY = 50
CONCURRENCY = 5

LANGUAGES = ["python", "javascript", "java", "c", "go", "php", "rust"]
PER_LANG = 100
TOTAL = len(LANGUAGES) * PER_LANG  # 700

VULN_TYPES = [
    "SQL Injection", "XSS", "Path Traversal", "Hardcoded Credentials",
    "Insecure Random", "Race Condition", "Missing Input Validation",
    "Missing Error Handling", "Memory Leak", "Authentication Bypass",
    "SSRF", "Command Injection", "Insecure Deserialization",
    "Missing Rate Limiting", "Information Disclosure",
]

SYSTEM_PROMPT = (
    "You are a senior software engineer and security expert performing code review. "
    "Analyze the given code for security vulnerabilities, bugs, and reliability issues. "
    "Always respond in valid JSON format with this structure: "
    "{\"issues\": [{\"type\": \"Security Vulnerability | Reliability Issue | Code Quality\", "
    "\"severity\": \"High | Medium | Low\", "
    "\"description\": \"Clear description of the issue\", "
    "\"suggestion\": \"How to fix it\", "
    "\"fixed_code\": \"The corrected code\"}], "
    "\"overall_score\": <1-10>, "
    "\"summary\": \"Brief overall assessment\"}. "
    "If no issues found, return empty issues array with high overall_score."
)

GEN_SYSTEM = """You generate realistic training data for a code review model.

Each example you produce must be a single JSON object with this exact structure:
{
  "user_message": "Review this <LANG> code for security vulnerabilities:\\n\\n<CODE>",
  "assistant_output": "<JSON review matching the exact schema below>"
}

The assistant_output must be a JSON STRING (not an object) containing valid JSON with:
  {"issues": [...], "overall_score": <1-10>, "summary": "..."}
Each issue has: type, severity, description, suggestion, fixed_code.

Rules:
- The code must be realistic, idiomatic for the target language, and 5-25 lines long.
- Code must be clearly self-contained (no ellipses, no "...", no placeholders like `<your key>`).
- If asked for a vulnerable example, inject exactly one primary vulnerability matching the requested type and severity. You may add 0-2 additional related issues.
- If asked for a clean example, the code must be correct and reasonably secure; return "issues": [] and overall_score 8-10.
- Never wrap the response in markdown fences. Output ONLY the top-level JSON object, nothing else.
- Ensure all JSON strings are properly escaped (newlines in code become \\n).
- The `fixed_code` field should show the corrected snippet, not the whole file."""


def load_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def extract_user_code(user_content: str) -> str:
    parts = user_content.split("\n\n", 1)
    return parts[1] if len(parts) == 2 else user_content


def few_shot_bank(entries: list[dict]) -> dict[str, list[dict]]:
    """Group entries by language, return {lang: [examples...]}."""
    by_lang: dict[str, list[dict]] = {}
    for e in entries:
        lang = e.get("metadata", {}).get("language", "python")
        by_lang.setdefault(lang, []).append(e)
    return by_lang


def sample_few_shots(bank: dict[str, list[dict]], lang: str, k: int, rng: random.Random) -> list[dict]:
    """Sample k few-shot examples for the given language. Fall back to python
    if the language has fewer than k examples."""
    pool = bank.get(lang, [])
    if len(pool) < k:
        pool = bank.get("python", [])
    if not pool:
        return []
    return rng.sample(pool, min(k, len(pool)))


def fewshot_block(examples: list[dict]) -> str:
    """Render few-shot examples as a demonstration block for the generator."""
    lines = []
    for ex in examples:
        user_msg = next((m["content"] for m in ex["messages"] if m["role"] == "user"), "")
        asst_msg = next((m["content"] for m in ex["messages"] if m["role"] == "assistant"), "")
        demo = {"user_message": user_msg, "assistant_output": asst_msg}
        lines.append(json.dumps(demo, ensure_ascii=False))
    return "\n\n".join(lines)


def plan_batch(lang: str) -> list[dict]:
    """Build 100 example specs for one language.
    Mix: 80 vulnerable (40% H, 45% M, 15% L), 20 clean."""
    specs = []
    vuln_total = int(PER_LANG * 0.8)  # 80
    clean_total = PER_LANG - vuln_total  # 20

    high = round(vuln_total * 0.40)   # 32
    med = round(vuln_total * 0.45)    # 36
    low = vuln_total - high - med     # 12

    severity_seq = (["High"] * high) + (["Medium"] * med) + (["Low"] * low)
    rng = random.Random(hash((lang, "plan")) & 0xFFFFFFFF)
    rng.shuffle(severity_seq)

    vuln_types_cycle = []
    while len(vuln_types_cycle) < vuln_total:
        shuffled = VULN_TYPES.copy()
        rng.shuffle(shuffled)
        vuln_types_cycle.extend(shuffled)
    vuln_types_cycle = vuln_types_cycle[:vuln_total]

    for i in range(vuln_total):
        specs.append({
            "language": lang,
            "kind": "vulnerable",
            "vuln_type": vuln_types_cycle[i],
            "severity": severity_seq[i],
        })
    for _ in range(clean_total):
        specs.append({
            "language": lang,
            "kind": "clean",
            "vuln_type": None,
            "severity": None,
        })

    rng.shuffle(specs)
    return specs


def build_user_prompt(spec: dict, few_shots: list[dict]) -> str:
    """Prompt the generator model to produce one training example."""
    demos = fewshot_block(few_shots) if few_shots else "(no demonstrations available)"
    lang = spec["language"]
    if spec["kind"] == "vulnerable":
        task = (
            f"Generate a NEW training example for language={lang}. "
            f"The code must contain a {spec['severity']}-severity {spec['vuln_type']} vulnerability "
            f"(and 0-2 related minor issues if natural). "
            f"The review must flag it correctly with severity='{spec['severity']}'."
        )
    else:
        task = (
            f"Generate a NEW training example for language={lang}. "
            f"The code must be CLEAN — no meaningful security or reliability issues. "
            f"The review must return issues: [] and overall_score between 8 and 10."
        )
    return (
        f"Here are 3 demonstration examples of the target format (user_message + assistant_output):\n\n"
        f"{demos}\n\n"
        f"---\n\n{task}\n\n"
        f"Return ONLY the JSON object with keys user_message and assistant_output. "
        f"Make the code different from the demonstrations."
    )


def strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def validate_generated(obj: dict) -> tuple[bool, str]:
    """Ensure the generated object conforms. Returns (ok, reason)."""
    if not isinstance(obj, dict):
        return False, "not a dict"
    um = obj.get("user_message")
    ao = obj.get("assistant_output")
    if not isinstance(um, str) or not um.strip():
        return False, "missing user_message"
    if not isinstance(ao, str) or not ao.strip():
        return False, "missing assistant_output (must be string)"
    try:
        inner = json.loads(ao)
    except Exception as e:
        return False, f"assistant_output not valid JSON: {e}"
    if not isinstance(inner, dict):
        return False, "assistant_output JSON is not an object"
    if "issues" not in inner or "overall_score" not in inner:
        return False, "missing keys issues/overall_score"
    if not isinstance(inner["issues"], list):
        return False, "issues is not a list"
    try:
        score = float(inner["overall_score"])
    except Exception:
        return False, "overall_score not numeric"
    if not (1 <= score <= 10):
        return False, f"overall_score out of range: {score}"
    return True, "ok"


def load_env_key():
    env_path = BASE_DIR / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("ANTHROPIC_API_KEY="):
                os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1].strip()


def load_progress() -> dict:
    if PROGRESS_PATH.exists():
        with open(PROGRESS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {"completed": [], "failed": []}


def save_progress(state: dict) -> None:
    tmp = PROGRESS_PATH.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False)
    os.replace(tmp, PROGRESS_PATH)


async def call_api(client, user_prompt: str) -> str:
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = await client.messages.create(
                model=MODEL,
                max_tokens=3000,
                system=GEN_SYSTEM,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return resp.content[0].text
        except Exception as e:
            last_err = e
            if attempt < MAX_RETRIES - 1:
                wait = BASE_BACKOFF * (2 ** attempt)
                await asyncio.sleep(wait)
    raise RuntimeError(f"API failed after {MAX_RETRIES} attempts: {last_err}")


async def generate_one(client, spec: dict, few_shots: list[dict], semaphore: asyncio.Semaphore) -> dict | None:
    """Generate + validate one example. Returns the training entry or None on failure."""
    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                prompt = build_user_prompt(spec, few_shots)
                raw = await call_api(client, prompt)
                text = strip_fences(raw)
                obj = json.loads(text)
                ok, reason = validate_generated(obj)
                if not ok:
                    if attempt < MAX_RETRIES - 1:
                        await asyncio.sleep(BASE_BACKOFF * (2 ** attempt))
                        continue
                    return None
                problem_type = spec["vuln_type"] if spec["kind"] == "vulnerable" else "Clean Code"
                return {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": obj["user_message"]},
                        {"role": "assistant", "content": obj["assistant_output"]},
                    ],
                    "metadata": {
                        "source": "synthetic_generated",
                        "language": spec["language"],
                        "problem_type": problem_type,
                    },
                }
            except Exception:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(BASE_BACKOFF * (2 ** attempt))
                    continue
                return None
        return None


async def run(resume: bool) -> None:
    load_env_key()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    from anthropic import AsyncAnthropic
    client = AsyncAnthropic()

    fixed = load_jsonl(DEFAULT_INPUT)
    bank = few_shot_bank(fixed)
    print(f"Loaded {len(fixed)} fixed entries for few-shot bank", flush=True)
    print(f"Per-language pool sizes: " + ", ".join(f"{k}={len(v)}" for k, v in bank.items()), flush=True)

    # Build all 700 specs up front, tagged with a stable id = f"{lang}:{i}"
    all_specs: list[tuple[str, dict]] = []
    for lang in LANGUAGES:
        specs = plan_batch(lang)
        for i, s in enumerate(specs):
            all_specs.append((f"{lang}:{i}", s))

    # Resume
    state = load_progress() if resume else {"completed": [], "failed": []}
    done_ids = {c["id"] for c in state.get("completed", [])}
    failed_ids = {f["id"] for f in state.get("failed", [])}

    pending = [(sid, spec) for sid, spec in all_specs if sid not in done_ids and sid not in failed_ids]
    print(f"Total specs: {len(all_specs)}. Already done: {len(done_ids)}. Prior failed: {len(failed_ids)}. Pending: {len(pending)}", flush=True)

    if not pending:
        print("Nothing to do. Writing final output.", flush=True)
        _write_output(state)
        return

    semaphore = asyncio.Semaphore(CONCURRENCY)
    rng = random.Random(42)

    t_start = time.time()
    completed_count = len(done_ids)
    save_counter = 0

    async def worker(sid: str, spec: dict):
        nonlocal completed_count, save_counter
        fs = sample_few_shots(bank, spec["language"], 3, rng)
        entry = await generate_one(client, spec, fs, semaphore)
        if entry is None:
            state["failed"].append({"id": sid, "spec": spec})
            print(f"  [FAIL] {sid} {spec['kind']}/{spec.get('vuln_type')}/{spec.get('severity')}", flush=True)
        else:
            state["completed"].append({"id": sid, "entry": entry})
            completed_count += 1
            save_counter += 1
            if save_counter % SAVE_EVERY == 0:
                save_progress(state)
                elapsed = time.time() - t_start
                rate = save_counter / elapsed if elapsed else 0
                remaining = len(pending) - save_counter
                eta = remaining / rate if rate else float("inf")
                print(f"  [{completed_count}/{TOTAL}] saved; rate={rate:.2f}/s eta={eta/60:.1f}m", flush=True)

    await asyncio.gather(*(worker(sid, spec) for sid, spec in pending))
    save_progress(state)
    _write_output(state)
    print(f"\nDone. Completed={len(state['completed'])} Failed={len(state['failed'])}", flush=True)


def _write_output(state: dict) -> None:
    entries = [c["entry"] for c in state.get("completed", [])]
    with open(DEFAULT_OUTPUT, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"Wrote {len(entries)} entries → {DEFAULT_OUTPUT.name}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true", default=True,
                    help="Resume from generation_progress.json if present (default: on)")
    ap.add_argument("--fresh", action="store_true",
                    help="Ignore existing progress and start over")
    args = ap.parse_args()
    resume = not args.fresh
    asyncio.run(run(resume=resume))


if __name__ == "__main__":
    main()
