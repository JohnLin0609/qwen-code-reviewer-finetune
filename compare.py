"""
Fine-tuned vs Original 模型比較
================================
用 subprocess 隔離兩個模型，避免 Unsloth 在同一 process 重複載入的問題。

執行：python compare.py
"""

import json
import sys
import subprocess
import tempfile
import os

# ── 設定 ──────────────────────────────────────────────────────────────────

BASE_MODEL    = "Qwen/Qwen2.5-Coder-7B-Instruct"
FINETUNED     = "./code-review-model-v7/lora"
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

test_cases = [
    ("SQL Injection", """
def get_user(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    return db.execute(query).fetchone()
"""),
    ("硬編碼密碼", """
def connect():
    return psycopg2.connect(
        host="localhost",
        user="admin",
        password="supersecret123"
    )
"""),
    ("沒有 exception handling", """
def read_config(path):
    with open(path) as f:
        return json.load(f)
"""),
    ("Race condition", """
vote_counts = {}

def vote(post_id):
    if post_id in vote_counts:
        vote_counts[post_id] += 1
    else:
        vote_counts[post_id] = 1
"""),
    ("Path traversal", """
def get_template(name):
    with open(f'templates/{name}.html') as f:
        return f.read()
"""),
]

# ── Subprocess worker ─────────────────────────────────────────────────────

WORKER_SCRIPT = '''
import warnings
warnings.filterwarnings("ignore")

import sys, json, torch
from unsloth import FastLanguageModel

model_name  = sys.argv[1]
cases_json  = sys.argv[2]
system_prompt = sys.argv[3]
output_file = sys.argv[4]

cases = json.loads(cases_json)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=2048,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

results = []
for i, code in enumerate(cases):
    print(f"  測試 {i+1}/{len(cases)}...", file=sys.stderr)
    messages = [
        {"role": "system",  "content": system_prompt},
        {"role": "user",    "content": f"Review this Python code for security vulnerabilities:\\n\\n{code}"},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
    )
    if hasattr(inputs, "input_ids"):
        input_ids = inputs.input_ids.to("cuda")
    else:
        input_ids = inputs.to("cuda")

    prompt_len = input_ids.shape[1]

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            use_cache=False,
        )

    generated = output[0][prompt_len:]
    results.append(tokenizer.decode(generated, skip_special_tokens=True))

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False)
'''


def run_model(model_name, label):
    print(f"{'=' * 60}")
    print(f"載入 {label}...")
    print(f"{'=' * 60}")

    codes = [code.strip() for _, code in test_cases]
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    tmp.close()

    try:
        result = subprocess.run(
            [sys.executable, "-c", WORKER_SCRIPT, model_name, json.dumps(codes), SYSTEM_PROMPT, tmp.name],
            timeout=1800,
        )

        if result.returncode != 0:
            print(f"  模型推理失敗 (returncode={result.returncode})")
            return None

        with open(tmp.name, encoding="utf-8") as f:
            return json.load(f)
    finally:
        os.unlink(tmp.name)


# ── 主程式 ────────────────────────────────────────────────────────────────

def main():
    ft_results   = run_model(FINETUNED, "Fine-tuned 模型")
    if ft_results is None:
        print("Fine-tuned 模型推理失敗")
        return

    print()
    orig_results = run_model(BASE_MODEL, "Original 模型")
    if orig_results is None:
        print("Original 模型推理失敗")
        return

    # ── 並排比較 ──
    print("\n")
    print("#" * 80)
    print("#  比較結果：Fine-tuned vs Original")
    print("#" * 80)

    for i, (name, code) in enumerate(test_cases):
        print(f"\n{'=' * 80}")
        print(f"測試 {i+1}: {name}")
        print(f"{'=' * 80}")
        print(f"程式碼：")
        print(code.strip())

        print(f"\n{'─' * 80}")
        print(f"Fine-tuned:")
        print(f"{'─' * 80}")
        print(ft_results[i])

        print(f"\n{'─' * 80}")
        print(f"Original (Qwen2.5-Coder-7B-Instruct):")
        print(f"{'─' * 80}")
        print(orig_results[i])

    print(f"\n{'=' * 80}")
    print("比較完成")


if __name__ == "__main__":
    main()
