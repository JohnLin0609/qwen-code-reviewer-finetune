"""
測試 fine-tuned 模型
====================
執行：python inference.py
"""

import warnings
warnings.filterwarnings("ignore")

import torch
from unsloth import FastLanguageModel

MODEL_PATH  = "./code-review-model-v7/lora"
MAX_SEQ_LEN = 2048

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

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)


def review_code(code: str, language: str = "Python") -> str:
    messages = [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": f"Review this {language} code for security vulnerabilities:\n\n{code}"},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    # apply_chat_template 可能回傳 tensor 或 BatchEncoding
    if hasattr(inputs, "input_ids"):
        input_ids = inputs.input_ids.to("cuda")
    else:
        input_ids = inputs.to("cuda")

    prompt_len = input_ids.shape[1]

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            use_cache=False,  # 關閉 KV cache，避免 Unsloth shape mismatch bug
        )

    # 只取模型生成的部分（去掉 input）
    generated = output[0][prompt_len:]
    return tokenizer.decode(generated, skip_special_tokens=True)


# ── 測試案例 ──────────────────────────────────────────────────────────────

test_cases = [
    # 單一問題：SQL Injection
    """
def get_user(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    return db.execute(query).fetchone()
""",
    # 多重問題：SQL Injection + 硬編碼密碼 + 連線洩漏
    """
def get_user(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    conn = psycopg2.connect(host='db.prod.internal', user='root', password='admin123')
    return conn.execute(query).fetchone()
""",
    # 多重問題：Command Injection + 敏感資訊寫入 Log + 無錯誤處理
    """
import subprocess
import logging

def deploy(repo_url, branch, token):
    logging.info(f'Deploying {repo_url} branch={branch} token={token}')
    cmd = f'git clone -b {branch} https://{token}@{repo_url} /deploy'
    subprocess.call(cmd, shell=True)
""",
    # 多重問題：路徑遍歷 + 無檔案類型驗證 + 無大小限制
    """
from flask import request

@app.route('/upload', methods=['POST'])
def upload():
    f = request.files['file']
    path = f'/uploads/{f.filename}'
    f.save(path)
    return f'Saved to {path}'
""",
]

for i, code in enumerate(test_cases, 1):
    print(f"\n{'='*60}")
    print(f"測試 {i}")
    print(f"程式碼：{code.strip()}")
    print(f"\nCode Review：")
    print(review_code(code.strip()))
