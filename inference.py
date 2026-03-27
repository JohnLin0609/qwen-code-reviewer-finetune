"""
測試 fine-tuned 模型
====================
執行：python inference.py
"""

import warnings
warnings.filterwarnings("ignore")

import torch
from unsloth import FastLanguageModel

MODEL_PATH  = "./code-review-model/lora"
MAX_SEQ_LEN = 2048

# 載入 fine-tuned 模型（Unsloth 載入省 VRAM）
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)


def review_code(code: str) -> str:
    messages = [
        {"role": "system",  "content": "你是資深軟體工程師，專精程式碼審查與資安。請提供具體、有建設性的 code review。"},
        {"role": "user",    "content": f"請對以下 Python 程式碼做 code review：\n\n```python\n{code}\n```"},
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
            max_new_tokens=512,
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
    # SQL Injection
    """
def get_user(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    return db.execute(query).fetchone()
""",
    # 硬編碼密碼
    """
def connect():
    return psycopg2.connect(
        host="localhost",
        user="admin",
        password="supersecret123"
    )
""",
    # 沒有 exception handling
    """
def read_config(path):
    with open(path) as f:
        return json.load(f)
""",
]

for i, code in enumerate(test_cases, 1):
    print(f"\n{'='*60}")
    print(f"測試 {i}")
    print(f"程式碼：{code.strip()}")
    print(f"\nCode Review：")
    print(review_code(code.strip()))
