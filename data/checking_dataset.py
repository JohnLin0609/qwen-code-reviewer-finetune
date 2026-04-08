import json
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).resolve().parent.parent

with open(BASE_DIR / "combined_training_data.json", encoding="utf-8") as f:
    data = json.load(f)

print(f"總筆數：{len(data)}")

# 來源分佈
sources = Counter(d["metadata"]["source"] for d in data)
for src, count in sources.items():
    print(f"  {src}: {count}")

# 輸入/輸出長度分佈
input_lens  = [len(d["input"]) for d in data]
output_lens = [len(d["output"]) for d in data]
print(f"\n輸入長度 avg={sum(input_lens)//len(input_lens)} max={max(input_lens)} min={min(input_lens)}")
print(f"輸出長度 avg={sum(output_lens)//len(output_lens)} max={max(output_lens)} min={min(output_lens)}")


### Find Anomaly
issues = []
for i, d in enumerate(data):
    # 欄位缺失
    if not d.get("instruction") or not d.get("input") or not d.get("output"):
        issues.append((i, "缺少欄位"))
    # 輸出太短（可能是沒過濾乾淨）
    elif len(d["output"]) < 20:
        issues.append((i, f"output 太短: {d['output']!r}"))
    # 輸入太長（超過模型 context 會被截斷）
    elif len(d["input"]) > 2000:
        issues.append((i, f"input 太長: {len(d['input'])} 字"))

print(f"異常筆數：{len(issues)}")
for i, reason in issues[:10]:
    print(f"  [{i}] {reason}")

### Random Pick
import random

samples = random.sample(data, 5)
for s in samples:
    print("="*60)
    print(f"來源: {s['metadata']['source']}")
    print(f"INPUT:\n{s['input'][:300]}")
    print(f"OUTPUT:\n{s['output'][:300]}")


### Check Format Valid
# 確認每筆都有三個必要欄位
valid = all(
    "instruction" in d and "input" in d and "output" in d
    for d in data
)
print(f"格式正確：{valid}")