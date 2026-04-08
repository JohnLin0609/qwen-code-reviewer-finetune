import json
import random
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# 讀入兩份資料
with open(BASE_DIR / "training_data" / "code_review_training_data.json", encoding="utf-8") as f:
    synthetic = json.load(f)

with open(BASE_DIR / "github_pr_training_data.json", encoding="utf-8") as f:
    github = json.load(f)

print(f"合成資料：{len(synthetic)} 筆")
print(f"GitHub PR：{len(github)} 筆")

# 合併 + 打亂
combined = synthetic + github
random.shuffle(combined)

# 存檔
with open(BASE_DIR / "combined_training_data.json", "w", encoding="utf-8") as f:
    json.dump(combined, f, ensure_ascii=False, indent=2)

print(f"合併後：{len(combined)} 筆")
print("✅ 儲存到 combined_training_data.json")