import json
import random

# 讀入兩份資料
with open("code_review_training_data.json", encoding="utf-8") as f:
    synthetic = json.load(f)

with open("github_pr_training_data.json", encoding="utf-8") as f:
    github = json.load(f)

print(f"合成資料：{len(synthetic)} 筆")
print(f"GitHub PR：{len(github)} 筆")

# 合併 + 打亂
combined = synthetic + github
random.shuffle(combined)

# 存檔
with open("combined_training_data.json", "w", encoding="utf-8") as f:
    json.dump(combined, f, ensure_ascii=False, indent=2)

print(f"合併後：{len(combined)} 筆")
print("✅ 儲存到 combined_training_data.json")