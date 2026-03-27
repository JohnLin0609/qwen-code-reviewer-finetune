import json

with open("combined_training_data.json", encoding="utf-8") as f:
    data = json.load(f)

# 已知的 bot 帳號
BOT_ACCOUNTS = {"sqla-tester", "github-actions", "codecov", "dependabot", "pre-commit-ci"}

# bot 回應的特徵字串
BOT_PATTERNS = [
    "sqla-tester", "setting up my work on behalf of",
    "codecov report", "coverage decreased", "pull request #",
    "this pull request", "ci bot", "github actions",
    "pre-commit", "lgtm bot", "mergify",
]

def is_bot_comment(text):
    text_lower = text.lower()
    return any(p in text_lower for p in BOT_PATTERNS)

MAX_INPUT_LEN = 2000

cleaned = []
removed = {"太長截斷": 0, "bot comment": 0, "output 空": 0}

for d in data:
    output = d["output"].strip()

    # 過濾 bot comment
    if is_bot_comment(output):
        removed["bot comment"] += 1
        continue

    # output 清洗後變空
    if not output:
        removed["output 空"] += 1
        continue

    # input 太長 → 截斷而非刪除
    if len(d["input"]) > MAX_INPUT_LEN:
        d["input"] = d["input"][:MAX_INPUT_LEN]
        removed["太長截斷"] += 1

    d["output"] = output
    cleaned.append(d)

print(f"清洗前：{len(data)} 筆")
print(f"清洗後：{len(cleaned)} 筆")
print(f"  bot comment 移除：{removed['bot comment']} 筆")
print(f"  output 空移除：{removed['output 空']} 筆")
print(f"  input 截斷（保留）：{removed['太長截斷']} 筆")

with open("cleaned_training_data.json", "w", encoding="utf-8") as f:
    json.dump(cleaned, f, ensure_ascii=False, indent=2)

print("✅ 儲存到 cleaned_training_data.json")