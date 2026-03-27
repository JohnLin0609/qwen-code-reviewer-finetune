"""
GitHub PR Code Review 資料抓取腳本 v2
======================================
改進：
- 更多 repo（20個）、更多 PR（每個 100 個）
- 同時抓 inline comments + review body（兩種來源）
- 放寬過濾條件，並顯示過濾原因統計
"""

import os
import json
import time
import requests
from dotenv import load_dotenv

load_dotenv()

# ── 設定 ──────────────────────────────────────────────────────────────────

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
if not GITHUB_TOKEN:
    raise EnvironmentError("請先設定 GITHUB_TOKEN 環境變數")

HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json",
}

TARGET_REPOS = [
    # 原有
    ("django", "django"),
    ("pallets", "flask"),
    ("psf", "requests"),
    ("tiangolo", "fastapi"),
    ("encode", "httpx"),
    ("python", "cpython"),
    ("celery", "celery"),
    # 新增
    ("numpy", "numpy"),
    ("pandas-dev", "pandas"),
    ("scikit-learn", "scikit-learn"),
    ("pytorch", "pytorch"),
    ("huggingface", "transformers"),
    ("pydantic", "pydantic"),
    ("sqlalchemy", "sqlalchemy"),
    ("aio-libs", "aiohttp"),
    ("pytest-dev", "pytest"),
    ("encode", "starlette"),
    ("python-poetry", "poetry"),
    ("pypa", "pip"),
    ("scrapy", "scrapy"),
]

MAX_PRS_PER_REPO = 100  # 從 30 提高到 100
OUTPUT_PATH = "github_pr_training_data.json"

# ── API ───────────────────────────────────────────────────────────────────

def api_get(url, params=None):
    while True:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=15)
        if resp.status_code == 403:
            reset_time = int(resp.headers.get("X-RateLimit-Reset", time.time() + 60))
            wait = max(reset_time - time.time(), 1)
            print(f"  ⚠ Rate limit，等待 {int(wait)}s...")
            time.sleep(wait)
            continue
        if resp.status_code in (404, 410):
            return None
        resp.raise_for_status()
        remaining = resp.headers.get("X-RateLimit-Remaining", "?")
        if remaining != "?" and int(remaining) < 50:
            print(f"  ⚠ API 剩餘配額僅剩 {remaining}，暫停 30s")
            time.sleep(30)
        return resp.json()


def get_closed_prs(owner, repo, max_count):
    prs = []
    page = 1
    while len(prs) < max_count:
        batch = api_get(
            f"https://api.github.com/repos/{owner}/{repo}/pulls",
            params={"state": "closed", "per_page": 100, "page": page,
                    "sort": "updated", "direction": "desc"},
        )
        if not batch:
            break
        prs.extend(batch)
        if len(batch) < 100:
            break
        page += 1
        time.sleep(0.3)
    return prs[:max_count]


def get_inline_comments(owner, repo, pr_number):
    """針對特定程式碼行的 review comments"""
    all_comments = []
    page = 1
    while True:
        batch = api_get(
            f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/comments",
            params={"per_page": 100, "page": page},
        ) or []
        all_comments.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return all_comments


def get_review_bodies(owner, repo, pr_number):
    """整份 review 的 body（非 inline，是整體評語）"""
    return api_get(
        f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/reviews",
        params={"per_page": 100},
    ) or []


def get_pr_files(owner, repo, pr_number):
    return api_get(
        f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/files",
        params={"per_page": 100},
    ) or []


# ── 過濾邏輯 ──────────────────────────────────────────────────────────────

SKIP_EXACT = {
    "lgtm", "+1", "-1", "👍", "👎", "✅", "done", "fixed", "thanks",
    "thank you", "agreed", "nice", "ok", "okay", ":+1:", ":-1:",
}

SKIP_CONTAINS = ["lgtm", "looks good to me", "ship it", ":shipit:"]

PYTHON_EXTENSIONS = {".py", ".pyi", ".pyx", ".pxd"}


def filter_reason(body):
    """回傳過濾原因字串，None 表示通過"""
    if not body or not body.strip():
        return "空白"
    text = body.strip()
    if len(text) < 20:
        return f"太短({len(text)}字)"
    if text.lower() in SKIP_EXACT:
        return "無意義回應"
    if any(kw in text.lower() for kw in SKIP_CONTAINS):
        return "LGTM 類"
    return None


def is_python_file(filename):
    return any(filename.endswith(ext) for ext in PYTHON_EXTENSIONS)


def clean_diff(diff_hunk):
    lines = [l for l in diff_hunk.split("\n") if not l.startswith("@@")]
    return "\n".join(lines).strip()


# ── 抓取邏輯 ──────────────────────────────────────────────────────────────

def fetch_repo(owner, repo):
    print(f"\n{'='*50}")
    print(f"🔍 {owner}/{repo}")

    result = []
    filter_stats = {}

    prs = get_closed_prs(owner, repo, MAX_PRS_PER_REPO)
    print(f"  PR 數量：{len(prs)}")

    for pr in prs:
        pr_number = pr["number"]
        pr_title  = pr.get("title", "")

        # ── 來源一：Inline comments（針對特定程式碼行）──
        for c in get_inline_comments(owner, repo, pr_number):
            path = c.get("path", "")
            if not is_python_file(path):
                continue

            body      = c.get("body", "")
            diff_hunk = c.get("diff_hunk", "")
            reason    = filter_reason(body)

            if reason:
                filter_stats[reason] = filter_stats.get(reason, 0) + 1
                continue
            if not diff_hunk or len(clean_diff(diff_hunk)) < 10:
                filter_stats["diff 太短"] = filter_stats.get("diff 太短", 0) + 1
                continue

            result.append({
                "instruction": "請對以下 Python 程式碼做 code review",
                "input": clean_diff(diff_hunk),
                "output": body.strip(),
                "metadata": {
                    "source": "github_inline_comment",
                    "repo": f"{owner}/{repo}",
                    "pr_number": pr_number,
                    "pr_title": pr_title,
                    "file": path,
                    "url": c.get("html_url", ""),
                }
            })

        # ── 來源二：Review body（整份 review 的整體評語）──
        files   = get_pr_files(owner, repo, pr_number)
        py_files = [f for f in files if is_python_file(f["filename"])]

        if py_files:
            first_patch = py_files[0].get("patch", "")
            context = clean_diff(first_patch)[:800] if first_patch else f"PR: {pr_title}"

            for review in get_review_bodies(owner, repo, pr_number):
                body   = review.get("body", "")
                state  = review.get("state", "")
                reason = filter_reason(body)

                if reason:
                    filter_stats[reason] = filter_stats.get(reason, 0) + 1
                    continue
                if len(body.strip()) < 50:
                    filter_stats["review body 太短"] = filter_stats.get("review body 太短", 0) + 1
                    continue

                result.append({
                    "instruction": "請對以下 Python 程式碼做整體 code review",
                    "input": context,
                    "output": body.strip(),
                    "metadata": {
                        "source": "github_review_body",
                        "repo": f"{owner}/{repo}",
                        "pr_number": pr_number,
                        "pr_title": pr_title,
                        "review_state": state,
                        "url": review.get("html_url", ""),
                    }
                })

        time.sleep(0.4)

    print(f"  ✅ 收集 {len(result)} 筆")
    if filter_stats:
        for reason, count in sorted(filter_stats.items(), key=lambda x: -x[1]):
            print(f"     過濾「{reason}」: {count} 筆")

    return result


# ── 主程式 ────────────────────────────────────────────────────────────────

def main():
    all_data = []

    for owner, repo in TARGET_REPOS:
        try:
            all_data.extend(fetch_repo(owner, repo))
        except Exception as e:
            print(f"  ❌ {owner}/{repo} 失敗：{e}")

    # 去重（同一個 comment url）
    seen, deduped = set(), []
    for d in all_data:
        url = d["metadata"].get("url", "")
        if url and url in seen:
            continue
        seen.add(url)
        deduped.append(d)

    print(f"\n{'='*50}")
    print(f"📊 總計：{len(all_data)} 筆，去重後 {len(deduped)} 筆")
    print(f"{'='*50}")
    repos = {}
    for d in deduped:
        r = d["metadata"]["repo"]
        repos[r] = repos.get(r, 0) + 1
    for r, c in sorted(repos.items(), key=lambda x: -x[1]):
        print(f"  {r}: {c} 筆")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(deduped, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 儲存到 {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
