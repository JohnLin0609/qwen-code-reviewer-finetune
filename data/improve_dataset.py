"""
Dataset Improvement: improve_dataset.py
========================================
Addresses three problems identified during evaluation:

  Problem 1 — 100% false positive rate: no clean-code negative examples.
              Fix: add ~80 clean-code samples where output is {"issues": []}.

  Problem 2 — GitHub data is mostly English free-text with low security relevance.
              Fix: filter GitHub samples to keep only security-relevant ones.

  Problem 3 — Ratio imbalance between high-quality handcrafted and noisy GitHub.
              Fix: after filtering, the dataset is smaller but much more focused.

Usage:
    python data/improve_dataset.py
    python data/improve_dataset.py --dry-run
    python data/improve_dataset.py --keep-all-github   # skip GitHub filtering
    python data/improve_dataset.py --clean-count 100   # more clean examples
"""

import json
import re
import random
import argparse
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = str(BASE_DIR / "claude_cleaned_training_data_v3.json")
DEFAULT_OUTPUT = str(BASE_DIR / "claude_cleaned_training_data_v4.json")
RANDOM_SEED = 42
EN_RATIO = 0.45

# ── Instruction pools (same as fix_dataset.py) ──────────────────────────────

INSTRUCTIONS_ZH_FREETEXT = [
    "請對以下 Python 程式碼做 code review",
    "請審查以下 Python 程式碼並指出問題",
    "請對以下程式碼提供詳細的 code review",
    "請仔細檢查以下程式碼並提供改進建議",
    "請對以下程式碼進行安全性與品質審查",
]

INSTRUCTIONS_ZH_JSON = [
    "請對以下 Python 程式碼做 code review，以 JSON 格式回覆",
    "請審查以下程式碼，以 JSON 格式列出所有問題，包含嚴重程度與修正方式",
    "請對以下程式碼進行結構化 code review，回覆格式為 JSON",
]

INSTRUCTIONS_EN_FREETEXT = [
    "Please review the following Python code and provide feedback.",
    "Perform a code review on the following Python code.",
    "Review the following code diff and provide constructive feedback.",
    "Analyze the following code change and suggest improvements.",
    "Please review this code and identify any issues or improvements.",
    "Conduct a thorough code review of the following Python snippet.",
]

INSTRUCTIONS_EN_JSON = [
    "Please review the following Python code and respond in JSON format with issues, severity, and fixes.",
    "Perform a structured code review of the following code. Return results as JSON.",
    "Review this code and output a JSON object listing all issues with severity ratings.",
]


def is_json_output(text: str) -> bool:
    try:
        obj = json.loads(text.strip())
        return isinstance(obj, dict) and "issues" in obj
    except (json.JSONDecodeError, TypeError):
        return False


def assign_instruction(entry: dict, rng: random.Random) -> dict:
    use_en = rng.random() < EN_RATIO
    json_out = is_json_output(entry["output"])
    if use_en:
        pool = INSTRUCTIONS_EN_JSON if json_out else INSTRUCTIONS_EN_FREETEXT
    else:
        pool = INSTRUCTIONS_ZH_JSON if json_out else INSTRUCTIONS_ZH_FREETEXT
    entry["instruction"] = rng.choice(pool)
    return entry


# ═══════════════════════════════════════════════════════════════════════
# STEP 1: GitHub Security Filter
# ═══════════════════════════════════════════════════════════════════════

# Tier 1: Security keywords
SECURITY_KEYWORDS = re.compile(
    r"sql.?injection|cross.site|injection|vulnerab|exploit|insecure|"
    r"credential|authenticat|authori[zs]|encrypt|decrypt|sanitiz|"
    r"traversal|deseriali[zs]|hardcod|shell\s*=\s*True|pickle\.loads|"
    r"unsafe|security|secret.?key|password|token.?leak|"
    r"\bxss\b|\bcsrf\b|\brce\b|\bssrf\b|\bidos\b|"
    r"\beval\s*\(|\bexec\s*\(|"
    r"安全|漏洞|注入|密碼|憑證|敏感|洩漏|攻擊|惡意|加密",
    re.IGNORECASE,
)

# Tier 2: Code quality (adjacent to security)
QUALITY_KEYWORDS = re.compile(
    r"race.?condition|thread.?safe|deadlock|memory.?leak|"
    r"uncaught.?exception|input.?validat|boundary.?check|"
    r"buffer.?overflow|null.?pointer|type.?check|"
    r"error.?handl|exception.?handl|"
    r"競態|例外處理|輸入驗證",
    re.IGNORECASE,
)


def classify_github_sample(entry: dict) -> str | None:
    """Return 'security', 'quality', or None."""
    text = entry["input"] + " " + entry["output"]
    if SECURITY_KEYWORDS.search(text):
        return "security"
    if QUALITY_KEYWORDS.search(text):
        return "quality"
    return None


# ═══════════════════════════════════════════════════════════════════════
# STEP 2: Clean-Code Negative Examples
# ═══════════════════════════════════════════════════════════════════════

CLEAN_SUMMARIES = [
    "此程式碼結構清晰、安全性良好，未發現明顯問題。",
    "程式碼遵循安全最佳實務，未發現安全漏洞或品質問題。",
    "程式碼品質良好，具有適當的錯誤處理與安全防護措施。",
    "未偵測到安全漏洞或可靠性問題，程式碼可安全使用。",
    "此程式碼實作方式安全且穩健，無需修正。",
    "程式碼符合安全標準，防禦性程式設計完善，無建議修改。",
    "此段程式碼品質優良，遵循最佳實務，無安全疑慮。",
    "程式碼安全無虞，邏輯清晰，錯誤處理完善。",
]

# (category, code_snippet)
CLEAN_CODE_EXAMPLES = [
    # ── Category 1: Safe SQL (parameterized queries) ──────────────────
    ("safe_sql", '''
def get_user_by_id(db, user_id: int):
    query = "SELECT * FROM users WHERE id = %s"
    return db.execute(query, (user_id,)).fetchone()
'''),
    ("safe_sql", '''
def search_users(db, name: str, role: str):
    query = "SELECT id, name, email FROM users WHERE name LIKE %s AND role = %s"
    return db.execute(query, (f"%{name}%", role)).fetchall()
'''),
    ("safe_sql", '''
from sqlalchemy import select
from models import User

def get_active_users(session):
    stmt = select(User).where(User.is_active == True).order_by(User.created_at.desc())
    return session.scalars(stmt).all()
'''),
    ("safe_sql", '''
def insert_order(db, user_id: int, product_id: int, quantity: int):
    query = """
        INSERT INTO orders (user_id, product_id, quantity, created_at)
        VALUES (%s, %s, %s, NOW())
        RETURNING id
    """
    result = db.execute(query, (user_id, product_id, quantity))
    db.commit()
    return result.fetchone()[0]
'''),
    ("safe_sql", '''
def update_email(db, user_id: int, new_email: str):
    query = "UPDATE users SET email = ? WHERE id = ?"
    db.execute(query, (new_email, user_id))
    db.commit()
'''),
    ("safe_sql", '''
def delete_expired_sessions(db, cutoff_date):
    query = "DELETE FROM sessions WHERE expires_at < %s"
    result = db.execute(query, (cutoff_date,))
    db.commit()
    return result.rowcount
'''),
    ("safe_sql", '''
import sqlite3

def get_product(db_path: str, product_id: int) -> dict | None:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            "SELECT * FROM products WHERE id = ?", (product_id,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None
'''),
    ("safe_sql", '''
from sqlalchemy.orm import Session
from models import Order

def get_user_orders(session: Session, user_id: int, limit: int = 20):
    return (
        session.query(Order)
        .filter(Order.user_id == user_id)
        .order_by(Order.created_at.desc())
        .limit(limit)
        .all()
    )
'''),

    # ── Category 2: Safe credential management ────────────────────────
    ("safe_credentials", '''
import os

def get_db_connection():
    return psycopg2.connect(
        host=os.environ["DB_HOST"],
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
        dbname=os.environ["DB_NAME"],
    )
'''),
    ("safe_credentials", '''
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    api_key: str
    db_url: str
    secret_key: str

    class Config:
        env_file = ".env"

settings = Settings()
'''),
    ("safe_credentials", '''
import os
import boto3

def get_secret(secret_name: str) -> str:
    client = boto3.client("secretsmanager", region_name=os.environ["AWS_REGION"])
    response = client.get_secret_value(SecretId=secret_name)
    return response["SecretString"]
'''),
    ("safe_credentials", '''
import os
from functools import lru_cache

@lru_cache(maxsize=1)
def get_config() -> dict:
    return {
        "api_key": os.environ["API_KEY"],
        "api_secret": os.environ["API_SECRET"],
        "base_url": os.environ.get("API_BASE_URL", "https://api.example.com"),
    }
'''),
    ("safe_credentials", '''
import keyring

def get_service_token(service_name: str) -> str:
    token = keyring.get_password(service_name, "api_token")
    if token is None:
        raise RuntimeError(f"No token found for service: {service_name}")
    return token
'''),
    ("safe_credentials", '''
from pathlib import Path
from dotenv import load_dotenv
import os

def init_config():
    env_path = Path(__file__).parent / ".env"
    load_dotenv(env_path)
    required = ["DATABASE_URL", "REDIS_URL", "JWT_SECRET"]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        raise EnvironmentError(f"Missing required env vars: {', '.join(missing)}")
'''),
    ("safe_credentials", '''
import os
import hashlib
import hmac

def verify_webhook(payload: bytes, signature: str) -> bool:
    secret = os.environ["WEBHOOK_SECRET"].encode()
    expected = hmac.new(secret, payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)
'''),
    ("safe_credentials", '''
import os
import jwt

def create_jwt_token(user_id: int, role: str) -> str:
    payload = {"sub": user_id, "role": role}
    return jwt.encode(payload, os.environ["JWT_SECRET"], algorithm="HS256")

def decode_jwt_token(token: str) -> dict:
    return jwt.decode(token, os.environ["JWT_SECRET"], algorithms=["HS256"])
'''),

    # ── Category 3: Proper exception handling ─────────────────────────
    ("safe_exception", '''
import json

def read_config(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise RuntimeError(f"Config file not found: {path}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON in {path}: {e}")
'''),
    ("safe_exception", '''
import logging

logger = logging.getLogger(__name__)

def fetch_data(url: str, timeout: int = 10) -> dict:
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.Timeout:
        logger.error("Request to %s timed out after %ds", url, timeout)
        raise
    except requests.HTTPError as e:
        logger.error("HTTP error from %s: %s", url, e.response.status_code)
        raise
'''),
    ("safe_exception", '''
from contextlib import contextmanager
import sqlite3

@contextmanager
def db_transaction(db_path: str):
    conn = sqlite3.connect(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
'''),
    ("safe_exception", '''
def parse_int_safe(value: str, default: int = 0) -> int:
    try:
        return int(value)
    except (ValueError, TypeError):
        return default
'''),
    ("safe_exception", '''
import csv
import logging

logger = logging.getLogger(__name__)

def read_csv_rows(path: str) -> list[dict]:
    rows = []
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                rows.append(row)
    except FileNotFoundError:
        logger.warning("CSV file not found: %s", path)
    except csv.Error as e:
        logger.error("CSV parsing error in %s at row %d: %s", path, i, e)
    return rows
'''),
    ("safe_exception", '''
import socket

def check_port(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False
'''),
    ("safe_exception", '''
from typing import Any

def safe_get_nested(data: dict, *keys: str, default: Any = None) -> Any:
    current = data
    for key in keys:
        try:
            current = current[key]
        except (KeyError, TypeError, IndexError):
            return default
    return current
'''),
    ("safe_exception", '''
import yaml
import logging

logger = logging.getLogger(__name__)

def load_yaml_config(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        if not isinstance(config, dict):
            raise ValueError(f"Expected dict, got {type(config).__name__}")
        return config
    except FileNotFoundError:
        logger.error("Config file missing: %s", path)
        raise
    except yaml.YAMLError as e:
        logger.error("YAML parse error in %s: %s", path, e)
        raise
'''),

    # ── Category 4: Safe file/path handling ───────────────────────────
    ("safe_path", '''
from pathlib import Path

TEMPLATES_DIR = Path(__file__).parent / "templates"

def get_template(name: str) -> str:
    safe_path = (TEMPLATES_DIR / name).resolve()
    if not str(safe_path).startswith(str(TEMPLATES_DIR.resolve())):
        raise ValueError(f"Invalid template name: {name}")
    return safe_path.read_text(encoding="utf-8")
'''),
    ("safe_path", '''
from pathlib import Path
from werkzeug.utils import secure_filename

UPLOAD_DIR = Path("/app/uploads")

def save_upload(filename: str, content: bytes) -> Path:
    safe_name = secure_filename(filename)
    if not safe_name:
        raise ValueError("Invalid filename")
    dest = UPLOAD_DIR / safe_name
    dest.write_bytes(content)
    return dest
'''),
    ("safe_path", '''
from pathlib import Path
import tempfile

def write_temp_file(data: str, suffix: str = ".txt") -> Path:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=suffix, delete=False, encoding="utf-8"
    ) as f:
        f.write(data)
        return Path(f.name)
'''),
    ("safe_path", '''
from pathlib import Path

def list_python_files(directory: str) -> list[Path]:
    base = Path(directory).resolve()
    if not base.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")
    return sorted(base.rglob("*.py"))
'''),
    ("safe_path", '''
import os
from pathlib import Path

ALLOWED_EXTENSIONS = {".csv", ".json", ".txt"}

def validate_upload_path(filename: str, upload_dir: str) -> Path:
    base = Path(upload_dir).resolve()
    target = (base / Path(filename).name).resolve()
    if not str(target).startswith(str(base)):
        raise ValueError("Path traversal detected")
    if target.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError(f"File type not allowed: {target.suffix}")
    return target
'''),
    ("safe_path", '''
from pathlib import Path

def read_log_file(log_dir: str, date_str: str) -> str:
    base = Path(log_dir).resolve()
    filename = f"app-{date_str}.log"
    if "/" in date_str or "\\\\" in date_str:
        raise ValueError("Invalid date string")
    path = (base / filename).resolve()
    if not str(path).startswith(str(base)):
        raise ValueError("Path traversal attempt")
    return path.read_text(encoding="utf-8")
'''),
    ("safe_path", '''
import shutil
from pathlib import Path

def safe_copy(src: str, dest_dir: str) -> Path:
    src_path = Path(src).resolve()
    dest_path = Path(dest_dir).resolve() / src_path.name
    if not src_path.is_file():
        raise FileNotFoundError(f"Source file not found: {src}")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dest_path)
    return dest_path
'''),
    ("safe_path", '''
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

def load_dataset(name: str) -> str:
    allowed = {"train", "test", "validation"}
    if name not in allowed:
        raise ValueError(f"Unknown dataset: {name}. Must be one of {allowed}")
    path = DATA_DIR / f"{name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    return path.read_text(encoding="utf-8")
'''),

    # ── Category 5: Safe subprocess usage ─────────────────────────────
    ("safe_subprocess", '''
import subprocess

def ping_host(host: str) -> bool:
    if not re.match(r"^[a-zA-Z0-9._-]+$", host):
        raise ValueError(f"Invalid hostname: {host}")
    result = subprocess.run(
        ["ping", "-c", "1", "-W", "3", host],
        capture_output=True, text=True, timeout=10,
    )
    return result.returncode == 0
'''),
    ("safe_subprocess", '''
import subprocess
import shlex

def run_git_command(repo_path: str, *args: str) -> str:
    cmd = ["git", "-C", repo_path] + list(args)
    result = subprocess.run(
        cmd, capture_output=True, text=True, check=True, timeout=30,
    )
    return result.stdout.strip()
'''),
    ("safe_subprocess", '''
import subprocess
from pathlib import Path

def run_linter(file_path: str) -> list[str]:
    path = Path(file_path).resolve()
    if not path.exists() or path.suffix != ".py":
        raise ValueError(f"Invalid Python file: {file_path}")
    result = subprocess.run(
        ["flake8", "--max-line-length=120", str(path)],
        capture_output=True, text=True, timeout=60,
    )
    return result.stdout.strip().splitlines()
'''),
    ("safe_subprocess", '''
import subprocess

def convert_image(input_path: str, output_path: str, width: int) -> None:
    if not isinstance(width, int) or width <= 0 or width > 4096:
        raise ValueError(f"Invalid width: {width}")
    subprocess.run(
        ["convert", input_path, "-resize", f"{width}x", output_path],
        check=True, timeout=30,
    )
'''),

    # ── Category 6: Secure random / crypto ────────────────────────────
    ("safe_crypto", '''
import secrets

def generate_api_key() -> str:
    return secrets.token_urlsafe(32)
'''),
    ("safe_crypto", '''
import secrets
import hashlib

def hash_password(password: str) -> tuple[str, str]:
    salt = secrets.token_hex(16)
    hashed = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000)
    return salt, hashed.hex()

def verify_password(password: str, salt: str, expected_hash: str) -> bool:
    hashed = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000)
    return secrets.compare_digest(hashed.hex(), expected_hash)
'''),
    ("safe_crypto", '''
import secrets
import string

def generate_temp_password(length: int = 16) -> str:
    alphabet = string.ascii_letters + string.digits + "!@#$%"
    return "".join(secrets.choice(alphabet) for _ in range(length))
'''),
    ("safe_crypto", '''
from cryptography.fernet import Fernet
import os

def encrypt_data(plaintext: str) -> bytes:
    key = os.environ["ENCRYPTION_KEY"].encode()
    f = Fernet(key)
    return f.encrypt(plaintext.encode())

def decrypt_data(ciphertext: bytes) -> str:
    key = os.environ["ENCRYPTION_KEY"].encode()
    f = Fernet(key)
    return f.decrypt(ciphertext).decode()
'''),

    # ── Category 7: Safe serialization ────────────────────────────────
    ("safe_serialization", '''
import json

def save_data(data: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_data(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
'''),
    ("safe_serialization", '''
import yaml

def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
'''),
    ("safe_serialization", '''
import json
from dataclasses import dataclass, asdict

@dataclass
class UserProfile:
    name: str
    email: str
    age: int

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, data: str) -> "UserProfile":
        obj = json.loads(data)
        return cls(**obj)
'''),
    ("safe_serialization", '''
from pydantic import BaseModel

class EventPayload(BaseModel):
    event_type: str
    timestamp: float
    data: dict

def parse_event(raw: str) -> EventPayload:
    return EventPayload.model_validate_json(raw)
'''),

    # ── Category 8: Thread-safe code ──────────────────────────────────
    ("thread_safe", '''
import threading

class Counter:
    def __init__(self):
        self._value = 0
        self._lock = threading.Lock()

    def increment(self) -> int:
        with self._lock:
            self._value += 1
            return self._value

    @property
    def value(self) -> int:
        with self._lock:
            return self._value
'''),
    ("thread_safe", '''
from queue import Queue
from threading import Thread

def process_items(items: list, num_workers: int = 4) -> list:
    q = Queue()
    results = []
    lock = threading.Lock()

    def worker():
        while True:
            item = q.get()
            if item is None:
                break
            result = transform(item)
            with lock:
                results.append(result)
            q.task_done()

    threads = [Thread(target=worker, daemon=True) for _ in range(num_workers)]
    for t in threads:
        t.start()
    for item in items:
        q.put(item)
    q.join()
    for _ in range(num_workers):
        q.put(None)
    for t in threads:
        t.join()
    return results
'''),
    ("thread_safe", '''
import threading
from functools import lru_cache

_init_lock = threading.Lock()
_client = None

def get_client():
    global _client
    if _client is None:
        with _init_lock:
            if _client is None:
                _client = create_expensive_client()
    return _client
'''),
    ("thread_safe", '''
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

logger = logging.getLogger(__name__)

def fetch_all_urls(urls: list[str], max_workers: int = 8) -> dict[str, str]:
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_url = {pool.submit(requests.get, url, timeout=10): url for url in urls}
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                results[url] = future.result().text
            except Exception as e:
                logger.error("Failed to fetch %s: %s", url, e)
    return results
'''),

    # ── Category 9: Input validation ──────────────────────────────────
    ("input_validation", '''
def create_user(name: str, email: str, age: int) -> dict:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Name must be a non-empty string")
    if not isinstance(email, str) or "@" not in email:
        raise ValueError("Invalid email address")
    if not isinstance(age, int) or age < 0 or age > 150:
        raise ValueError("Age must be between 0 and 150")
    return {"name": name.strip(), "email": email.lower().strip(), "age": age}
'''),
    ("input_validation", '''
import re

EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$")

def validate_email(email: str) -> str:
    email = email.strip().lower()
    if not EMAIL_REGEX.match(email):
        raise ValueError(f"Invalid email format: {email}")
    return email
'''),
    ("input_validation", '''
from enum import Enum

class SortOrder(str, Enum):
    ASC = "asc"
    DESC = "desc"

def get_sorted_items(items: list, sort_by: str, order: str = "asc") -> list:
    try:
        sort_order = SortOrder(order.lower())
    except ValueError:
        raise ValueError(f"Invalid sort order: {order}. Must be 'asc' or 'desc'")
    return sorted(items, key=lambda x: x.get(sort_by, ""), reverse=(sort_order == SortOrder.DESC))
'''),
    ("input_validation", '''
from typing import Any

def paginate(items: list[Any], page: int = 1, per_page: int = 20) -> dict:
    if page < 1:
        raise ValueError("Page must be >= 1")
    if per_page < 1 or per_page > 100:
        raise ValueError("per_page must be between 1 and 100")
    start = (page - 1) * per_page
    end = start + per_page
    return {
        "items": items[start:end],
        "total": len(items),
        "page": page,
        "per_page": per_page,
        "pages": (len(items) + per_page - 1) // per_page,
    }
'''),

    # ── Category 10: General clean code ───────────────────────────────
    ("clean_general", '''
from dataclasses import dataclass
from datetime import datetime

@dataclass
class LogEntry:
    level: str
    message: str
    timestamp: datetime

    def format(self) -> str:
        ts = self.timestamp.isoformat()
        return f"[{ts}] {self.level.upper()}: {self.message}"
'''),
    ("clean_general", '''
from contextlib import contextmanager
import time
import logging

logger = logging.getLogger(__name__)

@contextmanager
def timer(label: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info("%s took %.3fs", label, elapsed)
'''),
    ("clean_general", '''
from functools import wraps
import time
import logging

logger = logging.getLogger(__name__)

def retry(max_attempts: int = 3, delay: float = 1.0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts:
                        raise
                    logger.warning("Attempt %d/%d failed: %s", attempt, max_attempts, e)
                    time.sleep(delay * attempt)
        return wrapper
    return decorator
'''),
    ("clean_general", '''
from typing import Iterator

def chunked(iterable: list, size: int) -> Iterator[list]:
    if size <= 0:
        raise ValueError("Chunk size must be positive")
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]
'''),
    ("clean_general", '''
import hashlib
from pathlib import Path

def file_checksum(path: str, algorithm: str = "sha256") -> str:
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
'''),
    ("clean_general", '''
import logging
from typing import Any

logger = logging.getLogger(__name__)

class Registry:
    def __init__(self):
        self._handlers: dict[str, Any] = {}

    def register(self, name: str, handler: Any) -> None:
        if name in self._handlers:
            raise ValueError(f"Handler already registered: {name}")
        self._handlers[name] = handler
        logger.info("Registered handler: %s", name)

    def get(self, name: str) -> Any:
        if name not in self._handlers:
            raise KeyError(f"Unknown handler: {name}")
        return self._handlers[name]
'''),
    ("clean_general", '''
from datetime import datetime, timezone

def utc_now() -> datetime:
    return datetime.now(timezone.utc)

def format_relative_time(dt: datetime) -> str:
    delta = utc_now() - dt
    seconds = int(delta.total_seconds())
    if seconds < 60:
        return "剛剛"
    elif seconds < 3600:
        return f"{seconds // 60} 分鐘前"
    elif seconds < 86400:
        return f"{seconds // 3600} 小時前"
    else:
        return f"{seconds // 86400} 天前"
'''),
    ("clean_general", '''
from collections import defaultdict

def group_by(items: list[dict], key: str) -> dict[str, list[dict]]:
    groups = defaultdict(list)
    for item in items:
        groups[item[key]].append(item)
    return dict(groups)
'''),
]


def build_clean_output(rng: random.Random) -> str:
    """Build a structured JSON output for clean code."""
    score = rng.choice([8, 9, 9, 10])
    summary = rng.choice(CLEAN_SUMMARIES)
    obj = {"issues": [], "overall_score": score, "summary": summary}
    return json.dumps(obj, ensure_ascii=False, indent=2)


def generate_clean_examples(rng: random.Random, count: int) -> list[dict]:
    """Generate clean-code negative training examples."""
    examples = []
    pool = list(CLEAN_CODE_EXAMPLES)
    rng.shuffle(pool)

    for i in range(min(count, len(pool))):
        category, code = pool[i]
        entry = {
            "instruction": "",  # will be assigned later
            "input": code.strip(),
            "output": build_clean_output(rng),
            "metadata": {
                "source": "synthetic_clean",
                "category": category,
            },
        }
        examples.append(entry)

    return examples


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def print_report(
    original: list,
    handcrafted: list,
    github_kept: list,
    github_removed: int,
    github_tiers: Counter,
    clean_examples: list,
    final: list,
) -> None:
    print("\n" + "=" * 62)
    print("  DATASET IMPROVEMENT REPORT")
    print("=" * 62)

    print(f"\n  Original (v3)            : {len(original):>6}")

    # Handcrafted
    print(f"\n  Handcrafted (kept as-is) : {len(handcrafted):>6}")

    # GitHub filtering
    total_gh = len(github_kept) + github_removed
    print(f"\n  GitHub original          : {total_gh:>6}")
    print(f"    Kept (security)        : {github_tiers.get('security', 0):>6}")
    print(f"    Kept (quality)         : {github_tiers.get('quality', 0):>6}")
    print(f"    Removed (irrelevant)   : {github_removed:>6}")

    # Clean examples
    categories = Counter(d["metadata"]["category"] for d in clean_examples)
    print(f"\n  Clean-code negatives     : {len(clean_examples):>6}")
    for cat, cnt in sorted(categories.items()):
        print(f"    {cat:<24}: {cnt:>4}")

    # Final
    print(f"\n  {'─' * 40}")
    print(f"  Final dataset (v4)       : {len(final):>6}")
    print(f"  Change from v3           : {len(final) - len(original):>+6}")

    # Composition
    sources = Counter(d["metadata"]["source"] for d in final)
    print(f"\n  Source breakdown:")
    for src, cnt in sources.most_common():
        print(f"    {src:<28}: {cnt:>4} ({100*cnt/len(final):.1f}%)")

    # Instruction language
    zh = sum(1 for d in final if any(c > "\u4e00" for c in d["instruction"]))
    en = len(final) - zh
    print(f"\n  Instruction language:")
    print(f"    Chinese : {zh:>5} ({100*zh/len(final):.1f}%)")
    print(f"    English : {en:>5} ({100*en/len(final):.1f}%)")

    # Output format
    json_count = sum(1 for d in final if is_json_output(d["output"]))
    print(f"\n  Output format:")
    print(f"    Structured JSON : {json_count:>5} ({100*json_count/len(final):.1f}%)")
    print(f"    Free text       : {len(final)-json_count:>5} ({100*(len(final)-json_count)/len(final):.1f}%)")

    # Length stats
    out_lens = [len(d["output"]) for d in final]
    print(f"\n  Output length (chars):")
    print(f"    Min    : {min(out_lens)}")
    print(f"    Avg    : {sum(out_lens) // len(out_lens)}")
    print(f"    Median : {sorted(out_lens)[len(out_lens)//2]}")
    print(f"    Max    : {max(out_lens)}")

    print("=" * 62 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Improve dataset for v4")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-all-github", action="store_true",
                        help="Skip GitHub filtering (keep all)")
    parser.add_argument("--security-only", action="store_true",
                        help="Keep only tier-1 security matches from GitHub")
    parser.add_argument("--clean-count", type=int, default=80,
                        help="Number of clean-code examples to add")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # ── Load ──────────────────────────────────────────────────────────
    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples from {args.input}")

    # ── Split by source ───────────────────────────────────────────────
    handcrafted = [d for d in data if "handcrafted" in d["metadata"].get("source", "")]
    github = [d for d in data if "github" in d["metadata"].get("source", "")]
    print(f"  Handcrafted: {len(handcrafted)}, GitHub: {len(github)}")

    # ── Filter GitHub ─────────────────────────────────────────────────
    github_tiers = Counter()
    if args.keep_all_github:
        github_kept = github
        github_removed = 0
        print("GitHub filtering: SKIPPED (--keep-all-github)")
    else:
        github_kept = []
        for entry in github:
            tier = classify_github_sample(entry)
            if tier == "security":
                github_tiers["security"] += 1
                github_kept.append(entry)
            elif tier == "quality" and not args.security_only:
                github_tiers["quality"] += 1
                github_kept.append(entry)
        github_removed = len(github) - len(github_kept)
        print(f"GitHub filtering: {len(github_kept)} kept, {github_removed} removed")

    # ── Generate clean-code examples ──────────────────────────────────
    clean_examples = generate_clean_examples(rng, args.clean_count)
    print(f"Clean-code negatives generated: {len(clean_examples)}")

    # ── Combine ───────────────────────────────────────────────────────
    final = handcrafted + github_kept + clean_examples

    # ── Assign instructions ───────────────────────────────────────────
    for entry in final:
        assign_instruction(entry, rng)

    # ── Shuffle ───────────────────────────────────────────────────────
    rng.shuffle(final)

    # ── Report ────────────────────────────────────────────────────────
    print_report(data, handcrafted, github_kept, github_removed,
                 github_tiers, clean_examples, final)

    # ── Save ──────────────────────────────────────────────────────────
    if args.dry_run:
        print("[dry-run] No file saved.")
        return

    out_path = Path(args.output)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)
    print(f"Saved → {out_path.resolve()}")


if __name__ == "__main__":
    main()
