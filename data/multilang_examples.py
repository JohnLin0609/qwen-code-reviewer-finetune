"""
Multi-language security review training examples (English output).

Covers C, C++, Java, JavaScript, Go — the languages missing from the original
Python-only handcrafted dataset. Output is structured JSON in English to
complement the existing Chinese handcrafted examples.

Each example has the same schema as training_data/code_review_training_data.json:
    {
      "instruction": <str>,
      "input": <code>,
      "output": <JSON string>,
      "metadata": {"source": "handcrafted_multilang", "language": <str>, "cwe": <str>}
    }
"""

import json


def _pack(instruction, code, review_dict, language, cwe, source="handcrafted_multilang"):
    return {
        "instruction": instruction,
        "input": code,
        "output": json.dumps(review_dict, ensure_ascii=False, indent=2),
        "metadata": {"source": source, "language": language, "cwe": cwe},
    }


_INSTR = "Review this code for security vulnerabilities. Respond in JSON format with issues (type, severity, description, suggestion, fixed_code), overall_score (1-10), and summary."
_INSTR_CLEAN = "Review this code for security issues. Respond in JSON format."


# ══════════════════════════════════════════════════════════════════════════
# C — vulnerable examples
# ══════════════════════════════════════════════════════════════════════════

C_EXAMPLES = [
    _pack(_INSTR,
        '''#include <string.h>
void copy_name(const char *src) {
    char dest[10];
    strcpy(dest, src);
    printf("Name: %s\\n", dest);
}''',
        {
            "issues": [{
                "type": "Buffer Overflow",
                "severity": "High",
                "description": "strcpy copies unbounded input into a 10-byte stack buffer. Inputs longer than 9 bytes overflow adjacent stack memory, enabling arbitrary code execution via stack smashing.",
                "suggestion": "Use strncpy with an explicit size bound, or strlcpy on BSD/modern systems. Always reserve one byte for the null terminator.",
                "fixed_code": "#include <string.h>\nvoid copy_name(const char *src) {\n    char dest[10];\n    strncpy(dest, src, sizeof(dest) - 1);\n    dest[sizeof(dest) - 1] = '\\0';\n    printf(\"Name: %s\\n\", dest);\n}"
            }],
            "overall_score": 1,
            "summary": "VULNERABILITY FOUND: classic stack-buffer-overflow (CWE-121) via strcpy. Replace with bounded copy."
        },
        language="c", cwe="CWE-121"),

    _pack(_INSTR,
        '''#include <stdio.h>
void log_message(const char *user_input) {
    printf(user_input);
}''',
        {
            "issues": [{
                "type": "Format String Vulnerability",
                "severity": "High",
                "description": "printf is called with user-controlled data as the format string. Attackers can inject %s/%n specifiers to read arbitrary memory or write to arbitrary addresses.",
                "suggestion": "Always pass user data as arguments, never as the format string: printf(\"%s\", user_input).",
                "fixed_code": "#include <stdio.h>\nvoid log_message(const char *user_input) {\n    printf(\"%s\", user_input);\n}"
            }],
            "overall_score": 2,
            "summary": "VULNERABILITY FOUND: format-string vulnerability (CWE-134). User input must never be the format argument."
        },
        language="c", cwe="CWE-134"),

    _pack(_INSTR,
        '''#include <stdlib.h>
int *get_buffer(size_t n) {
    int *buf = malloc(n * sizeof(int));
    for (size_t i = 0; i < n; i++) buf[i] = i;
    return buf;
}''',
        {
            "issues": [
                {
                    "type": "Missing Null Check",
                    "severity": "High",
                    "description": "malloc may return NULL on allocation failure, but buf[i] is dereferenced without checking. This causes a NULL pointer dereference and crash, or worse on platforms where NULL is mapped.",
                    "suggestion": "Check malloc's return value before dereferencing.",
                    "fixed_code": "int *buf = malloc(n * sizeof(int));\nif (!buf) return NULL;"
                },
                {
                    "type": "Integer Overflow",
                    "severity": "High",
                    "description": "n * sizeof(int) can overflow size_t for large n, causing a smaller-than-expected allocation followed by out-of-bounds writes.",
                    "suggestion": "Validate n against SIZE_MAX / sizeof(int) before multiplying, or use calloc which performs overflow-safe multiplication.",
                    "fixed_code": "if (n > SIZE_MAX / sizeof(int)) return NULL;\nint *buf = calloc(n, sizeof(int));"
                }
            ],
            "overall_score": 2,
            "summary": "VULNERABILITY FOUND: two issues — missing malloc null check (CWE-476) and integer overflow in size calculation (CWE-190)."
        },
        language="c", cwe="CWE-476,CWE-190"),

    _pack(_INSTR,
        '''#include <stdio.h>
void read_file(const char *path) {
    FILE *f = fopen(path, "r");
    char buf[256];
    fgets(buf, sizeof(buf), f);
    printf("%s", buf);
    fclose(f);
}''',
        {
            "issues": [{
                "type": "Missing Null Check on fopen",
                "severity": "High",
                "description": "fopen returns NULL when the file cannot be opened (does not exist, no permission). fgets on NULL is undefined behavior and will crash.",
                "suggestion": "Check fopen's return value; return or handle error before using the FILE*.",
                "fixed_code": "FILE *f = fopen(path, \"r\");\nif (!f) { perror(\"fopen\"); return; }"
            }],
            "overall_score": 4,
            "summary": "VULNERABILITY FOUND: missing NULL check after fopen (CWE-476)."
        },
        language="c", cwe="CWE-476"),

    _pack(_INSTR,
        '''#include <stdlib.h>
#include <string.h>
char *duplicate(const char *s) {
    char *copy = malloc(strlen(s));
    strcpy(copy, s);
    return copy;
}''',
        {
            "issues": [
                {
                    "type": "Off-by-One Buffer Overflow",
                    "severity": "High",
                    "description": "malloc(strlen(s)) allocates one byte too few — strlen excludes the null terminator. strcpy then writes one byte past the allocation (heap overflow).",
                    "suggestion": "Allocate strlen(s) + 1 bytes, or use strdup which handles this correctly.",
                    "fixed_code": "char *copy = strdup(s);  // or: malloc(strlen(s) + 1)"
                },
                {
                    "type": "Missing Null Check",
                    "severity": "Medium",
                    "description": "malloc return value not checked before strcpy.",
                    "suggestion": "Check for NULL before writing.",
                    "fixed_code": "if (!copy) return NULL;"
                }
            ],
            "overall_score": 2,
            "summary": "VULNERABILITY FOUND: off-by-one heap overflow (CWE-193) plus missing null check (CWE-476)."
        },
        language="c", cwe="CWE-193"),
]


# ══════════════════════════════════════════════════════════════════════════
# C++ — vulnerable examples
# ══════════════════════════════════════════════════════════════════════════

CPP_EXAMPLES = [
    _pack(_INSTR,
        '''#include <vector>
void remove_evens(std::vector<int>& v) {
    for (auto it = v.begin(); it != v.end(); ++it) {
        if (*it % 2 == 0) v.erase(it);
    }
}''',
        {
            "issues": [{
                "type": "Iterator Invalidation",
                "severity": "High",
                "description": "vector::erase invalidates the iterator it and all iterators past the erase point. Continuing to use 'it' after erase leads to undefined behavior (often crashes or skipped elements).",
                "suggestion": "erase returns the next valid iterator; assign it back. Use the erase-remove idiom or C++20 std::erase_if for clarity.",
                "fixed_code": "for (auto it = v.begin(); it != v.end(); ) {\n    if (*it % 2 == 0) it = v.erase(it);\n    else ++it;\n}\n// Or (C++20): std::erase_if(v, [](int x){ return x % 2 == 0; });"
            }],
            "overall_score": 3,
            "summary": "VULNERABILITY FOUND: iterator invalidation after erase (CWE-416 variant)."
        },
        language="cpp", cwe="CWE-416"),

    _pack(_INSTR,
        '''#include <memory>
class Widget { public: int *data; Widget() { data = new int[100]; } };
void process() {
    Widget *w = new Widget();
    if (w->data[0] < 0) return;
    delete w;
}''',
        {
            "issues": [
                {
                    "type": "Memory Leak on Early Return",
                    "severity": "High",
                    "description": "When data[0] < 0, the function returns without deleting w, leaking the Widget and its internal buffer. Every negative input leaks memory.",
                    "suggestion": "Use std::unique_ptr for automatic cleanup regardless of exit path.",
                    "fixed_code": "auto w = std::make_unique<Widget>();\nif (w->data[0] < 0) return;  // cleanup is automatic"
                },
                {
                    "type": "Memory Leak in Destructor",
                    "severity": "Medium",
                    "description": "Widget allocates new int[100] in the constructor but has no destructor to delete[] it. Even successful deletion of Widget leaks the array.",
                    "suggestion": "Follow the Rule of Zero: use std::vector<int> instead of raw arrays.",
                    "fixed_code": "class Widget { std::vector<int> data = std::vector<int>(100); };"
                }
            ],
            "overall_score": 2,
            "summary": "VULNERABILITY FOUND: two memory leaks (CWE-401). Use RAII/smart pointers to fix both."
        },
        language="cpp", cwe="CWE-401"),

    _pack(_INSTR,
        '''int sum(int n) {
    int total;
    for (int i = 0; i < n; ++i) total += i;
    return total;
}''',
        {
            "issues": [{
                "type": "Use of Uninitialized Variable",
                "severity": "High",
                "description": "'total' is never initialized before being read and written in the loop. Reading an uninitialized automatic variable is undefined behavior — the result depends on stack contents.",
                "suggestion": "Always initialize local variables. Prefer brace-initialization to zero.",
                "fixed_code": "int total = 0;"
            }],
            "overall_score": 2,
            "summary": "VULNERABILITY FOUND: use of uninitialized stack variable (CWE-457)."
        },
        language="cpp", cwe="CWE-457"),
]


# ══════════════════════════════════════════════════════════════════════════
# Java — vulnerable examples
# ══════════════════════════════════════════════════════════════════════════

JAVA_EXAMPLES = [
    _pack(_INSTR,
        '''public User getUser(String username) {
    String sql = "SELECT * FROM users WHERE name = '" + username + "'";
    return jdbcTemplate.queryForObject(sql, new UserRowMapper());
}''',
        {
            "issues": [{
                "type": "SQL Injection",
                "severity": "High",
                "description": "username is concatenated directly into the SQL string. Attacker input like `' OR '1'='1` bypasses auth; `'; DROP TABLE users;--` destroys data.",
                "suggestion": "Use parameterized queries with PreparedStatement placeholders, or JdbcTemplate with '?' and an args array.",
                "fixed_code": "String sql = \"SELECT * FROM users WHERE name = ?\";\nreturn jdbcTemplate.queryForObject(sql, new Object[]{username}, new UserRowMapper());"
            }],
            "overall_score": 1,
            "summary": "VULNERABILITY FOUND: SQL injection via string concatenation (CWE-89)."
        },
        language="java", cwe="CWE-89"),

    _pack(_INSTR,
        '''import javax.xml.parsers.DocumentBuilderFactory;
public Document parseXml(String xml) throws Exception {
    DocumentBuilderFactory dbf = DocumentBuilderFactory.newInstance();
    return dbf.newDocumentBuilder().parse(new InputSource(new StringReader(xml)));
}''',
        {
            "issues": [{
                "type": "XML External Entity (XXE)",
                "severity": "High",
                "description": "Default DocumentBuilderFactory processes external entities and DOCTYPE declarations. An attacker can exfiltrate files via <!ENTITY % file SYSTEM 'file:///etc/passwd'> or trigger SSRF via external DTDs.",
                "suggestion": "Disable DTDs and external entities before creating the builder.",
                "fixed_code": "dbf.setFeature(\"http://apache.org/xml/features/disallow-doctype-decl\", true);\ndbf.setFeature(\"http://xml.org/sax/features/external-general-entities\", false);\ndbf.setFeature(\"http://xml.org/sax/features/external-parameter-entities\", false);\ndbf.setXIncludeAware(false);\ndbf.setExpandEntityReferences(false);"
            }],
            "overall_score": 1,
            "summary": "VULNERABILITY FOUND: XXE vulnerability (CWE-611) — parser defaults are unsafe."
        },
        language="java", cwe="CWE-611"),

    _pack(_INSTR,
        '''import java.io.*;
public Object loadFromFile(String path) throws Exception {
    try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(path))) {
        return ois.readObject();
    }
}''',
        {
            "issues": [{
                "type": "Insecure Deserialization",
                "severity": "High",
                "description": "ObjectInputStream.readObject deserializes arbitrary class types. Attacker-controlled files can invoke gadget chains (e.g., CommonsCollections, Spring) to achieve remote code execution.",
                "suggestion": "Avoid Java serialization for untrusted input. Use a safe format like JSON (Jackson) or Protobuf. If serialization is required, use ObjectInputFilter (Java 9+) to restrict allowed classes.",
                "fixed_code": "ObjectInputFilter filter = ObjectInputFilter.Config.createFilter(\"com.myapp.*;!*\");\nois.setObjectInputFilter(filter);"
            }],
            "overall_score": 1,
            "summary": "VULNERABILITY FOUND: unsafe Java deserialization (CWE-502) — a common RCE vector."
        },
        language="java", cwe="CWE-502"),

    _pack(_INSTR,
        '''public String hashPassword(String password) {
    MessageDigest md = MessageDigest.getInstance("MD5");
    byte[] hash = md.digest(password.getBytes());
    return Base64.getEncoder().encodeToString(hash);
}''',
        {
            "issues": [
                {
                    "type": "Weak Password Hashing",
                    "severity": "High",
                    "description": "MD5 is cryptographically broken and also not designed for password hashing — it is too fast, making brute-force trivial with modern GPUs.",
                    "suggestion": "Use a password-specific KDF: bcrypt, scrypt, or Argon2 (preferred). Include an automatic per-password salt.",
                    "fixed_code": "import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;\nPasswordEncoder encoder = new BCryptPasswordEncoder(12);\nString hash = encoder.encode(password);"
                },
                {
                    "type": "Missing Salt",
                    "severity": "High",
                    "description": "No salt is used, so identical passwords produce identical hashes — vulnerable to rainbow-table attacks.",
                    "suggestion": "Use a KDF that automatically salts (bcrypt, Argon2).",
                    "fixed_code": "// BCryptPasswordEncoder handles salt automatically"
                }
            ],
            "overall_score": 1,
            "summary": "VULNERABILITY FOUND: weak password hashing (CWE-327) and missing salt (CWE-759)."
        },
        language="java", cwe="CWE-327,CWE-759"),

    _pack(_INSTR,
        '''public byte[] readFile(String fileName) throws IOException {
    File base = new File("/var/www/uploads/");
    File target = new File(base, fileName);
    return Files.readAllBytes(target.toPath());
}''',
        {
            "issues": [{
                "type": "Path Traversal",
                "severity": "High",
                "description": "fileName is joined to the base directory without normalization. Input like '../../etc/passwd' resolves outside /var/www/uploads, allowing arbitrary file reads.",
                "suggestion": "Canonicalize the resolved path and verify it stays under the base directory.",
                "fixed_code": "Path base = Paths.get(\"/var/www/uploads/\").toAbsolutePath().normalize();\nPath target = base.resolve(fileName).toAbsolutePath().normalize();\nif (!target.startsWith(base)) throw new SecurityException(\"Path traversal\");\nreturn Files.readAllBytes(target);"
            }],
            "overall_score": 1,
            "summary": "VULNERABILITY FOUND: path traversal (CWE-22) — resolved path is not constrained to the base directory."
        },
        language="java", cwe="CWE-22"),
]


# ══════════════════════════════════════════════════════════════════════════
# JavaScript — vulnerable examples
# ══════════════════════════════════════════════════════════════════════════

JS_EXAMPLES = [
    _pack(_INSTR,
        '''function displayComment(comment) {
    const div = document.getElementById('comments');
    div.innerHTML += '<p>' + comment + '</p>';
}''',
        {
            "issues": [{
                "type": "Cross-Site Scripting (XSS)",
                "severity": "High",
                "description": "User-supplied 'comment' is inserted via innerHTML without escaping. Input like <img src=x onerror=alert(1)> executes attacker-controlled JavaScript in the victim's browser.",
                "suggestion": "Use textContent for plain text, or sanitize with DOMPurify if HTML is required.",
                "fixed_code": "const p = document.createElement('p');\np.textContent = comment;\ndiv.appendChild(p);\n\n// If HTML is required:\n// div.innerHTML += DOMPurify.sanitize('<p>' + comment + '</p>');"
            }],
            "overall_score": 1,
            "summary": "VULNERABILITY FOUND: DOM-based XSS (CWE-79) via innerHTML."
        },
        language="javascript", cwe="CWE-79"),

    _pack(_INSTR,
        '''function merge(target, source) {
    for (const key in source) {
        if (typeof source[key] === 'object') {
            merge(target[key] || {}, source[key]);
        } else {
            target[key] = source[key];
        }
    }
    return target;
}''',
        {
            "issues": [{
                "type": "Prototype Pollution",
                "severity": "High",
                "description": "The loop walks arbitrary keys from source, including '__proto__' and 'constructor'. Attacker input like {\"__proto__\": {\"isAdmin\": true}} pollutes Object.prototype, affecting every object in the program.",
                "suggestion": "Skip dangerous keys, use Object.create(null) for key-value stores, or use a library like lodash.merge with the security fix applied.",
                "fixed_code": "const BLOCKED = new Set(['__proto__', 'constructor', 'prototype']);\nfunction merge(target, source) {\n    for (const key in source) {\n        if (BLOCKED.has(key)) continue;\n        if (!Object.hasOwn(source, key)) continue;\n        ...\n    }\n}"
            }],
            "overall_score": 1,
            "summary": "VULNERABILITY FOUND: prototype pollution (CWE-1321) via unchecked key iteration."
        },
        language="javascript", cwe="CWE-1321"),

    _pack(_INSTR,
        '''function runCalculation(userExpr) {
    return eval(userExpr);
}''',
        {
            "issues": [{
                "type": "Code Injection",
                "severity": "High",
                "description": "eval executes arbitrary JavaScript from the caller. Any user-controlled input becomes remote code execution in the application context.",
                "suggestion": "Never eval user input. For math expressions, use a safe parser (mathjs with {evaluate: restricted} config or Function with a strict sandbox).",
                "fixed_code": "import { evaluate } from 'mathjs';\n// mathjs parses math safely and does not execute arbitrary JS\nreturn evaluate(userExpr);"
            }],
            "overall_score": 1,
            "summary": "VULNERABILITY FOUND: use of eval on user input (CWE-95 / CWE-94) — full code execution."
        },
        language="javascript", cwe="CWE-95"),

    _pack(_INSTR,
        '''const express = require('express');
const app = express();
app.get('/redirect', (req, res) => {
    res.redirect(req.query.url);
});''',
        {
            "issues": [{
                "type": "Open Redirect",
                "severity": "Medium",
                "description": "The 'url' query parameter is used directly as the redirect target. Attackers craft links like /redirect?url=https://phishing.example to trick users into trusting malicious sites.",
                "suggestion": "Validate the URL against an allow-list of hosts, or only permit relative paths.",
                "fixed_code": "const ALLOWED_HOSTS = ['example.com', 'www.example.com'];\napp.get('/redirect', (req, res) => {\n    const url = new URL(req.query.url, 'https://example.com');\n    if (!ALLOWED_HOSTS.includes(url.host)) return res.status(400).send('Invalid redirect');\n    res.redirect(url.toString());\n});"
            }],
            "overall_score": 3,
            "summary": "VULNERABILITY FOUND: unvalidated open redirect (CWE-601)."
        },
        language="javascript", cwe="CWE-601"),
]


# ══════════════════════════════════════════════════════════════════════════
# Go — vulnerable examples (modern server-side language, often in Micron/SRAM stack)
# ══════════════════════════════════════════════════════════════════════════

GO_EXAMPLES = [
    _pack(_INSTR,
        '''func getUser(db *sql.DB, name string) (*User, error) {
    query := fmt.Sprintf("SELECT * FROM users WHERE name = '%s'", name)
    row := db.QueryRow(query)
    ...
}''',
        {
            "issues": [{
                "type": "SQL Injection",
                "severity": "High",
                "description": "name is interpolated into the query via fmt.Sprintf. database/sql supports parameterized queries natively; the Sprintf pattern defeats that protection.",
                "suggestion": "Pass parameters separately to QueryRow.",
                "fixed_code": "row := db.QueryRow(\"SELECT * FROM users WHERE name = $1\", name)"
            }],
            "overall_score": 1,
            "summary": "VULNERABILITY FOUND: SQL injection via Sprintf (CWE-89)."
        },
        language="go", cwe="CWE-89"),

    _pack(_INSTR,
        '''resp, err := http.Get(userURL)
if err != nil { return err }
body, _ := io.ReadAll(resp.Body)
return body, nil''',
        {
            "issues": [
                {
                    "type": "Server-Side Request Forgery (SSRF)",
                    "severity": "High",
                    "description": "User-controlled URL is fetched without validation. Attackers can target internal services (http://169.254.169.254/latest/meta-data/ on AWS) to steal credentials.",
                    "suggestion": "Validate URL host against an allow-list, and block private/link-local/loopback IP ranges after DNS resolution.",
                    "fixed_code": "if !isAllowedURL(userURL) { return nil, errors.New(\"forbidden URL\") }\nresp, err := http.Get(userURL)"
                },
                {
                    "type": "Missing Body Close",
                    "severity": "Medium",
                    "description": "resp.Body is never closed, leaking the underlying TCP connection on every call. Under load this exhausts the connection pool.",
                    "suggestion": "Always defer resp.Body.Close() immediately after the error check.",
                    "fixed_code": "resp, err := http.Get(userURL)\nif err != nil { return err }\ndefer resp.Body.Close()"
                }
            ],
            "overall_score": 2,
            "summary": "VULNERABILITY FOUND: SSRF (CWE-918) plus resource leak from missing Body.Close()."
        },
        language="go", cwe="CWE-918"),
]


# ══════════════════════════════════════════════════════════════════════════
# Clean examples (negative signal) — one per language
# ══════════════════════════════════════════════════════════════════════════

CLEAN_EXAMPLES = [
    _pack(_INSTR_CLEAN,
        '''#include <string.h>
#include <stddef.h>
void copy_name(const char *src, char *dest, size_t dest_size) {
    if (!src || !dest || dest_size == 0) return;
    strncpy(dest, src, dest_size - 1);
    dest[dest_size - 1] = '\\0';
}''',
        {"issues": [], "overall_score": 9,
         "summary": "Clean code: bounded copy with null checks and explicit termination. No vulnerabilities identified."},
        language="c", cwe="none", source="synthetic_clean_multilang"),

    _pack(_INSTR_CLEAN,
        '''#include <memory>
#include <vector>
class Buffer {
    std::vector<int> data;
public:
    explicit Buffer(size_t n) : data(n) {}
    int& at(size_t i) { return data.at(i); }  // bounds-checked
};''',
        {"issues": [], "overall_score": 9,
         "summary": "Clean code: RAII via std::vector, bounds-checked access via at(), rule-of-zero compliant. No vulnerabilities identified."},
        language="cpp", cwe="none", source="synthetic_clean_multilang"),

    _pack(_INSTR_CLEAN,
        '''public User getUser(String username) {
    String sql = "SELECT * FROM users WHERE name = ?";
    return jdbcTemplate.queryForObject(sql, new Object[]{username}, new UserRowMapper());
}''',
        {"issues": [], "overall_score": 9,
         "summary": "Clean code: parameterized query prevents SQL injection. No vulnerabilities identified."},
        language="java", cwe="none", source="synthetic_clean_multilang"),

    _pack(_INSTR_CLEAN,
        '''function displayComment(comment) {
    const div = document.getElementById('comments');
    const p = document.createElement('p');
    p.textContent = comment;
    div.appendChild(p);
}''',
        {"issues": [], "overall_score": 9,
         "summary": "Clean code: textContent safely escapes user input. No XSS risk. No vulnerabilities identified."},
        language="javascript", cwe="none", source="synthetic_clean_multilang"),

    _pack(_INSTR_CLEAN,
        '''func getUser(db *sql.DB, name string) (*User, error) {
    row := db.QueryRow("SELECT id, name FROM users WHERE name = $1", name)
    var u User
    if err := row.Scan(&u.ID, &u.Name); err != nil {
        return nil, err
    }
    return &u, nil
}''',
        {"issues": [], "overall_score": 9,
         "summary": "Clean code: parameterized query with proper error handling. No vulnerabilities identified."},
        language="go", cwe="none", source="synthetic_clean_multilang"),
]


# ══════════════════════════════════════════════════════════════════════════
# English-output Python examples (parallel to existing Chinese ones)
# Address missed-CWE cases from the evaluation (CWE-200, CWE-400, CWE-730, CWE-943)
# ══════════════════════════════════════════════════════════════════════════

PYTHON_EN_EXAMPLES = [
    # CWE-200 Information Exposure — model previously used generic terms
    _pack(_INSTR,
        '''import logging
def login(username, password):
    if not authenticate(username, password):
        logging.warning(f"Failed login: user={username} password={password}")
        return False
    return True''',
        {
            "issues": [{
                "type": "Information Exposure (CWE-200)",
                "severity": "High",
                "description": "The plaintext password is written to the application log. Any person with log access — developers, ops, SIEM operators — can read user passwords. This also violates GDPR and PCI-DSS.",
                "suggestion": "Never log passwords, tokens, or other credentials. Log the username and outcome only.",
                "fixed_code": "logging.warning(f\"Failed login: user={username}\")"
            }],
            "overall_score": 2,
            "summary": "VULNERABILITY FOUND: information exposure through log file (CWE-200). Sensitive credential written to logs."
        },
        language="python", cwe="CWE-200"),

    # CWE-400 Resource Exhaustion
    _pack(_INSTR,
        '''from flask import request
@app.route('/api/search')
def search():
    query = request.json['query']
    results = db.session.execute(text(query)).all()  # ignore injection; focus on DoS
    return jsonify([r._asdict() for r in results])''',
        {
            "issues": [
                {
                    "type": "Resource Exhaustion (CWE-400)",
                    "severity": "High",
                    "description": "The query has no row limit. A caller can request SELECT * FROM large_table and exhaust DB memory, network bandwidth, and API response memory — a denial-of-service.",
                    "suggestion": "Enforce a server-side LIMIT, pagination, and per-endpoint rate limiting.",
                    "fixed_code": "# Force a LIMIT and pagination\npage = max(0, int(request.json.get('page', 0)))\nsize = min(int(request.json.get('size', 50)), 100)\nresults = db.session.execute(\n    text(query + ' LIMIT :size OFFSET :offset'),\n    {'size': size, 'offset': page * size}\n).all()"
                },
                {
                    "type": "SQL Injection (CWE-89)",
                    "severity": "High",
                    "description": "Raw SQL comes from request.json['query'] — full SQL injection.",
                    "suggestion": "Never accept raw SQL from clients; accept structured filter parameters and build the query server-side.",
                    "fixed_code": "# Replace raw SQL with ORM filters built from vetted params."
                }
            ],
            "overall_score": 1,
            "summary": "VULNERABILITY FOUND: uncontrolled resource consumption (CWE-400) plus SQL injection (CWE-89)."
        },
        language="python", cwe="CWE-400,CWE-89"),

    # CWE-730 ReDoS — model previously detected regex issue but missed DoS
    _pack(_INSTR,
        '''import re
def validate_email(email):
    pattern = r"^([a-zA-Z0-9]+)+@([a-zA-Z0-9]+)+\\.[a-zA-Z]{2,}$"
    return re.match(pattern, email) is not None''',
        {
            "issues": [{
                "type": "Regular Expression Denial of Service (ReDoS, CWE-730 / CWE-1333)",
                "severity": "High",
                "description": "The pattern contains nested quantifiers (a+)+ which causes catastrophic backtracking on inputs like 'a'*30 + '!'. Matching takes exponential time, allowing a single request to pin a worker CPU and deny service.",
                "suggestion": "Rewrite the regex without nested quantifiers. For email validation, prefer a library (email-validator) or a linear regex. Alternatively, set a timeout with regex (PyPI) or the 're2' library.",
                "fixed_code": "from email_validator import validate_email, EmailNotValidError\ntry:\n    validate_email(email, check_deliverability=False)\n    return True\nexcept EmailNotValidError:\n    return False"
            }],
            "overall_score": 2,
            "summary": "VULNERABILITY FOUND: ReDoS (CWE-1333) due to nested quantifiers causing catastrophic backtracking."
        },
        language="python", cwe="CWE-1333"),

    # CWE-943 NoSQL Injection — model previously detected injection but classified generically
    _pack(_INSTR,
        '''from flask import request
def login():
    username = request.json['username']
    password = request.json['password']
    user = db.users.find_one({"username": username, "password": password})
    return jsonify({"ok": user is not None})''',
        {
            "issues": [{
                "type": "NoSQL Injection (CWE-943)",
                "severity": "High",
                "description": "If request.json is decoded from JSON, username/password can be objects, not strings. Input {\"username\": \"admin\", \"password\": {\"$ne\": null}} bypasses auth because MongoDB interprets {\"$ne\": null} as 'not null'.",
                "suggestion": "Coerce inputs to strings before passing to MongoDB, or validate types with a schema (pydantic, marshmallow).",
                "fixed_code": "username = str(request.json['username'])\npassword = str(request.json['password'])\n# Better: store hashed passwords\nuser = db.users.find_one({\"username\": username})\nif user and bcrypt.checkpw(password.encode(), user['pw_hash']): ..."
            }],
            "overall_score": 1,
            "summary": "VULNERABILITY FOUND: NoSQL injection (CWE-943) — MongoDB operator injection bypasses authentication."
        },
        language="python", cwe="CWE-943"),
]


# ══════════════════════════════════════════════════════════════════════════
# Exported list
# ══════════════════════════════════════════════════════════════════════════

ALL_EXAMPLES = (
    C_EXAMPLES + CPP_EXAMPLES + JAVA_EXAMPLES + JS_EXAMPLES + GO_EXAMPLES +
    CLEAN_EXAMPLES + PYTHON_EN_EXAMPLES
)


if __name__ == "__main__":
    # Quick self-check
    from collections import Counter
    print(f"Total multilang examples: {len(ALL_EXAMPLES)}")
    print(f"By language: {Counter(e['metadata']['language'] for e in ALL_EXAMPLES)}")
    print(f"By source:   {Counter(e['metadata']['source'] for e in ALL_EXAMPLES)}")
    # Validate every output parses as JSON
    for i, e in enumerate(ALL_EXAMPLES):
        try:
            json.loads(e["output"])
        except Exception as exc:
            print(f"BAD JSON at {i}: {exc}")
            break
    else:
        print("All outputs parse as valid JSON")
