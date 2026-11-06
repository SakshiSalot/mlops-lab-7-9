"""Small helpers shared by the evidence scripts (standard library only, works on Windows)."""
from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXAMPLES = ROOT / "examples"
REPORTS = ROOT / "reports"
EVIDENCE = REPORTS / "evidence"


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def http(method: str, url: str, body=None, raw: bytes | None = None, timeout: float = 10) -> tuple[int, dict | str]:
    """Return (status, parsed JSON or text). Never raises for HTTP error codes."""
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            status, text = r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        status, text = e.code, e.read().decode()
    try:
        return status, json.loads(text)
    except json.JSONDecodeError:
        return status, text


def wait_for(url: str, timeout: float = 90, want: int = 200) -> float:
    """Poll url until it returns `want`; return seconds waited."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            if http("GET", url, timeout=3)[0] == want:
                return time.time() - start
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(1)
    raise TimeoutError(f"{url} did not return {want} within {timeout}s")


def run(cmd: list[str], log: Path | None = None, check: bool = True, env=None) -> subprocess.CompletedProcess:
    """Run a command, echo it, optionally save command + output to a text file."""
    print("$", " ".join(cmd), flush=True)
    res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    if log:
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "a", encoding="utf-8") as f:
            f.write(f"$ {' '.join(cmd)}\n{res.stdout}{res.stderr}\n")
    if check and res.returncode != 0:
        print(res.stdout[-3000:], res.stderr[-3000:], sep="\n")
        raise SystemExit(f"command failed ({res.returncode}): {' '.join(cmd)}")
    return res


def md_table(rows: list[dict], cols: list[str]) -> str:
    def cell(v):
        return str(v).replace("|", "/").replace("\n", " ")
    out = "| " + " | ".join(cols) + " |\n|" + "|".join("---" for _ in cols) + "|\n"
    return out + "".join("| " + " | ".join(cell(r.get(c, "")) for c in cols) + " |\n" for r in rows)


def load_json(name: str):
    return json.loads((EXAMPLES / name).read_text(encoding="utf-8"))


def short(obj, n: int = 140) -> str:
    s = json.dumps(obj) if not isinstance(obj, str) else obj
    return s if len(s) <= n else s[: n - 3] + "..."
