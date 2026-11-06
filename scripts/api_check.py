"""B5: send valid and invalid requests to a running API and record what it returned.

    python scripts/api_check.py --url http://127.0.0.1:8000 --label "local uvicorn"
    python scripts/api_check.py --url http://localhost:8000 --label "docker container" --out reports/api_validation_container.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import REPORTS, http, load_json, md_table, now, short

CASES = [
    ("Valid input", "Prediction returned (200)", "POST", "/predict", "request_valid.json", None, 200),
    ("Missing field (bedrooms)", "Validation error (422)", "POST", "/predict", "request_missing_field.json", None, 422),
    ("Wrong type (area_sqft as text)", "Validation error (422)", "POST", "/predict", "request_wrong_type.json", None, 422),
    ("Invalid business value (1 bedroom, 4 bathrooms)", "Controlled error (422)", "POST", "/predict",
     "request_business_rule.json", None, 422),
    ("Out-of-range value (year_built 2031)", "Validation error (422)", "POST", "/predict", "request_out_of_range.json", None, 422),
    ("Malformed JSON body", "Validation error (422)", "POST", "/predict", None, b'{"area_sqft": 1850,', 422),
    ("Health check", "Service reports healthy (200)", "GET", "/health", None, None, 200),
    ("Valid input again after the errors", "Still 200: service did not crash", "POST", "/predict", "request_valid.json", None, 200),
]


def summarize(status: int, body) -> str:
    if isinstance(body, dict):
        if "prediction" in body:
            return f"{status}: prediction {body['prediction']:.4f} {body['unit']} (model {body['model_version']})"
        if "details" in body:
            d = body["details"][0]
            return f"{status}: {d['field']} - {d['message']}"
        if "status" in body:
            return f"{status}: status={body['status']}, model_loaded={body['model_loaded']}"
    return f"{status}: {short(body, 80)}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--label", default="local uvicorn")
    ap.add_argument("--out", default=str(REPORTS / "api_validation.md"))
    args = ap.parse_args()

    rows, examples, ok_all = [], [], True
    for name, expected, method, path, file, raw, want in CASES:
        body = load_json(file) if file else None
        status, resp = http(method, args.url + path, body=body, raw=raw)
        ok = status == want
        ok_all &= ok
        rows.append({"Test": name, "Expected behaviour": expected, "Observed status/result": summarize(status, resp),
                     "Pass": "PASS" if ok else "FAIL"})
        if name.startswith(("Valid input", "Missing", "Invalid business")) and "again" not in name:
            examples.append((name, method, path, body, status, resp))
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {status}")

    text = [f"# API validation (B5): {args.label}", "",
            f"Target `{args.url}`, run {now()} with `python scripts/api_check.py`.", "",
            md_table(rows, ["Test", "Expected behaviour", "Observed status/result", "Pass"]),
            "## Example requests and responses", ""]
    for name, method, path, body, status, resp in examples:
        text += [f"**{name}** - `{method} {path}` -> `{status}`", "", "Request:", "```json",
                 json.dumps(body, indent=2), "```", "Response:", "```json", json.dumps(resp, indent=2), "```", ""]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(text) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    if not ok_all:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
