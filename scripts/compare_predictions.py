"""Part D: compare local predictions with the API (e.g. the Docker container) for the same inputs.

    python scripts/compare_predictions.py --url http://localhost:8000 --label "docker container"

Local = load models/model.joblib in this Python process and call src.predict.predict().
Remote = POST each input to <url>/predict. Tolerance: |local - remote| <= 1e-6 lakhs
(0.1 rupee). Identical artifacts and pinned library versions normally give a
difference of exactly 0; the tolerance allows for floating-point differences across
CPUs/BLAS builds.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from _common import REPORTS, http, load_json, md_table, now

TOLERANCE = 1e-6


def compare(url: str, model_path: str | None = None, inputs_file: str = "compare_inputs.json") -> dict:
    from src.predict import load_model, predict

    model = load_model(model_path)
    inputs = load_json(inputs_file)
    local = predict(model, [{**h, "has_garage": int(h["has_garage"])} for h in inputs])
    _, info = http("GET", url + "/model-info")
    rows, max_diff = [], 0.0
    for i, (house, loc) in enumerate(zip(inputs, local), start=1):
        status, body = http("POST", url + "/predict", body=house)
        remote = body.get("prediction") if status == 200 else None
        diff = abs(loc - remote) if remote is not None else float("inf")
        max_diff = max(max_diff, diff)
        rows.append({"#": i,
                     "Input": f"{house['area_sqft']} sqft, {house['bedrooms']}bd/{house['bathrooms']}ba, "
                              f"{house['year_built']}, {house['distance_to_center_km']} km, {house['locality']}",
                     "Local prediction": f"{loc:.10f}",
                     "API prediction": f"{remote:.10f}" if remote is not None else f"HTTP {status}",
                     "Abs diff": f"{diff:.2e}",
                     "Within tolerance": "yes" if diff <= TOLERANCE else "NO"})
    return {"rows": rows, "max_diff": max_diff, "local_sha": model.metadata["model_sha256"],
            "remote_sha": info.get("model_sha256") if isinstance(info, dict) else None,
            "remote_versions": info.get("library_versions") if isinstance(info, dict) else None,
            "local_versions": model.metadata["library_versions"]}


def render(result: dict, url: str, label: str) -> str:
    same_artifact = result["local_sha"] == result["remote_sha"]
    ok = result["max_diff"] <= TOLERANCE
    return "\n".join([
        f"# Local vs {label} predictions (Part D)", "",
        f"Run {now()} with `python scripts/compare_predictions.py --url {url}`. "
        f"Tolerance: absolute difference <= {TOLERANCE:g} lakhs.", "",
        md_table(result["rows"], ["#", "Input", "Local prediction", "API prediction", "Abs diff", "Within tolerance"]),
        f"**Max absolute difference:** {result['max_diff']:.2e} -> **{'consistent' if ok else 'NOT consistent'}**", "",
        f"Same model artifact: {'yes' if same_artifact else 'NO'} (local SHA-256 `{result['local_sha'][:16]}...`, "
        f"API reports `{(result['remote_sha'] or 'n/a')[:16]}...`).", "",
        "Libraries the model was trained with (from model_metadata.json): "
        + ", ".join(f"{k} {v}" for k, v in result["local_versions"].items()) + ".", "",
    ])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--label", default="docker container")
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--out", default=str(REPORTS / "local_vs_container.md"))
    args = ap.parse_args()
    result = compare(args.url, args.model_path)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(render(result, args.url, args.label), encoding="utf-8")
    for r in result["rows"]:
        print(r["#"], r["Local prediction"], r["API prediction"], r["Abs diff"], r["Within tolerance"])
    print(f"max abs diff {result['max_diff']:.2e}; wrote {args.out}")
    if result["max_diff"] > TOLERANCE:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
