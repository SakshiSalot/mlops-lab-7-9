"""Load configs/config.yaml and apply command-line overrides.

Overrides use dotted keys, e.g. ``--set model.type=random_forest --set training.test_size=0.25``.
Values are parsed as YAML, so numbers, booleans and null keep their types.
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"


def resolve(path: str | Path) -> Path:
    """Resolve repo-relative paths so commands work from any working directory."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override {item!r} must look like key.subkey=value")
        key, raw = item.split("=", 1)
        node = cfg
        parts = key.strip().split(".")
        for part in parts[:-1]:
            if not isinstance(node.get(part), dict):
                raise KeyError(f"unknown config section {part!r} in override {item!r}")
            node = node[part]
        if parts[-1] not in node:
            raise KeyError(f"unknown config key {key!r} (check configs/config.yaml)")
        node[parts[-1]] = yaml.safe_load(raw)
    return cfg


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> dict:
    with open(resolve(path or DEFAULT_CONFIG), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return apply_overrides(cfg, overrides)
