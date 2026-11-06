"""Generate the raw house-listings export (synthetic; no personal or confidential data).

Continues the Lab 3-6 house-price dataset, now as a *raw* export with the problems
real exports have, so the cleaning stage has real work to do:

  * locality typed inconsistently ("north", " NORTH ", "North") or missing
  * area_sqft sometimes text ("2,450" or "2450 sqft"), missing, or mistyped x10
  * has_garage as yes/no/Y/N/1/0/true/false
  * missing values in most feature columns
  * future construction years (invalid), missing or non-positive prices
  * exact duplicate rows

Output is byte-for-byte deterministic (fixed seeds, "\\n" line endings), so the
same command always produces the same md5 and DVC sees no change.

Usage:
    python -m src.make_dataset --out data/raw/houses_raw.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.config import resolve
from src.schema import LOCALITIES, REFERENCE_YEAR

LOCALITY_PREMIUM = {"Central": 18.0, "East": 0.0, "North": 6.0, "South": 4.0, "West": 2.0}
N_HOUSES = 1200
N_DUPLICATES = 30


def clean_houses(n: int = N_HOUSES, seed: int = 42) -> pd.DataFrame:
    """The ground-truth table before any data-entry problems are introduced."""
    rng = np.random.default_rng(seed)
    locality = rng.choice(LOCALITIES, size=n, p=[0.15, 0.2, 0.25, 0.2, 0.2])
    area = rng.integers(450, 4001, n)
    bedrooms = np.clip(np.round(area / 700 + rng.normal(0, 0.7, n)), 1, 6).astype(int)
    bathrooms = np.clip(bedrooms - rng.integers(0, 2, n), 1, 5).astype(int)
    year_built = rng.integers(1976, REFERENCE_YEAR, n)          # 1976..2025
    distance = np.round(np.clip(rng.gamma(2.0, 4.0, n), 0.5, 40.0), 2)
    garage = (rng.random(n) < 0.45).astype(int)
    age = REFERENCE_YEAR - year_built
    premium = np.array([LOCALITY_PREMIUM[x] for x in locality])

    # Price in INR lakhs: same structure as Lab 3-6 plus a locality premium.
    price = (
        8
        + 0.028 * area
        + 3.5 * bedrooms
        + 4.0 * bathrooms
        - 0.35 * age
        + 25 * np.exp(-distance / 6)
        + 0.012 * area * np.exp(-distance / 8)
        + 6 * garage
        + premium
        + rng.normal(0, 5, n)
    )
    return pd.DataFrame({
        "house_id": np.arange(1, n + 1),
        "locality": locality,
        "area_sqft": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "year_built": year_built,
        "distance_to_center_km": distance,
        "has_garage": garage,
        "price_lakhs": np.round(np.clip(price, 10, None), 2),
    })


def make_raw(seed: int = 7) -> pd.DataFrame:
    """Inject realistic data-entry problems into the clean table."""
    df = clean_houses().astype(object)
    n = len(df)
    rng = np.random.default_rng(seed)

    def pick(frac: float) -> np.ndarray:
        return rng.random(n) < frac

    # locality: inconsistent case/whitespace, some missing
    loc = df["locality"].astype(str)
    loc = np.where(pick(0.08), loc.str.lower(), loc)
    loc = np.where(pick(0.04), pd.Series(loc).str.upper(), loc)
    loc = np.where(pick(0.05), "  " + pd.Series(loc) + " ", loc)
    df["locality"] = np.where(pick(0.015), None, loc)

    # area_sqft: thousands separators, unit suffix, missing, 5 typos with an extra zero
    area = df["area_sqft"].astype(int)
    as_text = area.astype(str)
    comma = pick(0.04) & (area >= 1000)
    df.loc[comma, "area_sqft"] = area[comma].map(lambda v: f"{v:,}")
    suffix = pick(0.03) & ~comma
    df.loc[suffix, "area_sqft"] = as_text[suffix] + " sqft"
    df.loc[pick(0.015), "area_sqft"] = None
    typo_idx = rng.choice(n, 5, replace=False)
    df.loc[typo_idx, "area_sqft"] = area.iloc[typo_idx] * 10

    # missing values in other features
    for col, frac in [("bedrooms", 0.015), ("bathrooms", 0.015), ("year_built", 0.02),
                      ("distance_to_center_km", 0.02)]:
        df.loc[pick(frac), col] = None
    df.loc[rng.choice(n, 3, replace=False), "year_built"] = 2031      # impossible future year

    # has_garage: many spellings of yes/no, some missing
    yes, no = ["yes", "Yes", "Y", "1", "true"], ["no", "No", "N", "0", "false"]
    g = df["has_garage"].astype(int).to_numpy()
    choice = rng.integers(0, 5, n)
    df["has_garage"] = [yes[c] if v == 1 else no[c] for v, c in zip(g, choice)]
    df.loc[pick(0.01), "has_garage"] = None

    # target problems: missing and non-positive prices
    df.loc[rng.choice(n, 8, replace=False), "price_lakhs"] = None
    df.loc[rng.choice(n, 2, replace=False), "price_lakhs"] = 0.0

    # exact duplicate rows (an export that ran twice for some listings), then shuffle
    dups = df.iloc[rng.choice(n, N_DUPLICATES, replace=False)]
    df = pd.concat([df, dups], ignore_index=True)
    df = df.iloc[rng.permutation(len(df))].reset_index(drop=True)

    for col in ["bedrooms", "bathrooms", "year_built"]:
        df[col] = pd.array(df[col], dtype="Int64")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/raw/houses_raw.csv")
    args = ap.parse_args()
    df = make_raw()
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, lineterminator="\n")
    print(f"wrote {len(df)} rows x {df.shape[1]} columns -> {out}")


if __name__ == "__main__":
    main()
