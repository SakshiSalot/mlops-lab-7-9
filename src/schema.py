"""Data contract shared by training, batch prediction and the API.

One place defines the input columns, their types, the allowed ranges and the
business rule, so cleaning (src/data.py), the API request schema (src/api.py)
and batch scoring (src/predict.py) can never disagree.
"""
from __future__ import annotations

ID_COLUMN = "house_id"
TARGET = "price_lakhs"

# Raw input features the saved pipeline expects, in this order.
NUMERIC_FEATURES = ["area_sqft", "bedrooms", "bathrooms", "year_built", "distance_to_center_km"]
BINARY_FEATURES = ["has_garage"]
CATEGORICAL_FEATURES = ["locality"]
INPUT_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

LOCALITIES = ("Central", "East", "North", "South", "West")

# Feature engineering uses a fixed reference year (not "today") so the same
# input always produces the same features and the same prediction.
REFERENCE_YEAR = 2026

# Inclusive valid ranges. Values outside them are data errors.
VALID_RANGES: dict[str, tuple[float, float]] = {
    "area_sqft": (300, 10_000),
    "bedrooms": (1, 10),
    "bathrooms": (1, 10),
    "year_built": (1900, REFERENCE_YEAR),
    "distance_to_center_km": (0, 100),
}

# Business rule: a house cannot have more than one bathroom beyond its bedroom count.
MAX_EXTRA_BATHROOMS = 1

FEATURE_DESCRIPTIONS = {
    "area_sqft": "Built-up area in square feet",
    "bedrooms": "Number of bedrooms",
    "bathrooms": "Number of bathrooms (at most bedrooms + 1)",
    "year_built": "Year of construction",
    "distance_to_center_km": "Distance to the city centre in km",
    "has_garage": "Whether the house has a garage",
    "locality": "City zone: " + ", ".join(LOCALITIES),
}

EXAMPLE_HOUSE = {
    "area_sqft": 1850,
    "bedrooms": 3,
    "bathrooms": 2,
    "year_built": 2012,
    "distance_to_center_km": 6.5,
    "has_garage": True,
    "locality": "North",
}


def business_rule_violation(bedrooms: float, bathrooms: float) -> str | None:
    """Return a message if the bathrooms/bedrooms rule is broken, else None."""
    if bathrooms > bedrooms + MAX_EXTRA_BATHROOMS:
        return (f"bathrooms ({int(bathrooms)}) cannot exceed bedrooms + {MAX_EXTRA_BATHROOMS} "
                f"({int(bedrooms + MAX_EXTRA_BATHROOMS)})")
    return None
