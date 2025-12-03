"""Shared data cleaning utilities for the AICommerce demo."""

from __future__ import annotations

import pandas as pd


REQUIRED_TRAINING_COLUMNS = ["price", "trend_score", "category", "success_score"]
DEFAULT_MIN_TRAINING_SAMPLES = 20


def clean_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize categories and coerce numeric fields before training/analysis."""

    cleaned = df.copy()
    cleaned["category"] = (
        cleaned.get("category", "general")
        .fillna("general")
        .astype(str)
        .str.strip()
        .replace("", "general")
    )

    for col in ["price", "trend_score", "success_score"]:
        cleaned[col] = pd.to_numeric(cleaned.get(col), errors="coerce")

    return cleaned.dropna(subset=["price", "trend_score", "success_score"])


def validate_training_data(
    raw_df: pd.DataFrame, min_samples: int = DEFAULT_MIN_TRAINING_SAMPLES
):
    """Ensure data has the required columns and enough clean rows for modeling."""

    missing = [c for c in REQUIRED_TRAINING_COLUMNS if c not in raw_df.columns]
    if missing:
        return None, f"חסרות עמודות חובה: {missing}"

    cleaned = clean_training_frame(raw_df)
    if len(cleaned) < min_samples:
        return (
            None,
            f"צריך לפחות {min_samples} מוצרים נקיים (כרגע {len(cleaned)} מתוך {len(raw_df)})",
        )

    return cleaned, None

