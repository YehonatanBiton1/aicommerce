"""Shared data cleaning utilities for the AICommerce demo."""

from __future__ import annotations

import pandas as pd


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

