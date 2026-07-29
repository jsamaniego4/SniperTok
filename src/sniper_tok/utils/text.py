from __future__ import annotations

import re


def clean_text(value: str) -> str:
    value = (value or "").lower().strip()
    value = re.sub(r"http\S+", " ", value)
    value = re.sub(r"[^a-z0-9#\s]", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def merge_text_fields(caption: str, hashtags: str, product_name: str) -> str:
    return " ".join(
        part for part in [
            clean_text(caption),
            clean_text(hashtags),
            clean_text(product_name),
        ] if part
    )
