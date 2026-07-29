from __future__ import annotations

import pandas as pd

from ..db import get_connection, initialize_database


def get_top_trends(limit: int = 10) -> pd.DataFrame:
    initialize_database()
    query = """
    SELECT *
    FROM product_daily_metrics
    ORDER BY metric_date DESC, trend_score DESC
    LIMIT ?
    """
    with get_connection() as conn:
        df = pd.read_sql_query(query, conn, params=(limit,))
    return df


def get_trend_history(product_name: str) -> pd.DataFrame:
    initialize_database()
    query = """
    SELECT *
    FROM product_daily_metrics
    WHERE product_name = ?
    ORDER BY metric_date ASC
    """
    with get_connection() as conn:
        return pd.read_sql_query(query, conn, params=(product_name,))
