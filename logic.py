"""Pure data logic for the Rail Cascade Monitor (no Streamlit imports, so it can be tested alone)."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Label shown in the app -> columns in events.parquet
TARGETS = {
    "2+ trains delayed (cascade)": {"p": "p2", "p_raw": "p2_raw", "y": "y2"},
    "4+ trains delayed (severe cascade)": {"p": "p4", "p_raw": "p4_raw", "y": "y4"},
}


def window_events(events: pd.DataFrame, day, hour_range) -> pd.DataFrame:
    """Primary delays on one day between two hours (inclusive)."""
    h = events["actual_time"].dt.hour
    mask = (events["day"] == day) & (h >= hour_range[0]) & (h <= hour_range[1])
    return events.loc[mask]


def busiest_day(events: pd.DataFrame):
    """The day with the most primary delays (a good default for a demo)."""
    return events.groupby("day").size().idxmax()


def station_summary(win: pd.DataFrame, p_col: str, y_col: str) -> pd.DataFrame:
    """One row per station for the map: number of delays, mean predicted risk, observed cascade rate."""
    return win.groupby(["eva", "station_name", "lat", "lon"], as_index=False).agg(
        events=("event_id", "size"),
        mean_risk=(p_col, "mean"),
        max_risk=(p_col, "max"),
        observed=(y_col, "mean"),
    )


def alert_stats(win: pd.DataFrame, p_col: str, y_col: str, threshold: float) -> dict:
    """What happens if every event with predicted risk >= threshold raises an alert."""
    n = len(win)
    alerts = win[p_col] >= threshold
    y = win[y_col].astype(bool)
    n_alerts, n_cascades = int(alerts.sum()), int(y.sum())
    hits = int((alerts & y).sum())
    return {
        "events": n,
        "alerts": n_alerts,
        "cascades": n_cascades,
        "hits": hits,
        "precision": hits / n_alerts if n_alerts else np.nan,
        "recall": hits / n_cascades if n_cascades else np.nan,
        "base_rate": n_cascades / n if n else np.nan,
    }


def hourly_summary(win: pd.DataFrame, y_col: str) -> pd.DataFrame:
    """Primary delays and cascades per hour of the day, with all 24 hours present."""
    h = win["actual_time"].dt.hour
    g = win.groupby(h).agg(events=("event_id", "size"), cascades=(y_col, "sum"))
    return g.reindex(range(24), fill_value=0).rename_axis("hour").reset_index()


def reliability(df: pd.DataFrame, p_col: str, y_col: str, bins: int = 10) -> pd.DataFrame:
    """Predicted versus observed cascade rate in equally sized groups of predicted risk."""
    d = df[[p_col, y_col]].dropna().copy()
    d["bin"] = pd.qcut(d[p_col], bins, duplicates="drop")
    return (
        d.groupby("bin", observed=True)
        .agg(n=(y_col, "size"), predicted=(p_col, "mean"), actual=(y_col, "mean"))
        .reset_index(drop=True)
    )


def pct(x, digits: int = 0) -> str:
    """Format a share as a percentage, or a dash when it is undefined."""
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{digits}f}%"
