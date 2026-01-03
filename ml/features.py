# ml/features.py
import pandas as pd

BASE_COLS = ["pH", "EC_mS_cm", "temp_C", "do_mg_L"]

def add_features_one_tank(g: pd.DataFrame, window_steps: int) -> pd.DataFrame:
    """
    Adds rolling mean/std and window diffs. Must match training + inference.
    """
    g = g.sort_values("timestamp").copy()

    w = window_steps
    for col in BASE_COLS:
        g[f"{col}_mean"] = g[col].rolling(w, min_periods=1).mean()
        g[f"{col}_std"]  = g[col].rolling(w, min_periods=1).std().fillna(0.0)
        g[f"d{col}"]     = g[col].diff(w).fillna(0.0)

    g["temp_do_risk"] = g["temp_C"] / (g["do_mg_L"] + 0.1)
    return g


def add_features_all(df: pd.DataFrame, window_steps: int) -> pd.DataFrame:
    parts = []
    for tank_id, g in df.groupby("tank_id", sort=False):
        parts.append(add_features_one_tank(g, window_steps))
    return pd.concat(parts, ignore_index=True)
