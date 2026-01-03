# plot_results.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Config (same style as eval)
# ----------------------------
CSV_PATH = os.environ.get("EVAL_CSV", "ml_artifacts/with_anomaly_scores_test.csv")
OUT_DIR = os.environ.get("OUT_DIR", "pp1_charts")

AUTO_THRESHOLD = os.environ.get("AUTO_THRESHOLD", "1") == "1"
AUTO_PERCENTILE = float(os.environ.get("AUTO_PERCENTILE", "99.0"))
THRESHOLD = float(os.environ.get("THRESHOLD", "0.65"))

USE_PERSISTENCE = os.environ.get("USE_PERSISTENCE", "1") == "1"
PERSIST_MODE = os.environ.get("PERSIST_MODE", "2of3")  # "2of3" or "NofN"
PERSIST_N = int(os.environ.get("PERSIST_N", "2"))

POS_EVENTS = {"pump_off", "aeration_drop"}
NEG_EVENTS = {"none"}


def choose_threshold(df_eval: pd.DataFrame) -> float:
    if AUTO_THRESHOLD and "event" in df_eval.columns:
        normal_scores = df_eval[df_eval["event"].isin(NEG_EVENTS)]["anom_score_0_1"].dropna().values
        if len(normal_scores) == 0:
            return THRESHOLD
        return float(np.percentile(normal_scores, AUTO_PERCENTILE))
    return THRESHOLD


def apply_persistence(df_eval: pd.DataFrame, base_pred_col: str = "pred") -> pd.Series:
    if not USE_PERSISTENCE:
        return df_eval[base_pred_col].astype(int)

    if PERSIST_MODE == "NofN":
        n = max(1, PERSIST_N)
        return (
            df_eval.groupby("tank_id")[base_pred_col]
            .rolling(n, min_periods=n).sum()
            .reset_index(level=0, drop=True)
            .ge(n)
            .astype(int)
        )

    # default: 2-of-3
    return (
        df_eval.groupby("tank_id")[base_pred_col]
        .rolling(3, min_periods=3).sum()
        .reset_index(level=0, drop=True)
        .ge(2)
        .astype(int)
    )


def ensure_whs(df: pd.DataFrame) -> pd.DataFrame:
    """If WHS_0_100 is missing, compute it from sensors using ml/whs.py."""
    if "WHS_0_100" in df.columns:
        return df

    # Optional compute if you only have raw sensors
    try:
        from ml.whs import whs_from_row
    except Exception:
        print("WHS_0_100 not found and ml.whs import failed. Skipping WHS chart.")
        return df

    need = ["pH", "EC_mS_cm", "temp_C", "do_mg_L"]
    for c in need:
        if c not in df.columns:
            print("Missing columns for WHS:", need)
            return df

    whs_vals = []
    for r in df[need].itertuples(index=False):
        if any(pd.isna(x) for x in r):
            whs_vals.append(np.nan)
        else:
            whs, _ = whs_from_row(r[0], r[1], r[2], r[3])
            whs_vals.append(whs)
    df["WHS_0_100"] = whs_vals
    return df


def plot_whs(df: pd.DataFrame):
    if "WHS_0_100" not in df.columns:
        return
    for tank_id, g in df.groupby("tank_id"):
        g = g.sort_values("timestamp")
        plt.figure()
        plt.plot(g["timestamp"], g["WHS_0_100"])
        plt.title(f"WHS (0–100) vs Time — {tank_id}")
        plt.xlabel("Time")
        plt.ylabel("WHS_0_100")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"1_whs_{tank_id}.png"), dpi=200)
        plt.close()


def plot_do_with_alerts(df: pd.DataFrame):
    if "do_mg_L" not in df.columns:
        return
    for tank_id, g in df.groupby("tank_id"):
        g = g.sort_values("timestamp")
        plt.figure()
        plt.plot(g["timestamp"], g["do_mg_L"], label="DO (mg/L)")
        if "pred_final" in g.columns:
            alert_pts = g[g["pred_final"] == 1]
            if not alert_pts.empty:
                plt.scatter(alert_pts["timestamp"], alert_pts["do_mg_L"], label="Alert")
        plt.title(f"DO vs Time (alerts marked) — {tank_id}")
        plt.xlabel("Time")
        plt.ylabel("DO (mg/L)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"2_do_alerts_{tank_id}.png"), dpi=200)
        plt.close()


def plot_anom_score_with_threshold(df: pd.DataFrame, th: float):
    if "anom_score_0_1" not in df.columns:
        return
    for tank_id, g in df.groupby("tank_id"):
        g = g.sort_values("timestamp")
        plt.figure()
        plt.plot(g["timestamp"], g["anom_score_0_1"], label="Anomaly score (0–1)")
        plt.axhline(th, linestyle="--", label=f"Threshold ({th:.3f})")
        if "pred_final" in g.columns:
            alert_pts = g[g["pred_final"] == 1]
            if not alert_pts.empty:
                plt.scatter(alert_pts["timestamp"], alert_pts["anom_score_0_1"], label="Alert")
        plt.title(f"Anomaly Score vs Time — {tank_id}")
        plt.xlabel("Time")
        plt.ylabel("anom_score_0_1")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"3_anom_score_{tank_id}.png"), dpi=200)
        plt.close()


def plot_false_alarms_per_day(df: pd.DataFrame):
    # If we have event labels, we can compute false alarms/day.
    if "pred_final" not in df.columns:
        return

    df2 = df.copy()
    df2["date"] = df2["timestamp"].dt.date

    if "event" in df2.columns:
        df2["is_false_alarm"] = (df2["pred_final"] == 1) & (df2["event"].isin(NEG_EVENTS))
        fa = df2.groupby("date")["is_false_alarm"].sum()
        title = "False Alarms per Day"
        ylab = "False alarms"
        fname = "4_false_alarms_per_day.png"
    else:
        # fallback: just count alerts/day
        df2["is_alert"] = (df2["pred_final"] == 1)
        fa = df2.groupby("date")["is_alert"].sum()
        title = "Alerts per Day"
        ylab = "Alerts"
        fname = "4_alerts_per_day.png"

    plt.figure()
    fa.plot(kind="bar")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(ylab)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=200)
    plt.close()


def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    if "timestamp" not in df.columns or "tank_id" not in df.columns:
        raise ValueError("CSV must include at least: timestamp, tank_id")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["tank_id", "timestamp"]).reset_index(drop=True)

    df = ensure_whs(df)

    # Build predictions so plots can mark alerts
    if "anom_score_0_1" in df.columns:
        df_eval = df.copy()
        th = choose_threshold(df_eval)
        df_eval["pred"] = (df_eval["anom_score_0_1"] >= th).astype(int)
        df_eval["pred_final"] = apply_persistence(df_eval, "pred")
    else:
        df_eval = df
        th = THRESHOLD

    print("Using threshold:", round(th, 4))
    print("Saving charts to:", OUT_DIR)

    # 4 charts
    plot_whs(df_eval)
    plot_do_with_alerts(df_eval)
    plot_anom_score_with_threshold(df_eval, th)
    plot_false_alarms_per_day(df_eval)

    print("✅ Done. Files created in:", OUT_DIR)


if __name__ == "__main__":
    main()
