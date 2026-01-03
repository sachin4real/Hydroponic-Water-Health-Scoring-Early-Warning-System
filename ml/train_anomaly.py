# ml/train_anomaly.py
import os, json, joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from ml.features import add_features_all

CSV_PATH = os.environ.get("CSV_PATH", "data/greenhouse_hydroponics_2tanks_7days_10min.csv")
WINDOW_STEPS = int(os.environ.get("WINDOW_STEPS", "3"))      # 10-min -> 3 = 30 min
CONTAMINATION = float(os.environ.get("CONTAMINATION", "0.03"))
TRAIN_FRAC = float(os.environ.get("TRAIN_FRAC", "0.70"))     # time split per tank
OUT_DIR = os.environ.get("OUT_DIR", "ml_artifacts")

FEATURE_COLS = [
    "pH","EC_mS_cm","temp_C","do_mg_L",
    "pH_mean","pH_std","dpH",
    "EC_mS_cm_mean","EC_mS_cm_std","dEC_mS_cm",
    "temp_C_mean","temp_C_std","dtemp_C",
    "do_mg_L_mean","do_mg_L_std","ddo_mg_L",
    "temp_do_risk",
]

def add_time_split_flag(df: pd.DataFrame, train_frac: float) -> pd.DataFrame:
    parts = []
    for tank_id, g in df.groupby("tank_id", sort=False):
        g = g.sort_values("timestamp").reset_index(drop=True)
        cut = int(len(g) * train_frac)
        g["is_train"] = False
        if cut > 0:
            g.loc[:cut-1, "is_train"] = True
        parts.append(g)
    return pd.concat(parts, ignore_index=True)

def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    req = {"timestamp","tank_id","pH","EC_mS_cm","temp_C","do_mg_L"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["tank_id","timestamp"]).reset_index(drop=True)

    # if event not provided, create all-normal
    if "event" not in df.columns:
        df["event"] = "none"

    df = add_time_split_flag(df, TRAIN_FRAC)

    # features
    df = add_features_all(df, WINDOW_STEPS)

    # train on TRAIN + NORMAL only
    train_mask = (df["is_train"] == True) & (df["event"] == "none")
    train_df = df[train_mask].copy()
    if len(train_df) < 200:
        raise ValueError("Not enough TRAIN normal rows. Increase TRAIN_FRAC or data size.")

    X_train = train_df[FEATURE_COLS].values
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    model = IsolationForest(
        n_estimators=400,
        contamination=CONTAMINATION,
        random_state=42
    )
    model.fit(X_train_s)

    # score all
    X_all_s = scaler.transform(df[FEATURE_COLS].values)
    raw = -model.score_samples(X_all_s)  # higher = more anomalous

    # normalize using TRAIN normal only (no leakage)
    raw_train = raw[train_mask.values]
    p_low = float(np.percentile(raw_train, 5))
    p_high = float(np.percentile(raw_train, 99.9))

    scaled = (raw - p_low) / (p_high - p_low + 1e-9)
    scaled = np.clip(scaled, 0, 3)
    anom01 = np.tanh(scaled)

    df["anom_score_0_1"] = np.round(anom01, 4)

    # save artifacts
    joblib.dump(model, os.path.join(OUT_DIR, "isoforest.joblib"))
    joblib.dump(scaler, os.path.join(OUT_DIR, "scaler.joblib"))
    with open(os.path.join(OUT_DIR, "feature_list.json"), "w") as f:
        json.dump(FEATURE_COLS, f, indent=2)

    meta = {
        "csv_path": CSV_PATH,
        "window_steps": WINDOW_STEPS,
        "contamination": CONTAMINATION,
        "train_frac": TRAIN_FRAC,
        "score_norm_p_low": p_low,
        "score_norm_p_high": p_high,
    }
    with open(os.path.join(OUT_DIR, "train_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # save scored datasets
    df.to_csv(os.path.join(OUT_DIR, "with_anomaly_scores.csv"), index=False)
    df[df["is_train"] == True].to_csv(os.path.join(OUT_DIR, "with_anomaly_scores_train.csv"), index=False)
    df[df["is_train"] == False].to_csv(os.path.join(OUT_DIR, "with_anomaly_scores_test.csv"), index=False)

    print("✅ Training complete")
    print("Saved to:", OUT_DIR)
    print(" - isoforest.joblib")
    print(" - scaler.joblib")
    print(" - feature_list.json")
    print(" - train_meta.json")
    print(" - with_anomaly_scores.csv")
    print(" - with_anomaly_scores_train.csv")
    print(" - with_anomaly_scores_test.csv")
    print("Contamination used:", CONTAMINATION)

if __name__ == "__main__":
    main()
