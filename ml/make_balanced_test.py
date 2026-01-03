# ml/make_balanced_test.py
import os
import pandas as pd

IN_CSV  = os.environ.get("IN_CSV", "ml_artifacts/with_anomaly_scores_test.csv")
OUT_CSV = os.environ.get("OUT_CSV", "ml_artifacts/with_anomaly_scores_test_balanced.csv")
SEED    = int(os.environ.get("SEED", "42"))

POS_EVENTS = {"pump_off", "aeration_drop"}
NEG_EVENTS = {"none"}

def main():
    df = pd.read_csv(IN_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df_eval = df[df["event"].isin(POS_EVENTS | NEG_EVENTS)].copy()
    pos = df_eval[df_eval["event"].isin(POS_EVENTS)]
    neg = df_eval[df_eval["event"].isin(NEG_EVENTS)]

    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("Need both normal and failure rows to build a balanced test set.")

    n = min(len(pos), len(neg))
    pos_s = pos.sample(n=n, random_state=SEED)
    neg_s = neg.sample(n=n, random_state=SEED)

    out = pd.concat([pos_s, neg_s], ignore_index=True).sample(frac=1, random_state=SEED)
    out.to_csv(OUT_CSV, index=False)

    print("Saved:", OUT_CSV)
    print("Rows:", len(out), "| normal:", (out["event"]=="none").sum(), "| failure:", (out["event"]!="none").sum())

if __name__ == "__main__":
    main()
