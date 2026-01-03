# ml/run_stream_demo.py
import pandas as pd
from ml.whs import whs_from_row
from ml.anomaly_infer import infer_from_history

CSV = "data/greenhouse_hydroponics_rawonly_3tanks_30days_10min.csv"

def main():
    df = pd.read_csv(CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["tank_id","timestamp"])

    # simulate per tank streaming using last 6 rows history
    for tank_id, g in df.groupby("tank_id"):
        g = g.reset_index(drop=True)
        for i in range(6, len(g), 20):  # step 20 just to print less
            window = g.iloc[i-6:i].copy()

            # WHS on latest row
            latest = window.iloc[-1]
            whs, risk = whs_from_row(latest.pH, latest.EC_mS_cm, latest.temp_C, latest.do_mg_L)

            # Anomaly inference
            out = infer_from_history(window)

            print(tank_id, latest.timestamp, "WHS", whs, risk, "Anom", out["anom_score_0_1"], "Flag", out["anom_flag"])

if __name__ == "__main__":
    main()
