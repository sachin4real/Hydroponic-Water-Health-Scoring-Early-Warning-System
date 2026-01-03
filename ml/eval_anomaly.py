# ml/eval_anomaly.py
import os
import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
)

CSV_PATH = os.environ.get("EVAL_CSV", "ml_artifacts/with_anomaly_scores_test.csv")

AUTO_THRESHOLD = os.environ.get("AUTO_THRESHOLD", "0") == "1"
AUTO_PERCENTILE = float(os.environ.get("AUTO_PERCENTILE", "99.5"))
THRESHOLD = float(os.environ.get("THRESHOLD", "0.95"))

USE_PERSISTENCE = os.environ.get("USE_PERSISTENCE", "1") == "1"
PERSIST_MODE = os.environ.get("PERSIST_MODE", "2of3")  # "2of3" or "NofN"
PERSIST_N = int(os.environ.get("PERSIST_N", "2"))

POS_EVENTS = {"pump_off", "aeration_drop"}
NEG_EVENTS = {"none"}


def choose_threshold(df_eval: pd.DataFrame) -> float:
    if AUTO_THRESHOLD:
        normal_scores = df_eval[df_eval["event"].isin(NEG_EVENTS)]["anom_score_0_1"].values
        th = float(np.percentile(normal_scores, AUTO_PERCENTILE))
        print(f"Auto threshold from normal p{AUTO_PERCENTILE}: {th:.4f}")
        return th
    print(f"Fixed threshold: {THRESHOLD:.4f}")
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


def detection_delays_minutes(df_eval: pd.DataFrame, pred_col: str) -> pd.Series:
    delays = []
    for tank, g in df_eval.groupby("tank_id"):
        g = g.sort_values("timestamp").reset_index(drop=True)
        g["is_fail"] = g["event"].isin(POS_EVENTS)

        i = 0
        while i < len(g):
            if g.loc[i, "is_fail"]:
                start_ts = g.loc[i, "timestamp"]
                j = i
                first_flag = None

                while j < len(g) and g.loc[j, "is_fail"]:
                    if g.loc[j, pred_col] == 1 and first_flag is None:
                        first_flag = g.loc[j, "timestamp"]
                    j += 1

                if first_flag is not None:
                    delays.append((first_flag - start_ts).total_seconds() / 60.0)
                else:
                    delays.append(np.nan)
                i = j
            else:
                i += 1

    return pd.Series(delays, name="delay_minutes")


def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Eval CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["tank_id", "timestamp"])

    df_eval = df[df["event"].isin(POS_EVENTS.union(NEG_EVENTS))].copy()
    if df_eval.empty:
        raise ValueError("No eval rows found (check POS/NEG events).")

    y_true = df_eval["event"].isin(POS_EVENTS).astype(int).values
    y_score = df_eval["anom_score_0_1"].values

    th = choose_threshold(df_eval)
    df_eval["pred"] = (df_eval["anom_score_0_1"] >= th).astype(int)

    df_eval["pred_final"] = apply_persistence(df_eval, "pred")
    y_pred = df_eval["pred_final"].values

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    print("\n=== Evaluation ===")
    print(f"USE_PERSISTENCE={int(USE_PERSISTENCE)}  PERSIST_MODE={PERSIST_MODE}  PERSIST_N={PERSIST_N}")
    print("Confusion Matrix [[TN FP],[FN TP]]:")
    print(cm)
    print(f"Accuracy:          {acc:.3f}")
    print(f"Balanced Accuracy: {bal_acc:.3f}")
    print(f"Precision:         {precision:.3f}")
    print(f"Recall:            {recall:.3f}")
    print(f"F1-score:          {f1:.3f}")

    if len(np.unique(y_true)) == 2:
        auc = roc_auc_score(y_true, y_score)
        print(f"ROC-AUC:           {auc:.3f}")

    df_eval["is_false_alarm"] = (df_eval["pred_final"] == 1) & (df_eval["event"].isin(NEG_EVENTS))
    df_eval["date"] = df_eval["timestamp"].dt.date
    fa_per_day = df_eval.groupby("date")["is_false_alarm"].sum()

    print("\n=== False alarms/day ===")
    print(fa_per_day.describe())

    delays = detection_delays_minutes(df_eval, "pred_final")
    print("\n=== Detection delay (minutes) ===")
    print(delays.describe())
    print("Missed episodes:", int(delays.isna().sum()))


if __name__ == "__main__":
    main()
