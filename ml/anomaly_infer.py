# ml/anomaly_infer.py
import os, json, joblib
import numpy as np
import pandas as pd
from collections import deque
from ml.features import add_features_one_tank

ART_DIR = os.environ.get("ART_DIR", "ml_artifacts")

MODEL_PATH = os.path.join(ART_DIR, "isoforest.joblib")
SCALER_PATH = os.path.join(ART_DIR, "scaler.joblib")
FEATURES_PATH = os.path.join(ART_DIR, "feature_list.json")
META_PATH = os.path.join(ART_DIR, "train_meta.json")

# alert settings (change in env if needed)
THRESHOLD = float(os.environ.get("THRESHOLD", "0.73"))
USE_PERSISTENCE = os.environ.get("USE_PERSISTENCE", "1") == "1"
PERSIST_MODE = os.environ.get("PERSIST_MODE", "2of3")  # 2of3 or NofN
PERSIST_N = int(os.environ.get("PERSIST_N", "2"))

WINDOW_STEPS = int(os.environ.get("WINDOW_STEPS", "3"))  # must match training

_model = joblib.load(MODEL_PATH)
_scaler = joblib.load(SCALER_PATH)
_feature_cols = json.load(open(FEATURES_PATH))
_meta = json.load(open(META_PATH))

_p_low = float(_meta["score_norm_p_low"])
_p_high = float(_meta["score_norm_p_high"])

# keep last flags per tank (for persistence)
_state = {}

def _score_to_0_1(raw_score: float) -> float:
    scaled = (raw_score - _p_low) / (_p_high - _p_low + 1e-9)
    scaled = float(np.clip(scaled, 0, 3))
    return float(np.tanh(scaled))

def _persist_decision(tank_id: str, flag: int) -> int:
    if not USE_PERSISTENCE:
        return flag

    if tank_id not in _state:
        _state[tank_id] = deque(maxlen=max(3, PERSIST_N))
    _state[tank_id].append(flag)

    arr = list(_state[tank_id])

    if PERSIST_MODE == "NofN":
        n = max(1, PERSIST_N)
        if len(arr) < n:
            return 0
        return 1 if sum(arr[-n:]) == n else 0

    # default 2-of-3
    if len(arr) < 3:
        return 0
    return 1 if sum(arr[-3:]) >= 2 else 0


def infer_from_history(history_df: pd.DataFrame) -> dict:
    """
    history_df must contain last N rows for ONE tank:
    columns: timestamp,tank_id,pH,EC_mS_cm,temp_C,do_mg_L
    N should be >= WINDOW_STEPS (ideally 6 rows = 60 mins).
    """
    g = history_df.copy()
    g["timestamp"] = pd.to_datetime(g["timestamp"])
    g = g.sort_values("timestamp")

    # fill small missing gaps
    g[["pH","EC_mS_cm","temp_C","do_mg_L"]] = g[["pH","EC_mS_cm","temp_C","do_mg_L"]].ffill().bfill()

    g = add_features_one_tank(g, window_steps=WINDOW_STEPS)
    last = g.iloc[-1]

    X = g[_feature_cols].iloc[[-1]].values
    Xs = _scaler.transform(X)

    raw = -_model.score_samples(Xs)[0]
    score01 = _score_to_0_1(raw)

    base_flag = 1 if score01 >= THRESHOLD else 0
    tank_id = str(last["tank_id"])
    final_flag = _persist_decision(tank_id, base_flag)

    return {
        "tank_id": tank_id,
        "timestamp": str(last["timestamp"]),
        "anom_score_0_1": round(score01, 4),
        "anom_flag": int(final_flag),
        "threshold": THRESHOLD,
    }
