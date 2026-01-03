# ml/whs.py
import numpy as np

def whs_from_row(pH: float, EC: float, temp_C: float, do_mg_L: float):
    """
    Returns (whs_0_100, risk_level).
    Simple, explainable scoring for PP1.
    """

    def band_score(x, lo, hi, warn_lo, warn_hi):
        # 100 inside [lo,hi], falls to 60 at warn limits, then to 0 outside.
        if lo <= x <= hi:
            return 100.0
        if warn_lo <= x < lo:
            return 60.0 + 40.0 * (x - warn_lo) / (lo - warn_lo + 1e-9)
        if hi < x <= warn_hi:
            return 60.0 + 40.0 * (warn_hi - x) / (warn_hi - hi + 1e-9)
        if x < warn_lo:
            return max(0.0, 60.0 * (x / (warn_lo + 1e-9)))
        if x > warn_hi:
            return max(0.0, 60.0 * (warn_hi / (x + 1e-9)))
        return 0.0

    # Typical leafy-greens hydroponics bands (tune later if needed)
    s_ph = band_score(pH, 5.5, 6.5, 5.2, 6.8)
    s_ec = band_score(EC, 1.0, 2.0, 0.8, 2.3)
    s_t  = band_score(temp_C, 18.0, 26.0, 16.0, 28.0)

    # DO: >=6 ideal, 5–6 warning-ish, 4–5 low, <4 critical
    if do_mg_L >= 6.0:
        s_do = 100.0
    elif 5.0 <= do_mg_L < 6.0:
        s_do = 60.0 + 40.0 * (do_mg_L - 5.0)
    elif 4.0 <= do_mg_L < 5.0:
        s_do = 30.0 + 30.0 * (do_mg_L - 4.0)
    else:
        s_do = max(0.0, 30.0 * (do_mg_L / 4.0))

    # Weight DO + pH slightly more
    whs = 0.30 * s_do + 0.25 * s_ph + 0.25 * s_ec + 0.20 * s_t
    whs = float(np.clip(whs, 0, 100))

    if whs >= 80:
        risk = "SAFE"
    elif whs >= 60:
        risk = "WARNING"
    else:
        risk = "CRITICAL"

    return round(whs, 1), risk
