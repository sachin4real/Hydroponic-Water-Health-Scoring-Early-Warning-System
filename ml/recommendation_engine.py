# ml/recommendation_engine.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class Rec:
    priority: str              # HIGH / MED / LOW
    issue: str
    actions: List[str]
    reason: str
    confidence: str            # High / Medium / Low


def recommend(
    pH: float,
    EC: float,
    temp_C: float,
    do_mg_L: float,
    whs: Optional[float] = None,
    risk_level: Optional[str] = None,
    anomaly_score: Optional[float] = None,
    anomaly_flag: Optional[bool] = None,
    dpH: Optional[float] = None,     # change in last ~30 mins
    dEC: Optional[float] = None,
    dTemp: Optional[float] = None,
    dDO: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Returns a single best recommendation bundle + extra notes.
    You can show 'priority/issue/actions/reason' in the dashboard.
    """

    recs: List[Rec] = []
    notes: List[str] = []

    # ---------- Sensor sanity checks ----------
    if dpH is not None and dpH > 0.8:
        recs.append(Rec(
            priority="LOW",
            issue="Possible pH sensor spike",
            actions=[
                "Check pH probe wiring/connection",
                "Rinse and recalibrate pH probe",
                "Confirm pH using a manual test kit before dosing",
            ],
            reason=f"pH changed unusually fast (ΔpH={dpH:.2f} in ~30 min).",
            confidence="Medium",
        ))

    if dDO is not None and abs(dDO) > 3.0:
        recs.append(Rec(
            priority="LOW",
            issue="Possible DO sensor jump",
            actions=[
                "Check DO probe placement (bubbles can affect readings)",
                "Clean probe tip and recalibrate if available",
                "Confirm DO with a manual meter if possible",
            ],
            reason=f"DO changed unusually fast (ΔDO={dDO:.2f} mg/L in ~30 min).",
            confidence="Medium",
        ))

    # ---------- DO (critical) ----------
    if do_mg_L < 4.0 or (dDO is not None and dDO < -1.0 and (anomaly_flag or (anomaly_score or 0) > 0.7)):
        recs.append(Rec(
            priority="HIGH",
            issue="Low dissolved oxygen (DO)",
            actions=[
                "Check aeration pump power and airflow immediately",
                "Inspect air stones / tubing for blockage",
                "Increase aeration or water circulation",
            ],
            reason=f"DO is low or dropping fast (DO={do_mg_L:.2f} mg/L, ΔDO={dDO if dDO is not None else 0:.2f}).",
            confidence="High" if do_mg_L < 4.0 else "Medium",
        ))

    # ---------- Temperature ----------
    if temp_C > 28.0:
        recs.append(Rec(
            priority="HIGH",
            issue="High water temperature",
            actions=[
                "Add shading / reduce heat sources",
                "Increase circulation, consider cooling if available",
                "Re-check DO (warm water holds less oxygen)",
            ],
            reason=f"Temperature is high (temp={temp_C:.1f}°C).",
            confidence="High",
        ))
    elif temp_C > 26.0:
        recs.append(Rec(
            priority="MED",
            issue="Slightly high temperature",
            actions=[
                "Increase ventilation/shading",
                "Monitor DO closely",
            ],
            reason=f"Temperature above preferred range (temp={temp_C:.1f}°C).",
            confidence="Medium",
        ))

    # ---------- pH ----------
    if pH < 5.2:
        recs.append(Rec(
            priority="MED",
            issue="pH too low",
            actions=[
                "Add pH-up slowly in small steps",
                "Re-measure after mixing (10–15 minutes)",
            ],
            reason=f"pH below warning band (pH={pH:.2f}).",
            confidence="High",
        ))
    elif pH > 6.3:
        recs.append(Rec(
            priority="MED",
            issue="pH too high",
            actions=[
                "Add pH-down slowly in small steps",
                "Re-measure after mixing (10–15 minutes)",
            ],
            reason=f"pH above warning band (pH={pH:.2f}).",
            confidence="High",
        ))

    # ---------- EC ----------
    if EC < 1.0:
        recs.append(Rec(
            priority="MED",
            issue="EC too low (nutrients low)",
            actions=[
                "Add nutrient solution gradually",
                "Re-check EC after mixing",
            ],
            reason=f"EC below warning band (EC={EC:.2f} mS/cm).",
            confidence="High",
        ))
    elif EC > 2.0:
        recs.append(Rec(
            priority="MED",
            issue="EC too high (solution too strong)",
            actions=[
                "Dilute by adding fresh water",
                "Re-check EC after mixing",
            ],
            reason=f"EC above warning band (EC={EC:.2f} mS/cm).",
            confidence="High",
        ))

    # ---------- WHS-aware summary ----------
    if whs is not None and risk_level is not None:
        notes.append(f"WHS={whs:.1f} ({risk_level}).")
        if risk_level == "CRITICAL" and not any(r.priority == "HIGH" for r in recs):
            recs.append(Rec(
                priority="MED",
                issue="Overall water health is critical",
                actions=[
                    "Review pH/EC/temp/DO values and recent changes",
                    "Consider partial water change if conditions don’t stabilize",
                ],
                reason="WHS indicates critical health even if one parameter is not extreme.",
                confidence="Medium",
            ))

    # ---------- Anomaly-only note ----------
    if anomaly_flag and (whs is None or risk_level == "SAFE"):
        notes.append("Anomaly detected while WHS looks okay → possible early change or sensor noise. Re-check sensors.")

    # Choose the single best recommendation to display first:
    # HIGH > MED > LOW. If multiple, pick the first.
    priority_rank = {"HIGH": 3, "MED": 2, "LOW": 1}
    recs.sort(key=lambda r: priority_rank.get(r.priority, 0), reverse=True)

    best = recs[0] if recs else Rec(
        priority="LOW",
        issue="All parameters stable",
        actions=["Continue monitoring."],
        reason="No rule triggered.",
        confidence="High",
    )

    return {
        "priority": best.priority,
        "issue": best.issue,
        "actions": best.actions,
        "reason": best.reason,
        "confidence": best.confidence,
        "notes": notes,
    }
