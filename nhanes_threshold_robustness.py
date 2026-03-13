#!/usr/bin/env python3
"""
Robustness Check: Markov State Threshold (Q1 vs Q3 vs Median)
============================================================
Recalculates Night P_01 using 25th and 75th percentile thresholds
to demonstrate threshold-invariance of the finding.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "outputs" / "processed_data_physics_ultimate.csv"
PAXHR_PATH = BASE_DIR / "PAXHR_H.xpt"
ACTIVITY_COLS = ["PAXMTSH", "PAXINTEN"]
NIGHT_HOURS = set(range(22, 24)) | set(range(0, 6))


def load_xpt(fp: Path) -> pd.DataFrame:
    try:
        import pyreadstat
        df, _ = pyreadstat.read_xport(str(fp))
        return df
    except ImportError:
        return pd.read_sas(str(fp), format="xport")


def p01_from_sequence(act: np.ndarray, thresh: float) -> float:
    """P(0->1) = n_01 / n_0 for state = 1 if act > thresh."""
    state = (act > thresh).astype(int)
    if len(state) < 2:
        return np.nan
    n_01 = np.sum((state[:-1] == 0) & (state[1:] == 1))
    n_0 = np.sum(state[:-1] == 0)
    return n_01 / n_0 if n_0 > 0 else np.nan


def main():
    print("=" * 60)
    print("ROBUSTNESS CHECK: MARKOV THRESHOLD (Q1 / Q3)")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)
    seqn_set = set(df["SEQN"].dropna().astype(int))
    print(f"Target cohort: {len(df)} subjects")

    paxhr = load_xpt(PAXHR_PATH)
    activity_col = next((c for c in ACTIVITY_COLS if c in paxhr.columns), None)
    if activity_col is None:
        raise KeyError("No activity column in PAXHR_H")
    paxhr = paxhr[paxhr["SEQN"].isin(seqn_set)].copy()
    paxhr = paxhr[paxhr[activity_col].notna() & (paxhr[activity_col] >= 0)]
    paxhr = paxhr.sort_values(["SEQN", "PAXDAYH", "PAXSSNHP"])
    paxhr["_hour_idx"] = paxhr.groupby(["SEQN", "PAXDAYH"]).cumcount()

    rows = []
    for seqn, grp in paxhr.groupby("SEQN"):
        act_all = grp[activity_col].values
        if len(act_all) < 24:
            continue
        q1 = np.nanpercentile(act_all, 25)
        q3 = np.nanpercentile(act_all, 75)
        night_mask = grp["_hour_idx"].isin(NIGHT_HOURS)
        act_night = grp.loc[night_mask, activity_col].values
        if len(act_night) < 4:
            continue
        night_p01_q1 = p01_from_sequence(act_night, q1)
        night_p01_q3 = p01_from_sequence(act_night, q3)
        rows.append({
            "SEQN": seqn,
            "Night_P01_Q1": night_p01_q1,
            "Night_P01_Q3": night_p01_q3,
        })

    thresh_df = pd.DataFrame(rows)
    df = df.merge(thresh_df, on="SEQN", how="inner")
    df = df.dropna(subset=["Night_P01_Q1", "Night_P01_Q3"])
    print(f"Analytic sample: {len(df)} subjects with Q1/Q3 Night P_01")

    g0 = df[df["sleep_problem_reported"] == 0]
    g1 = df[df["sleep_problem_reported"] == 1]

    _, p_q1 = stats.mannwhitneyu(
        g0["Night_P01_Q1"].dropna(), g1["Night_P01_Q1"].dropna(),
        alternative="two-sided",
    )
    _, p_q3 = stats.mannwhitneyu(
        g0["Night_P01_Q3"].dropna(), g1["Night_P01_Q3"].dropna(),
        alternative="two-sided",
    )

    print("\n" + "=" * 60)
    print("STATISTICAL VALIDATION (Mann-Whitney U)")
    print("=" * 60)
    print(f"  Night_P01_Q1 (25th percentile threshold): p = {p_q1:.6f}")
    print(f"  Night_P01_Q3 (75th percentile threshold): p = {p_q3:.6f}")

    both_sig = p_q1 < 0.05 and p_q3 < 0.05
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    if both_sig:
        print("  Both Q1 and Q3 threshold models yield p < 0.05.")
        print("  The Night P_01 finding is THRESHOLD-INVARIANT and robust to binarization choice.")
    else:
        print(f"  Night_P01_Q1: p = {p_q1:.6f} {'(significant, p < 0.05)' if p_q1 < 0.05 else '(not significant)'}")
        print(f"  Night_P01_Q3: p = {p_q3:.6f} {'(significant, p < 0.05)' if p_q3 < 0.05 else '(not significant)'}")
        if p_q1 < 0.05:
            print("  The finding is robust when using Q1 (25th percentile); Q3 may dilute group differences.")


if __name__ == "__main__":
    main()
