#!/usr/bin/env python3
"""
NHANES Actigraphy: Statistical Physics of Complex Systems
=========================================================
Physica A | Advanced Time-Series Modeling

- Shannon Entropy (disorder metric)
- Markov State Transition Probabilities
- Statistical validation & publication figures
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
PROCESSED_PATH = BASE_DIR / "outputs" / "processed_data_refined.csv"
PAXHR_PATH = BASE_DIR / "PAXHR_H.xpt"
# Fallback if project reorganized
if not PROCESSED_PATH.exists():
    PROCESSED_PATH = BASE_DIR / "Cureus" / "outputs" / "processed_data_refined.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
DPI = 300
ACTIVITY_COLS = ["PAXMTSH", "PAXINTEN"]  # NHANES 2013-2014 uses PAXMTSH


def load_xpt(filepath: Path) -> pd.DataFrame:
    try:
        import pyreadstat
        df, _ = pyreadstat.read_xport(str(filepath))
        return df
    except ImportError:
        return pd.read_sas(str(filepath), format="xport")


def load_data():
    """Load processed cohort and raw hourly actigraphy, merged on SEQN."""
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    cohort = pd.read_csv(PROCESSED_PATH)
    print(f"Processed cohort: {len(cohort)} subjects from {PROCESSED_PATH}")
    seqn_set = set(cohort["SEQN"].dropna().astype(int))

    paxhr = load_xpt(PAXHR_PATH)
    activity_col = next((c for c in ACTIVITY_COLS if c in paxhr.columns), None)
    if activity_col is None:
        raise KeyError(f"Neither {ACTIVITY_COLS} found in PAXHR_H")
    print(f"Activity column: {activity_col}")
    paxhr = paxhr[paxhr["SEQN"].isin(seqn_set)].copy()
    paxhr = paxhr[paxhr[activity_col].notna() & (paxhr[activity_col] >= 0)]
    paxhr = paxhr.sort_values(["SEQN", "PAXDAYH", "PAXSSNHP"])
    return cohort, paxhr, activity_col


def shannon_entropy(activity: np.ndarray) -> float:
    """
    H = - sum(p_i * log2(p_i)), p_i = activity_i / sum(activity).
    Ignore p_i = 0.
    """
    activity = np.asarray(activity, dtype=float)
    total = np.nansum(activity)
    if total <= 0:
        return np.nan
    p = activity / total
    p = p[p > 0]
    return -np.sum(p * np.log2(p))


def _markov_from_threshold(activity: np.ndarray, threshold: float) -> tuple[float, float]:
    """P_01, P_10 using given threshold (median or mean)."""
    state = (activity > threshold).astype(int)
    if len(state) < 2:
        return np.nan, np.nan
    n_01 = np.sum((state[:-1] == 0) & (state[1:] == 1))
    n_10 = np.sum((state[:-1] == 1) & (state[1:] == 0))
    n_0 = np.sum(state[:-1] == 0)
    n_1 = np.sum(state[:-1] == 1)
    p_01 = n_01 / n_0 if n_0 > 0 else np.nan
    p_10 = n_10 / n_1 if n_1 > 0 else np.nan
    return p_01, p_10


def markov_transitions(activity: np.ndarray, use_median: bool = True) -> tuple[float, float]:
    """
    State 0 (Rest): activity <= threshold; State 1 (Active): activity > threshold.
    P_01 = P(0->1), P_10 = P(1->0). Threshold = median by default, or mean.
    """
    activity = np.asarray(activity, dtype=float)
    threshold = np.nanmedian(activity) if use_median else np.nanmean(activity)
    return _markov_from_threshold(activity, threshold)


def entropy_production_rate(pi0: float, pi1: float, p_01: float, p_10: float) -> float:
    """
    EPR (σ) = π₀ P₀₁ log(P₀₁/P₁₀) + π₁ P₁₀ log(P₁₀/P₀₁).
    Measures irreversibility / distance from equilibrium.
    Handle P_01=0 or P_10=0 to avoid log(0).
    """
    eps = 1e-12
    if np.isnan(pi0) or np.isnan(pi1) or np.isnan(p_01) or np.isnan(p_10):
        return np.nan
    if p_01 <= eps or p_10 <= eps:
        return 0.0
    term1 = pi0 * p_01 * np.log(p_01 / p_10) if p_10 > eps else 0.0
    term2 = pi1 * p_10 * np.log(p_10 / p_01) if p_01 > eps else 0.0
    return term1 + term2


def spectral_analysis(p_01: float, p_10: float) -> tuple[float, float, float, float]:
    """
    Build M = [[P_00, P_01], [P_10, P_11]], compute eigenvalues, spectral gap,
    stationary distribution (π0, π1), mixing time τ = 1/(1-λ2).
    """
    if np.isnan(p_01) or np.isnan(p_10):
        return np.nan, np.nan, np.nan, np.nan
    p_00, p_11 = 1 - p_01, 1 - p_10
    M = np.array([[p_00, p_01], [p_10, p_11]])
    eigvals = np.linalg.eigvals(M)
    eigvals = np.real(eigvals)
    eigvals = np.sort(eigvals)[::-1]
    lambda2 = eigvals[1] if len(eigvals) > 1 else np.nan
    spectral_gap = 1 - lambda2
    mixing_time = 1 / spectral_gap if spectral_gap > 1e-10 else np.nan
    denom = p_01 + p_10
    pi0 = p_10 / denom if denom > 0 else np.nan
    pi1 = p_01 / denom if denom > 0 else np.nan
    return spectral_gap, mixing_time, pi0, pi1


def compute_physics_features(
    paxhr: pd.DataFrame, activity_col: str, use_median: bool = True
) -> pd.DataFrame:
    """For each SEQN: Entropy, P_01, P_10, Spectral_Gap, Mixing_Time, pi0, pi1, EPR."""
    rows = []
    for seqn, grp in paxhr.groupby("SEQN"):
        act = grp[activity_col].values
        if len(act) < 24:
            continue
        ent = shannon_entropy(act)
        p_01, p_10 = markov_transitions(act, use_median=use_median)
        sg, tau, pi0, pi1 = spectral_analysis(p_01, p_10)
        epr = entropy_production_rate(pi0, pi1, p_01, p_10)
        rows.append({
            "SEQN": seqn,
            "Entropy": ent,
            "P_01": p_01,
            "P_10": p_10,
            "Spectral_Gap": sg,
            "Mixing_Time": tau,
            "pi0": pi0,
            "pi1": pi1,
            "EPR": epr,
        })
    return pd.DataFrame(rows)


def ancova_age_adjusted(df: pd.DataFrame, outcome: str) -> dict:
    """ANCOVA: outcome ~ sleep_problem_reported + age. Return p-value for sleep effect."""
    try:
        from statsmodels.formula.api import ols
        from statsmodels.stats.anova import anova_lm
    except ImportError:
        return {"p_sleep": np.nan, "p_age": np.nan, "ok": False}
    df_clean = df[[outcome, "sleep_problem_reported", "age"]].dropna()
    if len(df_clean) < 50:
        return {"p_sleep": np.nan, "p_age": np.nan, "ok": False}
    model = ols(f"{outcome} ~ C(sleep_problem_reported) + age", data=df_clean).fit()
    anova = anova_lm(model, typ=2)
    p_sleep = anova.loc["C(sleep_problem_reported)", "PR(>F)"]
    p_age = anova.loc["age", "PR(>F)"]
    return {"p_sleep": p_sleep, "p_age": p_age, "ok": True}


def statistical_validation(df: pd.DataFrame, vars: list = None) -> dict:
    """Mann-Whitney U between groups for specified variables."""
    vars = vars or ["Entropy", "P_01", "P_10", "Spectral_Gap", "Mixing_Time", "pi0", "EPR"]
    g0 = df[df["sleep_problem_reported"] == 0]
    g1 = df[df["sleep_problem_reported"] == 1]
    result = {}
    for var in vars:
        if var not in df.columns:
            continue
        a, b = g0[var].dropna(), g1[var].dropna()
        _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        result[var] = p
    return result


def plot_entropy_boxplot(df: pd.DataFrame, savepath: Path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    g0 = df[df["sleep_problem_reported"] == 0]["Entropy"].dropna()
    g1 = df[df["sleep_problem_reported"] == 1]["Entropy"].dropna()
    bp = ax.boxplot(
        [g0, g1],
        labels=["No sleep problem", "Sleep problem reported"],
        patch_artist=True,
    )
    bp["boxes"][0].set_facecolor("#2E86AB")
    bp["boxes"][1].set_facecolor("#E94F37")
    ax.set_ylabel("Shannon Entropy (bits)", fontsize=11)
    ax.set_title("Shannon Entropy: Circadian Disorder Metric", fontsize=12)
    plt.tight_layout()
    plt.savefig(savepath, dpi=DPI, bbox_inches="tight")
    plt.close()


def plot_spectral_gap_boxplot(df: pd.DataFrame, savepath: Path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    g0 = df[df["sleep_problem_reported"] == 0]["Spectral_Gap"].dropna()
    g1 = df[df["sleep_problem_reported"] == 1]["Spectral_Gap"].dropna()
    bp = ax.boxplot(
        [g0, g1],
        labels=["No sleep problem", "Sleep problem reported"],
        patch_artist=True,
    )
    bp["boxes"][0].set_facecolor("#2E86AB")
    bp["boxes"][1].set_facecolor("#E94F37")
    ax.set_ylabel("Spectral Gap (1 − λ₂)", fontsize=11)
    ax.set_title("Spectral Gap: Dynamical Instability of Circadian System", fontsize=12)
    plt.tight_layout()
    plt.savefig(savepath, dpi=DPI, bbox_inches="tight")
    plt.close()


def plot_stationary_dist(df: pd.DataFrame, savepath: Path):
    """Distribution of π₀ (Rest) and π₁ (Active) by group."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    labels = ["No sleep problem", "Sleep problem reported"]
    colors = ["#2E86AB", "#E94F37"]
    for i, (ax, var) in enumerate(zip(axes, ["pi0", "pi1"])):
        for g, lbl, c in zip([0, 1], labels, colors):
            vals = df[df["sleep_problem_reported"] == g][var].dropna()
            ax.hist(vals, bins=30, alpha=0.6, label=lbl, color=c, density=True)
        ax.set_xlabel(f"π_{var[-1]} ({'Rest' if var=='pi0' else 'Active'})", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.legend(loc="upper right", fontsize=9)
        ax.set_title(f"Stationary Distribution: {var}", fontsize=11)
    plt.suptitle("Stationary Distribution (π₀, π₁) by Sleep Problem Status", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(savepath, dpi=DPI, bbox_inches="tight")
    plt.close()


def plot_spectral_epr_scatter(df: pd.DataFrame, savepath: Path):
    """Scatter: Spectral_Gap vs EPR, colored by sleep disorder status."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 6))
    g0 = df[df["sleep_problem_reported"] == 0]
    g1 = df[df["sleep_problem_reported"] == 1]
    ax.scatter(g0["Spectral_Gap"], g0["EPR"], alpha=0.5, c="#2E86AB", label="No sleep problem", s=25)
    ax.scatter(g1["Spectral_Gap"], g1["EPR"], alpha=0.5, c="#E94F37", label="Sleep problem reported", s=25)
    ax.set_xlabel("Spectral Gap (1 − λ₂)", fontsize=11)
    ax.set_ylabel("Entropy Production Rate σ", fontsize=11)
    ax.set_title("Spectral Gap vs Entropy Production: Non-Equilibrium Signature", fontsize=12)
    ax.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.savefig(savepath, dpi=DPI, bbox_inches="tight")
    plt.close()


def plot_age_stratified_mixing(df: pd.DataFrame, savepath: Path):
    """Age-stratified boxplots of Mixing_Time."""
    import matplotlib.pyplot as plt
    df_plot = df.dropna(subset=["Mixing_Time", "age"])
    q1, q2, q3 = df_plot["age"].quantile([0.33, 0.66, 1.0]).values
    def age_stratum(a):
        if a <= q1:
            return "Young (≤33rd)"
        if a <= q2:
            return "Middle (33rd–66th)"
        return "Older (>66th)"
    df_plot = df_plot.copy()
    df_plot["age_stratum"] = df_plot["age"].apply(age_stratum)
    order = ["Young (≤33rd)", "Middle (33rd–66th)", "Older (>66th)"]
    strata = [s for s in order if s in df_plot["age_stratum"].unique()]
    fig, axes = plt.subplots(1, len(strata), figsize=(4 * len(strata), 5), sharey=True)
    if len(strata) == 1:
        axes = [axes]
    colors = {"No sleep problem": "#2E86AB", "Sleep problem reported": "#E94F37"}
    for ax, stratum in zip(axes, strata):
        sub = df_plot[df_plot["age_stratum"] == stratum]
        g0 = sub[sub["sleep_problem_reported"] == 0]["Mixing_Time"]
        g1 = sub[sub["sleep_problem_reported"] == 1]["Mixing_Time"]
        bp = ax.boxplot([g0, g1], labels=["No sleep problem", "Sleep problem"],
                        patch_artist=True)
        bp["boxes"][0].set_facecolor("#2E86AB")
        bp["boxes"][1].set_facecolor("#E94F37")
        ax.set_title(stratum, fontsize=11)
        ax.set_ylabel("Mixing Time τ" if ax == axes[0] else "")
    plt.suptitle("Age-Stratified Mixing Time", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(savepath, dpi=DPI, bbox_inches="tight")
    plt.close()


def plot_markov_scatter(df: pd.DataFrame, savepath: Path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 6))
    g0 = df[df["sleep_problem_reported"] == 0]
    g1 = df[df["sleep_problem_reported"] == 1]
    ax.scatter(g0["P_01"], g0["P_10"], alpha=0.5, c="#2E86AB", label="No sleep problem", s=25)
    ax.scatter(g1["P_01"], g1["P_10"], alpha=0.5, c="#E94F37", label="Sleep problem reported", s=25)
    ax.set_xlabel("P(0→1) Rest→Active", fontsize=11)
    ax.set_ylabel("P(1→0) Active→Rest", fontsize=11)
    ax.set_title("Markov Transition Probabilities", fontsize=12)
    ax.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.savefig(savepath, dpi=DPI, bbox_inches="tight")
    plt.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cohort, paxhr, activity_col = load_data()
    physics_df = compute_physics_features(paxhr, activity_col, use_median=True)
    df = cohort.merge(physics_df, on="SEQN", how="inner")
    required = ["Entropy", "P_01", "P_10", "Spectral_Gap", "Mixing_Time", "pi0", "pi1", "EPR"]
    df = df.dropna(subset=[c for c in required if c in df.columns])
    print(f"\nAnalytic sample with physics features: {len(df)} subjects")

    print("\n" + "=" * 60)
    print("STATISTICAL VALIDATION")
    print("=" * 60)
    pvals = statistical_validation(df)
    for var, p in pvals.items():
        print(f"Mann-Whitney U p-value, {var}: {p:.6f}")
    print("\n--- EPR (Entropy Production Rate) ---")
    print(f"Mann-Whitney U p-value, EPR: {pvals.get('EPR', np.nan):.6f}")

    print("\n" + "=" * 60)
    print("AGE-ADJUSTED ANALYSIS (ANCOVA)")
    print("=" * 60)
    for outcome in ["Spectral_Gap", "EPR"]:
        res = ancova_age_adjusted(df, outcome)
        if res.get("ok"):
            print(f"{outcome}: p(sleep)={res['p_sleep']:.6f}, p(age)={res['p_age']:.6f}")
        else:
            print(f"{outcome}: statsmodels not available; install with: pip install statsmodels")

    print("\n" + "=" * 60)
    print("THRESHOLD ROBUSTNESS (Mean vs Median)")
    print("=" * 60)
    physics_mean = compute_physics_features(paxhr, activity_col, use_median=False)
    df_mean = cohort.merge(physics_mean[["SEQN", "Spectral_Gap"]], on="SEQN", how="inner", suffixes=("", "_mean"))
    df_mean = df_mean.rename(columns={"Spectral_Gap": "Spectral_Gap_mean"})
    df_mean = df_mean.dropna(subset=["Spectral_Gap_mean"])
    _, p_sg_mean = stats.mannwhitneyu(
        df_mean[df_mean["sleep_problem_reported"] == 0]["Spectral_Gap_mean"].dropna(),
        df_mean[df_mean["sleep_problem_reported"] == 1]["Spectral_Gap_mean"].dropna(),
        alternative="two-sided",
    )
    print(f"Spectral_Gap (Mean threshold) Mann-Whitney p-value: {p_sg_mean:.6f}")
    print(f"Robust: p < 0.05 = {p_sg_mean < 0.05}")

    print("\n" + "=" * 60)
    print("GENERATING FIGURES (300 DPI)")
    print("=" * 60)
    plot_entropy_boxplot(df, OUTPUT_DIR / "fig5_entropy_boxplot.png")
    plot_markov_scatter(df, OUTPUT_DIR / "fig6_markov_scatter.png")
    plot_spectral_gap_boxplot(df, OUTPUT_DIR / "fig7_spectral_gap_comparison.png")
    plot_stationary_dist(df, OUTPUT_DIR / "fig8_stationary_dist_plot.png")
    plot_spectral_epr_scatter(df, OUTPUT_DIR / "fig9_spectral_epr_scatter.png")
    print(f"  Saved: {OUTPUT_DIR / 'fig9_spectral_epr_scatter.png'}")
    plot_age_stratified_mixing(df, OUTPUT_DIR / "fig10_age_stratified_mixing.png")
    print(f"  Saved: {OUTPUT_DIR / 'fig10_age_stratified_mixing.png'}")

    out_path = OUTPUT_DIR / "processed_data_physics_final.csv"
    df.to_csv(out_path, index=False)
    print(f"\nData saved: {out_path}")
    return df


if __name__ == "__main__":
    main()
