#!/usr/bin/env python3
"""
NHANES: Ultimate Non-Equilibrium Dynamics Pipeline for Physica A
================================================================
Integrates: EPR, Time-Varying Markov (Day/Night), Eigenvector Phase Space (PCA)
Input: processed_data_physics_deep.csv, PAXHR_H.xpt
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent
DEEP_PATH = BASE_DIR / "outputs" / "processed_data_physics_deep.csv"
if not DEEP_PATH.exists():
    DEEP_PATH = BASE_DIR / "Cureus" / "outputs" / "processed_data_physics_deep.csv"
PAXHR_PATH = BASE_DIR / "PAXHR_H.xpt"
OUTPUT_DIR = BASE_DIR / "outputs"
DPI = 300
PSEUDO = 1e-6
ACTIVITY_COLS = ["PAXMTSH", "PAXINTEN"]
DAY_HOURS = set(range(6, 22))   # 06:00-21:59
NIGHT_HOURS = set(range(22, 24)) | set(range(0, 6))  # 22:00-05:59


def load_xpt(fp: Path) -> pd.DataFrame:
    try:
        import pyreadstat
        df, _ = pyreadstat.read_xport(str(fp))
        return df
    except ImportError:
        return pd.read_sas(str(fp), format="xport")


def epr_pseudo(pi0: float, pi1: float, p_01: float, p_10: float, eps: float = PSEUDO) -> float:
    """σ = π₀ P₀₁ log(P₀₁/P₁₀) + π₁ P₁₀ log(P₁₀/P₀₁) with pseudo-count to avoid log(0)."""
    if np.isnan(pi0) or np.isnan(pi1) or np.isnan(p_01) or np.isnan(p_10):
        return np.nan
    p01 = max(p_01, eps)
    p10 = max(p_10, eps)
    return pi0 * p01 * np.log(p01 / p10) + pi1 * p10 * np.log(p10 / p01)


def _markov_threshold(act: np.ndarray, thresh: float) -> tuple[float, float]:
    state = (act > thresh).astype(int)
    if len(state) < 2:
        return np.nan, np.nan
    n_01 = np.sum((state[:-1] == 0) & (state[1:] == 1))
    n_10 = np.sum((state[:-1] == 1) & (state[1:] == 0))
    n_0 = np.sum(state[:-1] == 0)
    n_1 = np.sum(state[:-1] == 1)
    return (n_01 / n_0 if n_0 > 0 else np.nan), (n_10 / n_1 if n_1 > 0 else np.nan)


def time_varying_markov(paxhr: pd.DataFrame, activity_col: str) -> pd.DataFrame:
    """
    Day: 06:00-22:00 (hour indices 6..21).
    Night: 22:00-06:00 (hour indices 22,23,0..5).
    Per-subject threshold = median of full sequence.
    """
    paxhr = paxhr.sort_values(["SEQN", "PAXDAYH", "PAXSSNHP"])
    paxhr["_hour_idx"] = paxhr.groupby(["SEQN", "PAXDAYH"]).cumcount()
    rows = []
    for seqn, grp in paxhr.groupby("SEQN"):
        grp = grp.reset_index(drop=True)
        act_all = grp[activity_col].values
        if len(act_all) < 24:
            continue
        thresh = np.nanmedian(act_all)
        day_mask = grp["_hour_idx"].isin(DAY_HOURS)
        night_mask = grp["_hour_idx"].isin(NIGHT_HOURS)
        act_day = grp.loc[day_mask, activity_col].values
        act_night = grp.loc[night_mask, activity_col].values
        p01_d, p10_d = _markov_threshold(act_day, thresh) if len(act_day) >= 4 else (np.nan, np.nan)
        p01_n, p10_n = _markov_threshold(act_night, thresh) if len(act_night) >= 4 else (np.nan, np.nan)
        rows.append({
            "SEQN": seqn,
            "P_01_Day": p01_d, "P_10_Day": p10_d,
            "P_01_Night": p01_n, "P_10_Night": p10_n,
        })
    return pd.DataFrame(rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("ULTIMATE NON-EQUILIBRIUM DYNAMICS PIPELINE")
    print("=" * 60)

    # Load deep cohort
    df = pd.read_csv(DEEP_PATH)
    print(f"Loaded {len(df)} subjects from {DEEP_PATH.name}")
    seqn_set = set(df["SEQN"].dropna().astype(int))

    # EPR with pseudo-count (from existing pi0, pi1, P_01, P_10)
    df["EPR"] = df.apply(
        lambda r: epr_pseudo(r["pi0"], r["pi1"], r["P_01"], r["P_10"]),
        axis=1,
    )

    # Load PAXHR for time-varying analysis
    paxhr = load_xpt(PAXHR_PATH)
    activity_col = next((c for c in ACTIVITY_COLS if c in paxhr.columns), None)
    if activity_col is None:
        raise KeyError("No activity column in PAXHR_H")
    paxhr = paxhr[paxhr["SEQN"].isin(seqn_set)].copy()
    paxhr = paxhr[paxhr[activity_col].notna() & (paxhr[activity_col] >= 0)]

    tv_df = time_varying_markov(paxhr, activity_col)
    df = df.merge(tv_df, on="SEQN", how="inner")
    df = df.dropna(subset=["Entropy", "Spectral_Gap", "Mixing_Time", "EPR", "P_01_Night", "P_10_Night"])
    print(f"Analytic sample: {len(df)} subjects")

    # Statistical validation
    print("\n" + "=" * 60)
    print("STATISTICAL VALIDATION (Mann-Whitney U)")
    print("=" * 60)
    g0 = df[df["sleep_problem_reported"] == 0]
    g1 = df[df["sleep_problem_reported"] == 1]

    def pval(var):
        a, b = g0[var].dropna(), g1[var].dropna()
        _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        return p

    for v in ["EPR", "P_01_Day", "P_10_Day", "P_01_Night", "P_10_Night"]:
        print(f"  {v}: p = {pval(v):.6f}")

    # T-tests for Day/Night (alternative to Mann-Whitney)
    print("\n--- T-tests (Day/Night transitions) ---")
    for v in ["P_01_Day", "P_10_Day", "P_01_Night", "P_10_Night"]:
        t, p = stats.ttest_ind(g0[v].dropna(), g1[v].dropna())
        print(f"  {v}: t={t:.4f}, p = {p:.6f}")

    # PCA phase space
    print("\n" + "=" * 60)
    print("PCA EIGENVECTOR PHASE SPACE")
    print("=" * 60)
    feats = ["Entropy", "Spectral_Gap", "Mixing_Time", "EPR", "P_01_Night", "P_10_Night"]
    X = df[feats].dropna()
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_std)
    df_pca = df.loc[X.index].copy()
    df_pca["PC1"] = X_pca[:, 0]
    df_pca["PC2"] = X_pca[:, 1]
    print(f"Variance explained: PC1={pca.explained_variance_ratio_[0]:.2%}, PC2={pca.explained_variance_ratio_[1]:.2%}")

    # Figures
    print("\n" + "=" * 60)
    print("FIGURES (300 DPI)")
    print("=" * 60)
    import matplotlib.pyplot as plt

    # fig9: EPR boxplot
    fig, ax = plt.subplots(figsize=(6, 5))
    bp = ax.boxplot(
        [g0["EPR"].dropna(), g1["EPR"].dropna()],
        labels=["No sleep problem", "Sleep problem reported"],
        patch_artist=True,
    )
    bp["boxes"][0].set_facecolor("#2E86AB")
    bp["boxes"][1].set_facecolor("#E94F37")
    ax.set_ylabel("Entropy Production Rate σ", fontsize=11)
    ax.set_title("EPR by Sleep Disorder Status", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig9_EPR_boxplot.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'fig9_EPR_boxplot.png'}")

    # fig10: Day vs Night transitions (grouped bar)
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(4)
    w = 0.35
    m0 = [g0["P_01_Day"].mean(), g0["P_10_Day"].mean(), g0["P_01_Night"].mean(), g0["P_10_Night"].mean()]
    m1 = [g1["P_01_Day"].mean(), g1["P_10_Day"].mean(), g1["P_01_Night"].mean(), g1["P_10_Night"].mean()]
    ax.bar(x - w/2, m0, w, label="No sleep problem", color="#2E86AB")
    ax.bar(x + w/2, m1, w, label="Sleep problem reported", color="#E94F37")
    ax.set_xticks(x)
    ax.set_xticklabels(["P₀₁ Day", "P₁₀ Day", "P₀₁ Night", "P₁₀ Night"])
    ax.set_ylabel("Mean transition probability", fontsize=11)
    ax.set_title("Time-Varying Markov: Day vs Night Transitions", fontsize=12)
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig10_TimeVarying_Transitions.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'fig10_TimeVarying_Transitions.png'}")

    # fig11: PCA phase space with KDE contour
    try:
        from scipy.stats import gaussian_kde
    except ImportError:
        gaussian_kde = None
    fig, ax = plt.subplots(figsize=(8, 6))
    g0_pca = df_pca[df_pca["sleep_problem_reported"] == 0]
    g1_pca = df_pca[df_pca["sleep_problem_reported"] == 1]
    ax.scatter(g0_pca["PC1"], g0_pca["PC2"], alpha=0.4, c="#2E86AB", s=20, label="No sleep problem")
    ax.scatter(g1_pca["PC1"], g1_pca["PC2"], alpha=0.4, c="#E94F37", s=20, label="Sleep problem reported")
    if gaussian_kde is not None:
        for g, c, lbl in [(g1_pca, "#E94F37", "Sleep problem (Chaotic Phase)"), (g0_pca, "#2E86AB", "No problem")]:
            xy = g[["PC1", "PC2"]].dropna().values.T
            if xy.shape[1] > 10:
                try:
                    kde = gaussian_kde(xy, bw_method="scott")
                    xmin, xmax = xy[0].min(), xy[0].max()
                    ymin, ymax = xy[1].min(), xy[1].max()
                    xx = np.linspace(xmin, xmax, 80)
                    yy = np.linspace(ymin, ymax, 80)
                    Xg, Yg = np.meshgrid(xx, yy)
                    Z = kde(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)
                    ax.contour(Xg, Yg, Z, levels=4, colors=[c], alpha=0.7, linewidths=1.5)
                except Exception:
                    pass
    ax.set_xlabel("PC1", fontsize=11)
    ax.set_ylabel("PC2", fontsize=11)
    ax.set_title("Eigenvector Phase Space: Dynamic Attractors", fontsize=12)
    ax.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig11_Eigenvector_PhaseSpace.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'fig11_Eigenvector_PhaseSpace.png'}")

    out_path = OUTPUT_DIR / "processed_data_physics_ultimate.csv"
    df.to_csv(out_path, index=False)
    print(f"\nData saved: {out_path}")
    return df


if __name__ == "__main__":
    main()
