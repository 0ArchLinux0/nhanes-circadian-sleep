#!/usr/bin/env python3
"""
Multivariate Logistic Regression: Physics Metrics as Independent Predictors
===========================================================================
Validates Night P_01, Entropy (and Day P_10, EPR) after adjusting for Age, Gender.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "outputs" / "processed_data_physics_ultimate.csv"
if not DATA_PATH.exists():
    DATA_PATH = BASE_DIR / "outputs" / "processed_data_physics_deep.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
DPI = 300

PHYSICS_VARS = ["Entropy", "Night_P_01", "Day_P_10"]

def _safe_max(s, default=3.0):
    s = pd.Series(s).replace([np.inf, -np.inf], np.nan).dropna()
    return default if s.empty else float(s.max())


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("MULTIVARIATE LOGISTIC REGRESSION VALIDATION")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns={
        "age": "Age",
        "gender": "Gender",
        "P_01_Night": "Night_P_01",
        "P_10_Day": "Day_P_10",
    })
    if "Age" not in df.columns:
        df["Age"] = df.get("age", np.nan)
    if "Gender" not in df.columns:
        df["Gender"] = df.get("gender", np.nan)

    vars_to_std = ["Age", "Entropy", "Night_P_01", "Day_P_10", "EPR"]
    df_clean = df[["sleep_problem_reported", "Age", "Gender"] + [c for c in vars_to_std[1:] if c in df.columns]].copy()
    df_clean = df_clean.dropna()
    if "Night_P_01" not in df_clean.columns or "Day_P_10" not in df_clean.columns:
        print("ERROR: Night_P_01 or Day_P_10 not in dataset. Use processed_data_physics_ultimate.csv")
        return

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    cols_std = ["Age", "Entropy", "Night_P_01", "Day_P_10"]
    cols_std = [c for c in cols_std if c in df_clean.columns]
    df_clean[cols_std] = scaler.fit_transform(df_clean[cols_std].astype(float))
    df_clean["Gender"] = df_clean["Gender"].astype(int)

    import statsmodels.api as sm
    # EPR excluded: near-zero variance when standardized causes non-convergence
    x_cols = ["Age", "Gender", "Entropy", "Night_P_01", "Day_P_10"]
    X = df_clean[x_cols]
    X = sm.add_constant(X, has_constant="add")
    y = df_clean["sleep_problem_reported"]

    model = sm.Logit(y, X).fit(disp=0, maxiter=200)
    print("\nFormula: sleep_problem_reported ~ Age + Gender + Entropy + Night_P_01 + Day_P_10")
    print("(EPR excluded: near-zero variance causes model non-convergence)")
    print("\n" + "=" * 60)
    print("STATSMODELS LOGISTIC REGRESSION SUMMARY")
    print("=" * 60)
    print(model.summary())

    or_vals = np.exp(model.params)
    ci = np.exp(model.conf_int())
    ci.columns = ["CI_lower", "CI_upper"]
    results = pd.DataFrame({
        "variable": model.params.index,
        "coef": model.params.values,
        "OR": or_vals.values,
        "CI_lower": ci["CI_lower"].values,
        "CI_upper": ci["CI_upper"].values,
        "pvalue": model.pvalues.values,
    })
    results = results[results["variable"] != "const"]
    results = results.replace([np.inf, -np.inf], np.nan)
    results = results.dropna(subset=["CI_lower", "CI_upper"], how="all")

    print("\n" + "=" * 60)
    print("ODDS RATIOS (95% CI) & P-VALUES")
    print("=" * 60)
    for _, r in results.iterrows():
        print(f"  {r['variable']:14s} OR={r['OR']:.3f} [95%CI {r['CI_lower']:.3f}-{r['CI_upper']:.3f}] p={r['pvalue']:.6f}")

    night_p01_row = results[results["variable"] == "Night_P_01"]
    night_sig = len(night_p01_row) > 0 and night_p01_row["pvalue"].iloc[0] < 0.05
    print("\n" + "=" * 60)
    print("CONCLUSION: Night P_01 as Independent Predictor")
    print("=" * 60)
    if night_sig:
        print("  Night P_01 REMAINS statistically significant (p < 0.05) after controlling for Age and Gender.")
    else:
        pval = night_p01_row["pvalue"].iloc[0] if len(night_p01_row) > 0 else np.nan
        print(f"  Night P_01 is NOT significant after adjustment (p = {pval:.4f}).")

    # Forest plot
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 6))
    results_plot = results.copy()
    results_plot = results_plot.sort_values("OR", ascending=True)
    y_pos = np.arange(len(results_plot))
    colors = ["#E94F37" if v in PHYSICS_VARS else "#333333" for v in results_plot["variable"]]
    ax.barh(y_pos, results_plot["OR"], color=colors, alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.errorbar(
        results_plot["OR"], y_pos,
        xerr=[results_plot["OR"] - results_plot["CI_lower"], results_plot["CI_upper"] - results_plot["OR"]],
        fmt="none", ecolor="black", capsize=4,
    )
    ax.axvline(1, color="gray", linestyle="--", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(results_plot["variable"], fontsize=10)
    ax.set_xlabel("Odds Ratio (95% CI)", fontsize=11)
    ax.set_title("Forest Plot: Multivariate Logistic Regression\n(Physics variables highlighted in red)", fontsize=12)
    x_max = max(_safe_max(results_plot["CI_upper"]) * 1.15, 1.5)
    ax.set_xlim(0.25, x_max)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig12_forest_plot.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {OUTPUT_DIR / 'fig12_forest_plot.png'}")


if __name__ == "__main__":
    main()
