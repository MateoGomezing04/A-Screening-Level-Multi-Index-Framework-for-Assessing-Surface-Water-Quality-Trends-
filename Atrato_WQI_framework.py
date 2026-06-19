#!/usr/bin/env python3
"""
=============================================================================
Reproducibility Script for:
  "A Screening-Level Multi-Index Framework for Assessing Surface Water
   Quality Trends in the Atrato River Basin, Colombia, Under Data-Scarce
   Monitoring Conditions and Artisanal Gold Mining Pressure"

Authors: Marimón Bolívar W., Toussaint Jimenez N., Castro Arriaga A.,
         Gómez Espinel M.

Script version: 1.0 | June 2026
Python: 3.10+  |  Dependencies: numpy, pandas, scipy

Description:
  This script implements the complete analytical workflow described in the
  manuscript (Sections 2.3-2.5), including:
    1. Index orientation normalization (Eq. 4)
    2. Theil-Sen slope estimation (Eq. 5)
    3. Kendall tau coefficient and p-value (Eq. 6)
    4. Inter-period median difference (Eq. 7)
    5. Proportional threshold classification (Eq. 8)
    6. Bootstrap classification stability (1,000 iterations)
    7. Permutation significance test (1,000 iterations)
    8. Threshold sensitivity analysis (Table 6)

Input:
  CSV with columns: station_id, year (decimal), ICA, ICOMO, ICOMI,
  ICOSUS, ICOMINERIA, ICOTRO

Output:
  CSV with Theil-Sen slopes, Kendall tau, p-values, trend classes,
  bootstrap stability, and permutation p-values per station-index pair.

Usage:
  python atrato_wqi_framework.py
  python atrato_wqi_framework.py --input my_data.csv --output results.csv
  python atrato_wqi_framework.py --input my_data.csv --sensitivity

LICENSE: CC BY 4.0
=============================================================================
"""

import argparse
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from itertools import combinations

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
# Index orientation sign (s_k):
#   +1  higher value = better quality   (ICA)
#   -1  higher value = greater contamination (ICO family)
ORIENTATION = {
    "ICA":        +1,
    "ICOMO":      -1,
    "ICOMI":      -1,
    "ICOSUS":     -1,
    "ICOMINERIA": -1,
    "ICOTRO":     -1,
}

# Theoretical range R_k for proportional thresholds (Eq. 8)
#   ICA, ICOMO, ICOMI, ICOSUS, ICOMINERIA: [0,1] -> R_k = 1
#   ICOTRO: ordinal 1-4 -> R_k = 3
RANGE = {
    "ICA": 1.0, "ICOMO": 1.0, "ICOMI": 1.0,
    "ICOSUS": 1.0, "ICOMINERIA": 1.0, "ICOTRO": 3.0,
}

THETA_BETA_FRAC  = 0.02   # slope threshold fraction
THETA_DELTA_FRAC = 0.10   # median-diff threshold fraction
N_BOOTSTRAP      = 1000
N_PERMUTE        = 1000
MIN_OBS          = 4      # minimum valid observations for trend estimation


# ── STEP 1: INDEX ORIENTATION NORMALIZATION (Eq. 4) ──────────────────────────
def normalize_orientation(y, index_name):
    """
    q = s_k * y
    After normalization, positive increments in q always represent
    water-quality improvement regardless of original index convention.
    """
    s_k = ORIENTATION.get(index_name, +1)
    return s_k * np.array(y, dtype=float)


# ── STEP 2: THEIL-SEN SLOPE ESTIMATOR (Eq. 5) ────────────────────────────────
def theil_sen_slope(t, y):
    """
    beta_TS = median_{i<j} [ (y_j - y_i) / (t_j - t_i) ]
    Applied in q-space: positive = improving, negative = deteriorating.
    Breakdown point ~0.293; reliable at n >= 4.
    """
    t = np.array(t, dtype=float)
    y = np.array(y, dtype=float)
    slopes = []
    for (i, j) in combinations(range(len(t)), 2):
        dt = t[j] - t[i]
        if dt != 0:
            slopes.append((y[j] - y[i]) / dt)
    return float(np.median(slopes)) if slopes else np.nan


# ── STEP 3: KENDALL TAU AND P-VALUE (Eq. 6) ──────────────────────────────────
def kendall_tau_pvalue(t, y):
    """
    tau = (C - D) / [n(n-1)/2]
    C = concordant pairs, D = discordant pairs.
    Significance assessed at alpha = 0.05.
    """
    tau, p = kendalltau(np.array(t, float), np.array(y, float))
    return float(tau), float(p)


# ── STEP 4: INTER-PERIOD MEDIAN DIFFERENCE (Eq. 7) ───────────────────────────
def inter_period_median_diff(t, q, split_year=2022.5):
    """
    Delta_q = median(q | t > split_year) - median(q | t <= split_year)
    Periods: 2020-2022 (early) vs 2023-2025 (late) with default split=2022.5
    Positive Delta_q in q-space = quality improvement in second period.
    """
    t = np.array(t, dtype=float)
    q = np.array(q, dtype=float)
    early = q[t <= split_year]
    late  = q[t >  split_year]
    if len(early) == 0 or len(late) == 0:
        return np.nan
    return float(np.median(late) - np.median(early))


# ── STEP 5: TREND CLASSIFICATION (Eq. 8) ─────────────────────────────────────
def classify_trend(beta_ts, delta_q, index_name, n,
                   theta_beta_frac=THETA_BETA_FRAC,
                   theta_delta_frac=THETA_DELTA_FRAC):
    """
    Assign trend class using proportional thresholds in q-space.

    Decision rules:
      Insufficient data : n < MIN_OBS
      Stable            : |beta_ts| <= theta_beta AND |delta_q| <= theta_delta
      Improving         : beta_ts > theta_beta  AND delta_q >= -0.5*theta_delta
      Deteriorating     : beta_ts < -theta_beta AND delta_q <=  0.5*theta_delta
      Mixed/Unclear     : contradictory signals
    """
    if n < MIN_OBS or np.isnan(beta_ts) or np.isnan(delta_q):
        return "Insufficient data"
    R   = RANGE.get(index_name, 1.0)
    t_b = theta_beta_frac  * R
    t_d = theta_delta_frac * R
    if abs(beta_ts) <= t_b and abs(delta_q) <= t_d:
        return "Stable"
    if beta_ts >  t_b and delta_q >= -0.5 * t_d:
        return "Improving"
    if beta_ts < -t_b and delta_q <=  0.5 * t_d:
        return "Deteriorating"
    return "Mixed/Unclear"


# ── STEP 6: BOOTSTRAP CLASSIFICATION STABILITY ───────────────────────────────
def bootstrap_stability(t, y_raw, index_name, n_iter=N_BOOTSTRAP, seed=42):
    """
    Assess classification stability via bootstrap resampling (with replacement).
    Returns: percentage of n_iter iterations retaining the original class.
    Random seed fixed at 42 for reproducibility.
    """
    rng = np.random.default_rng(seed)
    t   = np.array(t, dtype=float)
    q   = normalize_orientation(y_raw, index_name)
    n   = len(t)
    if n < MIN_OBS:
        return np.nan
    ref_b   = theil_sen_slope(t, q)
    ref_d   = inter_period_median_diff(t, q)
    ref_cls = classify_trend(ref_b, ref_d, index_name, n)
    matches = 0
    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)
        b   = theil_sen_slope(t[idx], q[idx])
        d   = inter_period_median_diff(t[idx], q[idx])
        if classify_trend(b, d, index_name, n) == ref_cls:
            matches += 1
    return round(100.0 * matches / n_iter, 1)


# ── STEP 7: PERMUTATION SIGNIFICANCE TEST ────────────────────────────────────
def permutation_pvalue(t, y_raw, index_name, n_iter=N_PERMUTE, seed=42):
    """
    Evaluate whether observed Theil-Sen slope could arise from random
    temporal ordering by permuting time labels 1,000 times.
    Returns: empirical p-value = P(|beta_perm| >= |beta_obs|).
    Random seed fixed at 42 for reproducibility.
    """
    rng     = np.random.default_rng(seed)
    t       = np.array(t, dtype=float)
    q       = normalize_orientation(y_raw, index_name)
    n       = len(t)
    if n < MIN_OBS:
        return np.nan
    b_obs = theil_sen_slope(t, q)
    if np.isnan(b_obs):
        return np.nan
    count = sum(
        abs(theil_sen_slope(rng.permutation(t), q)) >= abs(b_obs)
        for _ in range(n_iter)
    )
    return round(count / n_iter, 4)


# ── FULL ANALYSIS WORKFLOW ────────────────────────────────────────────────────
def run_analysis(df, station_col="station_id", year_col="year",
                 indices=None, split_year=2022.5,
                 theta_beta_frac=THETA_BETA_FRAC,
                 theta_delta_frac=THETA_DELTA_FRAC):
    """
    Run the complete 7-step analytical framework on a station-year dataset.

    Expected input columns: station_id, year (decimal float),
    and one column per index (ICA, ICOMO, ICOMI, ICOSUS, ICOMINERIA, ICOTRO).

    Returns a DataFrame with one row per (station, index) combination
    containing: n, ts_slope_raw, ts_slope_qspace, kendall_tau,
    kendall_pvalue, delta_q, trend_class, bootstrap_stability_pct,
    permutation_pvalue, stat_significant.
    """
    if indices is None:
        indices = [c for c in
                   ["ICA","ICOMO","ICOMI","ICOSUS","ICOMINERIA","ICOTRO"]
                   if c in df.columns]
    records = []
    for station, grp in df.groupby(station_col):
        grp = grp.sort_values(year_col)
        t   = grp[year_col].values.astype(float)
        for idx in indices:
            if idx not in grp.columns:
                continue
            mask   = ~pd.isna(grp[idx].values)
            y_raw  = grp[idx].values[mask].astype(float)
            t_valid= t[mask]
            n      = len(y_raw)
            if n < MIN_OBS:
                records.append({
                    "station_id": station, "index": idx, "n": n,
                    "ts_slope_raw": np.nan, "ts_slope_qspace": np.nan,
                    "kendall_tau": np.nan, "kendall_pvalue": np.nan,
                    "delta_q": np.nan,
                    "trend_class": "Insufficient data",
                    "bootstrap_stability_pct": np.nan,
                    "permutation_pvalue": np.nan,
                    "stat_significant": False,
                })
                continue
            q       = normalize_orientation(y_raw, idx)
            b_raw   = theil_sen_slope(t_valid, y_raw)
            b_q     = theil_sen_slope(t_valid, q)
            tau, kp = kendall_tau_pvalue(t_valid, q)
            dq      = inter_period_median_diff(t_valid, q, split_year)
            cls     = classify_trend(b_q, dq, idx, n,
                                     theta_beta_frac, theta_delta_frac)
            bs      = bootstrap_stability(t_valid, y_raw, idx)
            pp      = permutation_pvalue(t_valid, y_raw, idx)
            records.append({
                "station_id":              station,
                "index":                   idx,
                "n":                       n,
                "ts_slope_raw":            round(b_raw, 5),
                "ts_slope_qspace":         round(b_q,   5),
                "kendall_tau":             round(tau,    3),
                "kendall_pvalue":          round(kp,     4),
                "delta_q":                 round(dq, 4) if not np.isnan(dq) else np.nan,
                "trend_class":             cls,
                "bootstrap_stability_pct": bs,
                "permutation_pvalue":      pp,
                "stat_significant":        kp < 0.05,
            })
    return pd.DataFrame(records)


# ── THRESHOLD SENSITIVITY ANALYSIS (Table 6) ─────────────────────────────────
def sensitivity_analysis(df, station_col="station_id", year_col="year",
                          indices=None):
    """
    Reproduce Table 6 of the manuscript: agreement (%) under conservative,
    base, and permissive threshold scenarios for theta_beta and theta_delta.
    """
    scenarios = {
        "theta_beta_Conservative":  {"theta_beta_frac": 0.01, "theta_delta_frac": 0.10},
        "theta_beta_Base":          {"theta_beta_frac": 0.02, "theta_delta_frac": 0.10},
        "theta_beta_Permissive":    {"theta_beta_frac": 0.03, "theta_delta_frac": 0.10},
        "theta_delta_Conservative": {"theta_beta_frac": 0.02, "theta_delta_frac": 0.05},
        "theta_delta_Base":         {"theta_beta_frac": 0.02, "theta_delta_frac": 0.10},
        "theta_delta_Permissive":   {"theta_beta_frac": 0.02, "theta_delta_frac": 0.15},
    }
    base = run_analysis(df, station_col, year_col, indices,
                        theta_beta_frac=0.02, theta_delta_frac=0.10)
    base_cls = base.set_index(["station_id", "index"])["trend_class"]
    rows = []
    for name, params in scenarios.items():
        res     = run_analysis(df, station_col, year_col, indices, **params)
        res_cls = res.set_index(["station_id", "index"])["trend_class"]
        common  = base_cls.index.intersection(res_cls.index)
        elig    = base_cls.loc[common]
        elig    = elig[elig != "Insufficient data"]
        matches = (res_cls.loc[elig.index] == elig).sum()
        agreement = round(100.0 * matches / len(elig), 2) if len(elig) else np.nan
        comp    = "theta_beta" if "beta" in name else "theta_delta"
        scen    = name.split("_")[-1]
        rows.append({
            "component": comp, "scenario": scen,
            "theta_beta_frac":  params["theta_beta_frac"],
            "theta_delta_frac": params["theta_delta_frac"],
            "agreement_pct":    agreement,
        })
    return pd.DataFrame(rows)


# ── DEMO DATASET (from manuscript Table 7 + Section 3.4) ─────────────────────
def build_demo_dataset():
    """
    Build a representative station-year dataset from manuscript reported values.
    Network annual means from Table 7; station deviations from Section 3.4.

    NOTE: Replace with actual CODECHOCO station-year data for full analysis.
    The CSV structure is: station_id, year (decimal), ICA, ICOMO, ICOMI,
    ICOSUS, ICOMINERIA, ICOTRO
    """
    network = {
        2020: {"ICA":0.654,"ICOMO":0.249,"ICOMI":0.024,"ICOSUS":0.325,"ICOMINERIA":0.191,"ICOTRO":1},
        2021: {"ICA":0.558,"ICOMO":0.176,"ICOMI":0.017,"ICOSUS":0.479,"ICOMINERIA":0.242,"ICOTRO":3},
        2022: {"ICA":0.582,"ICOMO":0.147,"ICOMI":0.455,"ICOSUS":0.661,"ICOMINERIA":0.303,"ICOTRO":2},
        2023: {"ICA":0.711,"ICOMO":0.145,"ICOMI":0.035,"ICOSUS":0.332,"ICOMINERIA":0.194,"ICOTRO":3},
        2024: {"ICA":0.621,"ICOMO":0.155,"ICOMI":0.027,"ICOSUS":0.523,"ICOMINERIA":0.405,"ICOTRO":1},
        2025: {"ICA":0.609,"ICOMO":0.175,"ICOMI":0.029,"ICOSUS":0.433,"ICOMINERIA":0.502,"ICOTRO":3},
    }
    adj = {
        "CA-R-AT-01":-0.08,"CA-R-AT-02":-0.06,"CA-R-AT-03":-0.09,
        "LL-R-AT-04": 0.02,"AT-R-AT-05": 0.10,"AT-R-AT-06": 0.09,
        "AT-R-AT-07": 0.04,"QB-R-AT-08": 0.05,"QB-R-AT-09": 0.07,
        "QB-R-AT-10": 0.18,"QB-R-AT-11": 0.15,"MA-R-AT-13": 0.06,
        "MA-R-AT-14": 0.12,"BY-R-AT-16": 0.09,"CD-R-AT-24": 0.03,
        "RU-R-AT-27": 0.05,"RU-R-AT-28": 0.01,
    }
    rng  = np.random.default_rng(42)
    rows = []
    for sid, a in adj.items():
        for yr, means in network.items():
            row = {"station_id": sid, "year": float(yr)}
            for k, v in means.items():
                if k == "ICOMINERIA":
                    row[k] = round(float(np.clip(v + a + rng.normal(0, 0.02), 0, 1)), 4)
                elif k == "ICOTRO":
                    row[k] = int(np.clip(round(v + rng.normal(0, 0.4)), 1, 4))
                else:
                    row[k] = round(float(np.clip(v + rng.normal(0, 0.03), 0, 1)), 4)
            rows.append(row)
    return pd.DataFrame(rows)


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Atrato River WQI multi-index trend framework")
    parser.add_argument("--input",  default=None,
        help="Input CSV (station_id, year, ICA, ICOMO, ICOMI, ICOSUS, "
             "ICOMINERIA, ICOTRO). If omitted, uses demo dataset.")
    parser.add_argument("--output", default="atrato_results.csv",
        help="Output CSV path (default: atrato_results.csv)")
    parser.add_argument("--sensitivity", action="store_true",
        help="Run threshold sensitivity analysis (reproduces Table 6)")
    parser.add_argument("--split_year", type=float, default=2022.5,
        help="Split year for inter-period comparison (default: 2022.5)")
    args = parser.parse_args()

    if args.input:
        print(f"Loading data from: {args.input}")
        data = pd.read_csv(args.input)
    else:
        print("No input file specified. Using demo dataset (Table 7 values).")
        data = build_demo_dataset()
        print(f"  Demo: {len(data)} station-year records, "
              f"{data['station_id'].nunique()} stations.")

    print("Running analytical framework (Steps 1-7)...")
    results = run_analysis(data, split_year=args.split_year)
    results.to_csv(args.output, index=False)
    print(f"  Results saved to: {args.output}")

    print("=== TREND CLASSIFICATION SUMMARY (Table 10) ===")
    for idx in ["ICA","ICOMO","ICOMI","ICOSUS","ICOMINERIA","ICOTRO"]:
        sub    = results[results["index"] == idx]
        counts = sub["trend_class"].value_counts()
        sig    = sub["stat_significant"].sum()
        print(
            f"  {idx:12s}: "
            f"Improving={counts.get('Improving',0):2d} | "
            f"Stable={counts.get('Stable',0):2d} | "
            f"Deteriorating={counts.get('Deteriorating',0):2d} | "
            f"Mixed={counts.get('Mixed/Unclear',0):2d} | "
            f"Insufficient={counts.get('Insufficient data',0):2d} | "
            f"Stat.Sig.(p<0.05)={sig}"
        )

    if args.sensitivity:
        print("Running threshold sensitivity analysis (Table 6)...")
        sens = sensitivity_analysis(data)
        print(sens.to_string(index=False))
        sens_path = args.output.replace(".csv", "_sensitivity.csv")
        sens.to_csv(sens_path, index=False)
        print(f"  Sensitivity results saved to: {sens_path}")

    print("Done.")
