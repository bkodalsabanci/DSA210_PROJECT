from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "processed"

FORM_ORDER = ["bad", "medium", "good"]
RESULT_ORDER = ["Away Win", "Draw", "Home Win"]
SEASON_ORDER = ["covid", "normal"]
CAPACITY_ORDER = ["0-20K", "20K-40K", "40K+"]
RESULT_LABEL_MAP = {"H": "Home Win", "D": "Draw", "A": "Away Win"}


def load_last3_df() -> pd.DataFrame:
    frames = []

    for file in sorted(PROCESSED_DIR.glob("*_last3_enriched.csv")):
        df = pd.read_csv(file)
        frames.append(df)

    last3_df = pd.concat(frames, ignore_index=True)
    last3_df["point_diff"] = (
        last3_df["home_last3_points"] - last3_df["away_last3_points"]
    )
    last3_df["home_win"] = (last3_df["result_match"] == "Home Win").astype(int)

    return last3_df


def load_covid_df() -> pd.DataFrame:
    frames = []

    for file in sorted(PROCESSED_DIR.glob("*_covid_comparison.csv")):
        df = pd.read_csv(file)
        frames.append(df)

    covid_df = pd.concat(frames, ignore_index=True)
    covid_df["home_win"] = (covid_df["FTR"] == "H").astype(int)

    return covid_df


def load_capacity_df() -> pd.DataFrame:
    capacity_df = pd.read_csv(PROCESSED_DIR / "football_eda_ready_complete_fixed_v2.csv")

    capacity_df = capacity_df[
        (capacity_df["season"] == "2024_2025")
        & (capacity_df["season_type"] == "normal")
    ].copy()

    capacity_df = capacity_df.dropna(subset=["capacity", "result"])

    capacity_df["capacity_interval"] = pd.cut(
        capacity_df["capacity"],
        bins=[0, 20000, 40000, np.inf],
        labels=CAPACITY_ORDER,
        include_lowest=True,
        right=False,
    )

    capacity_df["result_label"] = capacity_df["result"].map(RESULT_LABEL_MAP)
    capacity_df["home_win"] = (capacity_df["result"] == "H").astype(int)

    return capacity_df


def interpret_pvalue(p_value: float, alpha: float = 0.05) -> str:
    if p_value < alpha:
        return f"p-value = {p_value:.6f} < {alpha} -> H0 rejected."
    return f"p-value = {p_value:.6f} >= {alpha} -> H0 not rejected."


def cramers_v(table: pd.DataFrame) -> float:
    chi2, _, _, _ = stats.chi2_contingency(table)
    n = table.to_numpy().sum()
    r, k = table.shape
    return float(np.sqrt(chi2 / (n * min(r - 1, k - 1))))


def form_test(last3_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    table = (
        last3_df.groupby(["home_form", "result_match"], observed=False)
        .size()
        .unstack(fill_value=0)
        .reindex(index=FORM_ORDER, columns=RESULT_ORDER, fill_value=0)
    )

    chi2_stat, p_value, dof, _ = stats.chi2_contingency(table)

    result = pd.DataFrame(
        [
            {
                "test": "home_form_vs_result",
                "type": "chi_square",
                "statistic": round(float(chi2_stat), 4),
                "p_value": round(float(p_value), 6),
                "dof": int(dof),
                "effect": round(cramers_v(table), 4),
                "effect_name": "cramers_v",
                "decision": interpret_pvalue(float(p_value)),
            }
        ]
    )

    return table.reset_index(), result


def point_diff_test(last3_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    r_stat, p_value = stats.pearsonr(last3_df["point_diff"], last3_df["home_win"])

    summary = (
        last3_df.groupby("home_win", observed=False)["point_diff"]
        .agg(count="count", mean="mean", median="median")
        .reset_index()
    )

    summary["home_win"] = summary["home_win"].map({0: "No", 1: "Yes"})
    summary[["mean", "median"]] = summary[["mean", "median"]].round(3)

    result = pd.DataFrame(
        [
            {
                "test": "point_diff_vs_home_win",
                "type": "pearson",
                "statistic": round(float(r_stat), 4),
                "p_value": round(float(p_value), 6),
                "dof": pd.NA,
                "effect": round(float(r_stat), 4),
                "effect_name": "pearson_r",
                "decision": interpret_pvalue(float(p_value)),
            }
        ]
    )

    return summary, result


def covid_test(covid_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    home_win_counts = (
        covid_df.groupby("season_type", observed=False)["home_win"]
        .sum()
        .reindex(SEASON_ORDER)
    )

    match_counts = (
        covid_df.groupby("season_type", observed=False)["home_win"]
        .count()
        .reindex(SEASON_ORDER)
    )

    home_win_rates = (home_win_counts / match_counts).round(6)

    summary = pd.DataFrame(
        {
            "season_type": SEASON_ORDER,
            "home_wins": home_win_counts.values,
            "matches": match_counts.values,
            "home_win_rate": home_win_rates.values,
        }
    )

    z_stat, p_value = proportions_ztest(
        count=home_win_counts.values,
        nobs=match_counts.values,
    )

    result = pd.DataFrame(
        [
            {
                "test": "covid_vs_normal_home_win",
                "type": "z_test",
                "statistic": round(float(z_stat), 4),
                "p_value": round(float(p_value), 6),
                "dof": pd.NA,
                "effect": round(
                    float(home_win_rates.loc["normal"] - home_win_rates.loc["covid"]),
                    4,
                ),
                "effect_name": "rate_diff",
                "decision": interpret_pvalue(float(p_value)),
            }
        ]
    )

    return summary, result


def capacity_test(
    capacity_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    table = (
        capacity_df.groupby(["capacity_interval", "result_label"], observed=False)
        .size()
        .unstack(fill_value=0)
        .reindex(index=CAPACITY_ORDER, columns=RESULT_ORDER, fill_value=0)
    )

    chi2_stat, p_value, dof, _ = stats.chi2_contingency(table)

    summary = (
        capacity_df.groupby("capacity_interval", observed=False)["home_win"]
        .agg(matches="count", home_win_rate="mean")
        .reindex(CAPACITY_ORDER)
        .reset_index()
    )

    summary["home_win_rate"] = summary["home_win_rate"].round(6)

    result = pd.DataFrame(
        [
            {
                "test": "capacity_vs_result",
                "type": "chi_square",
                "statistic": round(float(chi2_stat), 4),
                "p_value": round(float(p_value), 6),
                "dof": int(dof),
                "effect": round(cramers_v(table), 4),
                "effect_name": "cramers_v",
                "decision": interpret_pvalue(float(p_value)),
            }
        ]
    )

    return table.reset_index(), summary, result
