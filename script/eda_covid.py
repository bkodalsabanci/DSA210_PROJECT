from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "processed"
RESULT_LABEL_MAP = {"H": "Home Win", "D": "Draw", "A": "Away Win"}


def load_covid_df() -> pd.DataFrame:
    covid_df = pd.read_csv(PROCESSED_DIR / "football_eda_ready_complete_fixed_v2.csv")

    covid_df["date"] = pd.to_datetime(covid_df["date"], errors="coerce")
    covid_df["total_goals"] = covid_df["home_goals"] + covid_df["away_goals"]
    covid_df["goal_diff"] = covid_df["home_goals"] - covid_df["away_goals"]
    covid_df["result_label"] = covid_df["result"].map(RESULT_LABEL_MAP)

    covid_df["home_win"] = (covid_df["result"] == "H").astype(int)
    covid_df["away_win"] = (covid_df["result"] == "A").astype(int)
    covid_df["draw"] = (covid_df["result"] == "D").astype(int)

    covid_df["home_points"] = np.select(
        [covid_df["result"] == "H", covid_df["result"] == "D"],
        [3, 1],
        default=0,
    )

    covid_df["away_points"] = np.select(
        [covid_df["result"] == "A", covid_df["result"] == "D"],
        [3, 1],
        default=0,
    )

    covid_df["home_advantage_points"] = (
        covid_df["home_points"] - covid_df["away_points"]
    )

    return covid_df


def build_season_summary(covid_df: pd.DataFrame) -> pd.DataFrame:
    return (
        covid_df.groupby("season_type", observed=False)
        .agg(
            matches=("result_label", "size"),
            avg_home_goals=("home_goals", "mean"),
            avg_away_goals=("away_goals", "mean"),
            avg_total_goals=("total_goals", "mean"),
            avg_goal_diff=("goal_diff", "mean"),
        )
        .round(3)
        .reset_index()
    )


def build_result_rates(covid_df: pd.DataFrame) -> pd.DataFrame:
    results = (
        covid_df.groupby(["season_type", "result_label"], observed=False)
        .size()
        .rename("count")
        .reset_index()
    )

    totals = results.groupby("season_type", observed=False)["count"].transform("sum")
    results["proportion"] = (results["count"] / totals).round(3)

    return results


def build_home_advantage_summary(covid_df: pd.DataFrame) -> pd.DataFrame:
    return (
        covid_df.groupby("season_type", observed=False)
        .agg(
            matches=("result_label", "size"),
            home_win_rate=("home_win", "mean"),
            away_win_rate=("away_win", "mean"),
            draw_rate=("draw", "mean"),
            avg_goal_diff=("goal_diff", "mean"),
            home_points_per_match=("home_points", "mean"),
            away_points_per_match=("away_points", "mean"),
            home_advantage_points=("home_advantage_points", "mean"),
        )
        .round(3)
        .reset_index()
    )


def build_league_home_win_table(covid_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        covid_df.groupby(["league", "season_type"], observed=False)["home_win"]
        .mean()
        .round(3)
        .reset_index()
    )

    leagues = sorted(grouped["league"].unique())
    season_types = ["covid", "normal"]
    rows = []

    for league in leagues:
        row = {"league": league}
        league_rows = grouped[grouped["league"] == league]

        for season_type in season_types:
            season_rows = league_rows[league_rows["season_type"] == season_type]
            row[season_type] = (
                season_rows["home_win"].iloc[0] if not season_rows.empty else pd.NA
            )

        rows.append(row)

    table = pd.DataFrame(rows)
    table["delta_normal_minus_covid"] = (
        table["normal"].astype(float) - table["covid"].astype(float)
    ).round(3)

    return table.sort_values("delta_normal_minus_covid", ascending=False)
