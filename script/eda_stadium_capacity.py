from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "processed"

CAPACITY_ORDER = ["0-20K", "20K-40K", "40K+"]
RESULT_ORDER = ["Away Win", "Draw", "Home Win"]
RESULT_LABEL_MAP = {"H": "Home Win", "D": "Draw", "A": "Away Win"}


def load_stadium_capacity_df() -> pd.DataFrame:
    stadium_df = pd.read_csv(PROCESSED_DIR / "football_eda_ready_complete_fixed_v2.csv")

    stadium_df = stadium_df[
        (stadium_df["season"] == "2024_2025")
        & (stadium_df["season_type"] == "normal")
    ].copy()

    stadium_df = stadium_df.dropna(subset=["capacity", "result"])

    stadium_df["capacity_interval"] = pd.cut(
        stadium_df["capacity"],
        bins=[0, 20000, 40000, np.inf],
        labels=CAPACITY_ORDER,
        include_lowest=True,
        right=False,
    )

    stadium_df["result_label"] = stadium_df["result"].map(RESULT_LABEL_MAP)
    stadium_df["home_win"] = (stadium_df["result"] == "H").astype(int)
    stadium_df["away_win"] = (stadium_df["result"] == "A").astype(int)
    stadium_df["draw"] = (stadium_df["result"] == "D").astype(int)
    stadium_df["capacity_group"] = stadium_df["capacity_interval"]

    return stadium_df


def build_result_distribution(stadium_df: pd.DataFrame) -> pd.DataFrame:
    result_dist = (
        stadium_df.groupby(["capacity_interval", "result_label"], observed=False)
        .size()
        .rename("count")
        .reset_index()
    )

    totals = result_dist.groupby("capacity_interval", observed=False)["count"].transform("sum")
    result_dist["proportion"] = (result_dist["count"] / totals).round(3)

    return result_dist


def build_result_matrix(result_dist: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for capacity in CAPACITY_ORDER:
        row = {"capacity_interval": capacity}
        capacity_rows = result_dist[result_dist["capacity_interval"] == capacity]

        for result_label in RESULT_ORDER:
            match = capacity_rows[capacity_rows["result_label"] == result_label]
            row[result_label] = match["proportion"].iloc[0] if not match.empty else pd.NA

        rows.append(row)

    return pd.DataFrame(rows).set_index("capacity_interval")


def build_home_win_by_capacity(stadium_df: pd.DataFrame) -> pd.DataFrame:
    return (
        stadium_df.groupby("capacity_interval", observed=False)
        .agg(
            match_count=("home_win", "count"),
            avg_capacity=("capacity", "mean"),
            home_win_rate=("home_win", "mean"),
            draw_rate=("draw", "mean"),
            away_win_rate=("away_win", "mean"),
        )
        .round(3)
        .reindex(CAPACITY_ORDER)
        .reset_index()
    )


def build_league_capacity_homewin(stadium_df: pd.DataFrame) -> pd.DataFrame:
    return (
        stadium_df.groupby(["league", "capacity_interval"], observed=False)
        .agg(
            match_count=("home_win", "count"),
            avg_capacity=("capacity", "mean"),
            home_win_rate=("home_win", "mean"),
            draw_rate=("draw", "mean"),
            away_win_rate=("away_win", "mean"),
        )
        .round(3)
        .reset_index()
    )


def build_summary_table(stadium_df: pd.DataFrame) -> pd.DataFrame:
    return (
        stadium_df.groupby(["league", "capacity_group"], observed=False)
        .agg(
            matches=("home_win", "count"),
            avg_capacity=("capacity", "mean"),
            home_win_rate=("home_win", "mean"),
        )
        .round(3)
        .reset_index()
    )
