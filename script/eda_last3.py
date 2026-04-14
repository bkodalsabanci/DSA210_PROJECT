from __future__ import annotations

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "processed"

FORM_ORDER = ["bad", "medium", "good"]
POINT_DIFF_ORDER = ["<= -4", "-3 to -1", "0 to 1", "2 to 4", "> 4"]


def load_last3_df() -> pd.DataFrame:
    frames = []

    for file in sorted(PROCESSED_DIR.glob("*_last3_enriched.csv")):
        df = pd.read_csv(file)
        frames.append(df)

    last3_df = pd.concat(frames, ignore_index=True)
    last3_df["Date"] = pd.to_datetime(last3_df["Date"], errors="coerce")
    last3_df["point_diff"] = (
        last3_df["home_last3_points"] - last3_df["away_last3_points"]
    )
    last3_df["home_win"] = (last3_df["result_match"] == "Home Win").astype(int)
    last3_df["away_win"] = (last3_df["result_match"] == "Away Win").astype(int)
    last3_df["draw"] = (last3_df["result_match"] == "Draw").astype(int)

    return last3_df


def add_point_diff_band(last3_df: pd.DataFrame) -> pd.DataFrame:
    df = last3_df.copy()

    df["point_diff_interval"] = pd.cut(
        df["point_diff"],
        bins=[-10, -4, -1, 1, 4, 10],
        labels=POINT_DIFF_ORDER,
        include_lowest=True,
    )

    return df


def build_point_diff_effect(last3_df: pd.DataFrame) -> pd.DataFrame:
    df = add_point_diff_band(last3_df)

    return (
        df.groupby("point_diff_interval", observed=False)
        .agg(
            matches=("result_match", "size"),
            home_win_rate=("home_win", "mean"),
            away_win_rate=("away_win", "mean"),
            draw_rate=("draw", "mean"),
        )
        .round(3)
        .reindex(POINT_DIFF_ORDER)
        .reset_index()
    )


def build_home_form_summary(last3_df: pd.DataFrame) -> pd.DataFrame:
    return (
        last3_df.groupby("home_form", observed=False)
        .agg(
            matches=("result_match", "size"),
            home_win_rate=("home_win", "mean"),
            avg_point_diff=("point_diff", "mean"),
        )
        .round(3)
        .reindex(FORM_ORDER)
        .reset_index()
    )


def build_matrix(
    frame: pd.DataFrame,
    row_col: str,
    col_col: str,
    value_col: str,
    row_order: list[str],
    col_order: list[str],
) -> pd.DataFrame:
    grouped = (
        frame.groupby([row_col, col_col], observed=False)[value_col]
        .mean()
        .round(3)
        .reset_index()
    )

    rows = []

    for row_value in row_order:
        row = {col_col: row_value}

        for col_value in col_order:
            match = grouped[
                (grouped[row_col] == row_value) & (grouped[col_col] == col_value)
            ]
            row[col_value] = match[value_col].iloc[0] if not match.empty else pd.NA

        rows.append(row)

    matrix = pd.DataFrame(rows).set_index(col_col)
    matrix.index.name = row_col

    return matrix


def build_point_diff_matrix(last3_df: pd.DataFrame) -> pd.DataFrame:
    return build_matrix(
        frame=last3_df,
        row_col="home_form",
        col_col="away_form",
        value_col="point_diff",
        row_order=FORM_ORDER,
        col_order=FORM_ORDER,
    )


def build_home_win_matrix(last3_df: pd.DataFrame) -> pd.DataFrame:
    return build_matrix(
        frame=last3_df,
        row_col="home_form",
        col_col="away_form",
        value_col="home_win",
        row_order=FORM_ORDER,
        col_order=FORM_ORDER,
    )
