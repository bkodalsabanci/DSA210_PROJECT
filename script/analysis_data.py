from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "processed"

RESULT_LABEL_MAP = {
    "H": "Home Win",
    "D": "Draw",
    "A": "Away Win",
}

STADIUM_FILE_LEAGUE_MAP = {
    "england_stadiums_FINAL.csv": "Premier League",
    "france_stadiums_FINAL.csv": "Ligue 1",
    "germany_stadiums_FINAL.csv": "Bundesliga",
    "italy_stadiums_FINAL.csv": "Serie A",
    "spain_stadiums_FINAL.csv": "La Liga",
}


def clean_capacity(value: object) -> float | np.nan:
    if pd.isna(value):
        return np.nan

    digits = "".join(ch for ch in str(value) if ch.isdigit())

    if not digits:
        return np.nan

    capacity = float(digits)

    while capacity > 200000:
        capacity = capacity / 10

    return round(capacity)


def read_stadium_file(file: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    with file.open(encoding="utf-8", errors="replace") as handle:
        next(handle, None)

        for line in handle:
            parts = [part.strip() for part in line.strip().split(",")]

            if len(parts) < 3:
                continue

            team = parts[0]
            stadium = ",".join(parts[1:-1]).strip()
            capacity = clean_capacity(parts[-1])

            if len(parts) > 3:
                stadium = parts[1]
                capacity = clean_capacity("".join(parts[2:]))

            rows.append(
                {
                    "team": team,
                    "stadium": stadium,
                    "capacity": capacity,
                }
            )

    return pd.DataFrame(rows)


def add_match_result_columns(
    df: pd.DataFrame,
    result_col: str,
    home_win_value: str,
    draw_value: str,
    away_win_value: str,
) -> pd.DataFrame:
    enriched = df.copy()
    enriched["home_win"] = (enriched[result_col] == home_win_value).astype(int)
    enriched["away_win"] = (enriched[result_col] == away_win_value).astype(int)
    enriched["draw"] = (enriched[result_col] == draw_value).astype(int)
    return enriched


def add_compatibility_columns(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()

    compatibility_map = {
        "home_form_group": "home_form",
        "away_form_group": "away_form",
        "result_label": "result_match",
        "date": "Date",
        "home_team": "HomeTeam",
        "away_team": "AwayTeam",
        "home_goals": "FTHG",
        "away_goals": "FTAG",
        "result": "FTR",
    }

    for source_col, alias_col in compatibility_map.items():
        if source_col in enriched.columns and alias_col not in enriched.columns:
            enriched[alias_col] = enriched[source_col]

    return enriched


def load_last3_analysis_df() -> pd.DataFrame:
    frames = []

    for file in sorted(PROCESSED_DIR.glob("*_last3_enriched.csv")):
        league_key = file.stem.split("_20")[0]

        df = pd.read_csv(file)
        df["league_file"] = league_key
        df["source_file"] = file.name

        frames.append(df)

    last3_df = pd.concat(frames, ignore_index=True)
    last3_df["Date"] = pd.to_datetime(last3_df["Date"], errors="coerce")
    last3_df["point_diff"] = (
        last3_df["home_last3_points"] - last3_df["away_last3_points"]
    )
    last3_df = add_match_result_columns(
        last3_df,
        result_col="result_match",
        home_win_value="Home Win",
        draw_value="Draw",
        away_win_value="Away Win",
    )
    last3_df = add_compatibility_columns(last3_df)

    return last3_df


def load_covid_comparison_df() -> pd.DataFrame:
    frames = []

    for file in sorted(PROCESSED_DIR.glob("*_covid_comparison.csv")):
        league_key = file.stem.replace("_covid_comparison", "")

        df = pd.read_csv(file)
        df["league_file"] = league_key
        df["source_file"] = file.name

        frames.append(df)

    covid_df = pd.concat(frames, ignore_index=True)
    covid_df["Date"] = pd.to_datetime(covid_df["Date"], errors="coerce")
    covid_df["total_goals"] = covid_df["FTHG"] + covid_df["FTAG"]
    covid_df["goal_diff"] = covid_df["FTHG"] - covid_df["FTAG"]
    covid_df["shot_diff"] = covid_df["HS"] - covid_df["AS"]
    covid_df["result_label"] = covid_df["FTR"].map(RESULT_LABEL_MAP)
    covid_df = add_match_result_columns(
        covid_df,
        result_col="FTR",
        home_win_value="H",
        draw_value="D",
        away_win_value="A",
    )
    covid_df["home_points"] = np.select(
        [covid_df["FTR"] == "H", covid_df["FTR"] == "D"],
        [3, 1],
        default=0,
    )
    covid_df["away_points"] = np.select(
        [covid_df["FTR"] == "A", covid_df["FTR"] == "D"],
        [3, 1],
        default=0,
    )
    covid_df["home_advantage_points"] = (
        covid_df["home_points"] - covid_df["away_points"]
    )
    covid_df = add_compatibility_columns(covid_df)

    return covid_df


def load_full_eda_df() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / "football_eda_ready_complete_fixed_v2.csv")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["total_goals"] = df["home_goals"] + df["away_goals"]
    df["goal_diff"] = df["home_goals"] - df["away_goals"]
    df["point_diff"] = df["home_last3_points"] - df["away_last3_points"]

    df["home_points"] = np.select(
        [df["result"] == "H", df["result"] == "D"],
        [3, 1],
        default=0,
    )
    df["away_points"] = np.select(
        [df["result"] == "A", df["result"] == "D"],
        [3, 1],
        default=0,
    )
    df["home_advantage_points"] = df["home_points"] - df["away_points"]
    df = add_match_result_columns(
        df,
        result_col="result",
        home_win_value="H",
        draw_value="D",
        away_win_value="A",
    )
    df = add_compatibility_columns(df)

    df["capacity_band"] = pd.qcut(
        df["capacity"],
        q=4,
        labels=["Low", "Medium", "High", "Very High"],
        duplicates="drop",
    )

    return df


def load_capacity_analysis_df(
    season: str = "2024_2025",
    season_type: str = "normal",
) -> pd.DataFrame:
    capacity_df = load_full_eda_df()
    capacity_df = capacity_df[
        (capacity_df["season"] == season) & (capacity_df["season_type"] == season_type)
    ].copy()
    capacity_df = capacity_df.dropna(subset=["capacity", "result"])
    capacity_df["capacity_interval"] = pd.cut(
        capacity_df["capacity"],
        bins=[0, 20000, 40000, np.inf],
        labels=["0-20K", "20K-40K", "40K+"],
        include_lowest=True,
        right=False,
    )

    return capacity_df


def load_stadium_lookup_df() -> pd.DataFrame:
    frames = []

    for file in sorted(PROCESSED_DIR.glob("*_stadiums_FINAL.csv")):
        df = read_stadium_file(file)
        df["league"] = STADIUM_FILE_LEAGUE_MAP.get(file.name, file.stem)
        frames.append(df)

    return pd.concat(frames, ignore_index=True)
