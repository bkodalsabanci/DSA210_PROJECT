from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "processed"
OUTPUT_DIR = BASE_DIR / "prepared_outputs"


def clean_capacity(value: object) -> float | np.nan:
    if pd.isna(value):
        return np.nan

    digits = ""

    for ch in str(value):
        if ch.isdigit():
            digits = digits + ch

    if digits == "":
        return np.nan

    capacity = float(digits)

    while capacity > 200000:
        capacity = capacity / 10

    return round(capacity)


def read_stadium_file(file: Path) -> pd.DataFrame:
    rows = []

    with file.open(encoding="utf-8", errors="replace") as handle:
        next(handle, None)

        for line in handle:
            line = line.strip()
            parts = line.split(",")

            cleaned_parts = []

            for part in parts:
                cleaned_parts.append(part.strip())

            if len(cleaned_parts) >= 3:
                team = cleaned_parts[0]
                stadium = ",".join(cleaned_parts[1:-1]).strip()
                capacity = clean_capacity(cleaned_parts[-1])

                if len(cleaned_parts) > 3:
                    stadium = cleaned_parts[1]
                    capacity_text = ""

                    for i in range(2, len(cleaned_parts)):
                        capacity_text = capacity_text + cleaned_parts[i]

                    capacity = clean_capacity(capacity_text)

                row = {
                    "team": team,
                    "stadium": stadium,
                    "capacity": capacity
                }
                rows.append(row)

    return pd.DataFrame(rows)


def load_covid_df() -> pd.DataFrame:
    frames = []

    for file in sorted(PROCESSED_DIR.glob("*_covid_comparison.csv")):
        league_key = file.stem.replace("_covid_comparison", "")

        df = pd.read_csv(file)
        df["league_file"] = league_key

        frames.append(df)

    covid_df = pd.concat(frames, ignore_index=True)

    covid_df["Date"] = pd.to_datetime(covid_df["Date"], errors="coerce")

    covid_df["total_goals"] = covid_df["FTHG"] + covid_df["FTAG"]
    covid_df["goal_diff"] = covid_df["FTHG"] - covid_df["FTAG"]
    covid_df["shot_diff"] = covid_df["HS"] - covid_df["AS"]

    covid_df["home_win"] = (covid_df["FTR"] == "H").astype(int)
    covid_df["away_win"] = (covid_df["FTR"] == "A").astype(int)
    covid_df["draw"] = (covid_df["FTR"] == "D").astype(int)

    covid_df["home_points"] = np.select(
        [covid_df["FTR"] == "H", covid_df["FTR"] == "D"],
        [3, 1],
        default=0
    )

    covid_df["away_points"] = np.select(
        [covid_df["FTR"] == "A", covid_df["FTR"] == "D"],
        [3, 1],
        default=0
    )

    covid_df["home_advantage_points"] = covid_df["home_points"] - covid_df["away_points"]

    return covid_df


def load_stadium_df() -> pd.DataFrame:
    frames = []

    country_map = {
        "england_stadiums_FINAL.csv": "Premier League",
        "france_stadiums_FINAL.csv": "Ligue 1",
        "germany_stadiums_FINAL.csv": "Bundesliga",
        "italy_stadiums_FINAL.csv": "Serie A",
        "spain_stadiums_FINAL.csv": "La Liga",
    }

    for file in sorted(PROCESSED_DIR.glob("*_stadiums_FINAL.csv")):
        df = read_stadium_file(file)
        df["league"] = country_map.get(file.name, file.stem)
        frames.append(df)

    stadium_df = pd.concat(frames, ignore_index=True)

    return stadium_df


def load_last3_df() -> pd.DataFrame:
    frames = []

    for file in sorted(PROCESSED_DIR.glob("*_last3_enriched.csv")):
        name = file.stem
        league_key = name.split("_20")[0]

        df = pd.read_csv(file)
        df["league_file"] = league_key
        df["point_diff"] = df["home_last3_points"] - df["away_last3_points"]

        frames.append(df)

    last3_df = pd.concat(frames, ignore_index=True)
    last3_df["Date"] = pd.to_datetime(last3_df["Date"], errors="coerce")

    return last3_df


def load_total_df() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / "football_eda_ready_complete_fixed_v2.csv")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["total_goals"] = df["home_goals"] + df["away_goals"]
    df["point_diff"] = df["home_last3_points"] - df["away_last3_points"]

    df["capacity_band"] = pd.qcut(
        df["capacity"],
        q=4,
        labels=["Low", "Medium", "High", "Very High"],
        duplicates="drop"
    )

    return df


def save_outputs(
    covid_df: pd.DataFrame,
    stadium_df: pd.DataFrame,
    last3_df: pd.DataFrame,
    total_df: pd.DataFrame
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    covid_df.to_csv(OUTPUT_DIR / "covid_prepared.csv", index=False)
    stadium_df.to_csv(OUTPUT_DIR / "stadium_prepared.csv", index=False)
    last3_df.to_csv(OUTPUT_DIR / "last3_prepared.csv", index=False)
    total_df.to_csv(OUTPUT_DIR / "total_prepared.csv", index=False)


def main() -> None:
    covid_df = load_covid_df()
    stadium_df = load_stadium_df()
    last3_df = load_last3_df()
    total_df = load_total_df()

    save_outputs(covid_df, stadium_df, last3_df, total_df)

    print("Prepared files saved.")
    print("covid_prepared.csv")
    print("stadium_prepared.csv")
    print("last3_prepared.csv")
    print("total_prepared.csv")


if __name__ == "__main__":
    main()
