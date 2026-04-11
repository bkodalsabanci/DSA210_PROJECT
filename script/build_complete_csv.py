from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "processed"


DATASETS = [
    {
        "league": "Premier League",
        "season": "2020_2021",
        "season_type": "covid",
        "covid": 1,
        "matches_file": "premier_league_2020_2021_last3_enriched.csv",
        "stadium_file": "england_stadiums_FINAL.csv",
    },
    {
        "league": "Premier League",
        "season": "2024_2025",
        "season_type": "normal",
        "covid": 0,
        "matches_file": "premier_league_2024_2025_last3_enriched.csv",
        "stadium_file": "england_stadiums_FINAL.csv",
    },
    {
        "league": "Bundesliga",
        "season": "2020_2021",
        "season_type": "covid",
        "covid": 1,
        "matches_file": "bundesliga_2020_2021_last3_enriched.csv",
        "stadium_file": "germany_stadiums_FINAL.csv",
    },
    {
        "league": "Bundesliga",
        "season": "2024_2025",
        "season_type": "normal",
        "covid": 0,
        "matches_file": "bundesliga_2024_2025_last3_enriched.csv",
        "stadium_file": "germany_stadiums_FINAL.csv",
    },
    {
        "league": "La Liga",
        "season": "2020_2021",
        "season_type": "covid",
        "covid": 1,
        "matches_file": "laliga_2020_2021_last3_enriched.csv",
        "stadium_file": "spain_stadiums_FINAL.csv",
    },
    {
        "league": "La Liga",
        "season": "2024_2025",
        "season_type": "normal",
        "covid": 0,
        "matches_file": "laliga_2024_2025_last3_enriched.csv",
        "stadium_file": "spain_stadiums_FINAL.csv",
    },
    {
        "league": "Ligue 1",
        "season": "2020_2021",
        "season_type": "covid",
        "covid": 1,
        "matches_file": "ligue1_2020_2021_last3_enriched.csv",
        "stadium_file": "france_stadiums_FINAL.csv",
    },
    {
        "league": "Ligue 1",
        "season": "2024_2025",
        "season_type": "normal",
        "covid": 0,
        "matches_file": "ligue1_2024_2025_last3_enriched.csv",
        "stadium_file": "france_stadiums_FINAL.csv",
    },
    {
        "league": "Serie A",
        "season": "2020_2021",
        "season_type": "covid",
        "covid": 1,
        "matches_file": "serieA_2020_2021_last3_enriched.csv",
        "stadium_file": "italy_stadiums_FINAL.csv",
    },
    {
        "league": "Serie A",
        "season": "2024_2025",
        "season_type": "normal",
        "covid": 0,
        "matches_file": "serieA_2024_2025_last3_enriched.csv",
        "stadium_file": "italy_stadiums_FINAL.csv",
    },
]


RENAME_MAP = {
    "Date": "date",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "FTR": "result",
    "HTHG": "ht_home_goals",
    "HTAG": "ht_away_goals",
    "HTR": "ht_result",
    "Referee": "referee",
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_shots_on_target",
    "AST": "away_shots_on_target",
    "HF": "home_fouls",
    "AF": "away_fouls",
    "HC": "home_corners",
    "AC": "away_corners",
    "HY": "home_yellow_cards",
    "AY": "away_yellow_cards",
    "HR": "home_red_cards",
    "AR": "away_red_cards",
}


FINAL_COLUMNS = [
    "date",
    "league",
    "season",
    "season_type",
    "covid",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "result",
    "home_last3_points",
    "away_last3_points",
    "result_label",
    "home_form_group",
    "away_form_group",
    "stadium",
    "capacity",
    "ht_home_goals",
    "ht_away_goals",
    "ht_result",
]


def load_and_merge(config: dict) -> pd.DataFrame:
    matches = pd.read_csv(PROCESSED_DIR / config["matches_file"])
    stadiums = pd.read_csv(PROCESSED_DIR / config["stadium_file"])

    stadiums = stadiums.drop_duplicates(subset=["team"], keep="first")
    merged = matches.merge(stadiums, left_on="HomeTeam", right_on="team", how="left")

    merged["league"] = config["league"]
    merged["season"] = config["season"]
    merged["season_type"] = config["season_type"]
    merged["covid"] = config["covid"]

    return merged


def main() -> None:
    frames = [load_and_merge(config) for config in DATASETS]
    complete = pd.concat(frames, ignore_index=True)
    complete = complete.rename(columns=RENAME_MAP)

    complete["stadium"] = complete["stadium"].fillna("NA")
    complete["stadium"] = complete["stadium"].replace(r"^\s*$", "NA", regex=True)
    complete["capacity"] = complete["capacity"].fillna("NA")
    complete["capacity"] = complete["capacity"].replace(r"^\s*$", "NA", regex=True)

    complete = complete[FINAL_COLUMNS].sort_values(["date", "league", "home_team"]).reset_index(drop=True)
    output_path = PROCESSED_DIR / "football_eda_ready_complete_fixed_v2.csv"
    complete.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")
    print(f"Rows: {len(complete)}")
    print(f"Columns: {len(complete.columns)}")


if __name__ == "__main__":
    main()
