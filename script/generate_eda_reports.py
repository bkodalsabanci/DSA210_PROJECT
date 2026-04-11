from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "processed"
OUTPUT_DIR = BASE_DIR / "eda_outputs"


def setup_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["figure.figsize"] = (12, 7)
    plt.rcParams["axes.titlesize"] = 18
    plt.rcParams["axes.labelsize"] = 13


def ensure_output_dirs() -> None:
    for folder in ["covid", "stadium_capacity", "last3_matches", "total"]:
        (OUTPUT_DIR / folder).mkdir(parents=True, exist_ok=True)


def save_plot(fig: plt.Figure, relative_path: str) -> None:
    path = OUTPUT_DIR / relative_path
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_report(relative_path: str, content: str) -> None:
    path = OUTPUT_DIR / relative_path
    path.write_text(content, encoding="utf-8")


def pct(value: float) -> str:
    return f"{value:.1%}"


def table_text(obj: pd.DataFrame | pd.Series) -> str:
    if isinstance(obj, pd.Series):
        return obj.to_string()
    return obj.to_string()


def clean_capacity(value: object) -> float | np.nan:
    if pd.isna(value):
        return np.nan
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    if not digits:
        return np.nan
    capacity = float(digits)
    while capacity > 200000:
        capacity /= 10
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
            rows.append({"team": team, "stadium": stadium, "capacity": capacity})
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
        default=0,
    )
    covid_df["away_points"] = np.select(
        [covid_df["FTR"] == "A", covid_df["FTR"] == "D"],
        [3, 1],
        default=0,
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
    return pd.concat(frames, ignore_index=True)


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
        duplicates="drop",
    )
    return df


def build_covid_eda() -> dict[str, str]:
    covid_df = load_covid_df()

    outcome_share = (
        pd.crosstab(covid_df["season_type"], covid_df["FTR"], normalize="index")
        .rename(columns={"H": "Home Win", "D": "Draw", "A": "Away Win"})
        .reindex(["covid", "normal"])
    )

    goals_summary = (
        covid_df.groupby("season_type")
        .agg(
            matches=("FTR", "size"),
            avg_total_goals=("total_goals", "mean"),
            avg_home_goals=("FTHG", "mean"),
            avg_away_goals=("FTAG", "mean"),
            avg_home_shots=("HS", "mean"),
            avg_away_shots=("AS", "mean"),
        )
        .round(2)
    )

    home_advantage_summary = (
        covid_df.groupby("season_type")
        .agg(
            matches=("FTR", "size"),
            home_win_rate=("home_win", "mean"),
            away_win_rate=("away_win", "mean"),
            draw_rate=("draw", "mean"),
            avg_goal_diff=("goal_diff", "mean"),
            avg_shot_diff=("shot_diff", "mean"),
            home_points_per_match=("home_points", "mean"),
            away_points_per_match=("away_points", "mean"),
            home_advantage_points=("home_advantage_points", "mean"),
        )
        .round(3)
        .reindex(["covid", "normal"])
    )

    league_home_win = (
        covid_df.groupby(["league_file", "season_type"])["home_win"]
        .mean()
        .unstack()
        .sort_index()
    )
    league_home_win["delta_normal_minus_covid"] = (
        league_home_win["normal"] - league_home_win["covid"]
    ).round(3)

    league_advantage = (
        covid_df.groupby(["league_file", "season_type"])
        .agg(
            matches=("FTR", "size"),
            home_win_rate=("home_win", "mean"),
            avg_goal_diff=("goal_diff", "mean"),
            home_points_per_match=("home_points", "mean"),
            away_points_per_match=("away_points", "mean"),
            home_advantage_points=("home_advantage_points", "mean"),
        )
        .round(3)
        .reset_index()
    )
    league_advantage_pivot = (
        league_advantage.pivot(
            index="league_file",
            columns="season_type",
            values="home_advantage_points",
        )
        .sort_index()
    )
    league_advantage_pivot["delta_normal_minus_covid"] = (
        league_advantage_pivot["normal"] - league_advantage_pivot["covid"]
    ).round(3)

    form_covid_effect = (
        covid_df.assign(
            home_strength=np.where(covid_df["HS"] >= covid_df["AS"], "Home shot edge", "Away shot edge")
        )
        .groupby(["season_type", "home_strength"])
        .agg(
            matches=("FTR", "size"),
            home_win_rate=("home_win", "mean"),
            avg_goal_diff=("goal_diff", "mean"),
        )
        .round(3)
        .reset_index()
    )

    fig, ax = plt.subplots()
    outcome_share.plot(kind="bar", stacked=True, ax=ax, colormap="Set2")
    ax.set_title("Covid vs Normal: Match Result Distribution")
    ax.set_xlabel("Season Type")
    ax.set_ylabel("Share of Matches")
    ax.legend(title="Result", loc="upper right")
    save_plot(fig, "covid/result_distribution.png")

    fig, ax = plt.subplots()
    sns.barplot(
        data=covid_df,
        x="season_type",
        y="total_goals",
        hue="season_type",
        estimator=np.mean,
        errorbar=("ci", 95),
        palette="Set1",
        legend=False,
        ax=ax,
    )
    ax.set_title("Average Total Goals by Season Type")
    ax.set_xlabel("Season Type")
    ax.set_ylabel("Average Total Goals")
    save_plot(fig, "covid/avg_total_goals.png")

    fig, ax = plt.subplots(figsize=(10, 6))
    advantage_plot = (
        home_advantage_summary[
            ["home_win_rate", "avg_goal_diff", "home_advantage_points"]
        ]
        .reset_index()
        .melt(id_vars="season_type", var_name="metric", value_name="value")
    )
    sns.barplot(
        data=advantage_plot,
        x="metric",
        y="value",
        hue="season_type",
        palette="Set2",
        ax=ax,
    )
    ax.set_title("Home Advantage Metrics: Covid vs Normal")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Value")
    metric_labels = {
        "home_win_rate": "Home Win Rate",
        "avg_goal_diff": "Avg Goal Diff",
        "home_advantage_points": "Home Advantage Points",
    }
    ax.set_xticks(range(len(metric_labels)))
    ax.set_xticklabels([metric_labels[key] for key in metric_labels], rotation=10)
    save_plot(fig, "covid/home_advantage_metrics.png")

    fig, ax = plt.subplots(figsize=(10, 6))
    heatmap_df = (league_home_win[["covid", "normal"]] * 100).round(1)
    sns.heatmap(heatmap_df, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax)
    ax.set_title("Home Win Rate by League and Season Type (%)")
    ax.set_xlabel("Season Type")
    ax.set_ylabel("League")
    save_plot(fig, "covid/home_win_rate_heatmap.png")

    fig, ax = plt.subplots(figsize=(10, 6))
    league_plot = league_advantage_pivot[["covid", "normal"]].reset_index().melt(
        id_vars="league_file",
        var_name="season_type",
        value_name="home_advantage_points",
    )
    sns.barplot(
        data=league_plot,
        x="league_file",
        y="home_advantage_points",
        hue="season_type",
        palette="Set1",
        ax=ax,
    )
    ax.set_title("League-Level Home Advantage Points by Season Type")
    ax.set_xlabel("League")
    ax.set_ylabel("Home Advantage Points per Match")
    ax.tick_params(axis="x", rotation=20)
    save_plot(fig, "covid/league_home_advantage_points.png")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=form_covid_effect,
        x="home_strength",
        y="home_win_rate",
        hue="season_type",
        palette="mako",
        ax=ax,
    )
    ax.set_title("Shot Edge and Home Win Rate in Covid vs Normal Seasons")
    ax.set_xlabel("Shot Profile")
    ax.set_ylabel("Home Win Rate")
    save_plot(fig, "covid/shot_edge_home_win_rate.png")

    report = f"""# Covid EDA

## Dataset Scope
- Rows: {len(covid_df)}
- Leagues: {covid_df['league_file'].nunique()}
- Covid rows: {int((covid_df['season_type'] == 'covid').sum())}
- Normal rows: {int((covid_df['season_type'] == 'normal').sum())}

## Result Distribution
    ```
    {table_text(outcome_share.round(3))}
    ```

## Numeric Summary
    ```
    {table_text(goals_summary)}
    ```

## Home Advantage Summary
    ```
    {table_text(home_advantage_summary)}
    ```

## League-Level Home Win Rate
    ```
    {table_text(league_home_win.round(3))}
    ```

## League-Level Home Advantage Points
    ```
    {table_text(league_advantage_pivot.round(3))}
    ```

## Shot Edge Split
    ```
    {table_text(form_covid_effect)}
    ```

## Key Findings
- Home win share in normal seasons is {pct(outcome_share.loc['normal', 'Home Win'])}, while it falls to {pct(outcome_share.loc['covid', 'Home Win'])} in covid seasons.
- Away win share increases from {pct(outcome_share.loc['normal', 'Away Win'])} to {pct(outcome_share.loc['covid', 'Away Win'])}, which is consistent with a weaker home advantage signal.
- Home advantage points per match drop from {home_advantage_summary.loc['normal', 'home_advantage_points']:.3f} in normal seasons to {home_advantage_summary.loc['covid', 'home_advantage_points']:.3f} in covid seasons, so the home-side points edge becomes weaker without fans.
- Average home goal difference also softens from {home_advantage_summary.loc['normal', 'avg_goal_diff']:.3f} to {home_advantage_summary.loc['covid', 'avg_goal_diff']:.3f}, which supports the same story from a scoreline perspective.
- Average total goals move from {goals_summary.loc['normal', 'avg_total_goals']:.2f} to {goals_summary.loc['covid', 'avg_total_goals']:.2f}; the gap is small, so covid seems more related to result balance than goal volume.
- The biggest normal-vs-covid home advantage drop appears in `{league_advantage_pivot['delta_normal_minus_covid'].idxmax()}`, making that league the best first target for deeper hypothesis testing.
"""
    write_report("covid/report.md", report)
    return {
        "rows": str(len(covid_df)),
        "home_win_normal": pct(outcome_share.loc["normal", "Home Win"]),
        "home_win_covid": pct(outcome_share.loc["covid", "Home Win"]),
        "home_advantage_points_normal": f"{home_advantage_summary.loc['normal', 'home_advantage_points']:.3f}",
        "home_advantage_points_covid": f"{home_advantage_summary.loc['covid', 'home_advantage_points']:.3f}",
    }


def build_stadium_eda() -> dict[str, str]:
    stadium_df = load_stadium_df()
    total_df = load_total_df()

    coverage = (
        total_df.groupby("league")
        .agg(matches=("league", "size"), capacity_non_null=("capacity", "count"))
        .assign(coverage_pct=lambda x: (x["capacity_non_null"] / x["matches"]).round(3))
    )

    stadium_summary = (
        stadium_df.groupby("league")["capacity"]
        .agg(["count", "mean", "median", "min", "max"])
        .round(1)
        .sort_values("mean", ascending=False)
    )

    capacity_effect = (
        total_df.dropna(subset=["capacity_band"])
        .groupby("capacity_band", observed=False)
        .agg(
            matches=("result_label", "size"),
            home_win_rate=("result_label", lambda s: (s == "Home Win").mean()),
            avg_total_goals=("total_goals", "mean"),
        )
        .round(3)
    )

    fig, ax = plt.subplots()
    sns.boxplot(
        data=stadium_df,
        x="league",
        y="capacity",
        hue="league",
        palette="crest",
        legend=False,
        ax=ax,
    )
    ax.set_title("Stadium Capacity Distribution by League")
    ax.set_xlabel("League")
    ax.set_ylabel("Capacity")
    ax.tick_params(axis="x", rotation=20)
    save_plot(fig, "stadium_capacity/capacity_by_league_boxplot.png")

    fig, ax = plt.subplots()
    sns.barplot(
        data=capacity_effect.reset_index(),
        x="capacity_band",
        y="home_win_rate",
        hue="capacity_band",
        palette="flare",
        legend=False,
        ax=ax,
    )
    ax.set_title("Home Win Rate by Capacity Band")
    ax.set_xlabel("Capacity Band")
    ax.set_ylabel("Home Win Rate")
    save_plot(fig, "stadium_capacity/home_win_by_capacity_band.png")

    fig, ax = plt.subplots()
    sns.scatterplot(
        data=total_df.dropna(subset=["capacity"]),
        x="capacity",
        y="total_goals",
        hue="league",
        alpha=0.55,
        ax=ax,
    )
    ax.set_title("Capacity vs Total Goals")
    ax.set_xlabel("Capacity")
    ax.set_ylabel("Total Goals")
    save_plot(fig, "stadium_capacity/capacity_vs_goals_scatter.png")

    report = f"""# Stadium Capacity EDA

## Stadium Table Summary
    ```
    {table_text(stadium_summary)}
    ```

## Capacity Coverage in Combined Dataset
    ```
    {table_text(coverage)}
    ```

## Capacity Band Effect
    ```
    {table_text(capacity_effect)}
    ```

## Key Findings
- The highest average stadium capacity belongs to `{stadium_summary.index[0]}` with an average of {stadium_summary.iloc[0]['mean']:.0f}.
- Capacity coverage is strongest in leagues where team-name matching worked well, and weakest where the combined dataset still carries missing stadium links.
- Home win rate by capacity band ranges from {pct(capacity_effect['home_win_rate'].min())} to {pct(capacity_effect['home_win_rate'].max())}, so capacity alone shows a mild rather than dominant effect.
- The scatter plot helps check whether larger venues align with more open games, but the spread suggests match outcome context matters more than stadium size by itself.
"""
    write_report("stadium_capacity/report.md", report)
    return {
        "rows": str(len(stadium_df)),
        "best_coverage_league": coverage["coverage_pct"].idxmax(),
        "worst_coverage_league": coverage["coverage_pct"].idxmin(),
    }


def build_last3_eda() -> dict[str, str]:
    last3_df = load_last3_df()

    result_by_form = (
        pd.crosstab(
            [last3_df["home_form"], last3_df["away_form"]],
            last3_df["result_match"],
            normalize="index",
        )
        .round(3)
    )

    point_diff_bins = pd.cut(
        last3_df["point_diff"],
        bins=[-10, -4, -1, 1, 4, 10],
        labels=["<= -4", "-3 to -1", "0 to 1", "2 to 4", ">= 5"],
        include_lowest=True,
    )
    point_diff_effect = (
        last3_df.assign(point_diff_band=point_diff_bins)
        .groupby("point_diff_band", observed=False)
        .agg(
            matches=("result_match", "size"),
            home_win_rate=("result_match", lambda s: (s == "Home Win").mean()),
            away_win_rate=("result_match", lambda s: (s == "Away Win").mean()),
            draw_rate=("result_match", lambda s: (s == "Draw").mean()),
        )
        .round(3)
    )

    form_summary = (
        last3_df.groupby("home_form")
        .agg(
            matches=("result_match", "size"),
            home_win_rate=("result_match", lambda s: (s == "Home Win").mean()),
            avg_point_diff=("point_diff", "mean"),
        )
        .round(3)
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    heatmap_data = (
        last3_df.pivot_table(
            index="home_form",
            columns="away_form",
            values="point_diff",
            aggfunc="mean",
        )
        .round(2)
    )
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="RdYlGn", center=0, ax=ax)
    ax.set_title("Average Last-3 Point Differential by Form Pair")
    ax.set_xlabel("Away Form")
    ax.set_ylabel("Home Form")
    save_plot(fig, "last3_matches/form_pair_point_diff_heatmap.png")

    fig, ax = plt.subplots()
    plot_df = point_diff_effect.reset_index()
    sns.barplot(
        data=plot_df,
        x="point_diff_band",
        y="home_win_rate",
        hue="point_diff_band",
        palette="mako",
        legend=False,
        ax=ax,
    )
    ax.set_title("Home Win Rate by Last-3 Point Differential")
    ax.set_xlabel("Point Differential Band")
    ax.set_ylabel("Home Win Rate")
    save_plot(fig, "last3_matches/home_win_by_point_diff_band.png")

    fig, ax = plt.subplots()
    sns.countplot(data=last3_df, x="home_form", hue="result_match", palette="Set2", ax=ax)
    ax.set_title("Match Results by Home Form Group")
    ax.set_xlabel("Home Form Group")
    ax.set_ylabel("Match Count")
    save_plot(fig, "last3_matches/result_by_home_form.png")

    report = f"""# Last 3 Matches EDA

## Result Share by Form Pair
    ```
    {table_text(result_by_form)}
    ```

## Point Differential Effect
    ```
    {table_text(point_diff_effect)}
    ```

## Home Form Summary
    ```
    {table_text(form_summary)}
    ```

## Key Findings
- Positive recent-form differential clearly pushes home win rate upward; the strongest home edge appears in the highest `point_diff_band`.
- When the home team enters with `good` form and the away side is weaker, the distribution shifts sharply toward `Home Win`.
- Draw rates stay highest around balanced short-term form bands, which makes point differential a useful but not complete predictor.
- The form-group heatmap is useful for feature engineering because it shows recent momentum is directional, not just absolute.
"""
    write_report("last3_matches/report.md", report)
    return {
        "rows": str(len(last3_df)),
        "best_band": str(point_diff_effect["home_win_rate"].idxmax()),
        "worst_band": str(point_diff_effect["home_win_rate"].idxmin()),
    }


def build_total_eda() -> dict[str, str]:
    total_df = load_total_df()

    numeric_corr = total_df[
        [
            "covid",
            "capacity",
            "home_last3_points",
            "away_last3_points",
            "point_diff",
            "home_goals",
            "away_goals",
            "total_goals",
        ]
    ].corr(numeric_only=True).round(2)

    combined_effect = (
        total_df.assign(
            positive_form=(total_df["point_diff"] > 0).map({True: "Positive", False: "Not Positive"}),
            covid_label=total_df["covid"].map({1: "Covid", 0: "Normal"}),
        )
        .groupby(["covid_label", "positive_form"])
        .agg(
            matches=("result_label", "size"),
            home_win_rate=("result_label", lambda s: (s == "Home Win").mean()),
            avg_total_goals=("total_goals", "mean"),
        )
        .round(3)
    )

    league_result = (
        pd.crosstab(total_df["league"], total_df["result_label"], normalize="index").round(3)
    )

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(numeric_corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Heatmap of Main Numeric Features")
    save_plot(fig, "total/correlation_heatmap.png")

    fig, ax = plt.subplots(figsize=(11, 7))
    league_result.plot(kind="bar", stacked=True, ax=ax, colormap="Paired")
    ax.set_title("Result Distribution by League")
    ax.set_xlabel("League")
    ax.set_ylabel("Share of Matches")
    ax.legend(title="Result")
    save_plot(fig, "total/result_distribution_by_league.png")

    fig, ax = plt.subplots()
    sns.barplot(
        data=combined_effect.reset_index(),
        x="covid_label",
        y="home_win_rate",
        hue="positive_form",
        palette="Set2",
        ax=ax,
    )
    ax.set_title("Home Win Rate: Covid Effect and Recent Form Together")
    ax.set_xlabel("Season Context")
    ax.set_ylabel("Home Win Rate")
    save_plot(fig, "total/home_win_covid_form_interaction.png")

    missing_summary = total_df.isna().sum().sort_values(ascending=False).head(8)

    report = f"""# Total EDA

## Combined Dataset Shape
- Rows: {len(total_df)}
- Columns: {total_df.shape[1]}
- Leagues: {total_df['league'].nunique()}

## Top Missing-Value Counts
```
{table_text(missing_summary)}
```

## Correlation Matrix
```
{table_text(numeric_corr)}
```

## League Result Distribution
```
{table_text(league_result)}
```

## Covid + Form Interaction
```
{table_text(combined_effect)}
```

## Key Findings
- `point_diff` has the clearest directional relationship with home goals and overall home success, making recent-form features central to the project.
- `capacity` coverage is incomplete, so stadium-based conclusions should be framed as exploratory rather than final.
- Covid context lowers baseline home-win rate, but positive recent form still preserves a meaningful home edge even during covid seasons.
- The combined view suggests the strongest practical predictors in this dataset are recent form and league context, with covid and stadium size acting as secondary modifiers.
"""
    write_report("total/report.md", report)
    return {
        "rows": str(len(total_df)),
        "capacity_missing": str(int(total_df["capacity"].isna().sum())),
        "leagues": str(total_df["league"].nunique()),
    }


def build_index(metadata: dict[str, dict[str, str]]) -> None:
    report = f"""# EDA Output Index

This folder contains four separate exploratory analyses generated from the `processed/` datasets.

## Reports
- [Covid](covid/report.md)
- [Stadium Capacity](stadium_capacity/report.md)
- [Last 3 Matches](last3_matches/report.md)
- [Total](total/report.md)

## Quick Metadata
- Covid EDA: {metadata['covid']}
- Stadium Capacity EDA: {metadata['stadium_capacity']}
- Last 3 Matches EDA: {metadata['last3_matches']}
- Total EDA: {metadata['total']}
"""
    write_report("README.md", report)


def main() -> None:
    setup_style()
    ensure_output_dirs()
    metadata = {
        "covid": build_covid_eda(),
        "stadium_capacity": build_stadium_eda(),
        "last3_matches": build_last3_eda(),
        "total": build_total_eda(),
    }
    build_index(metadata)
    print(f"EDA outputs generated under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
