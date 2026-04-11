from __future__ import annotations

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = BASE_DIR / "EDA_covid_comparison.ipynb"


def md(text: str):
    lines = text.strip().splitlines(keepends=True)
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines,
    }


def code(text: str):
    lines = text.strip().splitlines(keepends=True)
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines,
    }


def build_notebook() -> None:
    nb = {
        "cells": [
        md(
            """
            # EDA Covid Comparison

            Bu notebook, `EDA_stadium_capacity` akışına benzer şekilde önce genel veri kontrolü yapar,
            ardından temel `pandas` yapılarıyla bizi adım adım covid ve normal sezon karşılaştırmasına götürür.

            Ana hedefler:
            - veri setinin genel yapısını görmek
            - eksik değer ve temel dağılımları incelemek
            - `groupby`, `crosstab` ve `pivot` ile covid vs normal farklarını okumak
            - home advantage tarafında ilk çıkarımları üretmek
            """
        ),
        code(
            """
            from pathlib import Path

            import numpy as np
            import pandas as pd
            import seaborn as sns
            import matplotlib.pyplot as plt

            pd.set_option("display.max_columns", None)
            pd.set_option("display.width", 140)
            pd.set_option("display.float_format", lambda x: f"{x:.3f}")

            sns.set_theme(style="whitegrid", context="notebook")
            """
        ),
        md(
            """
            ## 1. Veriyi yükleme

            İlk adımda tüm `*_covid_comparison.csv` dosyalarını birleştiriyoruz.
            Burada amaç tek bir analiz tabanı oluşturmak.
            """
        ),
        code(
            """
            base_dir = Path.cwd()
            processed_dir = base_dir / "processed"

            frames = []
            for file in sorted(processed_dir.glob("*_covid_comparison.csv")):
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

            covid_df.shape
            """
        ),
        md("## 2. İlk bakış\nGenel yapı, örnek satırlar ve veri tipleri ile başlayalım."),
        code(
            """
            print("Shape:", covid_df.shape)
            print("League count:", covid_df["league_file"].nunique())
            print("Season type values:", covid_df["season_type"].unique())

            covid_df.head()
            """
        ),
        code(
            """
            covid_df.info()
            """
        ),
        md(
            """
            ## 3. Genel özet istatistikler

            Bu bölüm `describe`, benzersiz değer sayıları ve eksik değer kontrolü ile temel EDA iskeletini kurar.
            """
        ),
        code(
            """
            covid_df.describe(include="all").T.head(15)
            """
        ),
        code(
            """
            covid_df.isna().sum().sort_values(ascending=False).head(15)
            """
        ),
        code(
            """
            pd.DataFrame(
                {
                    "unique_count": covid_df.nunique(),
                    "missing_count": covid_df.isna().sum(),
                    "missing_ratio": (covid_df.isna().mean() * 100).round(2),
                }
            ).sort_values("missing_ratio", ascending=False).head(15)
            """
        ),
        md(
            """
            ## 4. Temel kategorik dağılımlar

            Önce sonucu ve sezon tipini ayrı ayrı okuyalım.
            Bu adım bizi karşılaştırmalı analiz öncesi veri dengesini anlamaya götürür.
            """
        ),
        code(
            """
            covid_df["season_type"].value_counts()
            """
        ),
        code(
            """
            covid_df["FTR"].value_counts().rename(index={"H": "Home Win", "D": "Draw", "A": "Away Win"})
            """
        ),
        code(
            """
            pd.crosstab(covid_df["league_file"], covid_df["season_type"])
            """
        ),
        md(
            """
            ## 5. Temel sayısal özetler

            Burada henüz covid-normal kıyasına tam geçmeden önce genel maç profiline bakıyoruz.
            """
        ),
        code(
            """
            numeric_cols = ["FTHG", "FTAG", "HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "total_goals", "goal_diff", "shot_diff"]
            covid_df[numeric_cols].describe().T
            """
        ),
        code(
            """
            covid_df.groupby("league_file")[["FTHG", "FTAG", "HS", "AS", "total_goals"]].mean().round(2)
            """
        ),
        md(
            """
            ## 6. Covid vs normal için ilk groupby karşılaştırmaları

            Şimdi ana soruya yaklaşıyoruz: covid sezonları ile normal sezonlar arasında temel fark var mı?
            """
        ),
        code(
            """
            season_summary = (
                covid_df.groupby("season_type")
                .agg(
                    matches=("FTR", "size"),
                    avg_home_goals=("FTHG", "mean"),
                    avg_away_goals=("FTAG", "mean"),
                    avg_total_goals=("total_goals", "mean"),
                    avg_home_shots=("HS", "mean"),
                    avg_away_shots=("AS", "mean"),
                    avg_goal_diff=("goal_diff", "mean"),
                    avg_shot_diff=("shot_diff", "mean"),
                )
                .round(3)
                .sort_index()
            )

            season_summary
            """
        ),
        code(
            """
            result_distribution = (
                pd.crosstab(covid_df["season_type"], covid_df["FTR"], normalize="index")
                .rename(columns={"H": "Home Win", "D": "Draw", "A": "Away Win"})
                .round(3)
            )

            result_distribution
            """
        ),
        code(
            """
            covid_df.groupby(["league_file", "season_type"]).size().unstack(fill_value=0)
            """
        ),
        md(
            """
            ## 7. Home advantage tarafına geçiş

            Covid comparison için en kritik hikaye genelde seyirci etkisinin zayıflaması olur.
            Bu yüzden home win, point advantage ve goal difference ölçülerini birlikte okuyalım.
            """
        ),
        code(
            """
            home_advantage_summary = (
                covid_df.groupby("season_type")
                .agg(
                    matches=("FTR", "size"),
                    home_win_rate=("home_win", "mean"),
                    away_win_rate=("away_win", "mean"),
                    draw_rate=("draw", "mean"),
                    avg_goal_diff=("goal_diff", "mean"),
                    home_points_per_match=("home_points", "mean"),
                    away_points_per_match=("away_points", "mean"),
                    home_advantage_points=("home_advantage_points", "mean"),
                )
                .round(3)
                .sort_index()
            )

            home_advantage_summary
            """
        ),
        code(
            """
            league_home_win = (
                covid_df.groupby(["league_file", "season_type"])["home_win"]
                .mean()
                .unstack()
                .round(3)
            )

            league_home_win["delta_normal_minus_covid"] = (
                league_home_win["normal"] - league_home_win["covid"]
            ).round(3)

            league_home_win.sort_values("delta_normal_minus_covid", ascending=False)
            """
        ),
        code(
            """
            league_advantage = (
                covid_df.groupby(["league_file", "season_type"])["home_advantage_points"]
                .mean()
                .unstack()
                .round(3)
            )

            league_advantage["delta_normal_minus_covid"] = (
                league_advantage["normal"] - league_advantage["covid"]
            ).round(3)

            league_advantage.sort_values("delta_normal_minus_covid", ascending=False)
            """
        ),
        md(
            """
            ## 8. Karşılaştırmayı biraz daha derinleştirme

            Burada temel `groupby` mantığıyla maç içi üstünlük sinyallerinin covid ve normal sezonda nasıl davrandığına bakıyoruz.
            """
        ),
        code(
            """
            covid_df["shot_edge_label"] = np.where(
                covid_df["shot_diff"] >= 0,
                "Home shot edge",
                "Away shot edge",
            )

            shot_edge_summary = (
                covid_df.groupby(["season_type", "shot_edge_label"])
                .agg(
                    matches=("FTR", "size"),
                    home_win_rate=("home_win", "mean"),
                    avg_goal_diff=("goal_diff", "mean"),
                )
                .round(3)
                .reset_index()
            )

            shot_edge_summary
            """
        ),
        code(
            """
            covid_df.pivot_table(
                index="league_file",
                columns="season_type",
                values="total_goals",
                aggfunc="mean",
            ).round(2)
            """
        ),
        md("## 9. Basit görseller\nNotebook akışını desteklemek için birkaç temel grafik ekleyelim."),
        code(
            """
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            result_distribution.plot(kind="bar", stacked=True, ax=axes[0], colormap="Set2")
            axes[0].set_title("Result Distribution: Covid vs Normal")
            axes[0].set_xlabel("Season Type")
            axes[0].set_ylabel("Match Share")
            axes[0].legend(title="Result")

            sns.barplot(
                data=home_advantage_summary.reset_index(),
                x="season_type",
                y="home_advantage_points",
                hue="season_type",
                palette="Set1",
                legend=False,
                ax=axes[1],
            )
            axes[1].set_title("Home Advantage Points")
            axes[1].set_xlabel("Season Type")
            axes[1].set_ylabel("Average Point Edge")

            plt.tight_layout()
            plt.show()
            """
        ),
        code(
            """
            plt.figure(figsize=(10, 5))
            sns.heatmap(league_home_win[["covid", "normal"]] * 100, annot=True, fmt=".1f", cmap="YlGnBu")
            plt.title("League-Level Home Win Rate (%)")
            plt.xlabel("Season Type")
            plt.ylabel("League")
            plt.tight_layout()
            plt.show()
            """
        ),
        md(
            """
            ## 10. Kısa yorum

            İlk EDA seviyesinde bu notebook bize şunları verir:
            - covid ve normal sezonlarda sonuç dağılımlarının nasıl değiştiği
            - home advantage metriklerinde düşüş olup olmadığı
            - hangi liglerin farkı daha güçlü taşıdığı
            - sonraki aşamada hangi hipotezleri test etmemiz gerektiği

            Bir sonraki adımda istersek bu yapının üstüne:
            - lig bazlı daha detaylı grafikler
            - istatistiksel testler
            - form / stadium capacity ile birlikte birleşik analiz
            ekleyebiliriz.
            """
        ),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.x",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    with NOTEBOOK_PATH.open("w", encoding="utf-8") as handle:
        json.dump(nb, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    build_notebook()
