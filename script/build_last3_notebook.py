from __future__ import annotations

import json
import textwrap
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = BASE_DIR / "EDA_last_3_matches.ipynb"


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": textwrap.dedent(text).strip().splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": textwrap.dedent(text).strip().splitlines(keepends=True),
    }


def build_notebook() -> None:
    nb = {
        "cells": [
            md(
                """
                # EDA Last 3 Matches Form

                Bu notebook'un amacı şu soruyu test etmek:

                `Bir takımın son 3 maçtaki formu maç sonucunu etkiliyor mu?`

                Hipotezi daha doğru kurarsak:
                - Eğer `last 3 matches form` güçlü bir sinyalse, maç sonucu form farkına daha çok duyarlı olur.
                - Eğer form etkisi zayıfsa, `home advantage` daha baskın görünür.
                - Ama bu iki etki birbirini tamamen dışlamak zorunda değil; ikisi aynı anda da çalışabilir.

                Bu yüzden burada iki şeye bakacağız:
                1. `home_last3_points - away_last3_points` farkı arttıkça sonuçlar nasıl değişiyor?
                2. Form farkı dengeli olduğunda bile ev sahibi avantajı devam ediyor mu?
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
                pd.set_option("display.width", 160)
                pd.set_option("display.float_format", lambda x: f"{x:.3f}")

                sns.set_theme(style="whitegrid", context="notebook")
                """
            ),
            md(
                """
                ## 1. Veriyi yükleme

                Tüm `*_last3_enriched.csv` dosyalarını birleştiriyoruz.
                Böylece ligler ve sezonlar üstünden tek bir analiz tabanı elde ediyoruz.
                """
            ),
            code(
                """
                base_dir = Path.cwd()
                processed_dir = base_dir / "processed"

                frames = []
                for file in sorted(processed_dir.glob("*_last3_enriched.csv")):
                    league_key = file.stem.replace("_last3_enriched", "")
                    df = pd.read_csv(file)
                    df["league_file"] = league_key
                    frames.append(df)

                last3_df = pd.concat(frames, ignore_index=True)
                last3_df["Date"] = pd.to_datetime(last3_df["Date"], errors="coerce")
                last3_df["point_diff"] = last3_df["home_last3_points"] - last3_df["away_last3_points"]
                last3_df["home_win"] = (last3_df["result_match"] == "Home Win").astype(int)
                last3_df["away_win"] = (last3_df["result_match"] == "Away Win").astype(int)
                last3_df["draw"] = (last3_df["result_match"] == "Draw").astype(int)

                last3_df.shape
                """
            ),
            md("## 2. İlk bakış"),
            code(
                """
                print("Shape:", last3_df.shape)
                print("League count:", last3_df["league_file"].nunique())
                print("Home form groups:", sorted(last3_df["home_form"].dropna().unique()))
                print("Away form groups:", sorted(last3_df["away_form"].dropna().unique()))

                last3_df.head()
                """
            ),
            code(
                """
                last3_df[[
                    "home_last3_points",
                    "away_last3_points",
                    "point_diff",
                    "result_match"
                ]].describe().T
                """
            ),
            md(
                """
                ## 3. Form gruplarına göre sonuç dağılımı

                Burada `home_form` ve `away_form` birlikte okunuyor.
                Eğer form önemliyse, `good vs bad` gibi eşleşmelerde sonuç dağılımı belirgin biçimde kaymalı.
                """
            ),
            code(
                """
                result_by_form_pair = (
                    pd.crosstab(
                        [last3_df["home_form"], last3_df["away_form"]],
                        last3_df["result_match"],
                        normalize="index",
                    )
                    .round(3)
                )

                result_by_form_pair
                """
            ),
            md(
                """
                ## 4. Son 3 maç puan farkı etkisi

                Asıl değişkenimiz:

                `point_diff = home_last3_points - away_last3_points`

                Pozitifse ev sahibi son 3 maçta daha formda demektir.
                Negatifse deplasman takımı daha formda demektir.
                """
            ),
            code(
                """
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

                point_diff_effect
                """
            ),
            code(
                """
                overall_result_rates = (
                    last3_df["result_match"]
                    .value_counts(normalize=True)
                    .rename("share")
                    .round(3)
                )

                overall_result_rates
                """
            ),
            md(
                """
                ## 5. Hipotezi daha net test etme

                Burada kritik soru şu:

                `Form farkı dengeli olduğunda home advantage hala görünüyor mu?`

                Eğer `0 to 1` bandında bile home win rate, away win rate'den anlamlı biçimde yüksekse
                bu bize form etkisinin güçlü olsa bile ev sahibi etkisini tamamen silmediğini söyler.
                """
            ),
            code(
                """
                balanced_band = last3_df[last3_df["point_diff"].between(-1, 1)]
                home_better = last3_df[last3_df["point_diff"] > 0]
                away_better = last3_df[last3_df["point_diff"] < 0]

                hypothesis_check = pd.DataFrame(
                    {
                        "sample": [
                            "All matches",
                            "Balanced form (-1 to 1)",
                            "Home better form (> 0)",
                            "Away better form (< 0)",
                        ],
                        "matches": [
                            len(last3_df),
                            len(balanced_band),
                            len(home_better),
                            len(away_better),
                        ],
                        "home_win_rate": [
                            (last3_df["result_match"] == "Home Win").mean(),
                            (balanced_band["result_match"] == "Home Win").mean(),
                            (home_better["result_match"] == "Home Win").mean(),
                            (away_better["result_match"] == "Home Win").mean(),
                        ],
                        "away_win_rate": [
                            (last3_df["result_match"] == "Away Win").mean(),
                            (balanced_band["result_match"] == "Away Win").mean(),
                            (home_better["result_match"] == "Away Win").mean(),
                            (away_better["result_match"] == "Away Win").mean(),
                        ],
                        "draw_rate": [
                            (last3_df["result_match"] == "Draw").mean(),
                            (balanced_band["result_match"] == "Draw").mean(),
                            (home_better["result_match"] == "Draw").mean(),
                            (away_better["result_match"] == "Draw").mean(),
                        ],
                    }
                ).round(3)

                hypothesis_check
                """
            ),
            code(
                """
                form_summary = (
                    last3_df.groupby("home_form")
                    .agg(
                        matches=("result_match", "size"),
                        home_win_rate=("result_match", lambda s: (s == "Home Win").mean()),
                        avg_point_diff=("point_diff", "mean"),
                    )
                    .round(3)
                )

                form_summary
                """
            ),
            md("## 6. Görselleştirme"),
            code(
                """
                fig, axes = plt.subplots(1, 2, figsize=(15, 5))

                sns.barplot(
                    data=point_diff_effect.reset_index(),
                    x="point_diff_band",
                    y="home_win_rate",
                    hue="point_diff_band",
                    palette="mako",
                    legend=False,
                    ax=axes[0],
                )
                axes[0].set_title("Home Win Rate by Last-3 Point Differential")
                axes[0].set_xlabel("Point Differential Band")
                axes[0].set_ylabel("Home Win Rate")

                result_by_form_pair["Home Win"].unstack().plot(
                    kind="bar",
                    ax=axes[1],
                    colormap="Set2",
                )
                axes[1].set_title("Home Win Share by Form Pair")
                axes[1].set_xlabel("Home Form")
                axes[1].set_ylabel("Home Win Share")
                axes[1].legend(title="Away Form")

                plt.tight_layout()
                plt.show()
                """
            ),
            code(
                """
                heatmap_data = (
                    last3_df.pivot_table(
                        index="home_form",
                        columns="away_form",
                        values="point_diff",
                        aggfunc="mean",
                    )
                    .round(2)
                )

                plt.figure(figsize=(8, 5))
                sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="RdYlGn", center=0)
                plt.title("Average Last-3 Point Differential by Form Pair")
                plt.xlabel("Away Form")
                plt.ylabel("Home Form")
                plt.tight_layout()
                plt.show()
                """
            ),
            md(
                """
                ## 7. Yorum

                Bu notebook'taki çıktıları şu şekilde okuyabiliriz:

                - `point_diff` yükseldikçe `home win rate` düzenli biçimde artıyorsa, son 3 maç formu sonuç üzerinde etkili demektir.
                - Ama dengeli form bandında bile `home win rate > away win rate` ise, home advantage hala devrededir.
                - Yani daha doğru sonuç şudur:

                `Last 3 matches form önemlidir, fakat home advantage etkisini tamamen ortadan kaldırmaz.`

                Senin cümleni proje diline daha akademik yazarsak:

                `Recent form is a meaningful predictor of match outcome; however, even when recent form is balanced, the home side still retains an observable advantage.`
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
