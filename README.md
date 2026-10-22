# Premier League transfer impact

Which Premier League signings actually made their clubs better, judged on performance alone, with transfer fees left out?

This project scores 223 headline signings (EUR 20m or more, summer 2016 to January 2023) by how many **league points above replacement (PAR)** each one added over his whole stay at the club. That is the football equivalent of baseball's WAR. Every score comes with a 90% bootstrap interval.

Everything runs on free, public data and takes about a minute on a laptop.

## Headline results

**Lifetime points above replacement, top 10**

| Rank | Player | Club | Stint | PAR | 90% interval | Rank range (90%) |
|---|---|---|---|---|---|---|
| 1 | Mohamed Salah | Liverpool | 2017-18 to 2024-25 | +35.1 | +28.3 to +42.3 | 1 to 3 |
| 2 | Virgil van Dijk | Liverpool | Jan 2018 to 2024-25 | +30.6 | +25.9 to +35.6 | 1 to 4 |
| 3 | Rodri | Man City | 2019-20 to 2024-25 | +28.3 | +23.5 to +33.2 | 2 to 6 |
| 4 | Ilkay Gündoğan | Man City | 2016-17 to 2022-23 | +24.9 | +20.2 to +29.9 | 2 to 9 |
| 5 | Alisson | Liverpool | 2018-19 to 2024-25 | +23.9 | +18.2 to +30.0 | 3 to 13 |
| 6 | Bruno Fernandes | Man United | Jan 2020 to 2024-25 | +21.0 | +16.5 to +25.6 | 4 to 16 |
| 7 | Erling Haaland | Man City | 2022-23 to 2024-25 | +20.5 | +15.2 to +26.5 | 4 to 20 |
| 8 | Marcos Alonso | Chelsea | 2016-17 to 2022-23 | +20.5 | +16.4 to +24.6 | 5 to 17 |
| 9 | Kyle Walker | Man City | 2017-18 to 2024-25 | +20.4 | +16.9 to +24.1 | 5 to 15 |
| 10 | N'Golo Kanté | Chelsea | 2016-17 to 2022-23 | +19.3 | +15.3 to +23.7 | 5 to 20 |

The top three are stable across 1,000 bootstrap reruns. Places 5 to 15 are statistically close: Kanté makes the top 10 in only 48% of reruns.

**Biggest upgrades over the previous player in the role** (signings from summer 2019, compared with the incumbent's average over the three seasons before)

| Signing | Replaced | Upgrade (points per season) | 90% interval |
|---|---|---|---|
| Alexander Isak (Newcastle) | Chris Wood | +4.4 | +2.7 to +6.1 |
| Erling Haaland (Man City) | Gabriel Jesus | +3.9 | +1.8 to +6.2 |
| Bruno Guimarães (Newcastle) | Isaac Hayden | +3.9 | +2.5 to +5.4 |
| Danny Ings (Southampton) | Shane Long | +3.6 | +2.2 to +5.2 |

Of the 119 signings with an incumbent, 18 are clearly better than him (the interval stays above zero), 45 are clearly worse, and 56 are too close to call.

Full tables are in [`results/`](results/).

## Method

1. **Signings.** Every permanent move into a Premier League club with a fee of EUR 20m or more (Transfermarkt, via `ewenme/transfers`). The fee only decides which moves count as headline signings. It never enters a score.
2. **Positions.** Nine roles from Transfermarkt: RB/RWB, LB/LWB, CB, DM, CM, AM, wide right, wide left, CF, plus GK. They cover 96% of Premier League minutes from 2016-17. The rest fall back to FPL's broad DEF or MID.
3. **Output.** FPL's Bonus Points System (BPS) per 90 minutes, shrunk toward the role average for small samples, and expressed as a z-score within the role.
4. **Team model.** Club points per game regressed on the minutes-weighted average player z-score across 180 club-seasons (2016-17 to 2024-25): `ppg = 0.71 × quality + 1.37`, R² = 0.81.
5. **Points above replacement.** For each season of the stint: (player's BPS/90 minus the replacement level of the role he played that season) × his share of team minutes × the team-model slope × 38. Replacement level is the 20th percentile of regulars in the role. If FPL relabels his position during the stint, that season is judged on the new role.
6. **Lifetime total.** PAR summed from arrival until he leaves, the club is relegated, or the data ends (2024-25). Longevity is the standard.
7. **One-and-done penalty.** A signing sold or loaned out after a single full season loses 1.47 points (the median PAR of a regular season by a headline signing). Relegation-ended and ongoing stints are exempt.
8. **Incumbent.** The player with the most minutes in the signing's role at the club the season before. The comparison flags whether the signing really took his place (the incumbent left, or his minutes halved) or was bought as depth.
9. **Uncertainty.** 1,000 bootstrap reruns, each resampling every player-season's matches and refitting the team model, give 90% intervals on scores, ranks and upgrades.

## Limitations

- **BPS is a fantasy-football metric.** It underrates tackling midfielders and defenders in losing teams. xG, xA and defensive actions (FBref/Opta) would be better.
- **The team model is partly circular,** because BPS rewards the clean sheets and goals that also win points. The R² of 0.81 flatters it.
- **Intervals cover match-to-match noise only.** They don't cover whether BPS is the right measure or whether a role is labelled correctly, so the true uncertainty is wider.
- **One position per player.** Role changes are only seen when FPL moves a player between DEF, MID and FWD.
- **The data window is limited.** Player data starts in 2016-17, so 2016 signings have no incumbent comparison. Transfer data ends in January 2023.
- **Shared credit.** Several signings in one window, and manager changes, are not separated.

## Reproduce

```bash
pip install -r requirements.txt
bash scripts/download_data.sh          # about 60 MB into ./data
python src/rank_transfers.py data results/single_season_scores.csv \
    results/lifetime_par_ranking_raw.csv results/lifetime_par_by_season.csv \
    results/lifetime_par_ranking.csv   # last argument turns on the bootstrap
```

The bootstrap uses a fixed seed, so reruns match the files in `results/` exactly.

## Data sources

- Transfers and positions: [ewenme/transfers](https://github.com/ewenme/transfers) (Transfermarkt)
- Match results: [jalapic/engsoccerdata](https://github.com/jalapic/engsoccerdata) and [openfootball/football.json](https://github.com/openfootball/football.json)
- Player stats: [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League)
