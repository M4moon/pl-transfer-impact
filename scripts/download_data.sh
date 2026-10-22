#!/usr/bin/env bash
# Downloads every free input the pipeline uses into ./data (about 60 MB).
set -euo pipefail
mkdir -p data && cd data
GH=https://raw.githubusercontent.com
for f in premier-league championship 1-bundesliga primera-division serie-a ligue-1 eredivisie liga-nos; do
  curl -fsSLO "$GH/ewenme/transfers/master/data/$f.csv"
done
curl -fsSLO "$GH/jalapic/engsoccerdata/master/data-raw/england.csv"
for l in en.1 en.2; do
  curl -fsSL -o "2022-23_$l.json" "$GH/openfootball/football.json/master/2022-23/$l.json"
done
for y in 2016 2017 2018 2019 2020 2021 2022 2023 2024; do
  s="$y-$(printf %02d $(( (y + 1) % 100 )))"
  curl -fsSL -o "players_raw_$s.csv" "$GH/vaastav/Fantasy-Premier-League/master/data/$s/players_raw.csv"
  curl -fsSL -o "gw_$s.csv" "$GH/vaastav/Fantasy-Premier-League/master/data/$s/gws/merged_gw.csv"
done
