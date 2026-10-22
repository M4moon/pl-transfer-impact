"""Premier League transfer impact, performance only.

Usage:
  python src/rank_transfers.py DATA_DIR single.csv lifetime.csv by_season.csv [bootstrap.csv]

Outputs, in order:
  1. single-season Transfer Impact Score and replacement counterfactual for each headline signing
  2. lifetime points above replacement (PAR), one-and-done penalty, incumbent comparisons
  3. the same PAR broken down by season
  4. (optional) the lifetime table again with 90% bootstrap intervals (1,000 reruns)

Data (all free, GitHub-hosted; fetch with scripts/download_data.sh):
  transfers  : ewenme/transfers (Transfermarkt scrape, 1992 to Jan 2023)
  results    : jalapic/engsoccerdata + openfootball/football.json for 2022-23
  players    : vaastav/Fantasy-Premier-League players_raw.csv and gws/merged_gw.csv (2016-17+)
Fee is used only to pick "headline" signings (>= EUR 20m), never in the score.
"""
import sys, re, unicodedata
import numpy as np, pandas as pd
from rapidfuzz import fuzz, process

D = sys.argv[1]
MIN_FEE = 20.0

# ---------- 1. league tables for every season, tiers 1-2 ----------
e = pd.read_csv(f"{D}/england.csv", low_memory=False)
e = e[(e.tier <= 2) & (e.Season >= 1991)]
# engsoccerdata skips 2022-23; patch it from openfootball/football.json
import json
for tier in (1, 2):
    js = json.load(open(f"{D}/2022-23_en.{tier}.json"))["matches"]
    clean = lambda n: n if n == "AFC Bournemouth" else re.sub(r" A?FC$", "", n)
    e = pd.concat([e, pd.DataFrame([dict(Season=2022, tier=tier, home=clean(x["team1"]),
                   visitor=clean(x["team2"]), hgoal=x["score"]["ft"][0], vgoal=x["score"]["ft"][1])
                   for x in js if x.get("round", "").startswith("Matchday")])])
rows = []
for (season, tier), g in e.groupby(["Season", "tier"]):
    for side, gf, ga in [("home", "hgoal", "vgoal"), ("visitor", "vgoal", "hgoal")]:
        x = g[[side, gf, ga]].rename(columns={side: "club", gf: "gf", ga: "ga"})
        x["pts"] = np.select([x.gf > x.ga, x.gf == x.ga], [3, 1], 0)
        x["season"], x["tier"] = season, tier
        rows.append(x)
m = pd.concat(rows)
tab = m.groupby(["season", "tier", "club"]).agg(p=("pts", "size"), pts=("pts", "sum"),
                                              gf=("gf", "sum"), ga=("ga", "sum")).reset_index()
tab["gd"] = tab.gf - tab.ga
tab = tab.sort_values(["season", "tier", "pts", "gd", "gf"], ascending=[True, True, False, False, False])
tab["pos"] = tab.groupby(["season", "tier"]).cumcount() + 1
# pyramid rank: PL 1-20(22), Championship continues below
tab["prank"] = tab.pos + np.where(tab.tier == 2, tab.groupby("season").apply(
    lambda g: (g.tier == 1).sum()).reindex(tab.season).values, 0)
tab["ppg"] = tab.pts / tab.p
tab["gdpg"] = tab.gd / tab.p

# regression-to-the-mean baseline: expected change in pyramid rank / ppg given last season
pairs = tab.merge(tab, on="club", suffixes=("_b", "_a"))
pairs = pairs[pairs.season_a == pairs.season_b + 1]
b_rank = np.polyfit(pairs.prank_b, pairs.prank_a - pairs.prank_b, 1)
pl = pairs[(pairs.tier_b == 1) & (pairs.tier_a == 1)]
b_ppg = np.polyfit(pl.ppg_b, pl.ppg_a - pl.ppg_b, 1)

# ---------- 2. club name harmonisation ----------
ALIAS = {"Arsenal FC": "Arsenal", "Chelsea FC": "Chelsea", "Liverpool FC": "Liverpool",
         "Everton FC": "Everton", "Southampton FC": "Southampton", "Burnley FC": "Burnley",
         "Watford FC": "Watford", "Fulham FC": "Fulham", "Brentford FC": "Brentford",
         "Sunderland AFC": "Sunderland"}
FPL_CODE = {3: "Arsenal", 7: "Aston Villa", 91: "AFC Bournemouth", 94: "Brentford",
            36: "Brighton & Hove Albion", 90: "Burnley", 8: "Chelsea", 31: "Crystal Palace",
            11: "Everton", 54: "Fulham", 2: "Leeds United", 13: "Leicester City", 14: "Liverpool",
            43: "Manchester City", 1: "Manchester United", 4: "Newcastle United",
            45: "Norwich City", 17: "Nottingham Forest", 49: "Sheffield United", 20: "Southampton",
            6: "Tottenham Hotspur", 57: "Watford", 35: "West Bromwich Albion", 21: "West Ham United",
            39: "Wolverhampton Wanderers", 102: "Luton Town", 40: "Ipswich Town",
            88: "Hull City", 25: "Middlesbrough", 110: "Stoke City", 80: "Swansea City",
            56: "Sunderland", 38: "Huddersfield Town", 97: "Cardiff City"}

# ---------- 3. player season totals ----------
def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z ]", "", s).strip()

ps = []
for y in range(2016, 2025):
    p = pd.read_csv(f"{D}/players_raw_{y}-{str(y + 1)[2:]}.csv")
    p["season"] = y
    p["club"] = p.team_code.map(FPL_CODE)
    p["full"] = (p.first_name + " " + p.second_name).map(norm)
    p["web"] = p.web_name.map(norm)
    ps.append(p[["season", "code", "club", "full", "web", "element_type", "minutes", "bps",
                 "goals_scored", "assists", "total_points"]])
ps = pd.concat(ps)

# ---------- 3a. fine positions from Transfermarkt (via ewenme transfer records) ----------
ROLE = {"Goalkeeper": "GK", "Right-Back": "RB/RWB", "Left-Back": "LB/LWB", "Centre-Back": "CB",
        "Defensive Midfield": "DM", "Central Midfield": "CM", "Attacking Midfield": "AM",
        "Right Winger": "Wide R", "Right Midfield": "Wide R", "Left Winger": "Wide L",
        "Left Midfield": "Wide L", "Centre-Forward": "CF", "Second Striker": "CF"}
# where to look for a replacement when nobody at the club shares the exact role
FAMILY = {"GK": ["GK"], "RB/RWB": ["RB/RWB", "LB/LWB", "CB", "DEF?"],
          "LB/LWB": ["LB/LWB", "RB/RWB", "CB", "DEF?"], "CB": ["CB", "DM", "DEF?"],
          "DM": ["DM", "CM", "CB", "MID?"], "CM": ["CM", "DM", "AM", "MID?"],
          "AM": ["AM", "CM", "Wide R", "Wide L", "MID?"], "Wide R": ["Wide R", "Wide L", "AM", "MID?", "CF"],
          "Wide L": ["Wide L", "Wide R", "AM", "MID?", "CF"], "CF": ["CF", "Wide R", "Wide L", "AM"],
          "DEF?": ["DEF?", "CB", "RB/RWB", "LB/LWB"], "MID?": ["MID?", "CM", "DM", "AM", "Wide R", "Wide L"]}
BROAD = {1: "GK", 2: "DEF?", 3: "MID?", 4: "CF"}
tm = pd.concat([pd.read_csv(f"{D}/{f}.csv") for f in
                ("premier-league", "championship", "1-bundesliga", "primera-division", "serie-a",
                 "ligue-1", "eredivisie", "liga-nos")])
tm["role"] = tm.position.map(ROLE)
tm = tm.dropna(subset=["role"]).sort_values("year")
tm["n"] = tm.player_name.map(norm)
tm["club"] = tm.club_name.replace(ALIAS)
tm_club = {c: g for c, g in tm.groupby("club")}
tm_exact = tm.groupby("n").role.last()

def fine_role(rows):
    """Latest Transfermarkt position for one FPL player (all his seasons), else None."""
    full, web = rows.full.iloc[-1], rows.web.iloc[-1]
    found = [tm_club[c] for c in rows.club.dropna().unique() if c in tm_club]
    if found:
        cands = pd.concat(found)
        names = cands.n.unique()
        ok = [c for c in names if fuzz.token_sort_ratio(full, c) >= 85 or c == web
              or (fuzz.token_set_ratio(full, c) == 100 and len(c.split()) >= 2)]
        if ok:
            return cands[cands.n.isin(ok)].role.iloc[-1]
    return tm_exact.get(full)

roles = ps.sort_values("season").groupby("code").apply(fine_role)
ps["role"] = ps.code.map(roles)
ps["grp"] = ps.role.fillna(ps.element_type.map(BROAD))
cov = ps[ps.role.notna()].minutes.sum() / ps.minutes.sum()
print(f"fine position found for {cov:.0%} of PL minutes", file=sys.stderr)

# shrunk bps per 90 (prior = role mean, weight = 5 full matches)
pos_mean = (ps.groupby("grp").bps.sum() / (ps.groupby("grp").minutes.sum() / 90))
ps["bps90"] = (ps.bps + 5 * ps.grp.map(pos_mean)) / (ps.minutes / 90 + 5)
reg = ps[ps.minutes >= 900]
mu, sd = reg.groupby("grp").bps90.mean(), reg.groupby("grp").bps90.std()
ps["out_z"] = (ps.bps90 - ps.grp.map(mu)) / ps.grp.map(sd)

# ---------- 3b. team model: points per game ~ minutes-weighted player quality ----------
ps["zmin"] = ps.out_z * ps.minutes
tq = ps.dropna(subset=["club"]).groupby(["season", "club"]).agg(zmin=("zmin", "sum"), mins=("minutes", "sum"))
tq["Q"] = tq.zmin / tq.mins
tq = tq.reset_index().merge(tab[tab.tier == 1][["season", "club", "ppg", "pts", "gd"]], on=["season", "club"])
b_q = np.polyfit(tq.Q, tq.ppg, 1)
r2 = np.corrcoef(tq.Q, tq.ppg)[0, 1] ** 2
print(f"team model: ppg = {b_q[0]:.2f}*Q + {b_q[1]:.2f}, R2 = {r2:.2f}, n = {len(tq)}", file=sys.stderr)
# replacement level = 20th percentile of regulars at the position
repl = reg.groupby("grp").bps90.quantile(.2)

def projected_z(code, season, pos):
    """Career-based projection: recency-weighted BPS/90 over all PL seasons up to `season`,
    shrunk toward replacement level by 10 full matches."""
    h = ps[(ps.code == code) & (ps.season <= season)]
    w = 0.5 ** (season - h.season)
    n90 = (h.minutes * w).sum() / 90
    rate = ((h.bps * w).sum() + 10 * repl[pos]) / (n90 + 10)
    return (rate - mu[pos]) / sd[pos], h.minutes.sum()

def counterfactual(a, role, club, season):
    """Swap the signing's minutes to the teammate in the same role ranked next by minutes;
    if nobody shares the role, step out through FAMILY (e.g. RB -> LB -> CB)."""
    squad = ps[(ps.season == season) & (ps.club == club) & (ps.code != a.code)]
    for r in FAMILY[role]:
        mates = squad[squad.grp == r].sort_values("minutes", ascending=False)
        if len(mates):
            break
    else:
        return {}
    below = mates[mates.minutes <= a.minutes]
    rp = (below if len(below) else mates).iloc[0]
    zr, career = projected_z(rp.code, season, rp.grp)  # z on his own role's scale
    row = tq[(tq.season == season) & (tq.club == club)]
    if row.empty:
        return dict(repl=rp.web, repl_role=rp.grp, repl_career_min=career)
    row = row.iloc[0]
    dQ = a.minutes * (zr - a.out_z) / row.mins
    d_pts = b_q[0] * dQ * 38                     # points the club would have gained (-) or lost
    cf_pts = row.pts + d_pts
    others = tab[(tab.season == season) & (tab.tier == 1) & (tab.club != club)]
    cf_pos = int((others.pts > cf_pts).sum() + 1)
    return dict(repl=rp.web, repl_role=rp.grp, repl_career_min=career, repl_z=zr, pts_added=-d_pts,
                cf_pos=f"PL {cf_pos}")

# ---------- 4. headline transfers into PL clubs ----------
t = pd.read_csv(f"{D}/premier-league.csv")
t = t[(t.transfer_movement == "in") & (t.fee_cleaned >= MIN_FEE)
      & ~t.fee.str.contains("loan", case=False, na=False) & (t.year >= 2016)].copy()
t["club"] = t.club_name.replace(ALIAS)
# window: summer of Y -> before = season Y-1, after = season Y
#         winter of Y/Y+1 -> before = Y-1 (last full season), after = Y+1 (first full season)
t["s_b"] = t.year - 1
t["s_a"] = np.where(t.transfer_period == "Summer", t.year, t.year + 1)

def match(r):
    pool = ps[(ps.season == r.s_a) & (ps.club == r.club)]
    if pool.empty:
        return None
    n = norm(r.player_name)
    # full-name agreement beats a shared first name (Gabriel Jesus vs Gabriel Magalhaes)
    sc = [max(fuzz.token_sort_ratio(n, f), fuzz.ratio(n, w), 0.95 * fuzz.token_set_ratio(n, f))
          for f, w in zip(pool.full, pool.web)]
    i = int(np.argmax(sc))
    return pool.iloc[i] if sc[i] >= 80 else None

out = []
for _, r in t.iterrows():
    a = match(r)
    tb = tab[(tab.club == r.club) & (tab.season == r.s_b)]
    ta = tab[(tab.club == r.club) & (tab.season == r.s_a)]
    rec = dict(player=r.player_name, club=r.club, window=f"Summer {r.year}" if r.transfer_period == "Summer" else f"Jan {r.year + 1}",
               after_season=f"{r.s_a}-{str(r.s_a + 1)[2:]}", fee_m=r.fee_cleaned, matched=a is not None)
    if len(tb) and len(ta):
        tb, ta = tb.iloc[0], ta.iloc[0]
        rec.update(pos_b=f"{'PL' if tb.tier == 1 else 'Ch'} {tb.pos}",
                   pos_a=f"{'PL' if ta.tier == 1 else 'Ch'} {ta.pos}",
                   rank_resid=-((ta.prank - tb.prank) - np.polyval(b_rank, tb.prank)),
                   ppg_resid=((ta.ppg - tb.ppg) - np.polyval(b_ppg, tb.ppg))
                   if tb.tier == 1 and ta.tier == 1 else np.nan)
    if a is not None:
        rec.update(minutes=a.minutes, share=min(a.minutes / (38 * 90), 1), goals=a.goals_scored,
                   assists=a.assists, out_z=a.out_z, pos=a.element_type)
        pb = ps[(ps.code == a.code) & (ps.season == r.s_b)]
        if len(pb) and pb.minutes.iloc[0] >= 900:
            rec["delta_z"] = a.out_z - pb.out_z.iloc[0]
        role = ROLE.get(r.position, a.grp)
        rec["role"] = role
        rec.update(code=a.code, start=r.year)
        rec.update(counterfactual(a, role, r.club, r.s_a))
    out.append(rec)
o = pd.DataFrame(out)
print(f"headline transfers: {len(o)}, matched to FPL: {o.matched.sum()}", file=sys.stderr)
o = o[o.matched].copy()

z = lambda s: (s - s.mean()) / s.std()
team = pd.concat([z(o.rank_resid), z(o.ppg_resid)], axis=1).mean(axis=1)
o["C_team"] = team * o.share            # team change, credited by share of minutes played
o["C_avail"] = z(o.share)
o["C_output"] = z(o.out_z) * np.sqrt(o.share.clip(0, 1))  # downweight tiny samples
o["C_delta"] = z(o.delta_z)
o["C_repl"] = z(o.pts_added)             # league points added over the projected replacement
W = {"C_team": .25, "C_repl": .25, "C_avail": .15, "C_output": .25, "C_delta": .10}
comp = o[list(W)]
w = comp.notna() * pd.Series(W)
o["TIS"] = (comp.fillna(0) * pd.Series(W)).sum(axis=1) / w.sum(axis=1)
o = o.sort_values("TIS", ascending=False)
o.to_csv(sys.argv[2], index=False)
cols = ["player", "role", "club", "window", "pos_b", "pos_a", "cf_pos", "repl", "repl_role", "pts_added", "TIS"]
pd.set_option("display.width", 200)
print(o[cols].head(15).round(2).to_string(index=False))
print(o[cols].tail(15).round(2).to_string(index=False))

# ---------- 6. whole-tenure points above replacement (WAR-style) ----------
# Every season from arrival (incl. a January arrival's half season) until he leaves the club,
# the club leaves the PL, or the data ends (2024-25). Points are league points, via the team model.
ALLOWED = {"GK": {1}, "RB/RWB": {2}, "LB/LWB": {2}, "CB": {2}, "DEF?": {2}, "DM": {3}, "CM": {3},
           "AM": {3, 4}, "Wide R": {3, 4}, "Wide L": {3, 4}, "MID?": {3}, "CF": {4, 3}}
PTS = b_q[0] * 38                       # league points per unit of team quality

def tenure(code, club, start, role):
    seasons, first_type = [], None
    for s in range(start, 2025):
        h = ps[(ps.code == code) & (ps.season == s) & (ps.club == club)]
        tr = tq[(tq.season == s) & (tq.club == club)]
        if h.empty or tr.empty:
            break
        a, tr = h.iloc[0], tr.iloc[0]
        # FPL re-classifies players each season: a switch during the stint (or a type that
        # contradicts his listed role) means he moved, so judge that season on the broad role
        first_type = first_type or a.element_type
        moved = a.element_type != first_type or a.element_type not in ALLOWED[role]
        r = BROAD[a.element_type] if moved else role
        zp = (a.bps90 - mu[r]) / sd[r]
        share = a.minutes / tr.mins
        par = PTS * share * (zp - (repl[r] - mu[r]) / sd[r])     # vs replacement level
        squad = ps[(ps.season == s) & (ps.club == club) & (ps.code != code)]
        mate, pam = None, np.nan
        for fr in FAMILY[r]:
            m = squad[squad.grp == fr].sort_values("minutes", ascending=False)
            if len(m):
                below = m[m.minutes <= a.minutes]
                rp = (below if len(below) else m).iloc[0]
                zr, _ = projected_z(rp.code, s, rp.grp)
                mate, pam = rp.web, PTS * share * (zp - zr)        # vs next man in the squad
                break
        seasons.append(dict(code=code, club=club, season=f"{s}-{str(s + 1)[2:]}", role_played=r,
                            role_change=r != role, minutes=a.minutes, par=par, vs_mate=pam, mate=mate))
    return seasons

per = pd.DataFrame([x for _, r in o.drop_duplicates(["code", "club"]).iterrows()
                    for x in tenure(r.code, r.club, int(r.start), r.role)])
ten = per.groupby(["code", "club"]).agg(seasons=("season", "size"), first=("season", "first"),
                                        last=("season", "last"), minutes=("minutes", "sum"),
                                        PAR=("par", "sum"), vs_mate=("vs_mate", "sum"),
                                        role_changes=("role_change", "sum")).reset_index()
ten["PAR_per_season"] = ten.PAR / ten.seasons
ten["to_date"] = ten["last"] == "2024-25"
t2 = o.drop_duplicates(["code", "club"]).merge(ten, on=["code", "club"], how="left", suffixes=("", "_ten"))
t2["PAR"] = t2.PAR.fillna(0)

# One-and-done penalty: sold after one full season (a January arrival's half season doesn't count)
# while the club stayed up. Relegation-ended and still-at-club stints are exempt; short stints of
# two or more seasons keep their total. The penalty is the season he didn't give: the median PAR
# of a regular (900+ minute) season by a headline signing.
ONE_AND_DONE = per[per.minutes >= 900].par.median()
end = t2["last"].str[:4].astype(float)
stayed_up = [((tq.season == e + 1) & (tq.club == c)).any() for e, c in zip(end, t2.club)]
full = t2.seasons - t2.window.str.startswith("Jan").astype(int)
t2["exit"] = np.select([t2.to_date == True, ~np.array(stayed_up, dtype=bool)], ["still there", "relegated"], "left")
# FPL files each season under the club a player finished it with, so a January exit in season two
# looks like a summer sale. Transfermarkt's winter "out" records catch those (data runs to Jan 2023).
tout = pd.read_csv(f"{D}/premier-league.csv")
tout = tout[(tout.transfer_movement == "out") & (tout.transfer_period == "Winter")]
tout = set(zip(tout.club_name.replace(ALIAS), tout.player_name.map(norm), tout.year))
jan_exit = [e == e and (c, norm(p), int(e) + 1) in tout for c, p, e in zip(t2.club, t2.player, end)]
t2["one_and_done"] = (full <= 1) & (t2.exit == "left") & ~np.array(jan_exit)
t2["PAR_adj"] = t2.PAR - ONE_AND_DONE * t2.one_and_done
print(f"one-and-done penalty: {ONE_AND_DONE:.2f} points, applied to {t2.one_and_done.sum()}", file=sys.stderr)
t2 = t2.sort_values("PAR_adj", ascending=False)

# ---------- 7. who did he displace? ----------
# The player at the club with the most minutes in the signing's role in the season before his
# first season. His PAR that season is the bar the signing had to clear; upgrade = signing's PAR per
# season minus it. Promoted clubs have no PL data for that season, so they get no comparison.
def season_par(code, club, s):
    h = ps[(ps.code == code) & (ps.season == s) & (ps.club == club)]
    tr = tq[(tq.season == s) & (tq.club == club)]
    if h.empty or tr.empty:
        return np.nan
    a, r = h.iloc[0], h.iloc[0].grp
    return PTS * a.minutes / tr.iloc[0].mins * ((a.bps90 - repl[r]) / sd[r])

def displaced(r):
    s = int(r.first[:4]) - 1 if isinstance(r.first, str) else int(r.start) - 1
    squad = ps[(ps.season == s) & (ps.club == r.club) & (ps.code != r.code) & (ps.minutes > 0)]
    if squad.empty:
        return dict(displaced_note="club not in PL the season before")
    for fr in FAMILY[r.role]:
        m = squad[squad.grp == fr].sort_values("minutes", ascending=False)
        if len(m):
            break
    else:
        return dict(displaced_note="nobody in that role")
    d = m.iloc[0]
    nxt = ps[(ps.code == d.code) & (ps.season == s + 1)]
    fate = ("stayed" if len(nxt) and nxt.club.iloc[0] == r.club else
            f"left for {nxt.club.iloc[0]}" if len(nxt) and isinstance(nxt.club.iloc[0], str) else "left the PL")
    # did the signing actually take his place? He left, or his minutes at least halved.
    kept = fate == "stayed" and nxt.minutes.iloc[0] >= 0.5 * d.minutes
    # his level over the three seasons before the signing's first, at whichever PL club he was
    hist = [season_par(d.code, c, y) for y in range(s - 2, s + 1)
            for c in ps[(ps.code == d.code) & (ps.season == y)].club.dropna()]
    hist = [h for h in hist if h == h]
    return dict(displaced=d.web.title(), displaced_code=d.code, displaced_role=d.grp, displaced_min=d.minutes,
                displaced_PAR_3yr=np.mean(hist) if hist else np.nan, displaced_seasons_3yr=len(hist),
                displaced_min_next=nxt.minutes.iloc[0] if fate == "stayed" else 0,
                took_his_place=not kept,
                displaced_PAR=season_par(d.code, r.club, s), displaced_fate=fate)

t2 = pd.concat([t2.reset_index(drop=True),
                pd.DataFrame([displaced(r) for _, r in t2.iterrows()])], axis=1)
t2["upgrade_per_season"] = t2.PAR_per_season - t2.displaced_PAR
# 2019+ view: signing's per-season PAR (after the one-and-done penalty) vs the incumbent's 3-season average
t2["upgrade_3yr"] = t2.PAR_adj / t2.seasons - t2.displaced_PAR_3yr
t2["since_2019"] = t2["first"].str[:4].astype(float).ge(2019) & ~((t2["first"] == "2018-19"))
print(t2.displaced.notna().sum(), "with an incumbent;", t2.took_his_place.sum(), "took his place;", t2.displaced_fate.value_counts().to_dict(), file=sys.stderr)
t2.to_csv(sys.argv[3], index=False)
per.merge(o[["code", "club", "player"]], on=["code", "club"]).to_csv(sys.argv[4], index=False)
tc = ["player", "role", "club", "window", "first", "last", "seasons", "PAR", "one_and_done",
      "PAR_adj", "PAR_per_season", "displaced", "displaced_PAR", "upgrade_per_season", "displaced_fate"]
print(t2[tc].head(15).round(1).to_string(index=False))
print(t2[tc].tail(15).round(1).to_string(index=False))

# ---------- 8. bootstrap uncertainty intervals ----------
# Two sources of noise: (1) each player-season's output is an average over a few dozen matches,
# so his matches are resampled with replacement; (2) the team model's slope is re-fitted on
# resampled club-seasons. Minutes played, roles, replacement level and the penalty stay fixed.
if len(sys.argv) > 5:
    B, rng = 1000, np.random.default_rng(2026)
    games = []
    for y in range(2016, 2025):
        tag = f"{y}-{str(y + 1)[2:]}"
        try:
            g = pd.read_csv(f"{D}/gw_{tag}.csv", usecols=["element", "minutes", "bps"])
        except UnicodeDecodeError:
            g = pd.read_csv(f"{D}/gw_{tag}.csv", usecols=["element", "minutes", "bps"], encoding="latin-1")
        ids = pd.read_csv(f"{D}/players_raw_{tag}.csv", usecols=["id", "code"]).set_index("id").code
        g = g[g.minutes > 0].assign(season=y, code=lambda x: x.element.map(ids))
        games.append(g)
    games = {k: (v.bps.to_numpy(float), v.minutes.to_numpy(float))
             for k, v in pd.concat(games).groupby(["season", "code"])}

    def boot_bps90(code, s, minutes, grp, base):
        """B draws of the shrunk BPS/90 for one player-season."""
        if (s, code) not in games or minutes <= 0:
            return np.full(B, base)
        bps, mins = games[(s, code)]
        idx = rng.integers(0, len(bps), (B, len(bps)))
        rate = bps[idx].sum(1) / np.maximum(mins[idx].sum(1), 1) * 90
        return (rate * minutes / 90 + 5 * pos_mean[grp]) / (minutes / 90 + 5)

    i = rng.integers(0, len(tq), (B, len(tq)))
    slope_b = np.array([np.polyfit(tq.Q.to_numpy()[k], tq.ppg.to_numpy()[k], 1)[0] for k in i]) * 38

    def boot_par(code, club, s, role):
        """B draws of one season's points above replacement, judged on `role`."""
        h = ps[(ps.code == code) & (ps.season == s) & (ps.club == club)]
        tr = tq[(tq.season == s) & (tq.club == club)]
        if h.empty or tr.empty:
            return None
        a = h.iloc[0]
        b90 = boot_bps90(code, s, a.minutes, a.grp, a.bps90)
        return slope_b * a.minutes / tr.iloc[0].mins * (b90 - repl[role]) / sd[role]

    # lifetime PAR for every signing
    per["s"] = per.season.str[:4].astype(int)
    draws = {}
    for (code, club), g in per.groupby(["code", "club"]):
        draws[(code, club)] = sum(boot_par(code, club, r.s, r.role_played) for _, r in g.iterrows())
    pen = ONE_AND_DONE * t2.one_and_done.to_numpy()
    M = np.vstack([draws.get((c, k), np.zeros(B)) for c, k in zip(t2.code, t2.club)]) - pen[:, None]
    ranks = (-M).argsort(0).argsort(0) + 1
    q = lambda x: np.percentile(x, [5, 95], axis=1)
    t2["PAR_adj_lo"], t2["PAR_adj_hi"] = q(M)
    t2["rank"] = np.arange(1, len(t2) + 1)
    t2["rank_lo"], t2["rank_hi"] = q(ranks)
    t2["p_top10"] = (ranks <= 10).mean(1)
    t2["p_bottom10"] = (ranks > len(t2) - 10).mean(1)

    # 3-season incumbent upgrade, signings from summer 2019
    U = np.full((len(t2), B), np.nan)
    for j, r in enumerate(t2.itertuples()):
        if not (r.since_2019 and r.upgrade_3yr == r.upgrade_3yr):
            continue
        s = int(r.first[:4]) - 1
        hist = [boot_par(r.displaced_code, c, y, ps[(ps.code == r.displaced_code) & (ps.season == y)].grp.iloc[0])
                for y in range(s - 2, s + 1)
                for c in ps[(ps.code == r.displaced_code) & (ps.season == y)].club.dropna()]
        hist = [h for h in hist if h is not None]
        U[j] = M[j] / r.seasons - np.mean(hist, axis=0)
    ok = ~np.isnan(U[:, 0])
    t2["upgrade_3yr_lo"], t2["upgrade_3yr_hi"] = np.nan, np.nan
    t2.loc[ok, "upgrade_3yr_lo"], t2.loc[ok, "upgrade_3yr_hi"] = q(U[ok])
    t2["p_upgrade"] = np.where(ok, (U > 0).mean(1), np.nan)
    ur = np.full(U.shape, np.nan)
    ur[ok] = (-U[ok]).argsort(0).argsort(0) + 1
    t2["upgrade_rank_lo"], t2["upgrade_rank_hi"] = np.nan, np.nan
    t2.loc[ok, "upgrade_rank_lo"], t2.loc[ok, "upgrade_rank_hi"] = q(ur[ok])
    print(f"bootstrap: slope 90% {np.percentile(slope_b / 38, [5, 95]).round(2)}", file=sys.stderr)
    t2.to_csv(sys.argv[5], index=False)
