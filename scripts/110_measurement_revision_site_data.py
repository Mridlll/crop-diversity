"""
110_measurement_revision_site_data.py

Site data for two new scenes on the measurement page and the revised national
series on the overview.

  (1) Count of crops with a minimum share of land. Area-weighted series for the
      districts reporting in every year of the analysis window (scripts/_years.py),
      from the district-year panel written by 104_d0_threshold_and_1997.py
      (outputs/revision_nov8/d0_threshold_panel.csv).
      A crop counts only where it holds at least 0.1, 0.5 or 1 percent of the
      district's gross cropped area that year. 1997-98 is dropped (see the
      1997-98 follow-ups in 108 and the referee checks in 109).

  (2) Season-separated diversity. Exact split of annual Shannon entropy into a
      within-season part and a between-season part, as in
      104_season_separated_diversity.py, under two treatments of Whole Year
      crops (sugarcane, coconut, banana):
        own   Whole Year as its own block (script 104's A_alpha)
        mon   Whole Year merged into the monsoon block, since those crops stand
              in the field alongside kharif crops (the referee's main variant)
      Season-reliable district-years only (script 104's season_ok flag),
      districts with at least 5 such years, inside the 606-district sample.
      District value = mean over its years of exp(H_alpha). Cropping systems
      come from site_monoculture.json (script 96).

Cleaning is script 104's (itself script 76's), executed from its source so the
two cannot drift.

Usage:
  C:/Users/mridu/anaconda3/envs/geo_env/python.exe scripts/110_measurement_revision_site_data.py
Output:
  docs/data/site_measurement_rev.json
  outputs/revision_nov8/measurement_rev_log.txt
"""
import json
import os

import numpy as np
import pandas as pd

from _years import keep_years, FIRST_YEAR, LAST_YEAR, N_YEARS, WINDOW_LABEL

REPO = r"D:/crop-diversity"
OUT = REPO + "/outputs/revision_nov8"
SITE = REPO + "/docs/data/site_measurement_rev.json"
LOG = []


def w(s=""):
    print(s)
    LOG.append(str(s))


# ------------------------------------------------------------------ (1) thresholds
# District-year counts at each threshold, from 104_d0_threshold_and_1997.py. The
# balanced panel is every district observed in all years of the analysis window.
tp = keep_years(pd.read_csv(OUT + "/d0_threshold_panel.csv"))
cnt = tp.groupby("district_key")["year_start"].nunique()
bp = tp[tp["district_key"].isin(cnt[cnt == N_YEARS].index)]
N_BAL = int(bp["district_key"].nunique())


def _aw(gg, c):
    return float(np.average(gg[c], weights=gg["gca"]))


series = [{"year": int(y), "raw": round(_aw(gg, "D0_raw"), 3), "t01": round(_aw(gg, "D0_0.1%"), 3),
           "t05": round(_aw(gg, "D0_0.5%"), 3), "t1": round(_aw(gg, "D0_1%"), 3), "D1": round(_aw(gg, "D1"), 3)}
          for y, gg in bp.groupby("year_start")]
by = {s["year"]: s for s in series}
first, last = by[FIRST_YEAR], by[LAST_YEAR]
w("Area-weighted, {} districts, {}".format(N_BAL, WINDOW_LABEL))
for k in ["raw", "t01", "t05", "t1", "D1"]:
    w("  {:4s} {:6.2f} -> {:6.2f}  ({:+.2f})".format(k, first[k], last[k], last[k] - first[k]))
under01 = (first["raw"] - first["t01"], last["raw"] - last["t01"])
w("  crops under 0.1% of land: {:.2f} -> {:.2f}".format(*under01))
start = {y: round(last["raw"] - by[y]["raw"], 2) for y in [1998, 1999, 2002]}
w("  raw rise to 2019 by start year: {}".format(start))


def win(k, a, b):
    return float(np.mean([by[y][k] for y in range(a, b + 1)]))


window = {k: round(win(k, 2013, 2019) - win(k, 1998, 2004), 3) for k in ["raw", "t01", "t05", "t1", "D1"]}
w("  window change (2013-19 mean minus 1998-2004 mean): {}".format(window))

# ------------------------------------------------------------------ (2) seasons
src = open(REPO + "/scripts/104_season_separated_diversity.py", encoding="utf-8").read()
src = src.split("# ------------------------------------------------------------------ helpers")[0]
src = src.split('"""', 2)[2]
ns = {}
exec(src, ns)
df = ns["df"]


def H(a):
    a = np.asarray(a, float)
    a = a[a > 0]
    if a.size == 0:
        return np.nan
    p = a / a.sum()
    return float(-(p * np.log(p)).sum())


cs = (df.groupby(["district_key", "year_start", "slot", "crop_name"])["area"].sum()
      .unstack("slot", fill_value=0.0))
for s in "KRSW":
    if s not in cs.columns:
        cs[s] = 0.0
rec = []
for (dk, y), g in cs.groupby(level=[0, 1]):
    g = g.droplevel([0, 1])
    tot = g.values.sum()
    if tot <= 0:
        continue
    ann = g.sum(axis=1).values
    r = {"district_key": dk, "year_start": y, "H_ann": H(ann)}
    for tag, blocks in [("own", {"K": g["K"], "R": g["R"], "S": g["S"], "W": g["W"]}),
                        ("mon", {"K": g["K"] + g["W"], "R": g["R"], "S": g["S"]})]:
        ha = 0.0
        for v in blocks.values():
            a = v.sum()
            if a > 0:
                ha += a / tot * H(v.values)
        r["Ha_" + tag] = ha
    rec.append(r)
A = pd.DataFrame(rec)

dy = pd.read_csv(OUT + "/seasons_district_year.csv", usecols=["district_key", "year_start", "season_ok",
                                                             "D1_ann", "A_alpha"])
A = A.merge(dy, on=["district_key", "year_start"], how="inner")
assert np.allclose(np.exp(A["H_ann"]), A["D1_ann"]), "annual D1 does not reproduce script 104"
assert np.allclose(np.exp(A["Ha_own"]), A["A_alpha"]), "within-season part does not reproduce script 104"
A = A[A["season_ok"]]
A["W_own"], A["W_mon"] = np.exp(A["Ha_own"]), np.exp(A["Ha_mon"])
g = A.groupby("district_key")
D = pd.DataFrame({"n": g.size(), "ann": g["D1_ann"].mean(), "own": g["W_own"].mean(),
                  "mon": g["W_mon"].mean(),
                  "bs_own": 1 - g["Ha_own"].sum() / g["H_ann"].sum(),
                  "bs_mon": 1 - g["Ha_mon"].sum() / g["H_ann"].sum()})
D = D[D["n"] >= 5]

sd = pd.read_csv(OUT + "/seasons_district.csv", usecols=["district_key", "in_final"])
D = D.reset_index().merge(sd, on="district_key", how="left")
D = D[D["in_final"] == True].copy()
mj = json.load(open(REPO + "/docs/data/site_monoculture.json", encoding="utf-8"))["districts"]
meta = pd.DataFrame([{"district_key": v["dk"], "d": v["d"], "s": v["s"], "sys": v["sys"]} for v in mj.values()])
D = D.merge(meta, on="district_key", how="left")
w("\nSeason sample: {} districts".format(len(D)))
assert len(D) > 540, len(D)

PAIRS = ['maize-wheat', 'soybean-wheat', 'bajra-wheat', 'bajra-mustard', 'rice-pulse', 'rice-oilseed',
         'rice-wheat-light', 'cotton-other', 'maize-other', 'soybean-gram', 'soybean-other', 'rice-other',
         'bajra-other']


def grp(s):
    if s == "rice-wheat":
        return "rw"
    if s == "rice-rice":
        return "rr"
    if s == "rice-only":
        return "r1"
    if s in PAIRS:
        return "pr"
    if s == "whole":
        return "wh"
    return "ot"


D["g"] = D["sys"].map(grp)
GROUPS = [("all", "all districts"), ("rw", "rice then wheat"), ("rr", "rice twice a year"),
          ("pr", "another two-season pair"), ("r1", "rice, little second season"),
          ("ot", "one main season or mixed")]
rows = []
for key, name in GROUPS:
    s = D if key == "all" else D[D["g"] == key]
    row = {"key": key, "name": name, "n": int(len(s))}
    for c in ["ann", "own", "mon", "bs_own", "bs_mon"]:
        row[c] = round(float(s[c].mean()), 4)
        row[c + "_med"] = round(float(s[c].median()), 4)
    rows.append(row)
    w("  {:4s} n={:3d} annual {:.2f}  within(own) {:.2f}  within(mon) {:.2f}  between share {:.3f}/{:.3f}".format(
        key, row["n"], row["ann"], row["own"], row["mon"], row["bs_own"], row["bs_mon"]))

# rank movers: annual against within-season, both treatments
for tag in ["own", "mon"]:
    ra = D["ann"].rank(ascending=False)
    rw_ = D[tag].rank(ascending=False)
    D["mv_" + tag] = ra - rw_          # negative = falls when seasons are separated
    fall50 = D.nsmallest(50, "mv_" + tag)
    w("  [{}] rice-then-wheat among 50 largest fallers: {}; mean move of rice-then-wheat {:.1f}; "
      "rice-twice median move {:.1f}".format(tag, int((fall50["g"] == "rw").sum()),
                                             D.loc[D["g"] == "rw", "mv_" + tag].mean(),
                                             D.loc[D["g"] == "rr", "mv_" + tag].median()))


# worked examples: a rice-then-wheat district and a rice-twice district at the low end
EXAMPLES = [("rw", "PUNJAB|LUDHIANA"), ("rr", "WEST BENGAL|PURBA BARDHAMAN")]
ex = []
for key, dk in EXAMPLES:
    r = D[D["district_key"] == dk]
    assert len(r) == 1 and r["g"].iloc[0] == key, dk
    r = r.iloc[0]
    ex.append({"key": key, "d": r["d"], "s": r["s"], "ann": round(r["ann"], 4), "own": round(r["own"], 4),
               "mon": round(r["mon"], 4), "n": int(r["n"])})
    w("  example {}: {}, {} annual {:.2f} within {:.2f}/{:.2f} over {} years".format(
        key, r["d"], r["s"], r["ann"], r["own"], r["mon"], int(r["n"])))

out = {
    "threshold": {"n": N_BAL, "series": series, "under01": [round(under01[0], 2), round(under01[1], 2)],
                  "rise_by_start": {str(k): v for k, v in start.items()}},
    "seasons": {"groups": rows, "examples": ex,
                "districts": [{"d": r.d, "s": r.s, "g": r.g, "ann": round(r.ann, 3), "own": round(r.own, 3),
                               "mon": round(r.mon, 3)} for r in D.itertuples()]},
}
with open(SITE, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
w("\nWROTE " + SITE)
with open(OUT + "/measurement_rev_log.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(LOG) + "\n")
