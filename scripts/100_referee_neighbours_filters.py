"""
100_referee_neighbours_filters.py

Referee check on 99_crop_groups_neighbours.py: sensitivity of the main lag coefficient
to the defect filter and to the first sample year. Reuses script 99's data build.
  F1 main (flag_annual_unreliable)
  F2 + drop f_gca_jump_mild unit-years
  F3 + drop 1997-98 levels (so first change is 1998-99)
  F4 + drop crop-zero level years (a group at exactly 0 in a unit-year where its
       unit median share is above 1 percent: a reporting gap, not a real zero)
  F5 + drop the Tamil Nadu single-season and whole-year >=20% unit-years
Output: outputs/crop_groups/referee_neighbours_filters.csv (+ printed)
Run: C:/Users/mridu/anaconda3/envs/geo_env/python.exe scripts/100_referee_neighbours_filters.py
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "scripts", "99_crop_groups_neighbours.py")
G = {"__file__": SRC, "__name__": "ref"}
exec(compile(open(SRC, encoding="utf-8").read().split("rows = []")[0], "99_head", "exec"), G)
pa, grid, GROUPS, nb_idx = G["pa"], G["grid"], G["GROUPS"], G["nb_idx"]
diff, shift, wavg, fit, RAIN, RN = G["diff"], G["shift"], G["wavg"], G["fit"], G["RAIN"], G["RAIN_NAMES"]
years = G["years"]
rc = {k: RAIN[k] for k in RN}

base = pa.flag_annual_unreliable == 0
f2 = base & (pa.f_gca_jump_mild == 0)
f3 = f2 & (pa.year_start > years[0])
f5 = f3 & (pa.f_tn_rice_single_season == 0) & (pa.f_whole_year_ge20 == 0)
rows = []
for g in ["pulses", "millets", "oilseeds", "maize", "other"]:
    col = f"share_gca_{g}"
    med = pa.groupby("unit_key")[col].transform("median")
    gap = (pa[col] == 0) & (med > 0.01)
    for lab, v in [("F1 main", base), ("F2 +mild GCA jump", f2), ("F3 +drop 1997-98", f3),
                   ("F4 +drop zero-gap years", f3 & ~gap), ("F5 F3 + TN rice + whole-year>=20%", f5)]:
        Sg = grid(pa, col, v)
        D = diff(Sg); WD = wavg(D, nb_idx)
        o = fit(D, {"WD_l1": shift(WD, 1), "own_l1": shift(D, 1), **rc})
        rows.append({"group": g, "filter": lab, "b": o["WD_l1_b"], "se": o["WD_l1_se"],
                     "p": o["WD_l1_p"], "n": o["n"]})
        print(f"{g:9s} {lab:36s} {o['WD_l1_b']:.4f} ({o['WD_l1_se']:.4f}) p={o['WD_l1_p']:.3f} n={o['n']}")
pd.DataFrame(rows).to_csv(os.path.join(G["OUT"], "referee_neighbours_filters.csv"), index=False)
