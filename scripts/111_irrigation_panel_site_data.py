"""
111_irrigation_panel_site_data.py

Site data for the "inside each district" scene on docs/irrigation.html.

Reads the district-year irrigation panel built by 104 (DES land-use statistics, net
irrigated over net sown area, 1998-99 to 2019-20) and the estimates written by 105 and
108, and writes the handful of numbers the scene draws:

  across   cross-section of district means with state fixed effects (re-estimated
           here so the 2x2 covariance of the linear and squared terms is available
           for the confidence band of the slope); errors clustered on 2011 parents
  within   straight-line slope with district and state x year fixed effects (105,
           "main, linear"), plus the same with state clusters (108)
  years    the cross-section re-estimated year by year (peak and se)
  longdiff 1998-2000 to 2017-2019 long differences with state fixed effects (108):
           effective number of crops and crops holding at least 0.25, 0.5, 1 and 2
           percent of cropped area, and the leading crop's share
  shape    the between-district slope term (m x dev) and the own-path term (dev^2)
           from 108, and the main quadratic's squared term with state clusters

Output: docs/data/site_irrigation_panel.json

Usage:
  C:/Users/mridu/anaconda3/envs/geo_env/python.exe scripts/111_irrigation_panel_site_data.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = r"D:/crop-diversity"
REV = REPO + "/outputs/revision_nov8"
OUT = REPO + "/docs/data/site_irrigation_panel.json"
sys.path.insert(0, REPO + "/scripts")
from _revision_fe import fe_ols, pval, turning_point  # noqa: E402


def r6(v):
    v = float(v)
    return None if (np.isnan(v) or np.isinf(v)) else round(v, 6)


# ------------------------------------------------------------------ panel
P = pd.read_csv(REV + "/irrigation_panel_district_year.csv")
P = P[P["in_final"] & P["year_start"].between(1998, 2019)].copy()
P["irr"] = P["irr_nsa"]

# ------------------------------------------------- across districts (district means)
X = (P.groupby("district_key")
     .agg(D1=("D1", "mean"), irr=("irr", "mean"), state_name=("state_name", "first"),
          shrug_key=("shrug_key", "first"))
     .reset_index().dropna(subset=["D1", "irr"]))
X["x"], X["x2"] = X["irr"], X["irr"] ** 2
r = fe_ols(X, "D1", ["x", "x2"], ["state_name"], "shrug_key")
b1, b2 = float(r["params"]["x"]), float(r["params"]["x2"])
V = r["vcov"].loc[["x", "x2"], ["x", "x2"]].to_numpy()
tp, tp_se = turning_point(r, "x", "x2")
# guard: must reproduce 105's cross-section on the land-use mean
est = pd.read_csv(REV + "/irrigation_panel_estimates.csv")
ref = est[(est["spec"] == "XS state FE: LUS 1998-2019 mean") & (est["y"] == "D1")].iloc[0]
assert abs(b1 - ref["b1"]) < 1e-8 and abs(b2 - ref["b2"]) < 1e-8, "cross-section does not reproduce 105"
print("across: b1 {:+.3f} b2 {:+.3f} peak {:.3f} ({:.3f}) n={}".format(b1, b2, tp, tp_se, r["n"]))

# ------------------------------------------------- year-by-year cross-sections
years = []
for yv, g in P.dropna(subset=["D1", "irr"]).groupby("year_start"):
    g = g.copy()
    g["x"], g["x2"] = g["irr"], g["irr"] ** 2
    ry = fe_ols(g, "D1", ["x", "x2"], ["state_name"], "shrug_key")
    t, ts = turning_point(ry, "x", "x2")
    years.append({"year": int(yv), "n": int(ry["n"]), "b2": r6(ry["params"]["x2"]),
                  "b2_se": r6(ry["se"]["x2"]), "peak": r6(t), "peak_se": r6(ts)})
pk = [y["peak"] for y in years]
print("year peaks {:.3f} to {:.3f}, median {:.3f}".format(min(pk), max(pk), float(np.median(pk))))

# ------------------------------------------------- within districts, straight line
lin = est[(est["spec"] == "main, linear") & (est["y"] == "D1")].iloc[0]
quad = est[(est["spec"] == "district FE + state x year FE") & (est["y"] == "D1")].iloc[0]
rf = pd.read_csv(REV + "/irrigation_referee.csv")


def rrow(test, y):
    s = rf[(rf["test"] == test) & (rf["y"] == y)]
    assert len(s) == 1, (test, y)
    return s.iloc[0]


lin_st = rrow("linear, state clusters", "D1")
within = {"b": r6(lin["b1"]), "se": r6(lin["se1"]), "p": r6(lin["p1"]),
          "se_state": r6(lin_st["se_irr"]), "p_state": r6(lin_st["p_irr"]),
          "n": int(lin["n"]), "districts": int(lin["units"])}
print("within: {:+.3f} ({:.3f}); per 10 points {:+.4f}, 95% {:+.4f} to {:+.4f}".format(
    within["b"], within["se"], within["b"] / 10,
    (within["b"] - 1.96 * within["se"]) / 10, (within["b"] + 1.96 * within["se"]) / 10))

dec = rrow("decomp irr + m*dev + dev^2", "D1")
qst = rrow("main, state clusters", "D1")
shape = {"mdev": r6(dec["b_mdev"]), "mdev_se": r6(dec["se_mdev"]), "mdev_p": r6(dec["p_mdev"]),
         "dev2": r6(dec["b_dev2"]), "dev2_se": r6(dec["se_dev2"]), "dev2_p": r6(dec["p_dev2"]),
         "quad_b2": r6(quad["b2"]), "quad_b2_se": r6(quad["se2"]),
         "quad_b2_p_state": r6(qst["p_irr2"])}

# ------------------------------------------------- long differences
LD = [("D1", "dD1", "effective number of crops"),
      ("D0_025", "dD0_025", "crops on at least 0.25% of cropped area"),
      ("D0_05", "dD0_05", "crops on at least 0.5% of cropped area"),
      ("D0_1", "dD0_1", "crops on at least 1% of cropped area"),
      ("D0_2", "dD0_2", "crops on at least 2% of cropped area")]
longdiff = []
for key, y, name in LD:
    a = rrow("LD state FE", y)
    b = rrow("LD state FE + baseline irr", y)
    longdiff.append({"key": key, "name": name, "n": int(a["n"]),
                     "b": r6(a["b_dirr"]), "se": r6(a["se_dirr"]), "p": r6(a["p_dirr"]),
                     "b_base": r6(b["b_dirr"]), "se_base": r6(b["se_dirr"]), "p_base": r6(b["p_dirr"])})
    print("LD {:7s} {:+.3f} ({:.3f}) p={:.3f}; + baseline irr {:+.3f} ({:.3f}) p={:.3f}".format(
        key, a["b_dirr"], a["se_dirr"], a["p_dirr"], b["b_dirr"], b["se_dirr"], b["p_dirr"]))
top = rrow("LD state FE", "dtop_share")
ldd = pd.read_csv(REV + "/irrigation_panel_longdiff_data.csv")
dirr_mean = float(ldd["dirr"].mean()) if "dirr" in ldd else None

out = {
    "note": ("Irrigation is DES land-use net irrigated area over net sown area. 'across' is the "
             "cross-section of 1998-2019 district means with state fixed effects; 'within' is the "
             "straight-line slope with district and state x year fixed effects; 'longdiff' slopes "
             "are per unit change in the irrigated share between 1998-2000 and 2017-2019, state "
             "fixed effects. Errors clustered on 2011 parent districts unless marked _state."),
    "across": {"b1": r6(b1), "b2": r6(b2), "V": [[r6(v) for v in row] for row in V],
               "peak": r6(tp), "peak_se": r6(tp_se), "n": int(r["n"]),
               "b2_p": r6(pval(b2, float(r["se"]["x2"]), r["G"]))},
    "years": years,
    "within": within,
    "shape": shape,
    "longdiff": longdiff,
    "top_share": {"b": r6(top["b_dirr"]), "se": r6(top["se_dirr"]), "p": r6(top["p_dirr"])},
    "ld_mean_change": r6(dirr_mean) if dirr_mean is not None else None,
    "irr_sd": {"overall": r6(P["irr"].std()),
               "within": r6((P["irr"] - P.groupby("district_key")["irr"].transform("mean")).std())},
}
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(out, f, separators=(",", ":"))
print("WROTE {} ({:,} bytes)".format(OUT, os.path.getsize(OUT)))
