"""
109_d0_referee_checks.py

Referee checks on 104_d0_threshold_and_1997.py and 108_d0_1997_followups.py.
Reads the district-year panel 104 wrote (d0_threshold_panel.csv) and the raw APY with
script 76's cleaning (copied from 108). Writes outputs/revision_nov8/d0_referee_raw.md.

Checks:
 1. State decomposition of the 1998->2019 area-weighted raw D0 rise: within-state
    (fixed weights) versus weight shifts. 108's "MP +2.92" mixes the two.
 2. "Drop collapsed state-years" endpoint on a COMMON district set (108 drops a
    district-year only on the side where it collapsed, so the 1998 and 2019 means
    cover different districts).
 3. Inference clustered on STATE (reporting shocks are state-year), cluster bootstrap.
 4. Guar Seed / Moth in Rajasthan; Other Oilseeds origin; Odisha 1997-98 seasons;
    Madhya Pradesh 1998-99 crop list.
 5. Balanced panel without the 1997 requirement (1998-2019) and which states it adds.
 6. How much of year-on-year raw D0 change is state-year common (reporting-shock claim).
 7. The top-10 gainer list quoted in d0_results.md against the CSV.

Usage:
  C:/Users/mridu/anaconda3/envs/geo_env/python.exe scripts/109_d0_referee_checks.py
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

RAW = r"E:/CEEW Project/outputs/all_crops_apy_1997_2021_india_data_portal.csv"
REPO = r"D:/crop-diversity"
OUT = REPO + "/outputs/revision_nov8"
R = []


def w(s=""):
    print(s)
    R.append(str(s))


rng = np.random.default_rng(109)
NB = 2000
MEAS = ["D0_raw", "D0_0.1%", "D0_0.5%", "D0_1%", "D1"]

pan = pd.read_csv(OUT + "/d0_threshold_panel.csv")
pan["state"] = pan["district_key"].str.split("|").str[0]
YRS = sorted(pan["year_start"].unique())
cnt = pan.groupby("district_key")["year_start"].nunique()
bal = set(cnt[cnt == len(YRS)].index)
bp = pan[pan["district_key"].isin(bal)].copy()
w("# Referee checks on the D0 analysis")
w("")
w("Balanced panel {} districts, {} states, {} parent clusters.".format(
    len(bal), bp["state"].nunique(), bp["cl"].nunique()))
w("")


def aw(g, m):
    return np.average(g[m], weights=g["gca"])


# ------------------------------------------------------------------ 1. state decomposition
a = bp[bp["year_start"] == 1998]
b = bp[bp["year_start"] == 2019]
G98, G19 = a["gca"].sum(), b["gca"].sum()
rows = []
for st in sorted(bp["state"].unique()):
    x, y = a[a["state"] == st], b[b["state"] == st]
    s98, s19 = x["gca"].sum() / G98, y["gca"].sum() / G19
    m98, m19 = aw(x, "D0_raw"), aw(y, "D0_raw")
    rows.append({"state": st, "n": len(x), "w98": s98, "w19": s19, "awD0_98": m98,
                 "awD0_19": m19, "contrib_108": s19 * m19 - s98 * m98,
                 "within": 0.5 * (s98 + s19) * (m19 - m98),
                 "between": (s19 - s98) * 0.5 * (m98 + m19)})
dec = pd.DataFrame(rows).sort_values("within", ascending=False)
tot = aw(b, "D0_raw") - aw(a, "D0_raw")
w("## 1. State decomposition of the +{:.2f} endpoint rise".format(tot))
w("")
w("within = average weight x change in the state's area-weighted D0; between = weight "
  "shift x average D0. Sum within {:+.2f}, sum between {:+.2f} (between terms net to "
  "the mean-level effect of reweighting).".format(dec["within"].sum(), dec["between"].sum()))
w("")
w("| state | n | weight 98 | weight 19 | aw D0 98 | aw D0 19 | 108 contribution | within | between |")
w("|---|---|---|---|---|---|---|---|---|")
for _, r in dec.iterrows():
    if abs(r["contrib_108"]) > 0.05 or abs(r["within"]) > 0.05:
        w("| {} | {} | {:.3f} | {:.3f} | {:.1f} | {:.1f} | {:+.2f} | {:+.2f} | {:+.2f} |".format(
            r["state"].title(), r["n"], r["w98"], r["w19"], r["awD0_98"], r["awD0_19"],
            r["contrib_108"], r["within"], r["between"]))
w("")
# counterfactual: hold MP at 1999 level instead of 1998
for st in ["MADHYA PRADESH"]:
    sub = bp[bp["state"] == st]
    w("{} area weight by year: 1997 {:.3f}, 1998 {:.3f}, 1999 {:.3f}, 2019 {:.3f}.".format(
        st.title(), *[sub.loc[sub.year_start == y, "gca"].sum() / bp.loc[bp.year_start == y, "gca"].sum()
                       for y in [1997, 1998, 1999, 2019]]))
w("")

# ------------------------------------------------------------------ 2. common district set
col = pd.read_csv(OUT + "/d0_state_collapses.csv")
colset = set(zip(col["state_name"].str.upper(), col["year_start"]))
bp["collapse"] = [(s, y) in colset for s, y in zip(bp["state"], bp["year_start"])]


def boot_endpoint(d, clcol, m, y0=1998, y1=2019, nb=NB):
    """aw change y0->y1 on districts present (non-missing) in BOTH years."""
    x0 = d[(d.year_start == y0) & d[m].notna()].set_index("district_key")
    x1 = d[(d.year_start == y1) & d[m].notna()].set_index("district_key")
    k = x0.index.intersection(x1.index)
    x0, x1 = x0.loc[k], x1.loc[k]
    cls = np.array(sorted(x0[clcol].unique()))
    ci = pd.Series(np.arange(len(cls)), index=cls)
    i0 = ci.loc[x0[clcol]].values
    nC = len(cls)
    S0 = np.bincount(i0, weights=x0["gca"] * x0[m], minlength=nC)
    G0 = np.bincount(i0, weights=x0["gca"], minlength=nC)
    S1 = np.bincount(i0, weights=x1["gca"] * x1[m], minlength=nC)
    G1 = np.bincount(i0, weights=x1["gca"], minlength=nC)
    pt = S1.sum() / G1.sum() - S0.sum() / G0.sum()
    cnts = rng.multinomial(nC, np.ones(nC) / nC, size=nb).astype(float)
    bt = cnts @ S1 / (cnts @ G1) - cnts @ S0 / (cnts @ G0)
    return pt, bt.std(), np.quantile(bt, .025), np.quantile(bt, .975), len(k), nC


def boot_window(d, clcol, m, weighted=True, nb=NB):
    e = d[d.year_start.between(1998, 2004) & d[m].notna()].groupby("district_key").agg(
        v=(m, "mean"), a=("gca", "mean"), c=(clcol, "first"))
    l_ = d[d.year_start.between(2013, 2019) & d[m].notna()].groupby("district_key")[m].mean()
    dd = e.join(l_.rename("v2"), how="inner")
    dd["d"] = dd["v2"] - dd["v"]
    wts = dd["a"].values if weighted else np.ones(len(dd))
    cls = np.array(sorted(dd["c"].unique()))
    ci = pd.Series(np.arange(len(cls)), index=cls)
    ii = ci.loc[dd["c"]].values
    nC = len(cls)
    S = np.bincount(ii, weights=wts * dd["d"], minlength=nC)
    G = np.bincount(ii, weights=wts, minlength=nC)
    cnts = rng.multinomial(nC, np.ones(nC) / nC, size=nb).astype(float)
    bt = cnts @ S / (cnts @ G)
    return S.sum() / G.sum(), bt.std(), np.quantile(bt, .025), np.quantile(bt, .975), len(dd), nC


def fmt(t):
    return "{:+.2f} ({:.2f}) [{:+.2f}, {:+.2f}]".format(*t[:4])


w("## 2. Endpoint with collapsed state-years removed from BOTH endpoints")
w("")
w("108 sets collapsed district-years missing and then compares all 1998 districts "
  "present against all 2019 districts present, so Madhya Pradesh, Andhra Pradesh, "
  "Chhattisgarh and Nagaland enter 2019 but not 1998. Here both endpoints use the "
  "same districts.")
w("")
w("| version | " + " | ".join(MEAS) + " | districts |")
w("|---|" + "---|" * (len(MEAS) + 1))
v = bp.copy()
v.loc[v["collapse"], MEAS] = np.nan
for lab, d in [("as published (common by construction)", bp), ("drop collapsed, common set", v)]:
    res = [boot_endpoint(d, "cl", m) for m in MEAS]
    w("| {} | ".format(lab) + " | ".join(fmt(r) for r in res) + " | {} |".format(res[0][4]))
# 108-style (non-common) for reference
x0 = v[(v.year_start == 1998) & v["D0_raw"].notna()]
x1 = v[(v.year_start == 2019) & v["D0_raw"].notna()]
w("| 108 construction (different sets) | {:+.2f} | | | | | {} vs {} |".format(
    aw(x1, "D0_raw") - aw(x0, "D0_raw"), len(x0), len(x1)))
# drop the states entirely
drop_states = {"MADHYA PRADESH", "ANDHRA PRADESH", "CHHATTISGARH", "NAGALAND"}
v3 = bp[~bp["state"].isin(drop_states)]
res = [boot_endpoint(v3, "cl", m) for m in MEAS]
w("| drop the four 1998-collapsed states in every year | " + " | ".join(fmt(r) for r in res) +
  " | {} |".format(res[0][4]))
v4 = bp[~bp["state"].isin(drop_states | {"ODISHA", "KARNATAKA"})]
res = [boot_endpoint(v4, "cl", m) for m in MEAS]
w("| also drop Odisha and Karnataka | " + " | ".join(fmt(r) for r in res) +
  " | {} |".format(res[0][4]))
w("")

# ------------------------------------------------------------------ 3. state clustering
w("## 3. Clustering on state instead of district/parent ({} states)".format(bp["state"].nunique()))
w("")
w("| statistic | " + " | ".join(MEAS) + " |")
w("|---|" + "---|" * len(MEAS))
for clc, lab in [("cl", "parent"), ("state", "state")]:
    w("| aw 1998->2019, {} | ".format(lab) + " | ".join(fmt(boot_endpoint(bp, clc, m)) for m in MEAS) + " |")
    w("| aw window, {} | ".format(lab) + " | ".join(fmt(boot_window(bp, clc, m)) for m in MEAS) + " |")
    w("| mean window, {} | ".format(lab) + " | ".join(fmt(boot_window(bp, clc, m, False)) for m in MEAS) + " |")
w("")
w("FE trends per decade, balanced 1998-2019, clustered on state:")
d = bp[bp.year_start >= 1998].copy()
d["yr10"] = (d.year_start - 2008) / 10
for m in MEAS:
    dm = d.copy()
    for c in [m, "yr10"]:
        dm[c + "_dm"] = dm[c] - dm.groupby("district_key")[c].transform("mean")
    r1 = smf.ols("Q('{}_dm') ~ yr10_dm - 1".format(m), data=dm).fit(
        cov_type="cluster", cov_kwds={"groups": dm["state"]})
    w("  {}: {:+.3f} (SE {:.3f}, p {:.3f})".format(m, r1.params.iloc[0], r1.bse.iloc[0], r1.pvalues.iloc[0]))
w("")

# ------------------------------------------------------------------ 5. balanced without 1997
Y2 = [y for y in YRS if y >= 1998]
p2 = pan[pan.year_start >= 1998]
c2 = p2.groupby("district_key")["year_start"].nunique()
bal2 = set(c2[c2 == len(Y2)].index)
bp2 = p2[p2["district_key"].isin(bal2)].copy()
added = bp2[~bp2.district_key.isin(bal)].groupby("state")["district_key"].nunique().sort_values(ascending=False)
w("## 5. Balanced panel over 1998-2019 only")
w("")
w("{} districts (against {}). Added by state: {}.".format(
    len(bal2), len(bal), "; ".join("{} {}".format(s.title(), n) for s, n in added.items())))
w("States with no district in the 1997-2019 balanced panel but present 1998-2019: {}.".format(
    ", ".join(sorted(s.title() for s in set(bp2.state) - set(bp.state)))))
w("")
w("| statistic | " + " | ".join(MEAS) + " |")
w("|---|" + "---|" * len(MEAS))
w("| levels 1998 / 2019 (aw) | " + " | ".join("{:.2f} / {:.2f}".format(
    aw(bp2[bp2.year_start == 1998], m), aw(bp2[bp2.year_start == 2019], m)) for m in MEAS) + " |")
w("| aw 1998->2019, parent | " + " | ".join(fmt(boot_endpoint(bp2, "cl", m)) for m in MEAS) + " |")
w("| aw window, parent | " + " | ".join(fmt(boot_window(bp2, "cl", m)) for m in MEAS) + " |")
w("| aw window, state | " + " | ".join(fmt(boot_window(bp2, "state", m)) for m in MEAS) + " |")
w("")

# ------------------------------------------------------------------ 6. state-year common share
x = bp[bp.year_start >= 1998].sort_values(["district_key", "year_start"]).copy()
for m in ["D0_raw", "D0_0.5%", "D1"]:
    x["d_" + m] = x.groupby("district_key")[m].diff()
x = x.dropna(subset=["d_D0_raw"])
x["sy"] = x["state"] + "_" + x["year_start"].astype(str)
w("## 6. Share of year-on-year district changes explained by state-year means (R2)")
w("")
for m in ["D0_raw", "D0_0.5%", "D1"]:
    y_ = x["d_" + m]
    fit = x.groupby("sy")["d_" + m].transform("mean")
    fit_y = x.groupby("year_start")["d_" + m].transform("mean")
    w("  {}: state-year R2 {:.2f}; year-only R2 {:.2f}".format(
        m, 1 - ((y_ - fit) ** 2).sum() / ((y_ - y_.mean()) ** 2).sum(),
        1 - ((y_ - fit_y) ** 2).sum() / ((y_ - y_.mean()) ** 2).sum()))
w("")

# ------------------------------------------------------------------ 4. raw-row checks
raw = pd.read_csv(RAW)
for c in ["year", "season", "state_name", "district_name", "crop_name", "crop_type"]:
    raw[c] = raw[c].astype(str).str.strip()
raw = raw.dropna(subset=["area"])
raw = raw[raw["area"] > 0]
REMAP = {("Madhya Pradesh", d): "Chhattisgarh" for d in
         ["Bastar", "Bilaspur", "Dhamtari", "Durg", "Jashpur", "Kanker",
          "Korba", "Mahasamund", "Raipur", "Rajnandgaon", "Surguja"]}
for (s_, d_), new in REMAP.items():
    mm = (raw["state_name"] == s_) & (raw["district_name"] == d_)
    raw.loc[mm, "state_name"] = new
raw["year_start"] = raw["year"].str.split("-").str[0].astype(int)
raw = raw[raw.year_start <= 2019]
w("## 4. Raw-row checks")
w("")
rj = raw[raw.state_name == "Rajasthan"]
t = rj[rj.crop_name.isin(["Moth", "Guar Seed", "Other Kharif Pulses", "Other Oilseeds"])].groupby(
    ["year_start", "crop_name"])["area"].sum().unstack().fillna(0) / 1e6
w("Rajasthan area (Mha), Moth / Guar Seed / Other Kharif Pulses / Other Oilseeds:")
for yy in [1997, 1998, 1999, 2000, 2001, 2010, 2019]:
    if yy in t.index:
        w("  {}: ".format(yy) + ", ".join("{} {:.2f}".format(c, t.loc[yy, c]) for c in t.columns))
nat = raw[raw.crop_name.isin(["Moth", "Guar Seed"])].groupby(["year_start", "crop_name"])["area"].sum().unstack().fillna(0) / 1e6
w("National Moth+Guar (Mha): " + ", ".join("{} {:.2f}".format(y, r.sum()) for y, r in nat.loc[1997:2002].iterrows()))
oo = raw[raw.crop_name == "Other Oilseeds"].groupby("year_start").agg(
    a=("area", "sum"), st=("state_name", "nunique"))
w("Other Oilseeds national (kha, states): " + ", ".join(
    "{} {:.0f}/{}".format(y, r["a"] / 1e3, r["st"]) for y, r in oo.iterrows() if y in [1998, 1999, 2000, 2005, 2010, 2015, 2016, 2017, 2018, 2019]))
oo19 = raw[(raw.crop_name == "Other Oilseeds") & (raw.year_start == 2019)].groupby("state_name")["area"].sum().sort_values(ascending=False).head(5) / 1e3
w("Other Oilseeds 2019 by state (kha): " + ", ".join("{} {:.0f}".format(s, a) for s, a in oo19.items()))
od = raw[(raw.state_name == "Odisha") & raw.year_start.isin([1997, 1998, 1999, 2000])].groupby(
    ["year_start", "season"])["area"].sum().unstack().fillna(0) / 1e6
w("Odisha area by season (Mha):")
for yy, r in od.iterrows():
    w("  {}: ".format(yy) + ", ".join("{} {:.2f}".format(c, v) for c, v in r.items()))
mp = raw[raw.state_name == "Madhya Pradesh"]
l98 = set(mp[mp.year_start == 1998].crop_name)
l99 = set(mp[mp.year_start == 1999].crop_name)
w("Madhya Pradesh 1998-99 names ({}): {}.".format(len(l98), "; ".join(sorted(l98))))
w("Madhya Pradesh: in 1999 not 1998: {}.".format("; ".join(sorted(l99 - l98))))
for c in ["Moong(Green Gram)", "Urad", "Masoor", "Peas & Beans (Pulses)", "Potato"]:
    w("  MP {}: in 1998 list {}".format(c, c in l98))
w("")

# ------------------------------------------------------------------ 7. top-10 list
cr = pd.read_csv(OUT + "/d0_endpoint_crop_drivers.csv", index_col=0)
top = cr.sort_values("change", ascending=False).head(10)
w("## 7. Actual top 10 endpoint gainers")
w("")
for c, r in top.iterrows():
    w("  {}: {:+.3f}, median share {:.3%}, 2019 presences <0.5% {:.0%}, area 98/19 {:,.0f}/{:,.0f} kha".format(
        c, r["change"], r["median_share_2019"], r["frac_2019_below_0.5%"], r["area_1998_kha"], r["area_2019_kha"]))
quoted = ["Coriander", "Guar Seed", "Garlic", "Other Oilseeds", "Sweet Potato",
          "Peas & Beans (Pulses)", "Onion", "Ginger", "Mesta", "Turmeric"]
w("Write-up list sum {:+.2f}; actual top-10 sum {:+.2f}.".format(
    cr.loc[quoted, "change"].sum(), top["change"].sum()))
pul = ["Moong(Green Gram)", "Masoor", "Urad", "Peas & Beans (Pulses)", "Potato"]
w("Pulses + potato with median 2019 share >0.25%: {:+.2f} of the rise.".format(cr.loc[pul, "change"].sum()))
w("")

with open(OUT + "/d0_referee_raw.md", "w", encoding="utf-8") as fh:
    fh.write("\n".join(R))
