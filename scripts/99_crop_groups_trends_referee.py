"""REFEREE COPY of 99_crop_groups_trends.py. Adds --fix (default on): removes two APY reporting
artefacts, Andhra Pradesh 'Niger Seed' 2015-16..2019-20 (0.9-1.3 Mha a year of phantom area, absent
before 2015, yields of 0.2 to 15.8 t/ha) and Gujarat 1998-99 Whole Year 'Moth' (2.9 Mha, against 0.03 Mha
in every other year). --w1end moves the end window. Writes referee_<tag>_* outputs only.

Q1: crop-group trends 1998-99 to 2019-20, and the Andhra Pradesh / Telangana rice check.

USAGE
    C:/Users/mridu/anaconda3/envs/geo_env/python.exe scripts/99_crop_groups_trends.py [--boot 1000]

Reads only the crop-group panel built by scripts 97 and 98 (APY records on 2011
parent districts), its adjacency and flags, the district geojson for maps, and the
IMD district rainfall (context only). No APCNF / PMDS / RySS data.

Design
- Sample: the 605 units holding the 606-district analysis sample (`in_606 == 1`).
- 1997-98 is dropped (weak in several states). Years 1998-99 .. 2019-20.
- A unit-year is usable when `flag_annual_unreliable == 0`, with one repair: the
  pre-split donor flag is lifted where every donor link is cross-state (a district
  cannot be carved out of a district in another state; Chittoor -> Chikkaballapura).
- Endpoint panel E: units with >= 2 usable years in BOTH windows
  W0 = 1998-99..2000-01 and W1 = 2017-18..2019-20. Window values are ratio of
  window-summed area to window-summed GCA.
- Area-weighted = aggregate share (sum area / sum GCA over units); bootstrap SE over
  units. District-mean = unweighted mean of unit changes; SE = sd/sqrt(n).
- Trend slopes: share (pp) on year with unit FE, SE clustered on unit (the 2011
  parent, which is the clustering group).

Outputs to outputs/crop_groups/: trends_*.csv, fig_trends_*.png, trends_results.md
(the md is written by hand from trends_log.txt).
"""
import argparse
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

ap_ = argparse.ArgumentParser()
ap_.add_argument("--boot", type=int, default=1000)
ap_.add_argument("--fix", type=int, default=1)
ap_.add_argument("--w1end", type=int, default=2019)
ap_.add_argument("--tag", default="fix")
ap_.add_argument("--seed", type=int, default=20261107)
ARGS = ap_.parse_args()
RNG = np.random.default_rng(ARGS.seed)

REPO = "D:/crop-diversity"
OUT = REPO + "/outputs/crop_groups"
G = ["rice", "wheat", "maize", "millets", "pulses", "oilseeds", "cotton", "sugarcane", "other"]
W0 = (1998, 1999, 2000)
W1 = (ARGS.w1end - 2, ARGS.w1end - 1, ARGS.w1end)
TS_PARENTS = {"ADILABAD", "KARIMNAGAR", "KHAMMAM", "MAHBUBNAGAR", "MEDAK", "NALGONDA",
              "NIZAMABAD", "RANGAREDDY", "WARANGAL", "HYDERABAD"}
LOG = []


def w(*s):
    line = " ".join(str(x) for x in s)
    print(line)
    LOG.append(line)


def yl(y):
    return "{}-{:02d}".format(y, (y + 1) % 100)


# ---------------------------------------------------------------- load + usable flag
a = pd.read_csv(OUT + "/crop_group_panel_annual.csv")
a = a[(a.in_606 == 1) & a.year_start.between(1998, 2019)].copy()
if ARGS.fix:
    raw = pd.read_csv("E:/CEEW Project/outputs/all_crops_apy_1997_2021_india_data_portal.csv")
    for c in ["year", "season", "state_name", "district_name", "crop_name"]:
        raw[c] = raw[c].astype(str).str.strip()
    raw = raw[raw.area > 0]
    raw["year_start"] = raw.year.str[:4].astype(int)
    bad = raw[((raw.state_name == "Andhra Pradesh") & (raw.crop_name == "Niger Seed") & (raw.year_start >= 2015)) |
              ((raw.state_name == "Gujarat") & (raw.crop_name == "Moth") & (raw.year_start == 1998) &
               (raw.season == "Whole Year"))].copy()
    bad["grp"] = np.where(bad.crop_name == "Moth", "pulses", "oilseeds")
    bad["div_key"] = bad.state_name.str.upper() + "|" + bad.district_name.str.upper()
    cmap = pd.read_csv(OUT + "/crop_group_child_map.csv")[["div_key", "unit_key"]]
    bad = bad.merge(cmap, on="div_key", how="left")
    w("FIX artefact rows:", len(bad), "unmatched:", int(bad.unit_key.isna().sum()),
      "area Mha by grp/year:", (bad.groupby(["grp", "year_start"]).area.sum() / 1e6).round(3).to_dict())
    adj_ = bad.groupby(["unit_key", "year_start", "grp"]).area.sum().unstack(fill_value=0).reset_index()
    a = a.merge(adj_, on=["unit_key", "year_start"], how="left")
    for g_ in ("oilseeds", "pulses"):
        a[g_] = a[g_].fillna(0)
        a["area_ha_" + g_] = a["area_ha_" + g_] - a[g_]
        a["gca_ha"] = a["gca_ha"] - a[g_]
    for g_ in ["rice", "wheat", "maize", "millets", "pulses", "oilseeds", "cotton", "sugarcane", "other"]:
        a["share_gca_" + g_] = a["area_ha_" + g_] / a["gca_ha"]
a["district"] = a.unit_key.str.split("|").str[1]
a["region"] = np.where(a.state.eq("ANDHRA PRADESH") & a.district.isin(TS_PARENTS),
                       "TELANGANA", a.state)

don = pd.read_csv(OUT + "/crop_group_presplit_donors.csv")
don["late_state"] = don.late_unit.str.split("|").str[0]
don["donor_state"] = don.donor_unit.str.split("|").str[0]
don["same_state"] = don.late_state.eq(don.donor_state)
w("pre-split donor links:", len(don), "of which cross-state:", int((~don.same_state).sum()))
w(don[~don.same_state][["late_unit", "entry_year", "donor_unit"]].to_string(index=False))
# a donor-year stays flagged only if some same-state late unit enters after that year
same = don[don.same_state]
keep_flag = set()
for _, r in same.iterrows():
    for y in range(1997, int(r.entry_year)):
        keep_flag.add((r.donor_unit, y))
a["f_pre_split_donor_fix"] = [int(f == 1 and (u, y) in keep_flag)
                              for f, u, y in zip(a.f_pre_split_donor, a.unit_key, a.year_start)]
a["unrel"] = ((a.f_odisha_summer_rice_2012 == 1) | (a.f_bihar_1999_rice_missing == 1) |
              (a.f_gca_jump_severe == 1) | (a.f_pre_split_donor_fix == 1) |
              (a.f_child_gap == 1) | (a.f_before_entry == 1)).astype(int)
w("unit-years 1998-2019 in 605 units:", len(a), "| usable (orig flag):",
  int((a.flag_annual_unreliable == 0).sum()), "| usable (cross-state donor fix):",
  int((a.unrel == 0).sum()))
a["ok"] = a.unrel == 0
for g in G:
    a["pp_" + g] = 100 * a["share_gca_" + g]
units_meta = a.groupby("unit_key").agg(state=("state", "first"), region=("region", "first"),
                                       district=("district", "first")).reset_index()


# ---------------------------------------------------------------- endpoint machinery
def window_frame(df, w0, w1, minyrs=2):
    """one row per unit: window-mean areas and shares; units with >= minyrs usable in both."""
    d = df[df.ok]
    rows = []
    for tag, win in (("0", w0), ("1", w1)):
        x = d[d.year_start.isin(win)]
        agg = x.groupby("unit_key").agg(n=("year_start", "size"), gca=("gca_ha", "mean"),
                                        **{"a_" + g: ("area_ha_" + g, "mean") for g in G})
        agg.columns = [c + tag for c in agg.columns]
        rows.append(agg)
    e = rows[0].join(rows[1], how="inner")
    e = e[(e.n0 >= minyrs) & (e.n1 >= minyrs)].copy()
    for g in G:
        e["s0_" + g] = 100 * e["a_" + g + "0"] / e.gca0
        e["s1_" + g] = 100 * e["a_" + g + "1"] / e.gca1
        e["ds_" + g] = e["s1_" + g] - e["s0_" + g]
        e["da_" + g] = e["a_" + g + "1"] - e["a_" + g + "0"]
        with np.errstate(divide="ignore", invalid="ignore"):
            e["dl_" + g] = np.log(e["a_" + g + "1"]) - np.log(e["a_" + g + "0"])
    e["dgca"] = e.gca1 - e.gca0
    return e.reset_index().merge(units_meta, on="unit_key", how="left")


def agg_share(e, g, t):
    return 100 * e["a_" + g + t].sum() / e["gca" + t].sum()


def boot(e, fn, B=None):
    B = B or ARGS.boot
    n = len(e)
    idx = RNG.integers(0, n, size=(B, n))
    vals = np.array([fn(e.iloc[i]) for i in idx])
    return np.nanstd(vals, ddof=1)


def summarise(e, label, B=None, groups=G):
    out = []
    for g in groups:
        s0, s1 = agg_share(e, g, "0"), agg_share(e, g, "1")
        dagg = s1 - s0
        se_agg = boot(e, lambda x: agg_share(x, g, "1") - agg_share(x, g, "0"), B)
        dm = e["ds_" + g]
        se_m = dm.std(ddof=1) / np.sqrt(len(dm))
        da = e["da_" + g].sum()
        se_da = boot(e, lambda x: x["da_" + g].sum(), B)
        out.append(dict(spec=label, group=g, n_units=len(e), agg_share0=s0, agg_share1=s1,
                        d_agg_pp=dagg, se_agg=se_agg, d_mean_pp=dm.mean(), se_mean=se_m,
                        d_area_kha=da / 1e3, se_area_kha=se_da / 1e3,
                        n_gain=int((dm > 0).sum()), n_lose=int((dm < 0).sum()),
                        sign_p=stats.binomtest(int((dm > 0).sum()), int((dm != 0).sum())).pvalue if (dm != 0).sum() > 0 else np.nan))
    return pd.DataFrame(out)


def fe_slope(df, g, weighted):
    """pp per decade, unit FE, cluster unit."""
    d = df[df.ok].copy()
    d["t"] = (d.year_start - 2008) / 10.0
    y = d["pp_" + g] - d.groupby("unit_key")["pp_" + g].transform("mean")
    x = d.t - d.groupby("unit_key").t.transform("mean")
    groups = pd.factorize(d.unit_key)[0]
    if weighted:
        wts = d.groupby("unit_key").gca_ha.transform("mean")
        m = sm.WLS(y.values, x.values, weights=wts.values).fit(cov_type="cluster",
                                                                 cov_kwds={"groups": groups})
    else:
        m = sm.OLS(y.values, x.values).fit(cov_type="cluster", cov_kwds={"groups": groups})
    return m.params[0], m.bse[0], d.unit_key.nunique(), len(d)


# ================================================================= 1. national
w("\n=== 1. NATIONAL ===")
E = window_frame(a, W0, W1)
w("endpoint panel E (>=2 usable yrs in each window):", len(E), "units")
strict_units = a[a.ok].groupby("unit_key").year_start.nunique()
strict_units = strict_units[strict_units == 22].index
w("strict balanced panel (all 22 years usable):", len(strict_units), "units")

nat = summarise(E, "main")
w(nat.round(3).to_string(index=False))

# FE slopes on E units, usable years
aE = a[a.unit_key.isin(E.unit_key)]
sl = []
for g in G:
    b_u, se_u, nu, nobs = fe_slope(aE, g, False)
    b_w, se_w, _, _ = fe_slope(aE, g, True)
    sl.append(dict(group=g, slope_mean_pp_dec=b_u, se_mean=se_u, slope_aw_pp_dec=b_w,
                   se_aw=se_w, n_units=nu, n_obs=nobs))
sl = pd.DataFrame(sl)
w("\nFE slopes (pp per decade), unit FE, cluster unit:")
w(sl.round(3).to_string(index=False))

# yearly series (strict balanced and E-usable)
ser = []
for lab, d in (("strict", a[a.unit_key.isin(strict_units) & a.ok]), ("E_usable", aE[aE.ok])):
    for y, x in d.groupby("year_start"):
        r = dict(panel=lab, year_start=y, year=yl(y), n_units=len(x), gca_mha=x.gca_ha.sum() / 1e6)
        for g in G:
            r["aw_" + g] = 100 * x["area_ha_" + g].sum() / x.gca_ha.sum()
            r["dm_" + g] = x["pp_" + g].mean()
            r["mha_" + g] = x["area_ha_" + g].sum() / 1e6
        ser.append(r)
ser = pd.DataFrame(ser)
ser.to_csv(OUT + "/referee_" + ARGS.tag + "_trends_national_series.csv", index=False)

# ---------------------------------------------------------------- robustness
w("\n=== ROBUSTNESS (area-weighted and district-mean change, pp) ===")
rob = [nat]
Es = window_frame(a[a.unit_key.isin(strict_units)], W0, W1, minyrs=3)
rob.append(summarise(Es, "strict_balanced", B=300))
E5 = window_frame(a, (1998, 1999, 2000, 2001, 2002), (2015, 2016, 2017, 2018, 2019), minyrs=3)
rob.append(summarise(E5, "5yr_windows", B=300))
Eno19 = window_frame(a, W0, (2016, 2017, 2018), minyrs=2)
rob.append(summarise(Eno19, "end_2016-18", B=300))
wy = a.groupby("unit_key").whole_year_share.mean()
Ewy = E[E.unit_key.isin(wy[wy < 0.2].index)]
rob.append(summarise(Ewy, "drop_wholeyear_ge20", B=300))
Eorig = window_frame(a.assign(ok=a.flag_annual_unreliable == 0), W0, W1)
rob.append(summarise(Eorig, "orig_flag", B=300))
rob = pd.concat(rob)
rob.to_csv(OUT + "/referee_" + ARGS.tag + "_trends_national_endpoint.csv", index=False)
piv = rob.pivot_table(index="group", columns="spec", values="d_agg_pp").loc[G]
w("area-weighted change (pp):\n" + piv.round(2).to_string())
w("n units by spec:", rob.groupby("spec").n_units.first().to_dict())
pivm = rob.pivot_table(index="group", columns="spec", values="d_mean_pp").loc[G]
w("district-mean change (pp):\n" + pivm.round(2).to_string())
sl.to_csv(OUT + "/referee_" + ARGS.tag + "_trends_national_fe_slopes.csv", index=False)

# rainfall context for the windows
rain = pd.read_csv(OUT + "/rainfall_allindia_check.csv")
w("\nall-India JJAS % departure:",
  {int(y): float(v) for y, v in rain.set_index("year").pct_dep_polygons.loc[list(W0) + list(W1)].items()})

# ================================================================= 2. states
w("\n=== 2. STATES ===")
st = []
for reg, e in E.groupby("region"):
    if len(e) < 3:
        continue
    s = summarise(e, reg, B=300)
    s["region"] = reg
    st.append(s)
st = pd.concat(st)
st["share_of_gca0"] = st.region.map(E.groupby("region").gca0.sum() / E.gca0.sum())
st.to_csv(OUT + "/referee_" + ARGS.tag + "_trends_state_endpoint.csv", index=False)
spiv = st.pivot_table(index="region", columns="group", values="d_agg_pp")[G]
spiv["n"] = st.groupby("region").n_units.first()
spiv["gca0_mha"] = E.groupby("region").gca0.sum() / 1e6
spiv = spiv.sort_values("gca0_mha", ascending=False)
w(spiv.round(1).to_string())
spiv.to_csv(OUT + "/referee_" + ARGS.tag + "_trends_state_change_pp.csv")

# ================================================================= 3. top districts
w("\n=== 3. TOP DISTRICTS ===")
keep = ["unit_key", "state", "region", "district", "gca0", "gca1"] + \
       [c for g in G for c in ("s0_" + g, "s1_" + g, "ds_" + g, "a_" + g + "0", "a_" + g + "1", "da_" + g)]
E[keep].to_csv(OUT + "/referee_" + ARGS.tag + "_trends_district_endpoint.csv", index=False)
w("top lists restricted to units with mean GCA >= 25,000 ha in both windows:", int(((E.gca0 >= 25000) & (E.gca1 >= 25000)).sum()))
tops = []
for g in ["pulses", "millets", "oilseeds"]:
    for metric, col in (("share_pp", "ds_" + g), ("area_ha", "da_" + g)):
        for direction, asc in (("gain", False), ("loss", True)):
            t = E[(E.gca0 >= 25000) & (E.gca1 >= 25000)].sort_values(col, ascending=asc).head(10)
            for rnk, (_, r) in enumerate(t.iterrows(), 1):
                tops.append(dict(group=g, metric=metric, direction=direction, rank=rnk,
                                 district=r.district.title(), state=r.region.title(),
                                 share0=r["s0_" + g], share1=r["s1_" + g], d_share_pp=r["ds_" + g],
                                 area0_kha=r["a_" + g + "0"] / 1e3, area1_kha=r["a_" + g + "1"] / 1e3,
                                 d_area_kha=r["da_" + g] / 1e3))
tops = pd.DataFrame(tops)
tops.to_csv(OUT + "/referee_" + ARGS.tag + "_trends_top_districts.csv", index=False)
for g in ["pulses", "millets", "oilseeds"]:
    for direction in ("gain", "loss"):
        t = tops[(tops.group == g) & (tops.metric == "share_pp") & (tops.direction == direction)]
        w("\n{} {} (share pp):".format(g, direction))
        w(t[["district", "state", "share0", "share1", "d_share_pp", "d_area_kha"]].round(1).to_string(index=False))
# concentration: share of national net area change from top-10 states
for g in ["pulses", "millets", "oilseeds"]:
    byst = E.groupby("region")["da_" + g].sum().sort_values()
    w("{} net area change kha by region (top/bottom 5):".format(g),
      (byst.head(5) / 1e3).round(0).to_dict(), (byst.tail(5) / 1e3).round(0).to_dict())

# ================================================================= 4. AP / Telangana rice
w("\n=== 4. ANDHRA PRADESH / TELANGANA RICE ===")
apE = E[E.region.isin(["ANDHRA PRADESH", "TELANGANA"])].copy()
w("AP/TS units in E:", len(apE), apE.groupby("region").size().to_dict())
missing = sorted(set(a[a.region.isin(["ANDHRA PRADESH", "TELANGANA"])].district) - set(apE.district))
w("AP/TS units not in E:", missing)

# national rank of AP/TS rice changes
E["pct_rank_ds_rice"] = E.ds_rice.rank(pct=True)
E["pct_rank_dl_rice"] = E.dl_rice.replace([np.inf, -np.inf], np.nan).rank(pct=True)
apE = E[E.region.isin(["ANDHRA PRADESH", "TELANGANA"])].copy()

# test: AP and TS vs rest of India
for y in ("ds_rice", "dl_rice"):
    d = E.replace([np.inf, -np.inf], np.nan).dropna(subset=[y])
    X = pd.DataFrame({"const": 1.0, "AP": (d.region == "ANDHRA PRADESH").astype(float),
                      "TS": (d.region == "TELANGANA").astype(float)})
    m = sm.OLS(d[y], X).fit(cov_type="HC1")
    w("{} on AP/TS dummies (HC1): const {:.3f} ({:.3f}); AP {:.3f} ({:.3f}) p={:.3f}; TS {:.3f} ({:.3f}) p={:.3f}; n={}".format(
        y, m.params.const, m.bse.const, m.params.AP, m.bse.AP, m.pvalues.AP, m.params.TS, m.bse.TS,
        m.pvalues.TS, int(m.nobs)))
    # restricted to rice-growing districts (rice >= 10% of GCA in W0)
    d2 = d[d.s0_rice >= 10]
    X2 = X.loc[d2.index]
    m2 = sm.OLS(d2[y], X2).fit(cov_type="HC1")
    w("   rice>=10% subsample: const {:.3f}; AP {:.3f} ({:.3f}) p={:.3f}; TS {:.3f} ({:.3f}) p={:.3f}; n={}".format(
        m2.params.const, m2.params.AP, m2.bse.AP, m2.pvalues.AP, m2.params.TS, m2.bse.TS, m2.pvalues.TS,
        int(m2.nobs)))

# Getis-Ord Gi* hot spots on change in rice share
adj = pd.read_csv(OUT + "/crop_group_adjacency_edges.csv")


def gi_star(e, col):
    keys = list(e.unit_key)
    pos = {k: i for i, k in enumerate(keys)}
    n = len(keys)
    W = np.eye(n)
    for u, v in zip(adj.unit_i, adj.unit_j):
        if u in pos and v in pos:
            W[pos[u], pos[v]] = 1
    x = e[col].values.astype(float)
    xb, S = x.mean(), np.sqrt((x ** 2).mean() - x.mean() ** 2)
    Wi = W.sum(1)
    num = W @ x - xb * Wi
    den = S * np.sqrt((n * Wi - Wi ** 2) / (n - 1))
    return num / den


E["gi_ds_rice"] = gi_star(E, "ds_rice")
dl = E.dl_rice.replace([np.inf, -np.inf], np.nan)
Edl = E[dl.notna()].copy()
Edl["gi_dl_rice"] = gi_star(Edl, "dl_rice")
E = E.merge(Edl[["unit_key", "gi_dl_rice"]], on="unit_key", how="left")
crit = stats.norm.ppf(1 - 0.05 / 2)
bonf = stats.norm.ppf(1 - 0.05 / (2 * len(E)))
for col in ("gi_ds_rice", "gi_dl_rice"):
    hot = E[E[col] > crit]
    hotb = E[E[col] > bonf]
    w("{}: hot spots z>1.96: {} units; by region {}".format(col, len(hot), hot.region.value_counts().head(8).to_dict()))
    w("   Bonferroni z>{:.2f}: {} units {}".format(bonf, len(hotb), hotb.region.value_counts().to_dict()))
    w("   cold spots z<-1.96: {} units; by region {}".format(
        int((E[col] < -crit).sum()), E[E[col] < -crit].region.value_counts().head(8).to_dict()))
apE = E[E.region.isin(["ANDHRA PRADESH", "TELANGANA"])].copy()

# season split for AP/TS rice (kharif vs rabi) by window
sp = pd.read_csv(OUT + "/crop_group_panel.csv",
                 usecols=["unit_key", "year_start", "season", "area_ha_rice", "flag_season_unreliable"])
sp = sp[sp.unit_key.isin(apE.unit_key)]
w("AP/TS season-unreliable unit-years 1998-2019:",
  int(sp[(sp.season == "annual") & sp.year_start.between(1998, 2019)].flag_season_unreliable.sum()))
ss = sp[sp.season.isin(["kharif", "rabi"])].pivot_table(index=["unit_key", "year_start"], columns="season",
                                                        values="area_ha_rice", aggfunc="sum").reset_index()
usable = a[a.ok][["unit_key", "year_start"]]
ss = ss.merge(usable, on=["unit_key", "year_start"])
for tag, win in (("0", W0), ("1", W1)):
    m = ss[ss.year_start.isin(win)].groupby("unit_key")[["kharif", "rabi"]].mean()
    m.columns = ["rice_kh" + tag, "rice_rb" + tag]
    apE = apE.merge(m, left_on="unit_key", right_index=True, how="left")
apE["d_rice_kh"] = apE.rice_kh1 - apE.rice_kh0
apE["d_rice_rb"] = apE.rice_rb1 - apE.rice_rb0

cols = ["region", "district", "s0_rice", "s1_rice", "ds_rice", "pct_rank_ds_rice", "a_rice0", "a_rice1",
        "da_rice", "dl_rice", "d_rice_kh", "d_rice_rb", "dgca", "gi_ds_rice"] + ["ds_" + g for g in G if g != "rice"] + \
       ["da_" + g for g in G if g != "rice"]
apE = apE.sort_values(["region", "ds_rice"], ascending=[True, False])
apE[cols].to_csv(OUT + "/referee_" + ARGS.tag + "_trends_apts_districts.csv", index=False)
show = apE[["region", "district", "s0_rice", "s1_rice", "ds_rice", "pct_rank_ds_rice", "da_rice",
            "d_rice_kh", "d_rice_rb", "dgca", "gi_ds_rice"]].copy()
for c in ("da_rice", "d_rice_kh", "d_rice_rb", "dgca"):
    show[c] = show[c] / 1e3
w(show.round(2).to_string(index=False))

# where did the land come from: share and area accounting by region
acc = []
for reg, e in apE.groupby("region"):
    for g in G:
        acc.append(dict(region=reg, group=g, share0=agg_share(e, g, "0"), share1=agg_share(e, g, "1"),
                        d_share_pp=agg_share(e, g, "1") - agg_share(e, g, "0"),
                        se_share=boot(e, lambda x: agg_share(x, g, "1") - agg_share(x, g, "0"), 500),
                        d_area_kha=e["da_" + g].sum() / 1e3,
                        se_area_kha=boot(e, lambda x: x["da_" + g].sum(), 500) / 1e3))
    acc.append(dict(region=reg, group="GCA", d_area_kha=e.dgca.sum() / 1e3,
                    se_area_kha=boot(e, lambda x: x.dgca.sum(), 500) / 1e3))
# districts where rice share rose >= 2 pp: pooled accounting
up = apE[apE.ds_rice >= 2]
for g in G:
    acc.append(dict(region="AP+TS rice-gain districts (n={})".format(len(up)), group=g,
                    share0=agg_share(up, g, "0"), share1=agg_share(up, g, "1"),
                    d_share_pp=agg_share(up, g, "1") - agg_share(up, g, "0"),
                    d_area_kha=up["da_" + g].sum() / 1e3))
acc.append(dict(region="AP+TS rice-gain districts (n={})".format(len(up)), group="GCA",
                d_area_kha=up.dgca.sum() / 1e3))
acc = pd.DataFrame(acc)
acc.to_csv(OUT + "/referee_" + ARGS.tag + "_trends_apts_accounting.csv", index=False)
w(acc.round(2).to_string(index=False))

# year series for AP and TS (all usable unit-years; panel composition fixed to apE units)
yr = a[a.unit_key.isin(apE.unit_key) & a.ok]
sp2 = pd.read_csv(OUT + "/crop_group_panel.csv", usecols=["unit_key", "year_start", "season"] +
                  ["area_ha_" + g for g in G])
sp2 = sp2[sp2.unit_key.isin(apE.unit_key) & sp2.season.isin(["kharif", "rabi", "annual"])]
sp2 = sp2.merge(yr[["unit_key", "year_start", "region"]], on=["unit_key", "year_start"])
yser = sp2.groupby(["region", "year_start", "season"])[["area_ha_" + g for g in G]].sum().reset_index()
nun = yr.groupby(["region", "year_start"]).size().rename("n_units").reset_index()
yser = yser.merge(nun, on=["region", "year_start"])
yser.to_csv(OUT + "/referee_" + ARGS.tag + "_trends_apts_year_series.csv", index=False)
ann = yser[yser.season == "annual"].copy()
ann["gca"] = ann[["area_ha_" + g for g in G]].sum(1)
ann["rice_share"] = 100 * ann.area_ha_rice / ann.gca
w("AP/TS annual rice area (kha) and share by year:")
w(ann.pivot_table(index="year_start", columns="region", values=["area_ha_rice", "rice_share", "n_units"])
  .assign(**{}).round(1).to_string())
# the end-window dependence on 2019-20
for reg in ("ANDHRA PRADESH", "TELANGANA"):
    e1 = window_frame(a[a.region == reg], W0, (2016, 2017, 2018))
    e2 = window_frame(a[a.region == reg], (2005, 2006, 2007), W1)
    e3 = window_frame(a[a.region == reg], (2008, 2009, 2010), (2016, 2017, 2018))
    w("{}: rice agg share change pp | main {:.2f} | end 2016-18 {:.2f} | start 2005-07 {:.2f} | 2008-10 to 2016-18 {:.2f}".format(
        reg, agg_share(apE[apE.region == reg], "rice", "1") - agg_share(apE[apE.region == reg], "rice", "0"),
        agg_share(e1, "rice", "1") - agg_share(e1, "rice", "0"),
        agg_share(e2, "rice", "1") - agg_share(e2, "rice", "0"),
        agg_share(e3, "rice", "1") - agg_share(e3, "rice", "0")))
    w("   rice area change kha | main {:.0f} | end 2016-18 {:.0f} | start 2005-07 {:.0f} | 2008-10 to 2016-18 {:.0f}".format(
        apE[apE.region == reg].da_rice.sum() / 1e3, e1.da_rice.sum() / 1e3, e2.da_rice.sum() / 1e3,
        e3.da_rice.sum() / 1e3))
# FE slope of rice share within AP / TS
for reg in ("ANDHRA PRADESH", "TELANGANA"):
    d = a[a.region == reg]
    b, se, nu, no = fe_slope(d, "rice", True)
    b2, se2, _, _ = fe_slope(d[d.year_start <= 2018], "rice", True)
    w("{} rice share FE slope (area-wtd) pp/decade: {:.2f} ({:.2f}) n_units={} n_obs={} | to 2018-19 only: {:.2f} ({:.2f})".format(
        reg, b, se, nu, no, b2, se2))

# rainfall context for AP/TS windows
rf = pd.read_csv(OUT + "/rainfall_district_year.csv", usecols=["district_key", "year", "pct_jjas", "qc_jjas"])
alias = {"YSR KADAPA": "KADAPA", "RANGAREDDY": "RANGAREDDI", "VISAKHAPATNAM": "VISAKHAPATANAM"}
rk = set(rf.district_key.dropna())
keymap = {}
for _, r in apE.iterrows():
    nm = alias.get(r.district, r.district)
    for pre in ("ANDHRA PRADESH|", "TELANGANA|"):
        if pre + nm in rk:
            keymap[pre + nm] = r.region
            break
rf = rf[rf.district_key.isin(keymap) & (rf.qc_jjas == 0)]
rf["region"] = rf.district_key.map(keymap)
w("rainfall matched AP/TS districts:", rf.district_key.nunique(), "of", len(apE))
w("mean JJAS % departure by year:\n" +
  rf[rf.year.isin(list(W0) + list(W1))].pivot_table(index="year", columns="region", values="pct_jjas").round(1).to_string())

# ================================================================= 5. millets -> maize
w("\n=== 5. MILLETS DOWN, MAIZE UP ===")
E["sub_share"] = (E.ds_millets <= -1) & (E.ds_maize >= 1)
E["sub_area"] = (E.da_millets < 0) & (E.da_maize > 0) & E.sub_share
base = E[E.s0_millets >= 5]
w("units with millets >= 5% of GCA in W0:", len(base))
w("of these, millet share fell >=1pp:", int((base.ds_millets <= -1).sum()),
  "| and maize share rose >=1pp:", int(base.sub_share.sum()),
  "| and millet area fell and maize area rose:", int(base.sub_area.sum()))
r, p = stats.pearsonr(base.ds_millets, base.ds_maize)
z = np.arctanh(r)
lo, hi = np.tanh(z - 1.96 / np.sqrt(len(base) - 3)), np.tanh(z + 1.96 / np.sqrt(len(base) - 3))
w("corr(d millets share, d maize share), millet-base units: r={:.3f} [{:.3f},{:.3f}] p={:.4f} n={}".format(r, lo, hi, p, len(base)))
m = sm.OLS(base.da_maize, sm.add_constant(base.da_millets)).fit(cov_type="HC1")
w("d maize area on d millet area (ha/ha), HC1: slope {:.3f} ({:.3f}) p={:.4f} n={}".format(
    m.params.da_millets, m.bse.da_millets, m.pvalues.da_millets, int(m.nobs)))
# where did lost millet share go, in millet-losing districts (pooled accounting)
loss = base[base.ds_millets <= -2]
acc2 = pd.DataFrame([dict(group=g, d_share_pp=agg_share(loss, g, "1") - agg_share(loss, g, "0"),
                          se=boot(loss, lambda x: agg_share(x, g, "1") - agg_share(x, g, "0"), 500),
                          d_area_kha=loss["da_" + g].sum() / 1e3) for g in G] +
                    [dict(group="GCA", d_area_kha=loss.dgca.sum() / 1e3)])
w("millet-losing (<= -2pp) districts n={}: pooled share/area accounting".format(len(loss)))
w(acc2.round(2).to_string(index=False))
acc2.to_csv(OUT + "/referee_" + ARGS.tag + "_trends_millet_loss_accounting.csv", index=False)
subd = base[base.sub_area].copy()
subd["score"] = np.minimum(-subd.ds_millets, subd.ds_maize)
subd = subd.sort_values("score", ascending=False)
w("substitution districts by region:", subd.region.value_counts().to_dict())
w(subd[["district", "region", "s0_millets", "s1_millets", "ds_millets", "s0_maize", "s1_maize", "ds_maize",
        "da_millets", "da_maize"]].head(25).round(1).to_string(index=False))
w("substitution districts: total millet area change kha {:.0f}, maize {:.0f}".format(
    subd.da_millets.sum() / 1e3, subd.da_maize.sum() / 1e3))
subd[["unit_key", "district", "region", "s0_millets", "s1_millets", "ds_millets", "s0_maize", "s1_maize",
      "ds_maize", "da_millets", "da_maize"]].to_csv(OUT + "/referee_" + ARGS.tag + "_trends_millet_maize_substitution.csv", index=False)

# within-unit panel relation, unit + year FE (alternating projections), cluster unit
def twfe(df, ycol, xcol, iters=200):
    d = df[["unit_key", "year_start", ycol, xcol]].dropna().copy()
    for c in (ycol, xcol):
        v = d[c].astype(float).copy()
        for _ in range(iters):
            v = v - v.groupby(d.unit_key).transform("mean")
            v = v - v.groupby(d.year_start).transform("mean")
        d[c + "_dm"] = v
    m = sm.OLS(d[ycol + "_dm"], d[xcol + "_dm"]).fit(cov_type="cluster",
                                                     cov_kwds={"groups": pd.factorize(d.unit_key)[0]})
    return m.params.iloc[0], m.bse.iloc[0], d.unit_key.nunique(), len(d)


pb = a[a.unit_key.isin(base.unit_key) & a.ok]
b, se, nu, no = twfe(pb, "pp_maize", "pp_millets")
w("TWFE maize share on millet share (pp/pp), millet-base units: {:.3f} ({:.3f}) units={} obs={}".format(b, se, nu, no))
pb2 = pb.assign(lmz=np.log1p(pb.area_ha_maize), lml=np.log1p(pb.area_ha_millets))
b2, se2, nu2, no2 = twfe(pb2, "lmz", "lml")
w("TWFE log(1+maize area) on log(1+millet area): {:.3f} ({:.3f})".format(b2, se2))
# state-level substitution: area changes
sst = base.groupby("region")[["da_millets", "da_maize"]].sum() / 1e3
sst = sst[(sst.da_millets.abs() > 20) | (sst.da_maize.abs() > 20)].sort_values("da_millets")
w("millet-base units, area change kha by region:\n" + sst.round(0).to_string())

# ================================================================= figures
cm = LinearSegmentedColormap.from_list("vc", ["#5B3FC4", "#F2F2F2", "#0E7F94"])
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

s = ser[ser.panel == "strict"]
fig, axs = plt.subplots(3, 3, figsize=(10, 8), sharex=True)
for ax, g in zip(axs.flat, G):
    ax.plot(s.year_start, s["aw_" + g], color="#222222", lw=1.6, label="area-weighted")
    ax.plot(s.year_start, s["dm_" + g], color="#888888", lw=1.2, ls="--", label="district mean")
    ax.set_title(g, loc="left")
    ax.set_ylim(bottom=0)
axs[0, 0].legend(frameon=False, fontsize=8)
for ax in axs[:, 0]:
    ax.set_ylabel("% of gross cropped area")
fig.suptitle("Crop-group shares of gross cropped area, {} districts observed in all 22 years".format(
    len(strict_units)), x=0.01, ha="left")
fig.tight_layout()
fig.savefig(OUT + "/referee_" + ARGS.tag + "_fig_trends_national_shares.png", dpi=180)
plt.close(fig)

hp = spiv[spiv.n >= 3].head(22)[G]
fig, ax = plt.subplots(figsize=(8, 7.5))
lim = np.nanmax(np.abs(hp.values))
im = ax.imshow(hp.values, cmap=cm, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
ax.set_xticks(range(len(G)), G, rotation=45, ha="right")
ax.set_yticks(range(len(hp)), [r.title() for r in hp.index])
for i in range(hp.shape[0]):
    for j in range(hp.shape[1]):
        ax.text(j, i, "{:+.1f}".format(hp.values[i, j]), ha="center", va="center", fontsize=7)
fig.colorbar(im, ax=ax, label="change in share of GCA, percentage points", shrink=0.6)
ax.set_title("Change in crop-group share, 1998-2000 to 2017-2019 (states by cropped area)", loc="left")
fig.tight_layout()
fig.savefig(OUT + "/referee_" + ARGS.tag + "_fig_trends_state_heatmap.png", dpi=180)
plt.close(fig)

try:
    import geopandas as gpd
    gj = gpd.read_file(REPO + "/docs/data/districts.geojson")[["n", "s", "k", "geometry"]]
    pm = pd.read_csv(OUT + "/crop_group_polygon_match.csv")
    gj = gj.reset_index().merge(pm[["n", "s", "k", "unit_key"]].reset_index(), on="index", suffixes=("", "_m"))
    gj["geometry"] = gj.geometry.buffer(0)
    gj = gj[gj.unit_key.notna()].dissolve("unit_key").reset_index().to_crs(7755)
    gm = gj.merge(E, on="unit_key", how="left")
    fig, axs = plt.subplots(2, 2, figsize=(10, 11))
    for ax, g in zip(axs.flat, ["rice", "pulses", "millets", "oilseeds"]):
        lim = 15
        gj.plot(ax=ax, color="#DDDDDD", lw=0)
        gm[gm["ds_" + g].notna()].plot(column="ds_" + g, ax=ax, cmap=cm, norm=TwoSlopeNorm(0, -lim, lim),
                                        lw=0.1, edgecolor="white")
        ax.set_title("{}: change in share, pp (clipped at +/-15)".format(g), loc="left")
        ax.set_axis_off()
    sm_ = plt.cm.ScalarMappable(cmap=cm, norm=TwoSlopeNorm(0, -15, 15))
    fig.colorbar(sm_, ax=axs, orientation="horizontal", fraction=0.03, pad=0.02,
                 label="percentage points of GCA, 1998-2000 to 2017-2019 (grey: not in endpoint panel)")
    fig.savefig(OUT + "/referee_" + ARGS.tag + "_fig_trends_change_maps.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
except Exception as ex:  # maps are optional
    w("map figure skipped:", repr(ex))

fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for ax, reg in zip(axs, ("ANDHRA PRADESH", "TELANGANA")):
    d = yser[yser.region == reg].pivot_table(index="year_start", columns="season", values="area_ha_rice") / 1e6
    ax.plot(d.index, d.annual, color="#222222", lw=1.8, label="annual")
    ax.plot(d.index, d.kharif, color="#0E7F94", lw=1.2, label="kharif")
    ax.plot(d.index, d.rabi, color="#5B3FC4", lw=1.2, label="rabi")
    for win in (W0, W1):
        ax.axvspan(win[0] - 0.4, win[-1] + 0.4, color="#EEEEEE", zorder=0)
    ax.set_title("{}: rice area, million ha ({} 2011 parents)".format(reg.title(), int((apE.region == reg).sum())),
                 loc="left")
axs[0].legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT + "/referee_" + ARGS.tag + "_fig_trends_apts_rice.png", dpi=180)
plt.close(fig)

fig, ax = plt.subplots(figsize=(5.5, 5))
ax.axhline(0, color="#999999", lw=0.6)
ax.axvline(0, color="#999999", lw=0.6)
ax.scatter(base.ds_millets, base.ds_maize, s=np.sqrt(base.gca0) / 15, color="#222222", alpha=0.5, lw=0)
ax.set_xlabel("change in millet share, pp")
ax.set_ylabel("change in maize share, pp")
ax.set_title("Districts with millets >= 5% of GCA in 1998-2000 (n={}), r = {:.2f}".format(len(base), r), loc="left",
             fontsize=8.5)
fig.tight_layout()
fig.savefig(OUT + "/referee_" + ARGS.tag + "_fig_trends_millet_maize.png", dpi=180)
plt.close(fig)

E.to_csv(OUT + "/referee_" + ARGS.tag + "_trends_endpoint_panel_full.csv", index=False)
open(OUT + "/referee_" + ARGS.tag + "_trends_log.txt", "w", encoding="utf-8").write("\n".join(LOG))
w("done")
