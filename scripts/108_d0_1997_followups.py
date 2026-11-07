"""
108_d0_1997_followups.py

Follow-ups to 104_d0_threshold_and_1997.py (read that first; same cleaning, same
balanced panel of 429 districts, area weights = district gross cropped area).

A. 1997-98: which rows make it fall. West Bengal 1997-98 Niger Seed (rabi) is
   filed at 50.8 Mha; season x state coverage against 1998-99 and against each
   state's 1999-2003 median; the 1997 national series recomputed without each
   defect.
B. The 21.2 -> 24.9 endpoint rise in raw D0:
   - sensitivity to the start and end year chosen,
   - state-years whose crop list collapses (names < 60% of the state's own median),
   - the series with collapsed state-years and the crop-group defect flags
     (outputs/crop_groups/crop_group_flags.csv, flag_annual_unreliable) removed,
   - which crops and which share bins carry the endpoint rise,
   - the same on the 606-district analysis sample.
   Cluster bootstrap on 2011 parent (district_crosswalk shrug_key), 2000 draws.

Usage:
  C:/Users/mridu/anaconda3/envs/geo_env/python.exe scripts/108_d0_1997_followups.py

Outputs (outputs/revision_nov8/):
  d0_followups_raw.md
  y1997_state_season_area.csv
  d0_state_collapses.csv
  d0_endpoint_crop_drivers.csv
"""
import numpy as np
import pandas as pd

RAW = r"E:/CEEW Project/outputs/all_crops_apy_1997_2021_india_data_portal.csv"
REPO = r"D:/crop-diversity"
DIV = REPO + "/outputs/crop_diversity_analysis"
COV = REPO + "/outputs/shrug_covariates"
CG = REPO + "/outputs/crop_groups"
OUT = REPO + "/outputs/revision_nov8"
R = []


def w(s=""):
    print(s)
    R.append(str(s))


rng = np.random.default_rng(108)
NB = 2000

# ------------------------------------------------------------------ cleaning (76)
raw = pd.read_csv(RAW)
for c in ["year", "season", "state_name", "district_name", "crop_name", "crop_type"]:
    raw[c] = raw[c].astype(str).str.strip()
raw = raw.dropna(subset=["area"])
raw = raw[raw["area"] > 0]
BOGUS = [("Delhi", "Chandigarh"), ("Goa", "Chandigarh"),
         ("The Dadra And Nagar Haveli And Daman And Diu", "Chandigarh"),
         ("Delhi", "Surguja"), ("Goa", "Surguja"),
         ("The Dadra And Nagar Haveli And Daman And Diu", "Surguja"),
         ("Andaman And Nicobar Islands", "Purulia")]
REMAP = {("Uttar Pradesh", d): "Uttarakhand" for d in
         ["Almora", "Bageshwar", "Chamoli", "Champawat", "Dehradun", "Haridwar",
          "Nainital", "Pauri Garhwal", "Pithoragarh"]}
REMAP.update({("Madhya Pradesh", d): "Chhattisgarh" for d in
              ["Bastar", "Bilaspur", "Dhamtari", "Durg", "Jashpur", "Kanker",
               "Korba", "Mahasamund", "Raipur", "Rajnandgaon", "Surguja"]})
REMAP.update({("Bihar", d): "Jharkhand" for d in ["Deoghar", "Dhanbad", "Garhwa"]})
pair = list(zip(raw["state_name"], raw["district_name"]))
raw = raw[[p not in BOGUS for p in pair]].copy()
for (s_, d_), new in REMAP.items():
    m = (raw["state_name"] == s_) & (raw["district_name"] == d_)
    raw.loc[m, "state_name"] = new
raw["year_start"] = raw["year"].str.split("-").str[0].astype(int)
raw["district_key"] = raw["state_name"].str.upper() + "|" + raw["district_name"].str.upper()
raw = raw[raw["year_start"] <= 2019]
K = ["district_key", "year", "season", "crop_name"]
df = raw.sort_values("area", ascending=False).drop_duplicates(subset=K, keep="first").copy()

corr = pd.read_csv(DIV + "/district_diversity_indices_corrected.csv")
keys725 = set(corr["district_key"])
cw = pd.read_csv(COV + "/district_crosswalk.csv")[["div_key", "shrug_key"]]
clmap = dict(zip(cw["div_key"], cw["shrug_key"]))
fp = pd.read_csv(COV + "/final_panel.csv")
k606 = set(fp.loc[fp["in_final"] == True, "district_key"])
assert len(k606) == 606


def hill1(a):
    p = a / a.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def build_panel(d):
    dy = d.groupby(["district_key", "year_start", "crop_name"])["area"].sum().reset_index()
    dy["gca"] = dy.groupby(["district_key", "year_start"])["area"].transform("sum")
    dy["share"] = dy["area"] / dy["gca"]
    g = dy.groupby(["district_key", "year_start"])
    pan = pd.DataFrame({"gca": g["area"].sum(),
                        "D1": g["area"].apply(lambda s: hill1(s.values)),
                        "D0_raw": g.size()})
    pan["D0_0.5%"] = dy[dy["share"] >= 0.005].groupby(["district_key", "year_start"]).size()
    pan["D0_0.1%"] = dy[dy["share"] >= 0.001].groupby(["district_key", "year_start"]).size()
    pan["D0_1%"] = dy[dy["share"] >= 0.01].groupby(["district_key", "year_start"]).size()
    pan = pan.fillna(0).reset_index()
    pan = pan[pan["district_key"].isin(keys725)].copy()
    pan["cl"] = pan["district_key"].map(clmap).fillna(pan["district_key"])
    return pan, dy


pan, dy = build_panel(df)
YRS = sorted(pan["year_start"].unique())
cnt = pan.groupby("district_key")["year_start"].nunique()
bal = set(cnt[cnt == len(YRS)].index)
bp = pan[pan["district_key"].isin(bal)].copy()
bp["state"] = bp["district_key"].str.split("|").str[0]
MEAS = ["D0_raw", "D0_0.1%", "D0_0.5%", "D0_1%", "D1"]


def aw(g, m):
    return np.average(g[m], weights=g["gca"])


# ===================================================================== PART A
w("# Follow-ups: 1997-98 and the D0 endpoint")
w("")
w("## A. 1997-98")
w("")
wb = df[(df["state_name"] == "West Bengal") & (df["year_start"].isin([1997, 1998]))]
ns = wb[wb["crop_name"] == "Niger Seed"].groupby("year_start").agg(
    area=("area", "sum"), rows=("area", "size"), prod=("production", "sum"))
w("West Bengal Niger Seed: 1997-98 {:,.0f} ha in {} rows (production {:,.0f}); "
  "1998-99 {:,.0f} ha in {} rows. West Bengal all crops 1997-98 {:.1f} Mha against "
  "{:.1f} Mha in 1998-99.".format(
      ns.loc[1997, "area"], int(ns.loc[1997, "rows"]), ns.loc[1997, "prod"],
      ns.loc[1998, "area"], int(ns.loc[1998, "rows"]),
      wb[wb["year_start"] == 1997]["area"].sum() / 1e6,
      wb[wb["year_start"] == 1998]["area"].sum() / 1e6))
nsd = df[(df["state_name"] == "West Bengal") & (df["year_start"] == 1997)
         & (df["crop_name"] == "Niger Seed")]
w("Niger Seed rows in West Bengal 1997-98 by district (ha): " + "; ".join(
    "{} {:,.0f}".format(r["district_name"], r["area"]) for _, r in
    nsd.sort_values("area", ascending=False).head(6).iterrows()) + ".")
dy97 = dy[(dy["year_start"] == 1997) & dy["district_key"].str.startswith("WEST BENGAL")]
w("Niger Seed's share of district area in West Bengal 1997-98: median {:.3f}, min {:.3f} "
  "({} districts).".format(
      dy97[dy97["crop_name"] == "Niger Seed"]["share"].median(),
      dy97[dy97["crop_name"] == "Niger Seed"]["share"].min(),
      int((dy97["crop_name"] == "Niger Seed").sum())))
nat97 = df[df["year_start"] == 1997]["area"].sum()
w("That one crop is {:.0%} of all area recorded nationally in 1997-98 ({:.1f} Mha).".format(
    ns.loc[1997, "area"] / nat97, nat97 / 1e6))
w("")

# state x season area against later years
sse = df.groupby(["state_name", "year_start", "season"])["area"].sum().unstack("season").fillna(0) / 1e6
later = sse.loc[(slice(None), list(range(1999, 2004))), :].groupby("state_name").median()
s97 = sse.xs(1997, level="year_start")
s98 = sse.xs(1998, level="year_start")
ss = []
for st in sorted(set(later.index)):
    for se in sse.columns:
        l_ = later.loc[st, se]
        a97 = s97.loc[st, se] if st in s97.index else 0.0
        a98 = s98.loc[st, se] if st in s98.index else 0.0
        if l_ >= 0.05 or a97 >= 0.05:
            ss.append({"state": st, "season": se, "area97_mha": a97, "area98_mha": a98,
                       "median_1999_2003_mha": l_, "ratio97": a97 / l_ if l_ > 0 else np.nan})
ss = pd.DataFrame(ss)
ss.to_csv(OUT + "/y1997_state_season_area.csv", index=False)
miss = ss[(ss["ratio97"] < 0.5) & (ss["median_1999_2003_mha"] >= 0.2)].sort_values(
    "median_1999_2003_mha", ascending=False)
w("State-seasons with at least 0.2 Mha in a typical 1999-2003 year that hold under half "
  "of it in 1997-98 ({} cases, {:.1f} Mha missing against the 1999-2003 median):".format(
      len(miss), (miss["median_1999_2003_mha"] - miss["area97_mha"]).sum()))
w("")
w("| state | season | 1997-98 Mha | 1998-99 Mha | 1999-2003 median Mha | ratio 1997 |")
w("|---|---|---|---|---|---|")
for _, r in miss.iterrows():
    w("| {} | {} | {:.2f} | {:.2f} | {:.2f} | {:.2f} |".format(
        r["state"], r["season"], r["area97_mha"], r["area98_mha"],
        r["median_1999_2003_mha"], r["ratio97"]))
w("")
over = ss[(ss["ratio97"] > 2) & (ss["area97_mha"] >= 0.2)]
w("State-seasons more than double their 1999-2003 median in 1997-98: " + "; ".join(
    "{} {} {:.1f} vs {:.1f} Mha".format(r["state"], r["season"], r["area97_mha"],
                                        r["median_1999_2003_mha"])
    for _, r in over.iterrows()) + ".")
tot_later = df[df["year_start"].between(1999, 2003)].groupby("year_start")["area"].sum().median() / 1e6
w("National recorded area: 1997-98 {:.1f} Mha; without West Bengal Niger Seed {:.1f} Mha; "
  "1999-2003 median {:.1f} Mha.".format(nat97 / 1e6, (nat97 - ns.loc[1997, "area"]) / 1e6,
                                       tot_later))
w("")

# district-level: kharif missing
dse = df[df["district_key"].isin(bal)].groupby(["district_key", "year_start", "season"])["area"].sum().unstack("season").fillna(0)
dse["kharif_like"] = dse.get("Kharif", 0) + dse.get("Autumn", 0)
k97 = dse.xs(1997, level="year_start")["kharif_like"]
kl = dse.loc[(slice(None), list(range(1999, 2004))), "kharif_like"].groupby("district_key").median()
kk = pd.DataFrame({"k97": k97, "kl": kl}).dropna()
kk = kk[kk["kl"] > 0]
kk["r"] = kk["k97"] / kk["kl"]
w("Balanced-panel districts ({}): kharif plus autumn area in 1997-98 under half of the "
  "district's 1999-2003 median in {}; zero in {}.".format(
      len(kk), int((kk["r"] < 0.5).sum()), int((kk["k97"] == 0).sum())))
bp97 = bp[bp["year_start"] == 1997].set_index("district_key")
lat = bp[bp["year_start"].between(1999, 2003)].groupby("district_key")[MEAS + ["gca"]].mean()
cmp_ = bp97[MEAS + ["gca"]].join(lat, rsuffix="_l")
cmp_["khmiss"] = kk["r"].reindex(cmp_.index) < 0.5
cmp_["wb"] = cmp_.index.str.startswith("WEST BENGAL")
for lab, sub in [("kharif under half", cmp_[cmp_["khmiss"]]),
                 ("West Bengal", cmp_[cmp_["wb"]]),
                 ("all other", cmp_[~cmp_["khmiss"] & ~cmp_["wb"]])]:
    w("  {} (n {}): raw D0 1997 {:.1f} vs 1999-2003 mean {:.1f}; D1 {:.2f} vs {:.2f}; "
      "area ratio median {:.2f}.".format(
          lab, len(sub), sub["D0_raw"].mean(), sub["D0_raw_l"].mean(), sub["D1"].mean(),
          sub["D1_l"].mean(), (sub["gca"] / sub["gca_l"]).median()))
w("")

# national 1997 series under corrections
w("### 1997-98 national value under corrections (balanced panel, area-weighted)")
w("")
w("| version | n districts | D0 raw | D0 0.5% | D1 | D1 unweighted |")
w("|---|---|---|---|---|---|")
ref = bp[bp["year_start"].between(1998, 2004)]
w("| 1998-2004 mean of annual values | 429 | {:.2f} | {:.2f} | {:.2f} | {:.2f} |".format(
    np.mean([aw(g, "D0_raw") for _, g in ref.groupby("year_start")]),
    np.mean([aw(g, "D0_0.5%") for _, g in ref.groupby("year_start")]),
    np.mean([aw(g, "D1") for _, g in ref.groupby("year_start")]),
    np.mean([g["D1"].mean() for _, g in ref.groupby("year_start")])))
b97 = bp[bp["year_start"] == 1997]
w("| 1997-98 as published | {} | {:.2f} | {:.2f} | {:.2f} | {:.2f} |".format(
    len(b97), aw(b97, "D0_raw"), aw(b97, "D0_0.5%"), aw(b97, "D1"), b97["D1"].mean()))
# drop WB Niger seed 1997 rows
df_fix = df[~((df["state_name"] == "West Bengal") & (df["year_start"] == 1997)
              & (df["crop_name"] == "Niger Seed"))]
pf, _ = build_panel(df_fix[df_fix["year_start"] == 1997])
pf = pf[pf["district_key"].isin(bal)]
w("| 1997-98 without West Bengal Niger Seed rows | {} | {:.2f} | {:.2f} | {:.2f} | {:.2f} |".format(
    len(pf), aw(pf, "D0_raw"), aw(pf, "D0_0.5%"), aw(pf, "D1"), pf["D1"].mean()))
x = b97[~b97["district_key"].str.startswith("WEST BENGAL")]
w("| 1997-98 without West Bengal | {} | {:.2f} | {:.2f} | {:.2f} | {:.2f} |".format(
    len(x), aw(x, "D0_raw"), aw(x, "D0_0.5%"), aw(x, "D1"), x["D1"].mean()))
okk = set(cmp_.index[~cmp_["khmiss"].fillna(False) & ~cmp_["wb"]])
x = b97[b97["district_key"].isin(okk)]
x2 = ref[ref["district_key"].isin(okk)]
w("| 1997-98, districts with kharif present and outside West Bengal | {} | {:.2f} | {:.2f} | {:.2f} | {:.2f} |".format(
    len(x), aw(x, "D0_raw"), aw(x, "D0_0.5%"), aw(x, "D1"), x["D1"].mean()))
w("| same districts, 1998-2004 mean | {} | {:.2f} | {:.2f} | {:.2f} | {:.2f} |".format(
    len(okk), np.mean([aw(g, "D0_raw") for _, g in x2.groupby("year_start")]),
    np.mean([aw(g, "D0_0.5%") for _, g in x2.groupby("year_start")]),
    np.mean([aw(g, "D1") for _, g in x2.groupby("year_start")]),
    np.mean([g["D1"].mean() for _, g in x2.groupby("year_start")])))
w("")
w("Share of the 1997-98 area weight carried by West Bengal districts in the balanced "
  "panel: {:.0%} (against {:.0%} in 1998-99).".format(
      b97.loc[b97["district_key"].str.startswith("WEST BENGAL"), "gca"].sum() / b97["gca"].sum(),
      bp.loc[(bp["year_start"] == 1998) & bp["district_key"].str.startswith("WEST BENGAL"), "gca"].sum()
      / bp.loc[bp["year_start"] == 1998, "gca"].sum()))
w("")

# ===================================================================== PART B
w("## B. The raw D0 endpoint rise")
w("")
ser = {y: {m: aw(g, m) for m in MEAS} for y, g in bp.groupby("year_start")}
w("### Sensitivity to the years compared (area-weighted raw D0, balanced 429)")
w("")
w("| start \\ end | " + " | ".join(str(e) for e in range(2015, 2020)) + " |")
w("|---|" + "---|" * 5)
for s in range(1998, 2003):
    w("| {} | ".format(s) + " | ".join("{:+.2f}".format(ser[e]["D0_raw"] - ser[s]["D0_raw"])
                                       for e in range(2015, 2020)) + " |")
w("")
w("Same grid for D0 at 0.5%: 1998->2019 {:+.2f}, 1999->2019 {:+.2f}, 2000->2018 {:+.2f}.".format(
    ser[2019]["D0_0.5%"] - ser[1998]["D0_0.5%"], ser[2019]["D0_0.5%"] - ser[1999]["D0_0.5%"],
    ser[2018]["D0_0.5%"] - ser[2000]["D0_0.5%"]))
w("")

# collapses
sdf = df[df["district_key"].isin(keys725)]
sc = sdf.groupby(["state_name", "year_start"]).agg(names=("crop_name", "nunique"),
                                                   districts=("district_key", "nunique")).reset_index()
med = sc[sc["year_start"] >= 1998].groupby("state_name")["names"].median()
sc["med"] = sc["state_name"].map(med)
sc["ratio"] = sc["names"] / sc["med"]
col = sc[(sc["ratio"] < 0.6) & (sc["med"] >= 8)].sort_values(["state_name", "year_start"])
col.to_csv(OUT + "/d0_state_collapses.csv", index=False)
w("### State-years whose crop list collapses (distinct crop names under 60% of the "
  "state's own 1998-2019 median; states with a median of at least 8 names)")
w("")
w("{} state-years in {} states. 1997-98 alone: {}. 1998-99: {}. 2019-20: {}.".format(
    len(col), col["state_name"].nunique(),
    ", ".join(col[col["year_start"] == 1997]["state_name"]),
    ", ".join(col[col["year_start"] == 1998]["state_name"]) or "none",
    ", ".join(col[col["year_start"] == 2019]["state_name"]) or "none"))
w("")
w("| state | years collapsed (names / state median) |")
w("|---|---|")
for st, g in col.groupby("state_name"):
    w("| {} | {} |".format(st, ", ".join("{} ({}/{:.0f})".format(y, n, m) for y, n, m in
                                         zip(g["year_start"], g["names"], g["med"]))))
w("")
colset = set(zip(col["state_name"].str.upper(), col["year_start"]))
bp["collapse"] = [(s, y) in colset for s, y in zip(bp["state"], bp["year_start"])]
bal_states = set(bp["state"])
w("Balanced-panel district-years in collapsed state-years: {} of {} ({:.1%}); by year: {}.".format(
    int(bp["collapse"].sum()), len(bp), bp["collapse"].mean(),
    ", ".join("{} {}".format(y, n) for y, n in bp.groupby("year_start")["collapse"].sum().items() if n > 0)))
w("")

# crop-group defect flags at 2011 parent
fl = pd.read_csv(CG + "/crop_group_flags.csv")
flu = fl[fl["flag_annual_unreliable"] == 1]
flagset = set(zip(flu["unit_key"], flu["year_start"]))
bp["dflag"] = [(c, y) in flagset for c, y in zip(bp["cl"], bp["year_start"])]
w("Balanced-panel district-years whose 2011 parent carries flag_annual_unreliable: {} "
  "({:.1%}); 1998-99 {}, 2019-20 {}.".format(
      int(bp["dflag"].sum()), bp["dflag"].mean(),
      int(bp.loc[bp["year_start"] == 1998, "dflag"].sum()),
      int(bp.loc[bp["year_start"] == 2019, "dflag"].sum())))
w("")


# ---- bootstrap machinery on cluster-level sums
def endpoint_and_window(d, keys, label):
    """d: balanced-type panel with possibly NaN measures (dropped district-years).
    Returns rows of point, SE, CI for aw endpoint change 1998->2019 and window changes."""
    d = d[d["district_key"].isin(keys)].copy()
    cls = np.array(sorted(d["cl"].unique()))
    ci = {c: i for i, c in enumerate(cls)}
    d["ci"] = d["cl"].map(ci)
    nC = len(cls)
    counts = rng.multinomial(nC, np.ones(nC) / nC, size=NB).astype(float)
    out = []
    for m in MEAS:
        # endpoint
        res = {}
        for y in [1998, 2019]:
            x = d[(d["year_start"] == y) & d[m].notna()]
            S = np.bincount(x["ci"], weights=x["gca"] * x[m], minlength=nC)
            G = np.bincount(x["ci"], weights=x["gca"], minlength=nC)
            res[y] = (S, G)
        pt = res[2019][0].sum() / res[2019][1].sum() - res[1998][0].sum() / res[1998][1].sum()
        bt = (counts @ res[2019][0]) / (counts @ res[2019][1]) - (counts @ res[1998][0]) / (counts @ res[1998][1])
        n_ = int(d[(d["year_start"].isin([1998, 2019])) & d[m].notna()]["district_key"].nunique())
        out.append([label, m, "aw 1998->2019", pt, bt.std(), *np.quantile(bt, [.025, .975]), n_])
        # windows
        e = d[d["year_start"].between(1998, 2004) & d[m].notna()].groupby("district_key").agg(v=(m, "mean"), a=("gca", "mean"), ci=("ci", "first"))
        l_ = d[d["year_start"].between(2013, 2019) & d[m].notna()].groupby("district_key")[m].mean()
        dd = e.join(l_.rename("v2"), how="inner")
        dd["d"] = dd["v2"] - dd["v"]
        for lab2, wts in [("aw window", dd["a"].values), ("mean window", np.ones(len(dd)))]:
            S = np.bincount(dd["ci"], weights=wts * dd["d"], minlength=nC)
            G = np.bincount(dd["ci"], weights=wts, minlength=nC)
            pt = S.sum() / G.sum()
            bt = (counts @ S) / (counts @ G)
            out.append([label, m, lab2, pt, bt.std(), *np.quantile(bt, [.025, .975]), len(dd)])
    return out


rows = []
base = bp.copy()
rows += endpoint_and_window(base, bal, "balanced 429, as published")
v = bp.copy()
v.loc[v["collapse"], MEAS] = np.nan
rows += endpoint_and_window(v, bal, "drop collapsed state-years")
v2 = v.copy()
v2.loc[v2["dflag"], MEAS] = np.nan
rows += endpoint_and_window(v2, bal, "drop collapsed + flagged")
rows += endpoint_and_window(base, bal & k606, "balanced within the 606 sample")
rows += endpoint_and_window(v2, bal & k606, "606, drop collapsed + flagged")
RT = pd.DataFrame(rows, columns=["sample", "measure", "statistic", "change", "se", "lo", "hi", "n"])
RT.to_csv(OUT + "/d0_endpoint_robustness.csv", index=False)
w("### Change in each measure, cluster bootstrap ({} draws on 2011 parent)".format(NB))
w("")
w("Dropped district-years are set missing; the endpoint uses the districts present in "
  "each endpoint year, windows use districts with at least one year in each window.")
w("")
w("| sample | statistic | D0 raw | D0 0.1% | D0 0.5% | D0 1% | D1 | n districts |")
w("|---|---|---|---|---|---|---|---|")
for (s, st), g in RT.groupby(["sample", "statistic"], sort=False):
    g = g.set_index("measure")
    w("| {} | {} | ".format(s, st) + " | ".join(
        "{:+.2f} ({:.2f}) [{:+.2f}, {:+.2f}]".format(g.loc[m, "change"], g.loc[m, "se"],
                                                    g.loc[m, "lo"], g.loc[m, "hi"])
        for m in MEAS) + " | {} |".format(int(g.loc["D0_raw", "n"])))
w("")

# ---- endpoint decomposition: share bins and crops, 1998 vs 2019
bdy = dy[dy["district_key"].isin(bal) & dy["year_start"].isin([1998, 2019])].copy()
BINS = [0, 0.001, 0.005, 0.01, 0.05, 1.01]
BL = ["<0.1%", "0.1-0.5%", "0.5-1%", "1-5%", ">=5%"]
bdy["bin"] = pd.cut(bdy["share"], BINS, labels=BL, right=False)
g98 = bp[bp["year_start"] == 1998].set_index("district_key")["gca"]
g19 = bp[bp["year_start"] == 2019].set_index("district_key")["gca"]
bdy["wt"] = np.where(bdy["year_start"] == 1998, bdy["district_key"].map(g98) / g98.sum(),
                     bdy["district_key"].map(g19) / g19.sum())
w("### Where the 1998->2019 area-weighted raw D0 rise sits")
w("")
w("Area-weighted mean count of crops per district in each share-of-district-area bin.")
w("")
w("| bin | 1998-99 | 2019-20 | change |")
w("|---|---|---|---|")
for b in BL:
    a = bdy[(bdy["bin"] == b) & (bdy["year_start"] == 1998)]["wt"].sum()
    c = bdy[(bdy["bin"] == b) & (bdy["year_start"] == 2019)]["wt"].sum()
    w("| {} | {:.2f} | {:.2f} | {:+.2f} |".format(b, a, c, c - a))
w("")
crp = bdy.groupby(["crop_name", "year_start"])["wt"].sum().unstack().fillna(0)
crp.columns = ["aw_presence_1998", "aw_presence_2019"]
crp["change"] = crp["aw_presence_2019"] - crp["aw_presence_1998"]
s19 = bdy[bdy["year_start"] == 2019].groupby("crop_name")
crp["median_share_2019"] = s19["share"].median()
crp["frac_2019_below_0.5%"] = s19["share"].apply(lambda s: (s < 0.005).mean())
crp["area_1998_kha"] = bdy[bdy["year_start"] == 1998].groupby("crop_name")["area"].sum() / 1e3
crp["area_2019_kha"] = s19["area"].sum() / 1e3
crp = crp.sort_values("change", ascending=False)
crp.to_csv(OUT + "/d0_endpoint_crop_drivers.csv")
w("Crops carrying the 1998->2019 change (area-weighted presence = area-weighted share of "
  "districts recording the crop; changes sum to the D0 change {:+.2f}). Gross gains {:+.2f}, "
  "gross losses {:+.2f}.".format(crp["change"].sum(), crp.loc[crp["change"] > 0, "change"].sum(),
                                 crp.loc[crp["change"] < 0, "change"].sum()))
w("")
w("| crop | aw presence 1998 | 2019 | change | median share 2019 when present | 2019 presences <0.5% | area 1998 kha | area 2019 kha |")
w("|---|---|---|---|---|---|---|---|")
for c, r in pd.concat([crp.head(15), crp.tail(5)]).iterrows():
    w("| {} | {:.2f} | {:.2f} | {:+.3f} | {:.3%} | {:.0%} | {:,.0f} | {:,.0f} |".format(
        c, r["aw_presence_1998"], r["aw_presence_2019"], r["change"],
        r["median_share_2019"] if pd.notna(r["median_share_2019"]) else 0,
        r["frac_2019_below_0.5%"] if pd.notna(r["frac_2019_below_0.5%"]) else 0,
        r["area_1998_kha"] if pd.notna(r["area_1998_kha"]) else 0,
        r["area_2019_kha"] if pd.notna(r["area_2019_kha"]) else 0))
w("")
top10 = crp.head(10)
w("Top 10 gaining crops together: {:+.2f} of the {:+.2f}; their 2019 area {:,.0f} kha, "
  "{:.1%} of the balanced panel's 2019 area.".format(
      top10["change"].sum(), crp["change"].sum(), top10["area_2019_kha"].fillna(0).sum(),
      top10["area_2019_kha"].fillna(0).sum() * 1e3 / g19.sum()))
# by state contribution to endpoint
st_c = []
for st, g in bp[bp["year_start"].isin([1998, 2019])].groupby("state"):
    a = g[g["year_start"] == 1998]
    b = g[g["year_start"] == 2019]
    st_c.append((st, (b["gca"] * b["D0_raw"]).sum() / g19.sum() - (a["gca"] * a["D0_raw"]).sum() / g98.sum(),
                 a["D0_raw"].mean(), b["D0_raw"].mean(), len(a)))
st_c = pd.DataFrame(st_c, columns=["state", "contrib", "D0_1998", "D0_2019", "n"]).sort_values("contrib", ascending=False)
w("State contributions to the area-weighted 1998->2019 raw D0 change (include the effect "
  "of shifting area weights): " + "; ".join(
      "{} {:+.2f} (mean D0 {:.1f} -> {:.1f}, n {})".format(r["state"].title(), r["contrib"], r["D0_1998"], r["D0_2019"], r["n"])
      for _, r in st_c.head(6).iterrows()) + ". Largest negative: " + "; ".join(
      "{} {:+.2f}".format(r["state"].title(), r["contrib"]) for _, r in st_c.tail(3).iterrows()) + ".")
w("")

# ---- year-on-year within district: D0 up and D1 not
x = bp[bp["year_start"] >= 1998].sort_values(["district_key", "year_start"]).copy()
x["dD0"] = x.groupby("district_key")["D0_raw"].diff()
x["dD05"] = x.groupby("district_key")["D0_0.5%"].diff()
x["dD1"] = x.groupby("district_key")["D1"].diff()
x = x.dropna(subset=["dD0"])
up = x["dD0"] > 0
w("### Year-on-year within district (balanced, 1998-2019, {:,} district-year changes)".format(len(x)))
w("")
w("Raw D0 rises in {:.0%} of changes; in {:.0%} of those D1 falls or holds. Among raw D0 "
  "rises of 3 or more crops ({:,}), D1 falls or holds in {:.0%}, and D0 at 0.5% does not "
  "rise in {:.0%}. Correlation of year-on-year changes: raw D0 with D1 {:.2f}; D0 0.5% with "
  "D1 {:.2f}.".format(
      up.mean(), (x.loc[up, "dD1"] <= 0).mean(), int((x["dD0"] >= 3).sum()),
      (x.loc[x["dD0"] >= 3, "dD1"] <= 0).mean(), (x.loc[x["dD0"] >= 3, "dD05"] <= 0).mean(),
      x["dD0"].corr(x["dD1"]), x["dD05"].corr(x["dD1"])))
w("")

with open(OUT + "/d0_followups_raw.md", "w", encoding="utf-8") as fh:
    fh.write("\n".join(R))
print("wrote", OUT + "/d0_followups_raw.md")
