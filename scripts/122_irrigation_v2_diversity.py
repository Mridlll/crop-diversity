"""
122_irrigation_v2_diversity.py

District-year and window-average diversity from the extended APY file
(data_raw/apy_extended_1997_2024.csv, script 120), with script 76's cleaning
copied here unchanged (bogus pairs, state remaps, duplicates collapsed by maximum,
seasons summed per crop, indices per district-year then averaged).

Crop-list variants (all computed per district-year):
  all     every crop, as on the site
  drop6   niger seed, coriander, garlic, turmeric, cashewnut, banana removed in EVERY
          year (the crops the source drops from 2020-21 on)
  stable  crop kept in a state only if the state reports it in every unflagged year of
          the window (state-specific stable list; window b only)

Flagged state-years (dropped in any window that reaches 2020-21; list agreed with the
user from apy_extend_log.md): Manipur, Madhya Pradesh, Jammu and Kashmir 2020-22;
Meghalaya 2021; Haryana 2022; DNH and DD 2021-22; Nagaland 2021.

Outputs (outputs/revision_nov8/irrigation_v2/):
  div_district_year.csv     district-year indices, variants all and drop6
  div_windows.csv           district averages per window x variant
  div_build_log.md

Usage:
  C:/Users/mridu/anaconda3/envs/geo_env/python.exe scripts/122_irrigation_v2_diversity.py
"""
import os

import numpy as np
import pandas as pd

REPO = r"D:/crop-diversity"
RAW = REPO + "/data_raw/apy_extended_1997_2024.csv"
OUT = REPO + "/outputs/revision_nov8/irrigation_v2"
os.makedirs(OUT, exist_ok=True)
LOG = []


def w(s=""):
    print(s)
    LOG.append(str(s))


# ------------------------------------------------------------ script 76 cleaning
df = pd.read_csv(RAW)
for c in ["year", "season", "state_name", "district_name", "crop_name", "crop_type"]:
    df[c] = df[c].astype(str).str.strip()
df = df.dropna(subset=["area"])
df = df[df["area"] > 0]
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
pair = list(zip(df["state_name"], df["district_name"]))
df = df[[p not in BOGUS for p in pair]]
for (s_, d_), new in REMAP.items():
    m = (df["state_name"] == s_) & (df["district_name"] == d_)
    df.loc[m, "state_name"] = new
df["year_start"] = df["year"].str.split("-").str[0].astype(int)
df = df[df["year_start"].between(1998, 2022)]          # 1997-98 dropped (_years.py)
df["district_key"] = df["state_name"].str.upper() + "|" + df["district_name"].str.upper()
K = ["district_key", "year", "season", "crop_name"]
df = df.sort_values("area", ascending=False).drop_duplicates(subset=K, keep="first")
w("# Diversity build on the extended APY file")
w("")
w("Rows after script 76 cleaning, 1998-99 to 2022-23: {:,}.".format(len(df)))

# ------------------------------------------------------------ flags
SU = df["state_name"].str.upper()
FLAG = {"MANIPUR": [2020, 2021, 2022], "MADHYA PRADESH": [2020, 2021, 2022],
        "JAMMU AND KASHMIR": [2020, 2021, 2022], "MEGHALAYA": [2021], "HARYANA": [2022],
        "THE DADRA AND NAGAR HAVELI AND DAMAN AND DIU": [2021, 2022], "NAGALAND": [2021]}
flag = np.zeros(len(df), bool)
for s, ys in FLAG.items():
    flag |= (SU == s).values & df["year_start"].isin(ys).values
df["flag_sy"] = flag
w("State names in the flag list found: {}.".format(
    sorted(set(FLAG) & set(SU.unique()))))
w("Rows in flagged state-years: {:,}.".format(int(flag.sum())))

DROP6 = ["Niger Seed", "Coriander", "Garlic", "Turmeric", "Cashewnut", "Banana"]
miss = [c for c in DROP6 if c not in set(df["crop_name"])]
assert not miss, miss
crop_type = df.groupby("crop_name")["crop_type"].agg(lambda s: s.mode().iloc[0])


def indices(dy):
    """dy: district_key, year_start, crop_name, area (seasons summed)."""
    g = dy.groupby(["district_key", "year_start"])["area"]
    gca = g.transform("sum")
    p = dy["area"] / gca
    t = dy.assign(p=p, plnp=-p * np.log(p), p2=p * p, ge05=(p >= 0.005).astype(int))
    a = t.groupby(["district_key", "year_start"]).agg(
        D0=("crop_name", "size"), H=("plnp", "sum"), S2=("p2", "sum"),
        D0_05=("ge05", "sum"), gca=("area", "sum")).reset_index()
    a["D1"] = np.exp(a["H"])
    a["D2"] = 1.0 / a["S2"]
    a["E"] = a["D1"] / a["D0"]
    return a.drop(columns=["H", "S2"])


def cats(dy):
    """pooled category shares over the rows given (district level)."""
    t = dy.assign(ct=dy["crop_name"].map(crop_type))
    s = t.groupby(["district_key", "ct"])["area"].sum().unstack(fill_value=0)
    s = s.div(s.sum(axis=1), axis=0)
    s.columns = ["share_" + c.lower().replace(" ", "_") for c in s.columns]
    return s[["share_cereals", "share_pulses", "share_oilseeds"]]


base = df.groupby(["district_key", "state_name", "year_start", "crop_name", "flag_sy"],
                  as_index=False)["area"].sum()
v_all = indices(base)
v_d6 = indices(base[~base["crop_name"].isin(DROP6)])
st = base.drop_duplicates(["district_key", "year_start"]).set_index(
    ["district_key", "year_start"])[["state_name", "flag_sy"]]
dyp = (v_all.set_index(["district_key", "year_start"])
       .join(v_d6.set_index(["district_key", "year_start"]), rsuffix="_d6")
       .join(st).reset_index())
dyp.to_csv(OUT + "/div_district_year.csv", index=False)
w("District-year rows: {:,}.".format(len(dyp)))

# ------------------------------------------------------------ reproduce the site
cor = pd.read_csv(REPO + "/outputs/crop_diversity_analysis/district_diversity_indices_corrected.csv")
full = dyp[dyp["year_start"].between(1998, 2019)].groupby("district_key")[
    ["D0", "D1", "D2", "E"]].mean()
chk = cor.set_index("district_key")[["D0_richness", "D1_exp_shannon", "D2_inv_simpson",
                                     "evenness_D1_D0"]].join(full, how="inner")
md = {a: float(np.abs(chk[a] - chk[b]).max()) for a, b in
      [("D0_richness", "D0"), ("D1_exp_shannon", "D1"), ("D2_inv_simpson", "D2"),
       ("evenness_D1_D0", "E")]}
w("Full period 1998-99 to 2019-20 rebuilt from the extended file against the site's "
  "corrected indices ({} districts): max abs difference {}.".format(
      len(chk), {k: round(v, 6) for k, v in md.items()}))
nbad = int((np.abs(chk["D1_exp_shannon"] - chk["D1"]) > 1e-6).sum())
w("Districts differing in D1 by more than 1e-6: {}.".format(nbad))

# ------------------------------------------------------------ windows
WIN = {"full": (1998, 2019), "w1517": (2015, 2019), "w1721": (2017, 2021),
       "cs0913": (2009, 2013), "e9802": (1998, 2002)}
MINY = {"full": 10, "w1517": 3, "w1721": 3, "cs0913": 3, "e9802": 3}
rows = []


def window(tag, variant, y0, y1, drop_flags, crop_filter=None):
    b = base[base["year_start"].between(y0, y1)]
    if drop_flags:
        b = b[~b["flag_sy"]]
    if crop_filter is not None:
        b = crop_filter(b)
    a = indices(b)
    m = a.groupby("district_key").agg(D0=("D0", "mean"), D1=("D1", "mean"), D2=("D2", "mean"),
                                      E=("E", "mean"), D0_05=("D0_05", "mean"),
                                      gca=("gca", "mean"), n_years=("year_start", "nunique"))
    m = m.join(cats(b))
    m = m[m["n_years"] >= MINY[tag]]
    m["window"], m["variant"] = tag, variant
    rows.append(m.reset_index())
    return m


for tag, (y0, y1) in WIN.items():
    window(tag, "all", y0, y1, False)
    window(tag, "drop6", y0, y1, False, lambda b: b[~b["crop_name"].isin(DROP6)])
# window b: the flag and stable-list variants
window("w1721", "flags", 2017, 2021, True)
window("w1721", "flags_drop6", 2017, 2021, True, lambda b: b[~b["crop_name"].isin(DROP6)])


def stable(b):
    ny = b.groupby("state_name")["year_start"].nunique()
    sc = b.groupby(["state_name", "crop_name"])["year_start"].nunique().reset_index()
    sc = sc[sc["year_start"] == sc["state_name"].map(ny)]
    keep = set(zip(sc["state_name"], sc["crop_name"]))
    return b[[k in keep for k in zip(b["state_name"], b["crop_name"])]]


window("w1721", "flags_stable", 2017, 2021, True, stable)
window("w1517", "stable", 2015, 2019, False, stable)
W = pd.concat(rows, ignore_index=True)
W.to_csv(OUT + "/div_windows.csv", index=False)
w("")
w("Window x variant district counts:")
w("")
w("| window | variant | districts | mean D0 | mean D0 0.5% | mean D1 | mean E |")
w("|---|---|---|---|---|---|---|")
for (t, v), g in W.groupby(["window", "variant"], sort=False):
    w("| {} | {} | {} | {:.2f} | {:.2f} | {:.3f} | {:.3f} |".format(
        t, v, len(g), g["D0"].mean(), g["D0_05"].mean(), g["D1"].mean(), g["E"].mean()))

# national mean crops per district-year, by variant, to show the 2020-21 break
w("")
w("Mean crops per district-year (unweighted), all crops / six crops removed:")
w("")
w("| year | D0 all | D0 drop6 | D0 0.5% all | D1 all | D1 drop6 |")
w("|---|---|---|---|---|---|")
for y, g in dyp[dyp["year_start"] >= 2013].groupby("year_start"):
    g2 = g[~g["flag_sy"]]
    w("| {}-{} | {:.2f} | {:.2f} | {:.2f} | {:.3f} | {:.3f} |".format(
        y, str(y + 1)[-2:], g2["D0"].mean(), g2["D0_d6"].mean(), g2["D0_05"].mean(),
        g2["D1"].mean(), g2["D1_d6"].mean()))
open(OUT + "/div_build_log.md", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
