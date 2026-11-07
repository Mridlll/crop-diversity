"""
120_apy_extend.py

Extends the district APY series past 2019-20.

SOURCE
  India Data Portal CKAN package "area-production-yield-apy"
  https://ckandev.indiadataportal.com/dataset/area-production-yield-apy
  resource f980409d-49a2-42ae-9eb0-182365005c04 "Crop Wise Area Production Yield"
  (DES district-wise data, 1997-98 to 2022-23, 115 crop names, hectares/tonnes).
  Downloaded to data_raw/apy_2020plus/dwes.csv. The package's second resource
  (UPAG, 30 crops, 2013-14 to 2024-25) is read only for a cross-check; its 30-crop
  list covers 28 of the 54 crop names the site uses (52 percent, apy_extend_log.md) and fails the project's own
  60 percent test, so it is not merged.

WHAT IT DOES
  1. Harmonises the new file to the old file's schema (crop names, season labels,
     units, year format, district keys).
  2. Writes data_raw/apy_extended_1997_2024.csv: old rows 1997-98..2019-20 unchanged,
     the 319-row 2020-21 stub removed, new 2020-21..2022-23 rows appended.
     (File name says 2024; the last year actually available from DES is 2022-23.)
  3. Overlap check on 1997-98..2019-20, collapse flags on 2020-21..2022-23, and a log.

Outputs
  data_raw/apy_extended_1997_2024.csv
  outputs/revision_nov8/apy_extend_log.md
  outputs/revision_nov8/apy_extend_state_year_flags.csv
  outputs/revision_nov8/apy_extend_overlap_districts.csv
  outputs/revision_nov8/apy_extend_unmapped.csv
"""
import difflib
import re

import numpy as np
import pandas as pd

OLD = r"E:/CEEW Project/outputs/all_crops_apy_1997_2021_india_data_portal.csv"
REPO = r"D:/crop-diversity"
NEW = REPO + "/data_raw/apy_2020plus/dwes.csv"
UPAG = REPO + "/data_raw/apy_2020plus/upag.csv"
OUTCSV = REPO + "/data_raw/apy_extended_1997_2024.csv"
OUT = REPO + "/outputs/revision_nov8"
NEW_YEARS = [2020, 2021, 2022]

L = []


def w(s=""):
    print(s)
    L.append(str(s))


def strip(df, cols):
    for c in cols:
        df[c] = df[c].astype(str).str.strip()
    return df


old = pd.read_csv(OLD)
old_raw = old.copy()
new = pd.read_csv(NEW)
cols = list(old.columns)
S = ["year", "season", "state_name", "district_name", "crop_name", "crop_type"]
old = strip(old, S)
new = strip(new, S)
old["y"] = old["year"].str[:4].astype(int)
new["y"] = new["year"].str[:4].astype(int)

w("# APY extension past 2019-20")
w("")
w("Old file: {:,} rows. New file (DES via India Data Portal): {:,} rows, years {}-{}.".format(
    len(old), len(new), new["y"].min(), new["y"].max()))

# ---------------------------------------------------------------- units
w("")
w("## Units")
w("")
w("Old area/production units: {} / {}. New: {} / {}. Both hectares and tonnes; no".format(
    sorted(old.area_unit.unique()), sorted(old.production_unit.unique()),
    sorted(new.area_unit.unique()), sorted(new.production_unit.unique())))
w("conversion needed. Yield unit in both: {} / {}.".format(
    sorted(old.yield_unit.unique()), sorted(new.yield_unit.unique())))

# --------------------------------------------------------------- seasons
w("")
w("## Seasons")
w("")
w("Old labels: {}. New labels: {}.".format(sorted(old.season.unique()),
                                          sorted(new.season.unique())))
tot = new[new.season == "Total"]
oth = new[new.season != "Total"]
only_tot = (set(zip(tot.year, tot.district_name, tot.crop_name)) -
            set(zip(oth.year, oth.district_name, oth.crop_name)))
w("The new file adds a season label 'Total' ({:,} rows, {:.1f} Mha) that is the sum of a".format(
    len(tot), tot.area.sum() / 1e6))
w("crop's seasonal rows in the same district-year; the old file has no such rows.")
w("Dropped to avoid double counting. Total-only cells with no seasonal row to stand in for them: {}.".format(
    len(only_tot)))
new = new[new.season != "Total"].copy()
pad = {s_.strip(): s_ for s_ in old_raw.season.unique()}
yr_fmt = old_raw["year"].iloc[0]
w("Season labels and year strings are written in the old file's padded format (e.g. {!r}, {!r}).".format(
    pad["Kharif"], yr_fmt))

# ---------------------------------------------------------- script 76 cleaning
BOGUS = [("Delhi", "Chandigarh"), ("Goa", "Chandigarh"),
         ("The Dadra And Nagar Haveli And Daman And Diu", "Chandigarh"),
         ("Delhi", "Surguja"), ("Goa", "Surguja"),
         ("The Dadra And Nagar Haveli And Daman And Diu", "Surguja"),
         ("Andaman And Nicobar Islands", "Purulia")]
pair = list(zip(new.state_name, new.district_name))
nb = sum(p in BOGUS for p in pair)
new = new[[p not in BOGUS for p in pair]].copy()
w("")
w("Bogus state-district pairs (script 76 list) found in new file: {} rows.".format(nb))

# ------------------------------------------------------------------ crops
OLD_CROPS = sorted(old.crop_name.unique())
CROP_MAP = {c: c for c in OLD_CROPS}
# the one genuine spelling difference: old file has a double space
CROP_MAP["Other Rabi Pulses"] = "Other  Rabi Pulses"
# Deliberately NOT mapped (would double count or were never in the old file):
#   'Arcanut (Processed)' is arecanut again in processed form (first appears 2020-21),
#   'Atcanut (Raw)', 'Cashewnut Raw/Processed' likewise; 'Dry Ginger' is ginger
#   again; 'Kapas' is unginned cotton (seed cotton) not lint; 'Jute & Mesta' pools two
#   crops the old file keeps apart.
crop_meta = (old.groupby("crop_name").agg(crop_code=("crop_code", lambda s: s.mode().iat[0]),
                                          crop_type=("crop_type", lambda s: s.mode().iat[0])))
new["crop_old"] = new["crop_name"].map(CROP_MAP)
um = new[new.crop_old.isna()]
umt = (um.groupby(["crop_type", "crop_name"]).agg(area_ha=("area", "sum"), rows=("area", "size"))
         .reset_index().sort_values("area_ha", ascending=False))
um_y = um[um.y >= 2020].groupby(["crop_name", "y"]).area.sum().unstack(fill_value=0)
umt.to_csv(OUT + "/apy_extend_unmapped.csv", index=False)
w("")
w("## Crop names")
w("")
w("Old file has {} crop names; new file {}. 53 are identical, 1 differs by whitespace".format(
    len(OLD_CROPS), new.crop_name.nunique()))
w("('Other Rabi Pulses' -> 'Other  Rabi Pulses'). {} new-file crop names have no old".format(
    umt.crop_name.nunique()))
w("counterpart and are excluded. Excluded area, all years: {:.1f} Mha of {:.1f} Mha ({:.2f} percent).".format(
    um.area.sum() / 1e6, new.area.sum() / 1e6, 100 * um.area.sum() / new.area.sum()))
w("")
w("Excluded crops with area in 2020-21..2022-23 (hectares; full list in apy_extend_unmapped.csv):")
w("")
w("| crop | " + " | ".join("{}-{}".format(y, str(y + 1)[-2:]) for y in NEW_YEARS) + " |")
w("|---|---|---|---|")
for c, r in um_y.sort_values(NEW_YEARS[-1], ascending=False).iterrows():
    if r.sum() > 0:
        w("| {} | {} |".format(c, " | ".join("{:,.0f}".format(r.get(y, 0)) for y in NEW_YEARS)))
w("")
w("Arecanut: from 2020-21 the new file carries BOTH 'Arecanut' and 'Arcanut (Processed)'")
w("(2022-23: 891k and 675k ha). Only 'Arecanut' is kept, so arecanut is not double counted.")
new = new[new.crop_old.notna()].copy()
new["crop_name"] = new["crop_old"]

# --------------------------------------------------------------- districts
old["dk"] = old.state_name.str.upper() + "|" + old.district_name.str.upper()
old_keys = sorted(old.dk.unique())
old_state = {k: s_ for k, s_ in zip(old.dk, old.state_name)}
old_dname = {k: d_ for k, d_ in zip(old.dk, old.district_name)}
old_codes = old.drop_duplicates("dk").set_index("dk")[["state_code", "district_code"]]


def norm(x):
    x = x.upper().replace("&", "AND")
    x = re.sub(r"[^A-Z ]", " ", x)
    x = re.sub(r"\b(DISTRICT|DIST)\b", " ", x)
    return re.sub(r"\s+", " ", x).strip()


# same alias list as script 104 (APY naming), plus the reverse direction
ALIAS = {"ANANTAPUR": "ANANTHAPURAMU", "KADAPA": "Y S R KADAPA", "DOHAD": "DAHOD",
         "GURGAON": "GURUGRAM", "MEWAT": "NUH", "BADGAM": "BUDGAM",
         "BANGALORE RURAL": "BENGALURU RURAL", "RAMANAGARA": "BENGALURU SOUTH",
         "HOSHANGABAD": "NARMADAPURAM", "KHANDWA": "KHANDWA EAST NIMAR",
         "KHARGONE": "KHARGONE WEST NIMAR", "NARSINGHPUR": "NARSIMHAPUR",
         "AHMEDNAGAR": "AHILYANAGAR", "OSMANABAD": "DHARASHIV", "SAIHA": "SIAHA",
         "BALESHWAR": "BALASORE", "KENDUJHAR": "KEONJHAR", "PONDICHERRY": "PUDUCHERRY",
         "MUKTSAR": "SRI MUKTSAR SAHIB", "TUTICORIN": "THOOTHUKKUDI",
         "BHADRADRI": "BHADRADRI KOTHAGUDEM", "JAYASHANKAR": "JAYASHANKAR BHUPALAPALLY",
         "JOGULAMBA": "JOGULAMBA GADWAL", "RAJANNA": "RAJANNA SIRCILLA",
         "YADADRI": "YADADRI BHUVANAGIRI", "ALLAHABAD": "PRAYAGRAJ", "FAIZABAD": "AYODHYA",
         "SANT RAVIDAS NAGAR": "BHADOHI", "BALODA BAZAR": "BALODABAZAR BHATAPARA",
         "KABIRDHAM": "KABEERDHAM", "KANKER": "UTTAR BASTAR KANKER",
         "GAURELLA PENDRA MARWAHI": "GAURELA PENDRA MARWAHI",
         "KARIMGANJ": "SRIBHUMI", "RANGAREDDI": "RANGA REDDY"}
ALIAS_REV = {v: k for k, v in ALIAS.items()}
# state-qualified aliases (script 104 writes these as "NAME|STATE"); (state, new name) -> old name
ALIAS_ST = {("MAHARASHTRA", "CHHATRAPATI SAMBHAJINAGAR"): "AURANGABAD",
            ("CHHATTISGARH", "BALRAMPUR RAMANUJGANJ"): "BALRAMPUR"}
old_by_state = {}
for k in old_keys:
    st, d_ = k.split("|")
    old_by_state.setdefault(st, {})[norm(d_)] = k

dm = {}
meth = {}
for st, d_ in sorted(set(zip(new.state_name, new.district_name))):
    key = st.upper() + "|" + d_.upper()
    if key in old_keys:
        dm[(st, d_)] = key
        meth[(st, d_)] = "exact"
        continue
    cands = old_by_state.get(st.upper(), {})
    n_ = norm(d_)
    hit = None
    for variant in (n_, ALIAS_ST.get((st.upper(), n_)), ALIAS.get(n_), ALIAS_REV.get(n_)):
        if variant and variant in cands:
            hit = cands[variant]
            how = "normalised" if variant == n_ else "alias"
            break
    if hit is None:
        best = difflib.get_close_matches(n_, list(cands), n=1, cutoff=0.85)
        if best:
            hit, how = cands[best[0]], "fuzzy"
    if hit is None:
        # same district filed under a different state in the old file (Telangana in AP, Ladakh in J&K)
        for st2 in ("ANDHRA PRADESH", "TELANGANA", "JAMMU AND KASHMIR", "LADAKH",
                    "MADHYA PRADESH", "CHHATTISGARH", "UTTAR PRADESH", "UTTARAKHAND",
                    "BIHAR", "JHARKHAND"):
            c2 = old_by_state.get(st2, {})
            if st2 != st.upper() and n_ in c2:
                hit, how = c2[n_], "cross_state"
                break
    if hit:
        dm[(st, d_)] = hit
        meth[(st, d_)] = how
    else:
        dm[(st, d_)] = None
        meth[(st, d_)] = "none"

new["dk"] = [dm[(a, b)] for a, b in zip(new.state_name, new.district_name)]
new["dmethod"] = [meth[(a, b)] for a, b in zip(new.state_name, new.district_name)]
mm = pd.Series(meth).value_counts().to_dict()
w("")
w("## Districts")
w("")
w("Matching rule is script 104's: exact state|district key, else normalised name within")
w("state, else alias list, else difflib >= 0.85 within state, else same name in a state the")
w("old file may have filed it under. New-file (state, district) pairs by method: {}.".format(mm))
new["dk_final"] = new["dk"].fillna(new.state_name.str.upper() + "|" + new.district_name.str.upper())
unm = new[new.dk.isna()]
w("")
if len(unm):
    ua = (unm[unm.y >= 2020].groupby(["state_name", "district_name"]).area.sum() / 3).round(0)
    allarea = new[new.y >= 2020].area.sum()
    w("Unmatched districts (no old key; kept in the output under their own name so they are")
    w("not lost, but they will not join to the existing district keys). Mean annual area")
    w("2020-23 in hectares; share of all new-year area {:.2f} percent:".format(
        100 * unm[unm.y >= 2020].area.sum() / allarea))
    w("")
    w("| state | district | mean annual ha | years present |")
    w("|---|---|---|---|")
    for (st, d_), v in ua.sort_values(ascending=False).items():
        yrs = sorted(unm[(unm.state_name == st) & (unm.district_name == d_)].y.unique())
        w("| {} | {} | {:,.0f} | {} |".format(st, d_, v, ",".join(str(y) for y in yrs)))
else:
    w("Every new-file district matched an old key.")
chg = [(k, dm[k], meth[k]) for k in dm if meth[k] in ("alias", "fuzzy", "cross_state", "normalised")]
if chg:
    w("")
    w("Non-exact matches (new -> old key):")
    w("")
    for (st, d_), k, m in chg:
        w("- {} | {} -> {} [{}]".format(st, d_, k, m))
# which old keys never appear in new 2020+
seen20 = set(new[new.y >= 2020].dk_final)
last_old = old[old.y == 2019].dk.unique()
gone = [k for k in last_old if k not in seen20]
w("")
w("Old-file districts present in 2019-20 but absent from 2020-23: {} of {}.".format(
    len(gone), len(last_old)))
if gone:
    ga = old[(old.y == 2019) & old.dk.isin(gone)].groupby("dk").area.sum().sort_values(ascending=False)
    w("Largest by 2019-20 area: " + "; ".join("{} ({:,.0f} ha)".format(k, v) for k, v in ga.head(10).items()))

# -------------------------------------------------- collapse duplicates for checks
K = ["dk_final", "y", "season", "crop_name"]
nd = int(new.duplicated(K).sum())
w("")
w("Duplicate (district, year, season, crop) rows in the new file after mapping: {}.".format(nd))
w("Left in the CSV as delivered; script 76 collapses them by maximum downstream.")

# ------------------------------------------------------------------ output
base = new[new.y.isin(NEW_YEARS)].copy()
rows = pd.DataFrame({
    "id": np.arange(old_raw["id"].max() + 1, old_raw["id"].max() + 1 + len(base)),
    "year": [yr_fmt.replace(yr_fmt.strip()[:7], "{}-{}".format(y, str(y + 1)[-2:]))
             for y in base.y],
    "state_name": [old_state.get(k, s_) for k, s_ in zip(base.dk, base.state_name)],
    "state_code": [old_codes.state_code.get(k, np.nan) if k else np.nan for k in base.dk],
    "district_name": [old_dname.get(k, d_) if k else d_ for k, d_ in zip(base.dk, base.district_name)],
    "district_code": [old_codes.district_code.get(k, np.nan) if k else np.nan for k in base.dk],
    "season": base.season.map(pad).values,
    "crop_code": base.crop_name.map(crop_meta.crop_code).values,
    "crop_name": base.crop_name.values,
    "crop_type": base.crop_name.map(crop_meta.crop_type).values,
    "area": base.area.values, "area_unit": "Hectare",
    "production": base.production.values, "production_unit": "Tonnes",
    "yield": base["yield"].values, "yield_unit": "Tonnes/Hectare",
})
# fall back to the new file's own codes where the old file has none
rows["state_code"] = rows["state_code"].fillna(pd.Series(base.state_code.values, index=rows.index))
rows["district_code"] = rows["district_code"].fillna(pd.Series(base.district_code.values, index=rows.index))
rows.index = base.index
rows["season"] = rows["season"].fillna(base.season)
if rows["season"].isna().any():
    raise SystemExit("season label missing from old file's set")
old_keep = old_raw[old.y.values <= 2019]
n_stub = int((old.y == 2020).sum())
ext = pd.concat([old_keep[cols], rows[cols]], ignore_index=True)
ext.to_csv(OUTCSV, index=False)
w("")
w("## Output")
w("")
w("`data_raw/apy_extended_1997_2024.csv`: {:,} old rows (1997-98..2019-20, byte-for-byte values,".format(len(old_keep)))
w("the {} stub rows of 2020-21 removed) + {:,} new rows for 2020-21..2022-23. File name".format(n_stub, len(rows)))
w("follows the request; no DES data past 2022-23 exists in this source.")
ny = rows.groupby("year").agg(rows=("area", "size"), mha=("area", lambda s: s.sum() / 1e6),
                              districts=("district_name", "nunique"), crops=("crop_name", "nunique"))
w("")
w("| year | rows | Mha | districts | crops |")
w("|---|---|---|---|---|")
for y, r in ny.iterrows():
    w("| {} | {:,} | {:.1f} | {} | {} |".format(y.strip(), int(r.rows), r.mha, int(r.districts), int(r.crops)))

# ---------------------------------------------------------------- overlap
w("")
w("## Overlap check, 1997-98 to 2019-20 (old file vs new file, same crops)")
w("")
o2 = old[old.y <= 2019].copy()
o2["dk_final"] = o2.dk
n2 = new[new.y <= 2019].copy()
Ko = ["dk_final", "y", "season", "crop_name"]
oc = o2.groupby(Ko).area.max().rename("area_old")
nc = n2.groupby(Ko).area.max().rename("area_new")
j = pd.concat([oc, nc], axis=1)
both = j.dropna()
w("Cells (district, year, season, crop): old {:,}, new {:,}, in both {:,}; only old {:,}; only new {:,}.".format(
    len(oc), len(nc), len(both), int(j.area_new.isna().sum()), int(j.area_old.isna().sum())))
rel = (both.area_new - both.area_old).abs() / both.area_old.clip(lower=1)
w("Cell level area: Pearson r = {:.5f}; identical in {:.1f} percent of shared cells;".format(
    both.area_old.corr(both.area_new), 100 * (both.area_new == both.area_old).mean()))
w("median abs percent difference {:.3f}; 99th percentile {:.2f} percent; cells differing by over 10 percent: {:,}.".format(
    100 * rel.median(), 100 * rel.quantile(.99), int((rel > .1).sum())))
bo = j.reset_index()
tot_o = bo.groupby(["dk_final", "y"]).area_old.sum(min_count=1)
tot_n = bo.groupby(["dk_final", "y"]).area_new.sum(min_count=1)
dt = pd.concat([tot_o, tot_n], axis=1).dropna()
dt["pct"] = 100 * (dt.area_new - dt.area_old) / dt.area_old.clip(lower=1)
w("District-year gross area totals ({:,} pairs): r = {:.5f}; median abs percent difference {:.3f};".format(
    len(dt), dt.area_old.corr(dt.area_new), dt.pct.abs().median()))
w("pairs differing by more than 5 percent: {:,}.".format(int((dt.pct.abs() > 5).sum())))
dd = dt.reset_index().sort_values("pct", key=lambda s: s.abs(), ascending=False)
dd.to_csv(OUT + "/apy_extend_overlap_districts.csv", index=False)
w("")
w("Worst ten district-years by absolute percent difference (full list in apy_extend_overlap_districts.csv):")
w("")
w("| district | year | old ha | new ha | pct |")
w("|---|---|---|---|---|")
for _, r in dd.head(10).iterrows():
    w("| {} | {}-{} | {:,.0f} | {:,.0f} | {:+.1f} |".format(
        r.dk_final, int(r.y), str(int(r.y) + 1)[-2:], r.area_old, r.area_new, r.pct))
# national year totals
yt = bo.groupby("y")[["area_old", "area_new"]].sum() / 1e6
yt["diff_pct"] = 100 * (yt.area_new - yt.area_old) / yt.area_old
w("")
w("National gross area by year, Mha (cells in both sources only would differ less; this is all cells):")
w("")
w("| year | old | new | pct |")
w("|---|---|---|---|")
for y, r in yt.iterrows():
    w("| {}-{} | {:.1f} | {:.1f} | {:+.1f} |".format(y, str(y + 1)[-2:], r.area_old, r.area_new, r.diff_pct))
# per crop
pc = bo.groupby("crop_name")[["area_old", "area_new"]].sum()
pc["pct"] = 100 * (pc.area_new - pc.area_old) / pc.area_old.clip(lower=1)
pcc = []
for c, g in bo.dropna().groupby("crop_name"):
    pcc.append((c, g.area_old.corr(g.area_new), (100 * (g.area_new - g.area_old).abs() / g.area_old.clip(lower=1)).median()))
pcc = pd.DataFrame(pcc, columns=["crop", "r", "med_pct"]).set_index("crop")
pc = pc.join(pcc)
w("")
w("Per crop, shared cells: lowest correlation / highest median difference (all 1997-2019):")
w("")
w("| crop | total old ha | total new ha | pct | r | median abs pct |")
w("|---|---|---|---|---|---|")
for c, r in pc.sort_values("r").head(8).iterrows():
    w("| {} | {:,.0f} | {:,.0f} | {:+.2f} | {:.4f} | {:.3f} |".format(c, r.area_old, r.area_new, r.pct, r.r, r.med_pct))
w("")
w("Largest single-cell disagreements (old vs new area, ha):")
w("")
bd = both.reset_index()
bd["d"] = (bd.area_new - bd.area_old).abs()
for _, r in bd.sort_values("d", ascending=False).head(12).iterrows():
    w("- {} {}-{} {} {}: old {:,.0f}, new {:,.0f}".format(r.dk_final, r.y, str(r.y + 1)[-2:], r.season, r.crop_name, r.area_old, r.area_new))
bd["rel"] = bd.d / bd.area_old.clip(lower=1)
w("")
w("Shared cells differing by more than 1 percent, by year: " + "; ".join(
    "{}:{}".format(y, int(n)) for y, n in bd[bd.rel > .01].groupby("y").size().items()))
w("Cells present in one source only, by year (old only / new only): " + "; ".join(
    "{}:{}/{}".format(y, int(g.area_new.isna().sum()), int(g.area_old.isna().sum()))
    for y, g in j.reset_index().groupby("y") if g.area_new.isna().sum() + g.area_old.isna().sum() > 150))
w("")
w("Years 2018-19 and 2019-20 only (the two years closest to the extension):")
rec = bo[bo.y >= 2018].dropna()
w("r = {:.5f}; identical cells {:.1f} percent; median abs percent difference {:.3f}.".format(
    rec.area_old.corr(rec.area_new), 100 * (rec.area_new == rec.area_old).mean(),
    (100 * (rec.area_new - rec.area_old).abs() / rec.area_old.clip(lower=1)).median()))

# ---------------------------------------------------------------- flags
w("")
w("## Collapse flags, state by state")
w("")
w("Computed on the extended file after script 76's collapse rule (duplicates -> maximum),")
w("seasons summed. Baseline is each state's 2015-16..2019-20 median. Thresholds:")
w("crop count < 60 percent of baseline; gross area outside 75-125 percent of baseline;")
w("districts reporting < 80 percent of baseline; any single crop in a district-year above")
w("1.5 times that district's baseline gross area (district area is not in the file, so the")
w("district's own 2015-20 gross cropped area stands in for it); any crop-level yield")
w("above 100 t/ha for a non-sugarcane, non-coconut, non-banana, non-tapioca crop.")
# build analysis frame on extended output (stripped), same cleaning
ex = ext.copy()
ex = strip(ex, S)
ex["y"] = ex["year"].str[:4].astype(int)
ex = ex[[(a, b) not in BOGUS for a, b in zip(ex.state_name, ex.district_name)]]
ex = ex[ex.area > 0]
ex["dk"] = ex.state_name.str.upper() + "|" + ex.district_name.str.upper()
ex = ex.sort_values("area", ascending=False).drop_duplicates(["dk", "y", "season", "crop_name"])
ex = ex[ex.y >= 1998]
sy = ex.groupby(["state_name", "y"]).agg(area=("area", "sum"),
                                         crops=("crop_name", "nunique"),
                                         dists=("dk", "nunique")).reset_index()
b = sy[sy.y.between(2015, 2019)].groupby("state_name").agg(
    b_area=("area", "median"), b_crops=("crops", "median"), b_dists=("dists", "median"))
sy = sy.join(b, on="state_name")
sy["crop_ratio"] = sy.crops / sy.b_crops
sy["area_ratio"] = sy.area / sy.b_area
sy["dist_ratio"] = sy.dists / sy.b_dists
dy = ex.groupby(["state_name", "dk", "y"]).area.sum().rename("gross").reset_index()
dbase = dy[dy.y.between(2015, 2019)].groupby("dk").gross.median().rename("d_base")
cy = ex.groupby(["state_name", "dk", "y", "crop_name"]).area.sum().reset_index().join(dbase, on="dk")
cy["big"] = cy.area > 1.5 * cy.d_base
bigc = cy[cy.big].groupby(["state_name", "y"]).size().rename("n_crop_gt_district")
sy = sy.join(bigc, on=["state_name", "y"])
sy["n_crop_gt_district"] = sy.n_crop_gt_district.fillna(0).astype(int)
yl = ex[~ex.crop_name.isin(["Sugarcane", "Coconut", "Banana", "Tapioca", "Potato", "Onion", "Sweet Potato"])
        & (ex["yield"] > 100)].groupby(["state_name", "y"]).size().rename("n_yield_gt_100")
sy = sy.join(yl, on=["state_name", "y"])
sy["n_yield_gt_100"] = sy.n_yield_gt_100.fillna(0).astype(int)


def reasons(r):
    out = []
    if r.crop_ratio < .6:
        out.append("crops {:.0f}pct of usual".format(100 * r.crop_ratio))
    if not (.75 <= r.area_ratio <= 1.25):
        out.append("area {:.0f}pct of 2015-19 median".format(100 * r.area_ratio))
    if r.dist_ratio < .8:
        out.append("districts {:.0f}pct of usual".format(100 * r.dist_ratio))
    if r.n_crop_gt_district > 0:
        out.append("{} crop-cells > 1.5x district baseline area".format(r.n_crop_gt_district))
    if r.n_yield_gt_100 > 0:
        out.append("{} yields > 100 t/ha".format(r.n_yield_gt_100))
    return "; ".join(out)


sy["flag_text"] = sy.apply(reasons, axis=1)
sy["flagged"] = sy["flag_text"] != ""
sy.to_csv(OUT + "/apy_extend_state_year_flags.csv", index=False)
bl_flag = sy[(sy.y.between(2015, 2019))]
w("")
w("Calibration: the same tests flag {} of {} state-years in 2015-16..2019-20 (by construction".format(
    int(bl_flag.flagged.sum()), len(bl_flag)))
w("mostly the single-crop and yield tests, since the baseline is those years).")
nsy = sy[sy.y.isin(NEW_YEARS)]
w("")
w("New years: {} of {} state-years flagged.".format(int(nsy.flagged.sum()), len(nsy)))
w("")
w("| state | year | crops | usual | area ratio | districts ratio | flags |")
w("|---|---|---|---|---|---|---|")
for _, r in nsy[nsy.flagged].sort_values(["state_name", "y"]).iterrows():
    w("| {} | {}-{} | {} | {:.0f} | {:.2f} | {:.2f} | {} |".format(
        r.state_name, r.y, str(r.y + 1)[-2:], r.crops, r.b_crops, r.area_ratio, r.dist_ratio, r.flag_text))
w("")
w("Unflagged new state-years: " + ", ".join(
    "{} {}".format(r.state_name, r.y) for _, r in nsy[~nsy.flagged].sort_values(["state_name", "y"]).iterrows()))
# national
w("")
w("National gross area, Mha: " + "; ".join(
    "{}-{} {:.1f}".format(y, str(y + 1)[-2:], v / 1e6) for y, v in
    sy.groupby("y").area.sum().loc[2015:].items()))
w("National mean crops per district-year: " + "; ".join(
    "{}-{} {:.1f}".format(y, str(y + 1)[-2:], v) for y, v in
    ex.groupby(["dk", "y"]).crop_name.nunique().groupby("y").mean().loc[2015:].items()))

# crop-level drops in new years
cb = ex.groupby(["crop_name", "y"]).area.sum().unstack()
base_c = cb[[2015, 2016, 2017, 2018, 2019]].median(axis=1)
w("")
w("Crops whose national area in a new year is under 60 percent of their 2015-19 median")
w("(baseline over 50,000 ha); these are the crop-list drops that move diversity indices:")
w("")
w("| crop | baseline ha | 2020-21 | 2021-22 | 2022-23 |")
w("|---|---|---|---|---|")
for c in cb.index:
    if base_c[c] > 5e4 and any(cb.loc[c, y] < .6 * base_c[c] for y in NEW_YEARS if y in cb.columns):
        w("| {} | {:,.0f} | {} |".format(c, base_c[c], " | ".join(
            "{:,.0f} ({:.0f}pct)".format(cb.loc[c, y], 100 * cb.loc[c, y] / base_c[c]) for y in NEW_YEARS)))
# state-crop drops
sc = ex.groupby(["state_name", "crop_name", "y"]).area.sum().unstack(fill_value=0)
bsc = sc[[2015, 2016, 2017, 2018, 2019]].median(axis=1)
w("")
w("State-crop pairs with baseline over 50,000 ha that fall under 25 percent of baseline in a new year:")
w("")
for (st, c), v in bsc[bsc > 5e4].items():
    for y in NEW_YEARS:
        if sc.loc[(st, c), y] < .25 * v:
            w("- {} / {} {}-{}: {:,.0f} ha vs baseline {:,.0f}".format(
                st, c, y, str(y + 1)[-2:], sc.loc[(st, c), y], v))

# ---------------------------------------------------------------- UPAG check
w("")
w("## Cross-check against the UPAG resource (same package)")
w("")
try:
    up = pd.read_csv(UPAG)
    up = strip(up, ["year", "season", "state_name", "district_name", "crop_name"])
    up["y"] = up.year.str[:4].astype(int)
    w("UPAG: 30 crop names, seasons {}, years {}-{} to {}-{}, {:,} rows.".format(
        sorted(up.season.unique()), up.y.min(), str(up.y.min() + 1)[-2:], up.y.max(), str(up.y.max() + 1)[-2:], len(up)))
    w("Area column is described as hectares; national totals below are over 30 crops only.")
    um_ = {"Tur": "Arhar/Tur", "Moong": "Moong(Green Gram)", "Rapeseed & Mustard": "Rapeseed &Mustard",
           "Soybean": "Soyabean", "Cotton": "Cotton(Lint)", "Castorseed": "Castor Seed",
           "Guarseed": "Guar Seed", "Nigerseed": "Niger Seed", "Sannhemp": "Sannhamp"}
    up["crop_name"] = up.crop_name.replace(um_)
    up = up[up.crop_name.isin(set(old.crop_name))]
    up["dk"] = up.state_name.str.upper() + "|" + up.district_name.str.upper()
    ua_ = up.groupby(["dk", "y", "crop_name"]).area.sum().rename("a_up")
    ea_ = ex.groupby(["dk", "y", "crop_name"]).area.sum().rename("a_des")
    jj = pd.concat([ua_, ea_], axis=1).dropna().reset_index()
    jj = jj[jj.y.between(2013, 2022)]
    w("")
    w("| year | shared cells | r | median abs pct diff | UPAG Mha | DES Mha (same cells) |")
    w("|---|---|---|---|---|---|")
    for y, g in jj.groupby("y"):
        w("| {}-{} | {:,} | {:.4f} | {:.2f} | {:.1f} | {:.1f} |".format(
            y, str(y + 1)[-2:], len(g), g.a_up.corr(g.a_des),
            (100 * (g.a_up - g.a_des).abs() / g.a_des.clip(lower=1)).median(),
            g.a_up.sum() / 1e6, g.a_des.sum() / 1e6))
    upn = up.groupby("y").crop_name.nunique()
    w("")
    w("UPAG covers {} of the 54 old crop names ({:.0f} percent), so its 2023-24 and 2024-25 years".format(
        up.crop_name.nunique(), 100 * up.crop_name.nunique() / 54))
    w("cannot extend the 54-crop series without a crop-list break; they are not merged.")
except Exception as e:  # keep the log honest if the cross-check fails
    w("UPAG cross-check failed: {!r}".format(e))

with open(OUT + "/apy_extend_log.md", "w", encoding="utf-8") as f:
    f.write("\n".join(L) + "\n")
print("wrote log")
