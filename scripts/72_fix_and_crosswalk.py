"""
72_fix_and_crosswalk.py

Two corrections that the validation battery forced, plus a district crosswalk.

CORRECTION 1 - irrigation denominator.
  Script 70 divided Antyodaya's `area_irrigated_in_hac` by gross cropped area.
  That was wrong. Diagnostics show:
    - irrigated + unirrigated reconstructs NET SOWN area (ratio 1.15), not gross (0.66)
    - against published state net-irrigated shares, irr/(irr+unirr) gives MAE 0.086
      and bias +0.051, while irr/GCA gives MAE 0.181 and bias -0.129
  So `area_irrigated_in_hac` is NET irrigated area and the correct share is
  irr / (irr + unirr), which is internally closed and needs no other field.

CORRECTION 2 - district identity.
  SHRUG is on PC11 (2011) boundaries. The diversity panel is on post-2011 names.
  Telangana does not exist in SHRUG: all 10 of its pre-2014 districts sit under
  Andhra Pradesh. Post-2011 carve-outs add another ~60 mismatches.
  Handled by: exact match -> explicit parent map -> fuzzy match within state.
  Every fuzzy acceptance is logged with its score for review.

Outputs:
  outputs/shrug_covariates/district_crosswalk.csv
  outputs/shrug_covariates/crosswalk_review.csv        (fuzzy matches to eyeball)
  outputs/shrug_covariates/analysis_panel.csv          (the file analysis uses)
"""
import difflib
import numpy as np
import pandas as pd

REPO = r"D:/crop-diversity"
OUT = REPO + "/outputs/shrug_covariates"
DIV = REPO + "/outputs/crop_diversity_analysis"

sh = pd.read_csv(OUT + "/shrug_district_covariates.csv")
dv = pd.read_csv(DIV + "/district_diversity_indices.csv")


# =============================================================== correction 1
print("=" * 78)
print("CORRECTION 1: irrigation share")
print("=" * 78)
irr = pd.to_numeric(sh["ay_area_irrigated_in_hac"], errors="coerce")
un = pd.to_numeric(sh["ay_total_unirrigated_land_area_in_h"], errors="coerce")
sh["irr_share"] = np.where((irr + un) > 0, irr / (irr + un), np.nan)

vi = pd.to_numeric(sh["pc11_vd_land_src_irr"], errors="coerce")
vu = pd.to_numeric(sh["pc11_vd_land_un_irr"], errors="coerce")
sh["irr_share_vd11"] = np.where((vi + vu) > 0, vi / (vi + vu), np.nan)

sh["irr_share_disagree"] = (sh["irr_share"] - sh["irr_share_vd11"]).abs()
print("irr_share (Antyodaya 2019): p10={:.3f} p50={:.3f} p90={:.3f}".format(
    sh["irr_share"].quantile(.1), sh["irr_share"].median(), sh["irr_share"].quantile(.9)))
print("irr_share_vd11 (PC11 2011): p10={:.3f} p50={:.3f} p90={:.3f}".format(
    sh["irr_share_vd11"].quantile(.1), sh["irr_share_vd11"].median(),
    sh["irr_share_vd11"].quantile(.9)))
print("districts where the two sources differ by >0.30: {}".format(
    int((sh["irr_share_disagree"] > 0.30).sum())))
bad_states = (sh.groupby("state_name")["irr_share_disagree"].median()
                .sort_values(ascending=False).head(6))
print("states with worst median source disagreement:")
print(bad_states.round(3).to_string())
sh["flag_irr_source_conflict"] = sh["irr_share_disagree"] > 0.30

# deprecate the wrong columns but keep them for audit
sh = sh.rename(columns={"irr_share_gca": "DEPRECATED_irr_share_gca",
                        "vd_irr_share": "DEPRECATED_vd_irr_share",
                        "irr_share_nsa": "DEPRECATED_irr_share_nsa"})


# =============================================================== correction 2
print()
print("=" * 78)
print("CORRECTION 2: district crosswalk")
print("=" * 78)


def norm(s):
    s = (s.astype(str).str.lower().str.strip()
         .str.replace(r"[^a-z0-9 ]", " ", regex=True)
         .str.replace(r"\s+", " ", regex=True).str.strip())
    return s


ALIAS_STATE = {
    "orissa": "odisha", "pondicherry": "puducherry", "uttaranchal": "uttarakhand",
    "jammu and kashmir": "jammu kashmir", "nct of delhi": "delhi",
    "andaman and nicobar islands": "andaman nicobar islands",
    "dadra and nagar haveli": "dadra nagar haveli", "daman and diu": "daman diu",
    "the dadra and nagar haveli and daman and diu": "dadra nagar haveli",
    "delhi": "delhi",
}

# child district (post-2011) -> (SHRUG state, SHRUG parent district on PC11 boundaries)
# Telangana: SHRUG carries all of these under andhra pradesh.
TG = "andhra pradesh"
PARENT = {
    # --- Telangana, created 2014 and reorganised 2016
    ("telangana", "adilabad"): (TG, "adilabad"),
    ("telangana", "komaram bheem asifabad"): (TG, "adilabad"),
    ("telangana", "mancherial"): (TG, "adilabad"),
    ("telangana", "nirmal"): (TG, "adilabad"),
    ("telangana", "nizamabad"): (TG, "nizamabad"),
    ("telangana", "kamareddy"): (TG, "nizamabad"),
    ("telangana", "karimnagar"): (TG, "karimnagar"),
    ("telangana", "jagitial"): (TG, "karimnagar"),
    ("telangana", "peddapalli"): (TG, "karimnagar"),
    ("telangana", "rajanna"): (TG, "karimnagar"),
    ("telangana", "medak"): (TG, "medak"),
    ("telangana", "sangareddy"): (TG, "medak"),
    ("telangana", "siddipet"): (TG, "medak"),
    ("telangana", "warangal"): (TG, "warangal"),
    ("telangana", "hanumakonda"): (TG, "warangal"),
    ("telangana", "jangaon"): (TG, "warangal"),
    ("telangana", "jayashankar"): (TG, "warangal"),
    ("telangana", "mahabubabad"): (TG, "warangal"),
    ("telangana", "mulugu"): (TG, "warangal"),
    ("telangana", "khammam"): (TG, "khammam"),
    ("telangana", "bhadradri"): (TG, "khammam"),
    ("telangana", "nalgonda"): (TG, "nalgonda"),
    ("telangana", "suryapet"): (TG, "nalgonda"),
    ("telangana", "yadadri"): (TG, "nalgonda"),
    ("telangana", "mahabubnagar"): (TG, "mahbubnagar"),
    ("telangana", "nagarkurnool"): (TG, "mahbubnagar"),
    ("telangana", "wanaparthy"): (TG, "mahbubnagar"),
    ("telangana", "jogulamba"): (TG, "mahbubnagar"),
    ("telangana", "narayanapet"): (TG, "mahbubnagar"),
    ("telangana", "rangareddi"): (TG, "rangareddy"),
    ("telangana", "vikarabad"): (TG, "rangareddy"),
    ("telangana", "medchal malkajgiri"): (TG, "rangareddy"),
    ("telangana", "hyderabad"): (TG, "hyderabad"),
    # --- Andhra Pradesh residual naming
    ("andhra pradesh", "kadapa"): ("andhra pradesh", "ysr kadapa"),
    ("andhra pradesh", "visakhapatanam"): ("andhra pradesh", "visakhapatnam"),
    ("andhra pradesh", "rangareddi"): ("andhra pradesh", "rangareddy"),
    ("andhra pradesh", "nellore"): ("andhra pradesh", "sri potti sriramulu nellore"),
    # --- West Bengal, spelling and 2017 splits
    ("west bengal", "coochbehar"): ("west bengal", "koch bihar"),
    ("west bengal", "hooghly"): ("west bengal", "hugli"),
    ("west bengal", "howrah"): ("west bengal", "haora"),
    ("west bengal", "purulia"): ("west bengal", "puruliya"),
    ("west bengal", "darjeeling"): ("west bengal", "darjiling"),
    ("west bengal", "north 24 parganas"): ("west bengal", "north twenty four parganas"),
    ("west bengal", "south 24 parganas"): ("west bengal", "south twenty four parganas"),
    ("west bengal", "pashchim medinipur"): ("west bengal", "paschim medinipur"),
    ("west bengal", "purba bardhaman"): ("west bengal", "barddhaman"),
    ("west bengal", "paschim bardhaman"): ("west bengal", "barddhaman"),
    ("west bengal", "alipurduar"): ("west bengal", "jalpaiguri"),
    ("west bengal", "kalimpong"): ("west bengal", "darjiling"),
    ("west bengal", "jhargram"): ("west bengal", "paschim medinipur"),
    # --- Chhattisgarh, carved 2012 and 2020
    ("chhattisgarh", "balod"): ("chhattisgarh", "durg"),
    ("chhattisgarh", "bemetara"): ("chhattisgarh", "durg"),
    ("chhattisgarh", "baloda bazar"): ("chhattisgarh", "raipur"),
    ("chhattisgarh", "gariyaband"): ("chhattisgarh", "raipur"),
    ("chhattisgarh", "mungeli"): ("chhattisgarh", "bilaspur"),
    ("chhattisgarh", "gaurella pendra marwahi"): ("chhattisgarh", "bilaspur"),
    ("chhattisgarh", "balrampur"): ("chhattisgarh", "surguja"),
    ("chhattisgarh", "surajpur"): ("chhattisgarh", "surguja"),
    ("chhattisgarh", "korea"): ("chhattisgarh", "koriya"),
    ("chhattisgarh", "kabirdham"): ("chhattisgarh", "kabeerdham"),
    ("chhattisgarh", "kanker"): ("chhattisgarh", "uttar bastar kanker"),
    ("chhattisgarh", "kondagaon"): ("chhattisgarh", "bastar"),
    ("chhattisgarh", "sukma"): ("chhattisgarh", "dakshin bastar dantewada"),
    # --- Uttar Pradesh
    ("uttar pradesh", "amethi"): ("uttar pradesh", "sultanpur"),
    ("uttar pradesh", "hapur"): ("uttar pradesh", "ghaziabad"),
    ("uttar pradesh", "sambhal"): ("uttar pradesh", "moradabad"),
    ("uttar pradesh", "shamli"): ("uttar pradesh", "muzaffarnagar"),
    ("uttar pradesh", "bhadohi"): ("uttar pradesh", "sant ravidas nagar bhadohi"),
    ("uttar pradesh", "kushi nagar"): ("uttar pradesh", "kushinagar"),
    ("uttar pradesh", "maharajganj"): ("uttar pradesh", "mahrajganj"),
    ("uttar pradesh", "sant kabeer nagar"): ("uttar pradesh", "sant kabir nagar"),
    ("uttar pradesh", "shravasti"): ("uttar pradesh", "shrawasti"),
    ("uttar pradesh", "siddharth nagar"): ("uttar pradesh", "siddharthnagar"),
    ("uttar pradesh", "barabanki"): ("uttar pradesh", "bara banki"),
    # these three are Uttarakhand districts mislabelled UP in the diversity panel
    ("uttar pradesh", "rudra prayag"): ("uttarakhand", "rudraprayag"),
    ("uttar pradesh", "udam singh nagar"): ("uttarakhand", "udham singh nagar"),
    ("uttar pradesh", "uttar kashi"): ("uttarakhand", "uttarkashi"),
    ("uttar pradesh", "amroha"): ("uttar pradesh", "jyotiba phule nagar"),
    ("uttar pradesh", "hathras"): ("uttar pradesh", "mahamaya nagar"),
    ("uttar pradesh", "kasganj"): ("uttar pradesh", "kanshiram nagar"),
    # --- Karnataka 2014 renamings
    ("karnataka", "ballari"): ("karnataka", "bellary"),
    ("karnataka", "belagavi"): ("karnataka", "belgaum"),
    ("karnataka", "bengaluru urban"): ("karnataka", "bangalore"),
    ("karnataka", "kalaburagi"): ("karnataka", "gulbarga"),
    ("karnataka", "mysuru"): ("karnataka", "mysore"),
    ("karnataka", "vijayapura"): ("karnataka", "bijapur"),
    # --- Madhya Pradesh (three of these are Chhattisgarh districts
    #     carrying an MP state label in the diversity panel)
    ("madhya pradesh", "khandwa"): ("madhya pradesh", "east nimar"),
    ("madhya pradesh", "khargone"): ("madhya pradesh", "west nimar"),
    ("madhya pradesh", "agar malwa"): ("madhya pradesh", "shajapur"),
    ("madhya pradesh", "niwari"): ("madhya pradesh", "tikamgarh"),
    ("madhya pradesh", "dakshin bastar dantewada"):
        ("chhattisgarh", "dakshin bastar dantewada"),
    ("madhya pradesh", "kabirdham"): ("chhattisgarh", "kabeerdham"),
    ("madhya pradesh", "korea"): ("chhattisgarh", "koriya"),
    # --- Jharkhand
    ("jharkhand", "east singhbum"): ("jharkhand", "purbi singhbhum"),
    ("jharkhand", "west singhbhum"): ("jharkhand", "pashchimi singhbhum"),
    # --- Gujarat, seven districts carved in 2013
    ("gujarat", "aravalli"): ("gujarat", "sabar kantha"),
    ("gujarat", "botad"): ("gujarat", "bhavnagar"),
    ("gujarat", "chhotaudepur"): ("gujarat", "vadodara"),
    ("gujarat", "devbhumi dwarka"): ("gujarat", "jamnagar"),
    ("gujarat", "gir somnath"): ("gujarat", "junagadh"),
    ("gujarat", "mahisagar"): ("gujarat", "panch mahals"),
    ("gujarat", "morbi"): ("gujarat", "rajkot"),
    ("gujarat", "dang"): ("gujarat", "the dangs"),
    # --- Maharashtra
    ("maharashtra", "beed"): ("maharashtra", "bid"),
    ("maharashtra", "palghar"): ("maharashtra", "thane"),
    # --- Punjab, two carved 2011
    ("punjab", "fazilka"): ("punjab", "firozpur"),
    ("punjab", "pathankot"): ("punjab", "gurdaspur"),
    ("punjab", "s a s nagar"): ("punjab", "sahibzada ajit singh nagar"),
    # --- Tamil Nadu, five carved 2019
    ("tamil nadu", "tuticorin"): ("tamil nadu", "thoothukkudi"),
    ("tamil nadu", "chengalpattu"): ("tamil nadu", "kancheepuram"),
    ("tamil nadu", "kallakurichi"): ("tamil nadu", "viluppuram"),
    ("tamil nadu", "ranipet"): ("tamil nadu", "vellore"),
    ("tamil nadu", "tenkasi"): ("tamil nadu", "tirunelveli"),
    ("tamil nadu", "tirupathur"): ("tamil nadu", "vellore"),
    # --- Odisha, Haryana, J&K, Ladakh
    ("odisha", "sonepur"): ("odisha", "subarnapur"),
    ("haryana", "charki dadri"): ("haryana", "bhiwani"),
    ("jammu kashmir", "poonch"): ("jammu kashmir", "punch"),
    ("ladakh", "kargil"): ("jammu kashmir", "kargil"),
    ("ladakh", "leh ladakh"): ("jammu kashmir", "leh ladakh"),
    # --- Assam, five carved 2015-16
    ("assam", "biswanath"): ("assam", "sonitpur"),
    ("assam", "charaideo"): ("assam", "sivasagar"),
    ("assam", "hojai"): ("assam", "nagaon"),
    ("assam", "majuli"): ("assam", "jorhat"),
    ("assam", "south salmara mancachar"): ("assam", "dhubri"),
    # --- Sikkim, renamed 2021
    ("sikkim", "gangtok"): ("sikkim", "east district"),
    ("sikkim", "gyalshing"): ("sikkim", "west district"),
    ("sikkim", "mangan"): ("sikkim", "north district"),
    ("sikkim", "namchi"): ("sikkim", "south district"),
    # --- Tripura, four carved 2012
    ("tripura", "gomati"): ("tripura", "south tripura"),
    ("tripura", "khowai"): ("tripura", "west tripura"),
    ("tripura", "sepahijala"): ("tripura", "west tripura"),
    ("tripura", "unakoti"): ("tripura", "north tripura"),
    # --- Arunachal Pradesh, carved 2012-2018
    ("arunachal pradesh", "kamle"): ("arunachal pradesh", "upper subansiri"),
    ("arunachal pradesh", "kra daadi"): ("arunachal pradesh", "kurung kumey"),
    ("arunachal pradesh", "leparada"): ("arunachal pradesh", "west siang"),
    ("arunachal pradesh", "shi yomi"): ("arunachal pradesh", "west siang"),
    ("arunachal pradesh", "longding"): ("arunachal pradesh", "tirap"),
    ("arunachal pradesh", "namsai"): ("arunachal pradesh", "lohit"),
    ("arunachal pradesh", "pakke kessang"): ("arunachal pradesh", "east kameng"),
    ("arunachal pradesh", "siang"): ("arunachal pradesh", "east siang"),
    # --- corrections to fuzzy false positives (see REJECT below)
    ("uttarakhand", "pauri garhwal"): ("uttarakhand", "garhwal"),
    ("meghalaya", "north garo hills"): ("meghalaya", "east garo hills"),
    ("meghalaya", "south west garo hills"): ("meghalaya", "west garo hills"),
    ("telangana", "jangoan"): (TG, "warangal"),
    # --- others
    ("haryana", "mewat"): ("haryana", "mewat"),
    ("bihar", "kaimur bhabua"): ("bihar", "kaimur bhabua"),
    ("madhya pradesh", "hoshangabad"): ("madhya pradesh", "hoshangabad"),
    ("punjab", "sahibzada ajit singh nagar"): ("punjab", "sahibzada ajit singh nagar"),
}

# Fuzzy matches that look plausible on string distance but are the wrong district.
# Each of these is a genuinely different district that happens to share a stem,
# and each already exists separately in the diversity panel.
REJECT_FUZZY = {
    ("uttarakhand", "pauri garhwal"),      # would grab TEHRI GARHWAL
    ("tamil nadu", "tirupathur"),          # would grab TIRUPPUR
    ("meghalaya", "north garo hills"),     # would grab SOUTH GARO HILLS
    ("meghalaya", "south west garo hills"),  # would grab SOUTH GARO HILLS
}

sh["s_n"] = norm(sh["state_name"]).replace(ALIAS_STATE)
sh["d_n"] = norm(sh["district_name"])
dv["s_n"] = norm(dv["state_name"]).replace(ALIAS_STATE)
dv["d_n"] = norm(dv["district_name"])

sh_lookup = {(r.s_n, r.d_n): r.district_key for r in sh.itertuples()}
by_state = sh.groupby("s_n")["d_n"].apply(list).to_dict()

# every manual target must actually exist in SHRUG, or the map is silently wrong
missing = sorted({v for v in PARENT.values() if v not in sh_lookup})
if missing:
    print("\nWARNING: {} manual parent targets do not exist in SHRUG:".format(len(missing)))
    for v in missing:
        print("   {} | {}".format(*v))
else:
    print("\nall {} manual parent targets resolve to a real SHRUG district".format(len(PARENT)))

rows = []
for r in dv.itertuples():
    k = (r.s_n, r.d_n)
    if k in sh_lookup:
        rows.append((r.district_key, sh_lookup[k], "exact", 1.0)); continue
    if k in PARENT and PARENT[k] in sh_lookup:
        rows.append((r.district_key, sh_lookup[PARENT[k]], "parent_map", 1.0)); continue
    if k in REJECT_FUZZY:
        rows.append((r.district_key, np.nan, "rejected_fuzzy", 0.0)); continue
    cand = by_state.get(r.s_n, [])
    if cand:
        best = difflib.get_close_matches(r.d_n, cand, n=1, cutoff=0.75)
        if best:
            sc = difflib.SequenceMatcher(None, r.d_n, best[0]).ratio()
            meth = "fuzzy_auto" if sc >= 0.86 else "fuzzy_review"
            rows.append((r.district_key, sh_lookup[(r.s_n, best[0])], meth, round(sc, 3)))
            continue
    rows.append((r.district_key, np.nan, "unmatched", 0.0))

cw = pd.DataFrame(rows, columns=["div_key", "shrug_key", "method", "score"])
print(cw["method"].value_counts().to_string())
matched = cw["shrug_key"].notna().sum()
print("\nmatched {} of {} diversity districts ({:.1%})".format(
    matched, len(dv), matched / len(dv)))

rev = cw[cw["method"].isin(["fuzzy_auto", "fuzzy_review"])].sort_values("score")
rev.to_csv(OUT + "/crosswalk_review.csv", index=False)
print("\nfuzzy matches written for review ({} rows). Lowest 15 scores:".format(len(rev)))
print(rev.head(15).to_string(index=False))

# how many diversity districts share one SHRUG parent?
share = cw.dropna(subset=["shrug_key"]).groupby("shrug_key")["div_key"].count()
cw["n_sharing_parent"] = cw["shrug_key"].map(share)
print("\nSHRUG districts serving more than one diversity district: {}".format(
    int((share > 1).sum())))
print("diversity districts on a shared parent: {}".format(
    int(cw["n_sharing_parent"].fillna(1).gt(1).sum())))
cw.to_csv(OUT + "/district_crosswalk.csv", index=False)

still = cw[cw["method"] == "unmatched"]["div_key"].tolist()
print("\nstill unmatched ({}): {}".format(len(still), ", ".join(sorted(still)[:40])))


# ================================================================ build panel
print()
print("=" * 78)
print("ANALYSIS PANEL")
print("=" * 78)
sh_use = sh.drop(columns=["s_n", "d_n"]).rename(columns={"district_key": "shrug_key"})
dv_use = dv.drop(columns=["s_n", "d_n"]).rename(columns={"district_key": "div_key"})

panel = dv_use.merge(cw, on="div_key", how="left")
n0 = len(panel)
panel = panel.merge(sh_use, on="shrug_key", how="left", suffixes=("", "_sh"))
assert len(panel) == n0, "panel merge multiplied rows"

panel["flag_shared_parent"] = panel["n_sharing_parent"].fillna(1) > 1
panel["w_parent"] = 1.0 / panel["n_sharing_parent"].fillna(1)
panel["in_analysis"] = (
    panel["shrug_key"].notna()
    & panel["irr_share"].notna()
    & ~panel["flag_low_ay_coverage"].fillna(True)
    & ~panel["flag_no_antyodaya"].fillna(True)
    & panel["agro_biodiversity_index"].notna()
)
print("panel rows: {}".format(len(panel)))
print("with a SHRUG match: {}".format(int(panel["shrug_key"].notna().sum())))
print("in_analysis (match + irrigation + adequate coverage + ABI): {}".format(
    int(panel["in_analysis"].sum())))
print("of those, on a shared parent: {}".format(
    int((panel["in_analysis"] & panel["flag_shared_parent"]).sum())))
print("states represented in analysis set: {}".format(
    panel.loc[panel["in_analysis"], "state_name"].nunique()))

panel.to_csv(OUT + "/analysis_panel.csv", index=False)
sh.drop(columns=["s_n", "d_n"]).to_csv(OUT + "/shrug_district_covariates.csv", index=False)
print("\nWROTE analysis_panel.csv {} and updated shrug_district_covariates.csv".format(panel.shape))
