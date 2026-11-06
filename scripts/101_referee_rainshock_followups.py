"""
101_referee_rainshock_followups.py

Follow-up referee checks on Q3 (script 99), after script 100. Reuses script 100's copy
of the script 99 data build and estimator (executed up to the referee block), then:
  A. rabi rice in the units that grow rabi rice: which states, which crops fill the gap
  B. next-kharif millet share after an excess monsoon: 1997 and data-flag sensitivity,
     intensive margin, lead placebo on the intensive margin
  C. low-irrigation tercile: wheat area vs wheat share, and without the hill states
Output: outputs/crop_groups/referee_rainshock_followups_log.txt
"""
import io

SRC = "D:/crop-diversity/scripts/100_referee_rainshock_checks.py"
code = io.open(SRC, encoding="utf-8").read().split("# =========================================================================== REFEREE CHECKS")[0]
code = code.replace('"/referee_rainshock_log.txt"', '"/referee_rainshock_followups_log.txt"')
exec(compile(code, SRC, "exec"))


def show(tag, r, v):
    b, s = r["b"][v], r["se"][v]
    p = 2 * stats.t.sf(abs(b / s), r["g"] - 1)
    say(f"{tag:60s} {v:8s} b={b:+.3f} se={s:.3f} p={p:.3f} n={r['n']} g={r['g']} ymean={r['ymean']:.2f}")


D = {name: build(df, lag, ctrl) for name, (df, lag, ctrl) in SAMPLES.items()}
X0 = ["s_def", "s_exc"]
X1 = ["s_def", "s_exc", "c_def", "c_exc"]

say("### A. rabi rice-growing units (mean rabi rice share >= 2 pp)")
d = D["rabi_t"]
msh = d.groupby("unit_key")["sh_rice"].transform("mean")
dr = d[msh >= 2].copy()
say("units by state:", dr.groupby("state")["unit_key"].nunique().sort_values(ascending=False).to_dict())
for g in G:
    show(f"share {g} in rabi-rice units", fe_ols(dr, "sh_" + g, X0), "s_def")
for g in ["rice", "pulses", "millets", "oilseeds", "maize"]:
    show(f"log1p area {g} in rabi-rice units", fe_ols(dr, "la_" + g, X0), "s_def")
show("log total in rabi-rice units", fe_ols(dr, "la_total", X0), "s_def")
show("cluster state: share rice in rabi-rice units", fe_ols(dr, "sh_rice", X0, cluster="state"), "s_def")
dr2 = dr[(dr["flag_annual_unreliable"] == 0) & (dr["year_start"] > 1997) & (dr["area_ha_rice"] > 0)].copy()
dr2["ln_rice"] = np.log(dr2["area_ha_rice"])
show("log rice area, rabi-rice units, drop flags+1997", fe_ols(dr2, "ln_rice", X0), "s_def")
show("same, cluster state", fe_ols(dr2, "ln_rice", X0, cluster="state"), "s_def")
# leave out AP / Telangana parents
dr3 = dr2[~dr2["state"].str.upper().isin(["ANDHRA PRADESH", "TELANGANA"])]
show("log rice area, rabi-rice units, excl AP+TG", fe_ols(dr3, "ln_rice", X0), "s_def")
# all units outside rabi-rice units: rice share there
show("share rice in units with mean rabi rice < 2pp", fe_ols(d[msh < 2], "sh_rice", X0), "s_def")

say("\n### B. next-kharif millet share after an excess monsoon")
d = D["kharif_t1"]
say("excess unit-years with year_start 1997 (shock JJAS 1996):", int(d.loc[d.year_start == 1997, "s_exc"].sum()),
    "of", int((d.year_start == 1997).sum()), "rows; by state:",
    d[(d.year_start == 1997) & (d.s_exc == 1)].groupby("state").size().to_dict())
for tag, dd in [("baseline", d), ("drop 1997", d[d.year_start > 1997]),
                ("drop annual flag", d[d.flag_annual_unreliable == 0]),
                ("drop annual flag + 1997", d[(d.flag_annual_unreliable == 0) & (d.year_start > 1997)]),
                ("drop annual flag + mild jump + 1997",
                 d[(d.flag_annual_unreliable == 0) & (d.f_gca_jump_mild == 0) & (d.year_start > 1997)])]:
    show(f"share millets [{tag}]", fe_ols(dd, "sh_millets", X1), "s_exc")
    show(f"share millets [{tag}] cluster state", fe_ols(dd, "sh_millets", X1, cluster="state"), "s_exc")
    m = dd.groupby("unit_key")["sh_millets"].transform("mean")
    di = dd[(m >= 2) & (dd["area_ha_millets"] > 0)].copy()
    di["ln_m"] = np.log(di["area_ha_millets"])
    show(f"log millet area, millet units [{tag}]", fe_ols(di, "ln_m", X1), "s_exc")
# bins after cleaning
dd = d[(d.flag_annual_unreliable == 0) & (d.year_start > 1997)]
rb = fe_ols(dd, "sh_millets", ["s_b_" + l for l in BLAB if l != "normal"] + ["c_def", "c_exc"])
for l in ["p10_p20", "p20_p40", "ge_p40", "le_m40", "m40_m20"]:
    show(f"bin share millets [drop annual flag + 1997]", rb, "s_b_" + l)
# lead placebo on intensive margin
lead2 = R[["unit_key", "yr", "s_def", "s_exc", "s_qc"]].rename(
    columns={"yr": "year_start", "s_def": "ld_def", "s_exc": "ld_exc", "s_qc": "ld_qc"})
lead2["year_start"] = lead2["year_start"] - 1
dl = d.merge(lead2, on=["unit_key", "year_start"], how="left")
dl = dl[dl["ld_qc"] == 0]
m = dl.groupby("unit_key")["sh_millets"].transform("mean")
di = dl[(m >= 2) & (dl["area_ha_millets"] > 0)].copy()
di["ln_m"] = np.log(di["area_ha_millets"])
r = fe_ols(di, "ln_m", X1 + ["ld_def", "ld_exc"])
show("lead JJAS(t+2) placebo, log millet area millet units", r, "ld_exc")
show("   same regression, actual JJAS(t)", r, "s_exc")
r = fe_ols(dl, "sh_millets", X1 + ["ld_def", "ld_exc"])
show("lead JJAS(t+2) placebo, millet share", r, "ld_exc")
show("   same regression, actual JJAS(t)", r, "s_exc")
dl2 = dl[(dl.flag_annual_unreliable == 0) & (dl.year_start > 1997)]
r = fe_ols(dl2, "sh_millets", X1 + ["ld_def", "ld_exc"])
show("lead placebo, millet share [drop flag + 1997]", r, "ld_exc")
show("   same regression, actual JJAS(t)", r, "s_exc")

say("\n### C. low-irrigation tercile, rabi")
d = D["rabi_t"].dropna(subset=["irr_t"])
Xi = [f"def_T{t}" for t in (1, 2, 3)] + [f"exc_T{t}" for t in (1, 2, 3)]
for y in ["sh_wheat", "la_wheat", "sh_pulses", "la_pulses", "la_total", "sh_millets"]:
    show(f"{y} T1 all", fe_ols(d, y, Xi), "def_T1")
HILL = ["JAMMU KASHMIR", "HIMACHAL PRADESH", "UTTARAKHAND", "ARUNACHAL PRADESH", "ASSAM",
        "MEGHALAYA", "NAGALAND", "MANIPUR", "MIZORAM", "TRIPURA", "SIKKIM"]
dn = d[~d["state"].str.upper().isin(HILL)]
say("low-tercile units left after dropping hill/NE states:", dn.loc[dn.irr_t == 1, "unit_key"].nunique(),
    "of", d.loc[d.irr_t == 1, "unit_key"].nunique())
for y in ["sh_wheat", "la_wheat", "sh_pulses", "la_pulses", "la_total"]:
    show(f"{y} T1 excl hill+NE", fe_ols(dn, y, Xi), "def_T1")
dc = d[(d.flag_annual_unreliable == 0) & (d.year_start > 1997)]
for y in ["sh_wheat", "sh_pulses", "la_pulses", "la_total", "sh_millets"]:
    r = fe_ols(dc, y, Xi)
    show(f"{y} T1 drop flag+1997", r, "def_T1")
    show(f"{y} T2 drop flag+1997", r, "def_T2")
    show(f"{y} T3 drop flag+1997", r, "def_T3")
LOG.close()
