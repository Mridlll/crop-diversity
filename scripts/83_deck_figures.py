"""
83_deck_figures.py

Figures for the crop diversity deck, drawn in the CEEW palette.

Figures carry data only. No titles, no captions, no legends that restate the
slide. The slide title names the page and the footnote line carries definitions.

Output: deck/figs/*.png at 300 dpi
"""
import os
import sys
import warnings

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore")

REPO = r"D:/crop-diversity"
COV = REPO + "/outputs/shrug_covariates"
DIV = REPO + "/outputs/crop_diversity_analysis"
OUT = REPO + "/deck/figs"
os.makedirs(OUT, exist_ok=True)

sys.path.insert(0, r"D:/Alternative Proteins/demand_pathways/deck")
from ceew_palette import CEEW  # noqa: E402

INK, MUTED, FAINT = CEEW["grey_dark"], CEEW["grey_light"], CEEW["grey_lighter"]
ORANGE, BLUE, GREEN = CEEW["orange"], CEEW["blue"], CEEW["green"]
BLUE_D, ORANGE_D = CEEW["blue_dark"], CEEW["orange_dark"]

mpl.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "font.family": "sans-serif",
    "font.sans-serif": ["Calibri", "Segoe UI", "DejaVu Sans"], "font.size": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": FAINT, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False,
})

SEQ_BLUE = LinearSegmentedColormap.from_list("ceew_blue", ["#EAF4FA", BLUE_D])
SEQ_ORANGE = LinearSegmentedColormap.from_list("ceew_orange", ["#FDEDE6", ORANGE_D])
DIV_BO = LinearSegmentedColormap.from_list("ceew_div", [ORANGE_D, "#F2F2F1", BLUE_D])


def save(fig, name):
    p = "{}/{}.png".format(OUT, name)
    fig.savefig(p, facecolor="white")
    plt.close(fig)
    print("  {:34s} {:>7.0f} KB".format(name + ".png", os.path.getsize(p) / 1024))


# ------------------------------------------------------------------- data
corr = pd.read_csv(DIV + "/district_diversity_indices_corrected.csv")
fp = pd.read_csv(COV + "/final_panel.csv")
d = fp[fp["in_final"]].copy()
mkc = pd.read_csv(COV + "/market_covariates.csv")
for t in (fp, mkc):
    for c in ["pc11_state_id", "pc11_district_id"]:
        t[c] = pd.to_numeric(t[c], errors="coerce")
dm = fp.merge(mkc, on=["pc11_state_id", "pc11_district_id"], how="left")
dm = dm[dm["in_final"] & dm["idx_output_market"].notna()].copy()
geo = gpd.read_file(REPO + "/docs/data/districts.geojson")
print("figures")


# --------------------------------------------------- 1. counting vs weighting
fig, ax = plt.subplots(figsize=(5.0, 3.5))
ax.scatter(corr["D0_richness"], corr["D1_exp_shannon"], s=7, alpha=.42,
           color=BLUE_D, linewidths=0)
lim = corr["D0_richness"].max() * 1.04
ax.plot([0, lim], [0, lim], color=MUTED, lw=1.1, ls="--")
ax.annotate("even cropping", xy=(lim * .60, lim * .60), xytext=(lim * .40, lim * .76),
            color=MUTED, fontsize=8,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=.8))
ax.set(xlim=(0, lim), ylim=(0, lim),
       xlabel="Crops grown in an average year", ylabel="Effective number of crops")
save(fig, "f1_count_vs_effective")


# -------------------------------------------------------------- 2. D1 map
def mapfig(col, cmap, name, vmin=None, vmax=None, fmt="{:.0f}", label=""):
    fig, ax = plt.subplots(figsize=(4.3, 4.8))
    g = geo.copy()
    ok = g[col].notna()
    lo = vmin if vmin is not None else g.loc[ok, col].quantile(.02)
    hi = vmax if vmax is not None else g.loc[ok, col].quantile(.98)
    g[~ok].plot(ax=ax, color="#F2F2F1", edgecolor="white", linewidth=.15)
    g[ok].plot(ax=ax, column=col, cmap=cmap, vmin=lo, vmax=hi,
               edgecolor="white", linewidth=.15)
    ax.set_axis_off()
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=mpl.colors.Normalize(vmin=lo, vmax=hi))
    sm._A = []
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=.032,
                      pad=.01, aspect=26)
    cb.outline.set_edgecolor(FAINT)
    cb.ax.tick_params(labelsize=7.5, colors=MUTED, length=2)
    cb.set_ticks([lo, (lo + hi) / 2, hi])
    cb.set_ticklabels([fmt.format(v) for v in [lo, (lo + hi) / 2, hi]])
    if label:
        cb.set_label(label, fontsize=7.5, color=MUTED, labelpad=3)
    save(fig, name)


mapfig("D1", SEQ_BLUE, "f2_map_effective_crops", fmt="{:.1f}",
       label="effective number of crops")


# ---------------------------------------------------------- 3. dominant crop
dom = (corr.groupby("dominant_crop")
           .agg(n=("district_key", "size"), sh=("dominant_crop_share", "mean"))
           .sort_values("n", ascending=False).head(8).sort_values("n"))
fig, ax = plt.subplots(figsize=(5.0, 2.9))
ax.barh(range(len(dom)), dom["n"], color=BLUE_D, height=.66)
ax.set_yticks(range(len(dom)))
ax.set_yticklabels([c.replace("(Lint)", "").strip().lower() for c in dom.index])
for i, (n, sh) in enumerate(zip(dom["n"], dom["sh"])):
    ax.text(n + 5, i, "{:.0f}% of land".format(sh * 100), va="center",
            fontsize=7.6, color=MUTED)
ax.set(xlabel="Districts where this crop takes the most land",
       xlim=(0, dom["n"].max() * 1.30))
ax.grid(axis="y", b=False)
save(fig, "f3_dominant_crop")


# ----------------------------------------------------------------- 4. trend
import json  # noqa: E402
ov = json.load(open(REPO + "/docs/data/site_overview.json", encoding="utf-8"))
tr = [t for t in ov["trend_balanced"] if t["year"] >= 1998]
fig, ax = plt.subplots(figsize=(5.0, 2.7))
ax.plot([t["year"] for t in tr], [t["D0"] for t in tr], color=BLUE_D, lw=1.9,
        marker="o", ms=2.6)
ax.plot([t["year"] for t in tr], [t["D1"] for t in tr], color=ORANGE, lw=1.9,
        marker="s", ms=2.6)
ax.text(tr[-1]["year"] + .3, tr[-1]["D0"], "crops grown", color=BLUE_D,
        fontsize=8, va="center")
ax.text(tr[-1]["year"] + .3, tr[-1]["D1"], "effective crops", color=ORANGE,
        fontsize=8, va="center")
ax.set(ylim=(0, max(t["D0"] for t in tr) * 1.25), xlim=(1997.5, 2024),
       ylabel="Number of crops")
ax.set_xticks([1998, 2004, 2010, 2016, 2019])
save(fig, "f4_trend")


# ------------------------------------------------------- 5. irrigation shape
fig, ax = plt.subplots(figsize=(5.0, 3.2))
ax.scatter(d["irr_share"], d["D1_exp_shannon"], s=7, alpha=.35, color=MUTED,
           linewidths=0)
cf = np.polyfit(d["irr_share"], d["D1_exp_shannon"], 2)
xs = np.linspace(d["irr_share"].min(), d["irr_share"].max(), 200)
ax.plot(xs, np.polyval(cf, xs), color=ORANGE, lw=2.4)
tp = -cf[1] / (2 * cf[0])
ax.axvline(tp, color=ORANGE, lw=1, ls=":")
ax.annotate("turns at {:.0f}%".format(tp * 100), xy=(tp, ax.get_ylim()[1] * .93),
            xytext=(tp + .04, ax.get_ylim()[1] * .93), color=ORANGE, fontsize=8.4)
ax.set(xlabel="Share of sown area irrigated", ylabel="Effective number of crops",
       ylim=(0, d["D1_exp_shannon"].max() * 1.05))
ax.set_xticks([0, .2, .4, .6, .8, 1.0])
ax.set_xticklabels(["0", "20%", "40%", "60%", "80%", "100%"])
save(fig, "f5_irrigation_shape")

mapfig("irr", SEQ_ORANGE, "f6_map_irrigation", vmin=0, vmax=1,
       fmt="{:.0%}", label="share of sown area irrigated")


# ------------------------------------------------------ 7. irrigation source
src = pd.read_csv(COV + "/final_source_results.csv")
LBL = {"D1_exp_shannon": "effective crops", "D0_richness": "crops grown",
       "D2_inv_simpson": "dominant-weighted", "evenness_D1_D0": "evenness",
       "share_cereals": "cereal share", "share_pulses": "pulse share",
       "share_oilseeds": "oilseed share"}
src = src[src["outcome"].isin(LBL)].iloc[::-1]
y = np.arange(len(src))
fig, ax = plt.subplots(figsize=(5.4, 3.0))
ax.axvline(0, color=INK, lw=1)
for off, bcol, pcol, colr, nm in [(-.17, "b_canal", "p_canal", BLUE_D, "canal"),
                                  (.17, "b_surface", "p_surface", ORANGE, "surface water")]:
    sig = src[pcol] < .05
    ax.scatter(src.loc[sig, bcol], y[sig.values] + off, s=34, color=colr, zorder=3)
    ax.scatter(src.loc[~sig, bcol], y[~sig.values] + off, s=34, facecolors="white",
               edgecolors=colr, linewidths=1.2, zorder=3)
ax.set_yticks(y)
ax.set_yticklabels([LBL[o] for o in src["outcome"]])
ax.set(xlabel="Coefficient against groundwater")
ax.grid(axis="y", b=False)
ax.text(.02, .04, "filled = significant at 5%", transform=ax.transAxes,
        fontsize=7.4, color=MUTED)
ax.text(.02, .96, "canal", transform=ax.transAxes, fontsize=8.4, color=BLUE_D,
        va="top")
ax.text(.02, .89, "surface water", transform=ax.transAxes, fontsize=8.4,
        color=ORANGE, va="top")
save(fig, "f7_irrigation_source")


# ---------------------------------------------------------- 8. facility corr
FAC = [("mandi", "m_mandi_vshare"), ("regular market", "m_regular_market_vshare"),
       ("weekly haat", "m_weekly_haat_vshare"), ("fertiliser shop", "m_fert_shop_vshare"),
       ("seed centre", "m_seed_centre_vshare"), ("soil testing", "m_soil_test_vshare"),
       ("custom hiring", "m_custom_hire_vshare"), ("cold storage", "m_storage_vshare"),
       ("farm-gate proc.", "m_farmgate_proc_vshare"), ("FPO", "m_fpo_vshare")]
cm = dm[[c for _, c in FAC]].corr().values
labs = [l for l, _ in FAC]
fig, ax = plt.subplots(figsize=(4.6, 4.0))
im = ax.imshow(cm, cmap=DIV_BO, vmin=-1, vmax=1)
ax.set_xticks(range(len(labs))); ax.set_yticks(range(len(labs)))
ax.set_xticklabels(labs, rotation=45, ha="right", fontsize=7.4)
ax.set_yticklabels(labs, fontsize=7.4)
for i in range(len(labs)):
    for j in range(len(labs)):
        ax.text(j, i, "{:.2f}".format(cm[i, j]).replace("0.", "."),
                ha="center", va="center", fontsize=5.9,
                color="white" if abs(cm[i, j]) > .55 else INK)
ax.grid(False)
for s in ax.spines.values():
    s.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=.036, pad=.02)
cb.outline.set_edgecolor(FAINT)
cb.ax.tick_params(labelsize=7, colors=MUTED, length=2)
save(fig, "f8_facility_correlation")

mapfig("haat", SEQ_BLUE, "f9_map_haat", vmin=0, vmax=.5, fmt="{:.0%}",
       label="share of villages holding a weekly haat")


# ------------------------------------------------------------- 10. solo effects
# Both panels carry the SAME row order, set once by the crops-grown effect, so the
# shared label column on the left applies to the panel on the right as well.
solo = pd.read_csv(COV + "/market_results_solo.csv")
order = (solo[solo["outcome"] == "D0_richness"]
         .sort_values("b")["facility"].tolist())
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))
for ax, out, xl in [(axes[0], "D0_richness", "Change in crops grown"),
                    (axes[1], "share_pulses", "Change in pulse share")]:
    t = solo[solo["outcome"] == out].set_index("facility").loc[order].reset_index()
    ax.axvline(0, color=INK, lw=1)
    for i, r in enumerate(t.itertuples()):
        c = (ORANGE if r.facility == "weekly haat"
             else (GREEN if r.facility == "FPO" else BLUE_D))
        if r.p < .05:
            ax.scatter(r.b, i, s=36, color=c, zorder=3)
        else:
            ax.scatter(r.b, i, s=36, facecolors="white", edgecolors=MUTED,
                       linewidths=1.1, zorder=3)
    ax.set_ylim(-0.8, len(order) - 0.2)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order if ax is axes[0] else [], fontsize=8)
    ax.set(xlabel=xl)
    ax.grid(axis="y", b=False)
fig.text(0.5, -0.02, "filled = significant at 5%", ha="center", fontsize=7.4,
         color=MUTED)
fig.subplots_adjust(wspace=.10)
save(fig, "f10_market_solo")


# --------------------------------------------------------- 11. interaction
inter = pd.read_csv(COV + "/market_results_interaction.csv").iloc[::-1]
fig, ax = plt.subplots(figsize=(5.4, 2.7))
yy = np.arange(len(inter))
for i, r in enumerate(inter.itertuples()):
    ax.plot([r.curv_lo, r.curv_hi], [i, i], color=FAINT, lw=2, zorder=1)
    ax.scatter(r.curv_lo, i, s=38, facecolors="white", edgecolors=MUTED,
               linewidths=1.3, zorder=3)
    ax.scatter(r.curv_hi, i, s=38, color=ORANGE, zorder=3)
ax.set_yticks(yy)
ax.set_yticklabels([m.replace(" village share", "").replace(" index", "")
                    for m in inter["measure"]], fontsize=8)
ax.set(xlabel="Coefficient on irrigation squared")
ax.grid(axis="y", b=False)
ax.text(.03, .06, "hollow = low density, filled = high density",
        transform=ax.transAxes, fontsize=7.4, color=MUTED)
save(fig, "f11_interaction")

print("\n{} figures in {}".format(len(os.listdir(OUT)), OUT))
