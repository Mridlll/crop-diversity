"""
84_build_deck.py

The crop diversity deck, CEEW house style, 13.333 by 7.5 inches.

Imports the house furniture rather than restating it. Figures come from
83_deck_figures.py and carry data only: the slide title names the page, the
strip carries the finding, the footnote carries definitions.

The running order is an argument the reader follows from ignorance to conclusion.
Act one asks what rides on a district's crop mix. Act two describes: the measure,
the national picture, the geography, the concentration in two cereals, and whether
the balance has moved. Act three opens on the irrigation hump, which is the pattern
that generates every candidate explanation, and screens thirteen candidates down to
five. Act four states those five. Act five tests them one at a time and keeps the
one that fails on the page with the four that held.

Page shapes vary deliberately. Eight of the fourteen pages carry a shape that
appears once: a lede over three panels with a statistics band, a summary-statistics
page of three blocks, a large map with a side rail, a full-width figure over a
statistics band, a hypothesis table, a figure over a table, a scorecard and a page
of two stacked tables.

VOCABULARY IS LOCKED. It is "the effective number of crops" everywhere, "crops
grown in an average year", "evenness", "the irrigation share of sown area",
"weekly haat", "mandi", "regular market" and "producer organisation". The composite
index appears once, in the footnote to slide two, as the reason a count is used.

Basis (rebuilt 7 November 2026): window 1998-99 to 2019-20 (22 years), 724
districts and 606 in the analysis sample. Irrigation is read from fifths of the
irrigation share and a two-piece line (docs/data/site_irrigation_v2.json), not from
a fitted hump. Market coefficients are on population-weighted shares (version W,
outputs/revision_nov8). Every figure that can be computed is computed below, not
typed.

Output: deck/crop_diversity.pptx
"""
import os
import sys
from pathlib import Path

from PIL import Image
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

FURN = r"D:/Alternative Proteins/demand_pathways/deck"
sys.path.insert(0, FURN)

from deck_furniture import (  # noqa: E402
    plain_slide, chrome, source, footnote, highlight_strip, titled_panel,
    panel, box, set_text, check_fits, wrapped_lines, stack_in, prs,
    RUNNING_ORDER, LINE_PITCH_IN, TWOUP_X, TWOUP_W, STRIP_X, STRIP_W,
    PANEL_ML, PANEL_MR, PANEL_MB, INK, MUTED, TINT, WHITE, ACCENT, LINE)

REPO = Path(r"D:/crop-diversity")
FIGS = REPO / "deck" / "figs"
OUT = REPO / "deck" / "crop_diversity.pptx"

SRC = "Ministry of Agriculture and Farmers Welfare; SHRUG 2.1, Development Data Lab"


# ================================================================== the numbers
import json  # noqa: E402
import re  # noqa: E402

import pandas as pd  # noqa: E402

COV = REPO / "outputs" / "shrug_covariates"
DIVD = REPO / "outputs" / "crop_diversity_analysis"
PWD = REPO / "outputs" / "revision_nov8"

CORR = pd.read_csv(DIVD / "district_diversity_indices_corrected.csv")
FP = pd.read_csv(COV / "final_panel.csv")
D606 = FP[FP["in_final"]].copy()
DIM = pd.read_csv(COV / "final_dimension_results.csv").set_index("dimension")
SRCR = pd.read_csv(COV / "final_source_results.csv").set_index("outcome")
OVJ = json.load(open(REPO / "docs/data/site_overview.json", encoding="utf-8"))
IRRJ = json.load(open(REPO / "docs/data/site_irrigation_v2.json", encoding="utf-8"))
_pw = pd.read_csv(PWD / "market_pw_coefs.csv")
PWW = _pw[_pw["version"] == "W"]
HAATW = pd.read_csv(PWD / "market_pw_robust_haat.csv")
HAATW = HAATW[HAATW["version"] == "W"].reset_index(drop=True)
INTW = pd.read_csv(PWD / "market_pw_interaction.csv")
INTW = INTW[INTW["version"] == "W"].reset_index(drop=True)
MKT_MD = open(COV / "market_analysis.md", encoding="utf-8").read()


def sg(x, d=2):
    return "{:+.{d}f}".format(x, d=d)


def stars(p):
    return "***" if p < .01 else ("**" if p < .05 else ("*" if p < .10 else ""))


def pf(p):
    """p value as the deck prints it: two places from 0.10, three to 0.001, else four."""
    return "{:.2f}".format(p) if p >= .1 else (
        "{:.3f}".format(p) if p >= .001 else ("{:.4f}".format(p) if p >= .0001 else "<0.0001"))


def PW(block, fac, out):
    t = PWW[(PWW["block"] == block) & (PWW["facility"] == fac) & (PWW["outcome"] == out)]
    return t.iloc[0]


# 724 districts, 606 sample
N_ALL, N_SAMPLE = len(CORR), len(D606)
assert (N_ALL, N_SAMPLE) == (724, 606), (N_ALL, N_SAMPLE)
DOM1 = CORR["dominant_crop_share"].mean() * 100
TOP3 = CORR["top3_crops_share"].mean() * 100
Q = lambda c, p: D606[c].quantile(p)   # noqa: E731
MEAN = {c: D606[c].mean() for c in
        ["D0_richness", "D1_exp_shannon", "D2_inv_simpson", "evenness_D1_D0"]}

# states with at least four districts, unweighted district means
_g = (CORR.groupby("state_name")
      .agg(D1=("D1_exp_shannon", "mean"), D0=("D0_richness", "mean"),
           E=("evenness_D1_D0", "mean"), n=("district_key", "size")))
_g = _g[_g["n"] >= 4]
STATE_TOP = _g.sort_values("D1", ascending=False).head(5)
STATE_BOT = _g.sort_values("D1").head(5)
_tc = lambda x: x.title().replace("And", "and")   # noqa: E731
CG = _g.sort_values("D0").iloc[-1]
CG_NAME = _tc(_g.sort_values("D0").index[-1])

# national category shares (area weighted) and quarters
_cats = [("cereals", "share_cereals"), ("oilseeds", "share_oilseeds"),
         ("pulses", "share_pulses"), ("fibre crops", "share_fiber_crops"),
         ("sugar", "share_sugar"), ("fodder", "share_fodder"),
         ("spices", "share_spices"), ("vegetables", "share_vegetable"),
         ("fruit", "share_fruits")]
_w = CORR["mean_annual_cropped_area"]
NAT = sorted([(n, (CORR[c] * _w).sum() / _w.sum() * 100) for n, c in _cats],
             key=lambda t: -t[1])
CORR["_q"] = pd.qcut(CORR["D1_exp_shannon"], 4, labels=False)
QTR = CORR.groupby("_q")[[c for _, c in _cats]].mean() * 100

_dom = (CORR.groupby("dominant_crop")
        .agg(n=("district_key", "size"), sh=("dominant_crop_share", "mean"))
        .sort_values("n", ascending=False).head(12))
_NM = {"Cotton(Lint)": "Cotton", "Rapeseed &Mustard": "Rapeseed and mustard"}
DOMROWS = [[_NM.get(k, k), str(int(r["n"])), "{:.1f}%".format(r["sh"] * 100)]
           for k, r in _dom.iterrows()]
RICEWHEAT = int(CORR["dominant_crop"].isin(["Rice", "Wheat"]).sum())

# trend panel
CH = OVJ["change"]
TRB = [t for t in OVJ["trend_balanced"] if t["year"] >= 1998]
NPAN = CH["n"]

# irrigation
FIF = IRRJ["fifths"]["full"]["rows"]
SPL = IRRJ["spline"]["full"]
TEN = IRRJ["tenths"]["rows"]


# The three-across span, which is box_row's own left and right edges with its own
# gap, so a three-panel row lines up with the two-up pages and the four-slot band.
BAND_GAP = 0.34
THIRD_W = (12.88 - 0.43 - 2 * BAND_GAP) / 3          # 3.9233
THIRD_X = [0.43 + i * (THIRD_W + BAND_GAP) for i in range(3)]


def place(slide, name, x, y, max_w, max_h, centre=True):
    """Place a figure scaled to fit its box, never stretched."""
    p = FIGS / (name + ".png")
    w_px, h_px = Image.open(p).size
    ar = w_px / h_px
    w, h = max_w, max_w / ar
    if h > max_h:
        h, w = max_h, max_h * ar
    if centre:
        x = x + (max_w - w) / 2
    slide.shapes.add_picture(str(p), Inches(x), Inches(y + (max_h - h) / 2),
                             width=Inches(w), height=Inches(h))


def line(text):
    """A strip body: one line, ink."""
    return [[(text, {"color": INK})]]


# ============================================================ the table shape
#
# WHY THIS IS WRITTEN HERE AND NOT IN THE FURNITURE. The furniture carries no table
# helper because the deck it was cut for has no tables on it. This one has seven, so
# the shape is written once and every table on the deck is this call.
#
# IT RAISES RATHER THAN OVERFLOWS, on the same principle as check_fits and
# titled_panel. A cell that is a tenth of an inch too wide prints across its
# neighbour's column and reads as a table that lost a value, and nobody catches it
# until the export. Measured against Calibri's own advance widths, not counted.
#
# THE COLOUR VOCABULARY IS THREE VALUES AND NO MORE. Header text is MUTED over an
# INK rule. Data rules are LINE. A row or a cell that the page wants found is filled
# TINT, which is the same signal the highlight strip uses, so a tinted row means
# "this is the row the headline is about" and nothing else. ACCENT appears on a
# verdict chip and nowhere inside a table.

CELL_PAD = 0.05                 # left and right inset inside a cell, each side
RULE_H = 0.008                  # a data rule
HEAD_RULE_H = 0.013             # the heavier rule under the header row


def _cell(slide, x, y, w, h, text, pt, bold, colour, right, wrap):
    tb = box(slide, x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = wrap
    tf.vertical_anchor = MSO_ANCHOR.TOP if wrap else MSO_ANCHOR.MIDDLE
    tf.margin_left = Inches(CELL_PAD)
    tf.margin_right = Inches(CELL_PAD)
    tf.margin_top = Inches(0.02)
    tf.margin_bottom = Inches(0.02)
    set_text(tf, [[(text, {"size": pt, "bold": bold, "color": colour})]],
             align=PP_ALIGN.RIGHT if right else PP_ALIGN.LEFT,
             space_after=0, line_spacing=1.06)
    return tb


def _rule(slide, x, y, w, h, colour):
    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y),
                                Inches(w), Inches(h))
    sh.fill.solid()
    sh.fill.fore_color.rgb = colour
    sh.line.fill.background()
    sh.shadow.inherit = False
    return sh


def _fill(slide, x, y, w, h, colour):
    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y),
                                Inches(w), Inches(h))
    sh.fill.solid()
    sh.fill.fore_color.rgb = colour
    sh.line.fill.background()
    sh.shadow.inherit = False
    return sh


def table(slide, x, y, widths, header, rows, row_h=0.28, head_h=0.28, align=None,
          pt=9.5, head_pt=9.0, row_fill=None, cell_fill=None, bold_rows=(),
          rail=None, rail_w=0.0, wrap=False, where="table"):
    """A header row and data rows laid out from panel primitives.

    widths are inches per column and must be given for every column. align is one
    character per column, l or r, and numeric columns take r. row_fill maps a row
    index to a fill colour, cell_fill maps (row, column) to one. rail is a list of
    (label, row count) that names groups of rows in a column to the left of the
    table, which is how a table gets a grouping without spending a row on each
    group name.

    Returns the y of the table's bottom edge.
    """
    ncol = len(widths)
    if len(header) != ncol:
        raise ValueError("{}: {} headers against {} columns.".format(
            where, len(header), ncol))
    align = align or ("l" + "r" * (ncol - 1))
    if len(align) != ncol:
        raise ValueError("{}: alignment string is {} against {} columns.".format(
            where, len(align), ncol))
    row_fill = row_fill or {}
    cell_fill = cell_fill or {}
    dx = x + rail_w
    total = sum(widths)
    inner = [w - 2 * CELL_PAD for w in widths]

    # MEASURE EVERYTHING FIRST. A table that is going to fail should fail before it
    # has put half of itself on the page.
    for j, h in enumerate(header):
        check_fits(h, head_pt, inner[j], "{} header {!r}".format(where, h), True)
    for i, r in enumerate(rows):
        if len(r) != ncol:
            raise ValueError("{}: row {} has {} cells against {} columns.".format(
                where, i, len(r), ncol))
        for j, c in enumerate(r):
            if wrap:
                n = wrapped_lines(str(c), pt, inner[j])
                need = 0.04 + n * pt * LINE_PITCH_IN
                if need > row_h:
                    raise ValueError(
                        "{}: cell ({}, {}) wraps to {} lines and stacks {:.3f} in "
                        "against the {:.2f} in row height, so it would print on the "
                        "row below. Shorten it or give the rows more height. Text: "
                        "{!r}".format(where, i, j, n, need, row_h, str(c)))
            else:
                check_fits(str(c), pt, inner[j],
                           "{} cell ({}, {})".format(where, i, j))
    if rail is not None:
        if sum(n for _, n in rail) != len(rows):
            raise ValueError("{}: the rail covers {} rows against {}.".format(
                where, sum(n for _, n in rail), len(rows)))

    # HEADER. Text over a rule, with no fill, so the tint stays available to mean
    # one thing only: the row this page is about.
    cx = dx
    for j, h in enumerate(header):
        _cell(slide, cx, y, widths[j], head_h, h, head_pt, True, MUTED,
              align[j] == "r", False)
        cx += widths[j]
    _rule(slide, x, y + head_h, total + rail_w, HEAD_RULE_H, INK)

    ry = y + head_h + HEAD_RULE_H
    for i, r in enumerate(rows):
        if i in row_fill:
            _fill(slide, dx, ry, total, row_h, row_fill[i])
        cx = dx
        for j, c in enumerate(r):
            if (i, j) in cell_fill:
                _fill(slide, cx, ry, widths[j], row_h, cell_fill[(i, j)])
            _cell(slide, cx, ry, widths[j], row_h, str(c), pt, i in bold_rows,
                  INK, align[j] == "r", wrap)
            cx += widths[j]
        _rule(slide, dx, ry + row_h, total, RULE_H, LINE)
        # THE SEPARATOR GOES WHITE WHERE THE FILL IS. The house rule colour is
        # grey_lighter, which disappears against the tint, so two tinted rows one
        # above the other read as a single block and the reader loses a row boundary
        # exactly where the page is asking them to look.
        if i in row_fill:
            _rule(slide, dx, ry + row_h, total, RULE_H, WHITE)
        else:
            cx = dx
            for j in range(ncol):
                if (i, j) in cell_fill:
                    _rule(slide, cx, ry + row_h, widths[j], RULE_H, WHITE)
                cx += widths[j]
        ry += row_h + RULE_H

    # THE RAIL, drawn last so its group rules sit over the data rules they replace.
    if rail is not None:
        gy = y + head_h + HEAD_RULE_H
        for k, (lab, n) in enumerate(rail):
            gh = n * (row_h + RULE_H) - RULE_H
            lines = wrapped_lines(lab, pt, rail_w - 2 * CELL_PAD, True)
            tb = box(slide, x, gy, rail_w - CELL_PAD, gh)
            tf = tb.text_frame
            tf.word_wrap = True
            tf.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf.margin_left = Inches(0.0)
            tf.margin_right = Inches(CELL_PAD)
            set_text(tf, [[(lab, {"size": pt, "bold": True, "color": MUTED})]],
                     space_after=0, line_spacing=1.06)
            if lines * pt * LINE_PITCH_IN > gh:
                raise ValueError("{}: rail label {!r} does not fit its group.".format(
                    where, lab))
            if k:
                _rule(slide, x, gy - RULE_H, total + rail_w, RULE_H, MUTED)
            gy += n * (row_h + RULE_H)
    return ry


# ---------------------------------------------------------- the smaller shapes


def block_head(slide, x, y, w, text, pt=11.5):
    """The name over a statistics block or one half of a two-table page."""
    check_fits(text, pt, w, "block head")
    tb = box(slide, x, y, w, 0.26)
    set_text(tb.text_frame, [[(text, {"size": pt, "bold": True, "color": INK})]],
             space_after=0, line_spacing=1.06)
    return tb


def verdict(slide, x, y, w, text, pt=11.5):
    """A tinted chip carrying the verdict on the hypothesis the page tests."""
    check_fits(text, pt, w - 0.34, "verdict chip", True)
    band = _fill(slide, x, y, w, 0.30, TINT)
    _fill(slide, x, y, 0.055, 0.30, ACCENT)
    tb = box(slide, x + 0.20, y + 0.025, w - 0.24, 0.26)
    set_text(tb.text_frame, [[(text, {"size": pt, "bold": True, "color": INK})]],
             space_after=0, line_spacing=1.06)
    return band


def stack_panel(slide, x, y, w, h, paras, gap_pt=3, where="panel"):
    """A bordered panel carrying several measured paragraphs.

    paras is a list of (text, point size, bold, colour). The furniture's
    titled_panel takes exactly a lead-in and one body paragraph at 13 pt, and half
    the pages here want two paragraphs or a smaller size, so this is the general
    case of it and it raises on the same principle.
    """
    col = w - PANEL_ML - PANEL_MR
    spec = []
    for k, (text, pt, bold, _c) in enumerate(paras):
        spec.append((wrapped_lines(text, pt, col, bold), pt,
                     0 if k == len(paras) - 1 else gap_pt))
    need = stack_in(spec, 0.13, PANEL_MB)
    if need > h:
        raise ValueError(
            "{}: the text stacks {:.3f} in against the {:.2f} in the box has, so its "
            "last line would print on the bottom rule. Shorten it or give the box "
            "more height. Text: {!r}".format(where, need, h, paras[0][0][:52]))
    sh = panel(slide, x, y, w, h)
    set_text(sh.text_frame,
             [[(t, {"size": p, "bold": b, "color": c})] for t, p, b, c in paras],
             space_after=0, line_spacing=1.06)
    for k in range(len(paras) - 1):
        sh.text_frame.paragraphs[k].space_after = Pt(gap_pt)
    return sh


def bare_lines(slide, x, y, w, h, paras, gap_pt=3, where="lines"):
    """The same stack with no border, for a note that sits under a table."""
    spec = [(wrapped_lines(t, p, w, b), p, 0 if k == len(paras) - 1 else gap_pt)
            for k, (t, p, b, _c) in enumerate(paras)]
    need = stack_in(spec, 0.02, 0.02)
    if need > h:
        raise ValueError("{}: the text stacks {:.3f} in against {:.2f} in. "
                         "Text: {!r}".format(where, need, h, paras[0][0][:52]))
    tb = box(slide, x, y, w, h)
    tb.text_frame.word_wrap = True
    set_text(tb.text_frame,
             [[(t, {"size": p, "bold": b, "color": c})] for t, p, b, c in paras],
             space_after=0, line_spacing=1.06)
    for k in range(len(paras) - 1):
        tb.text_frame.paragraphs[k].space_after = Pt(gap_pt)
    return tb


def stat_band(slide, y, h, cells, num_pt=19.0, lab_pt=10.5):
    """A row of bordered cells, each a figure over what the figure counts."""
    n = len(cells)
    w = (12.88 - 0.43 - BAND_GAP * (n - 1)) / n
    for i, (num, lab) in enumerate(cells):
        x = 0.43 + i * (w + BAND_GAP)
        col = w - PANEL_ML - PANEL_MR
        need = stack_in([(1, num_pt, 3),
                         (wrapped_lines(lab, lab_pt, col), lab_pt, 0)], 0.10, 0.08)
        if need > h:
            raise ValueError("statistics band cell {}: stacks {:.3f} in against "
                             "{:.2f} in. Text: {!r}".format(i, need, h, lab))
        check_fits(num, num_pt, col, "statistics band figure", True)
        sh = panel(slide, x, y, w, h, margin_top=0.10)
        sh.text_frame.margin_bottom = Inches(0.08)
        set_text(sh.text_frame,
                 [[(num, {"size": num_pt, "bold": True, "color": ACCENT})],
                  [(lab, {"size": lab_pt, "color": MUTED})]],
                 space_after=0, line_spacing=1.06)
        sh.text_frame.paragraphs[0].space_after = Pt(3)


def stat_row(slide, y, h, cells, name_pt=11.5, body_pt=12.0):
    """A row of bordered cells, each a measure over what happened to it."""
    n = len(cells)
    w = (12.88 - 0.43 - BAND_GAP * (n - 1)) / n
    for i, (name, body) in enumerate(cells):
        stack_panel(slide, 0.43 + i * (w + BAND_GAP), y, w, h,
                    [(name, name_pt, True, MUTED), (body, body_pt, False, INK)],
                    where="statistics row cell {}".format(i))


def page(title, subtitle, headline, strip, note=None, src=SRC):
    """Open a page and register it. The strip and the chrome close it."""
    s = plain_slide(title, subtitle)
    chrome(s)
    return s


def close(s, title, headline, strip, note=None, src=SRC):
    highlight_strip(s, headline, line(strip))
    if note:
        footnote(s, note)
    source(s, src)
    RUNNING_ORDER.append((title, headline))


# ===================================================================== 1. why
T = "What rides on a district's crop mix"
s = page(T, "What a district's crop mix decides, and what this settles", "", "")
stack_panel(s, STRIP_X, 1.38, STRIP_W, 0.94, [(
    "A district that puts four fifths of its cropped area under one crop is exposed "
    "to that crop's price and that crop's water demand. The mix sits underneath "
    "groundwater draw, procurement exposure and what a farming household earns in a "
    "bad year. District statistics report area by crop every year and carry no "
    "measure of how that area is split.", 13.0, False, INK)], where="s1 lede")
titled_panel(s, THIRD_X[0], 2.38, THIRD_W, 1.96, "What was open",
             "Whether a district's crop mix follows the water it has, the markets it "
             "can reach or its agrarian structure. Cropping statistics carry area by "
             "crop and village records carry the rest, and the two have not been read "
             "together.", where="s1 a")
titled_panel(s, THIRD_X[1], 2.38, THIRD_W, 1.96, "The record",
             "Ministry of Agriculture district statistics: area and production by crop "
             "and season, 54 crops, 22 agricultural years, 13,605 district-year "
             "records. Village records on irrigation, markets, input supply and "
             "agrarian structure, aggregated to district.", where="s1 b")
titled_panel(s, THIRD_X[2], 2.38, THIRD_W, 1.96, "What this settles",
             "A measure of the crop mix that reads as a count of crops, the national "
             "picture and its geography, thirteen candidate explanations screened on "
             "one sample, and five taken forward with the coefficient that decides "
             "each.", where="s1 c")
stat_band(s, 4.46, 0.78, [(str(N_ALL), "districts with a cropping record"),
                          (str(N_SAMPLE), "carrying village records too"),
                          ("54", "crops"),
                          ("22", "agricultural years")])
close(s, T, "Half of a district's cropped area sits under its single largest crop",
      "The largest crop takes {:.1f} percent of cropped area across the {} districts, "
      "and the top three take {:.1f} percent.".format(DOM1, N_ALL, TOP3),
      "The join matches 99.6 percent of the cropping record, and 64 of the 606 "
      "districts sit on a shared pre-2011 parent, which every estimate is tested "
      "against.")

# ============================================================== 2. resolution
T = "Counting crops and weighting them"
s = page(T, "The effective number of crops against crops grown in an average year, "
            "by district", "", "")
place(s, "f1_count_vs_effective", TWOUP_X[0], 1.50, TWOUP_W, 3.40)
table(s, TWOUP_X[1], 1.42, [2.30, 0.65, 3.105],
      ["Measure", "Mean", "What it reads as"],
      [["Crops grown in an average year", "{:.1f}".format(MEAN["D0_richness"]),
        "crops with any recorded area"],
       ["Effective number of crops", "{:.1f}".format(MEAN["D1_exp_shannon"]),
        "equally common crops giving the same diversity"],
       ["Effective number weighted to the dominant crops",
        "{:.1f}".format(MEAN["D2_inv_simpson"]),
        "the same, with rare crops discounted further"],
       ["Evenness", "{:.3f}".format(MEAN["evenness_D1_D0"]),
        "the effective number over the count"]],
      row_h=0.38, head_h=0.28, align="lrl", pt=9.0, wrap=True, where="s2 scale")
stack_panel(s, TWOUP_X[1], 3.30, TWOUP_W, 1.95, [
    ("Across the 606 districts the effective number of crops runs {:.1f} at the "
     "tenth percentile, {:.1f} at the median and {:.1f} at the ninetieth.".format(
         Q("D1_exp_shannon", .1), Q("D1_exp_shannon", .5), Q("D1_exp_shannon", .9)),
     13.0, False, INK),
    ("The effective number of crops is the exponential of the Shannon entropy of a "
     "district's cropped-area shares, taken within each year and then averaged. It "
     "reads as a count. A district growing twenty crops with four fifths of its land "
     "under paddy farms like one growing three, and the measure says three.",
     13.0, False, INK)], where="s2 note")
close(s, T, "The average district is effectively growing five crops",
      "It grows {:.1f} crops in an average year, and evenness across the 606 "
      "districts averages {:.3f}.".format(MEAN["D0_richness"], MEAN["evenness_D1_D0"]),
      "Measured within an agricultural year and then averaged. The composite index "
      "used in earlier work is a min-max blend whose district scores move by 0.0032 "
      "on average and by up to 0.0190 when the sample changes.")

# ============================================================== 3. resolution
T = "The national picture in numbers"
s = page(T, "Cropping, composition and rural structure across 606 districts in 30 "
            "states", "", "")
AX, BX, CX = 0.43, 5.93, 9.33
AW, BW, CW = 5.30, 3.20, 3.55
block_head(s, AX, 1.36, AW, "Cropping, across the 606 districts")
table(s, AX, 1.66, [2.30, 0.75, 0.75, 0.75, 0.75],
      ["Measure", "Mean", "10th", "Median", "90th"],
      [["Crops grown in an average year"] + [
          "{:.1f}".format(v) for v in (MEAN["D0_richness"], Q("D0_richness", .1),
                                       Q("D0_richness", .5), Q("D0_richness", .9))],
       ["Effective number of crops"] + [
          "{:.1f}".format(v) for v in (MEAN["D1_exp_shannon"], Q("D1_exp_shannon", .1),
                                       Q("D1_exp_shannon", .5), Q("D1_exp_shannon", .9))],
       ["Evenness"] + [
          "{:.3f}".format(v) for v in (MEAN["evenness_D1_D0"],
                                       Q("evenness_D1_D0", .1), Q("evenness_D1_D0", .5),
                                       Q("evenness_D1_D0", .9))],
       ["Irrigation share of sown area", "0.546", "0.235", "0.529", "0.879"],
       ["Cropping intensity", "1.75", "1.44", "1.77", "2.03"],
       ["Mean holding size, hectares", "1.96", "0.61", "1.35", "2.95"]],
      row_h=0.30, where="s3 block a")
stack_panel(s, AX, 3.86, AW, 1.39, [
    ("Eight districts in ten sit between 0.24 and 0.88 on the irrigation share of "
     "sown area, so the sample spans the range of Indian water availability rather "
     "than one end of it. Mean holding size is inferred from village records rather "
     "than measured. It is carried through every estimate on this deck and is not "
     "distinguishable from zero in any of them.",
     12.0, False, INK)], where="s3 block a note")
block_head(s, BX, 1.36, BW, "Share of national cropped area")
table(s, BX, 1.66, [1.95, 1.25], ["Category", "Share"],
      [[n, "{:.1f}".format(v)] for n, v in NAT],
      row_h=0.30, row_fill={0: TINT}, where="s3 block b")
block_head(s, CX, 1.36, CW, "Share of a district's villages")
table(s, CX, 1.66, [2.30, 1.25], ["Facility", "Mean share"],
      [["producer organisation", "0.231"], ["fertiliser shop", "0.175"],
       ["weekly haat", "0.143"], ["farm-gate processing", "0.114"],
       ["regular market", "0.098"], ["seed centre", "0.098"],
       ["custom hiring", "0.092"], ["cold storage", "0.084"],
       ["soil testing", "0.043"], ["mandi", "0.030"]],
      row_h=0.29, where="s3 block c")
bare_lines(s, CX, 5.01, CW, 0.28,
           [("40 districts have no mandi and 30 have no weekly haat.",
             10.5, False, MUTED)], where="s3 block c tail")
close(s, T, "Cereals take {:.0f} percent of India's cropped area".format(NAT[0][1]),
      "Fruit and vegetables together take {:.1f} percent, and a mandi sits in 3 of "
      "every hundred villages against a weekly haat in 14.".format(
          dict(NAT)["fruit"] + dict(NAT)["vegetables"]),
      "Percentiles are across the 606 districts. Crop shares are weighted by each "
      "district's mean annual cropped area. Village shares come from the 2019 "
      "village records.")

# ============================================================== 4. resolution
T = "Effective number of crops by district"
s = page(T, "Averaged over each district's years, 1998-99 to 2019-20", "", "")
place(s, "f12_map_effective_large", 0.43, 1.24, 6.05, 4.02)
RAIL_X, RAIL_W2 = 6.90, 5.98
# The two state tables sit side by side inside the rail beside the map, so the
# district count comes out and the note below carries it instead.
TW = 2.80
block_head(s, RAIL_X, 1.34, TW, "Widest")
table(s, RAIL_X, 1.62, [1.72, 1.08],
      ["State", "Effective crops"],
      [[_tc(k), "{:.1f}".format(r["D1"])] for k, r in STATE_TOP.iterrows()],
      row_h=0.30, where="s4 widest")
block_head(s, RAIL_X + TW + 0.38, 1.34, TW, "Narrowest")
table(s, RAIL_X + TW + 0.38, 1.62, [1.72, 1.08],
      ["State", "Effective crops"],
      [[_tc(k), "{:.1f}".format(r["D1"])] for k, r in STATE_BOT.iterrows()],
      row_h=0.30, where="s4 narrowest")
stack_panel(s, RAIL_X, 3.52, RAIL_W2, 1.73, [
    ("{} grows {:.1f} crops in an average year, more than any other state, "
     "and is effectively growing {:.1f} on an evenness of {:.3f}.".format(
         CG_NAME, CG["D0"], CG["D1"], CG["E"]), 12.0, False, INK),
    ("Punjab reaches the narrow end through wheat and Odisha through paddy. Both "
     "grow enough crops, and the land sits under one of them.", 12.0, False, INK),
    ("The dark districts run through the peninsular and central belt, and the pale "
     "ones through the eastern rice belt and the irrigated northwest.",
     12.0, False, INK)], where="s4 rail note")
close(s, T, "Karnataka districts average {:.1f} effective crops, the highest of "
            "any state".format(_g.loc["KARNATAKA", "D1"]),
      "Odisha districts average {:.1f} and Punjab districts {:.1f}, reaching the "
      "same place through paddy and through wheat.".format(
          _g.loc["ODISHA", "D1"], _g.loc["PUNJAB", "D1"]),
      "State figures are unweighted means of their districts, for states with at "
      "least four, over {} districts in Karnataka, {} in Madhya Pradesh, {} in "
      "Odisha and {} in Punjab. Grey districts carry no agricultural record.".format(
          int(_g.loc["KARNATAKA", "n"]), int(_g.loc["MADHYA PRADESH", "n"]),
          int(_g.loc["ODISHA", "n"]), int(_g.loc["PUNJAB", "n"])))

# ============================================================== 5. resolution
T = "The crop that takes the most land"
s = page(T, "Largest single land use by district, and what each quarter of districts "
            "plants", "", "")
block_head(s, STRIP_X, 1.30, 6.10, "Largest land use, of the {} districts".format(N_ALL))
HEAD5 = ["Crop", "Districts", "Share of cropped area"]
table(s, 0.43, 1.56, [2.60, 1.10, 2.40], HEAD5, DOMROWS[:6],
      row_h=0.25, head_h=0.26, row_fill={0: TINT}, where="s5 crops left")
table(s, 6.78, 1.56, [2.60, 1.10, 2.40], HEAD5, DOMROWS[6:],
      row_h=0.25, head_h=0.26, where="s5 crops right")
block_head(s, STRIP_X, 3.46, 8.00,
           "Composition by quarter of the effective number of crops, percent of "
           "cropped area")
table(s, STRIP_X, 3.76, [2.45] + [1.25] * 8,
      ["Quarter", "cereals", "oilseeds", "pulses", "fibre crops", "spices", "sugar",
       "vegetables", "fruit"],
      [[nm] + ["{:.1f}".format(QTR.loc[k, c]) for c in
               ["share_cereals", "share_oilseeds", "share_pulses", "share_fiber_crops",
                "share_spices", "share_sugar", "share_vegetable", "share_fruits"]]
       for k, nm in [(0, "Least diverse quarter"), (1, "Second"), (2, "Third"),
                     (3, "Most diverse quarter")]],
      row_h=0.29, cell_fill={(i, 1): TINT for i in range(4)}, where="s5 quarters")
close(s, T, "Rice or wheat is the largest land use in {} of {} districts".format(
    RICEWHEAT, N_ALL),
      "Where cereal area gives way it goes to pulses and oilseeds, and fruit and "
      "vegetables stay near a fiftieth of the land.",
      "Districts are sorted into quarters on the effective number of crops across the "
      "{} with a cropping record. Category shares are averaged within the quarter.".format(N_ALL))

# ============================================================== 6. resolution
T = "Crop diversity over two decades"
s = page(T, "Fixed panel of {} districts reporting in every year, 1998-99 to "
            "2019-20".format(NPAN), "", "")
place(s, "f13_trend_wide", STRIP_X, 1.40, STRIP_W, 2.90)
stat_row(s, 4.42, 0.80, [
    ("Effective number of crops", "{} districts up, {} down, median {}".format(
        CH["D1_up"], CH["D1_down"], sg(CH["dD1_median"]))),
    ("Crops grown in an average year", "{} up, {} down, median {}".format(
        CH["D0_up"], CH["D0_down"], sg(CH["dD0_median"]))),
    ("Evenness", "{} up, {} down, median {}".format(
        CH["E_up"], CH["E_down"], sg(CH["dE_median"], 3)))])
close(s, T, "The effective number of crops held at about {:.1f} across two decades".format(
          sum(t["D1"] for t in TRB) / len(TRB)),
      "Crops grown rose from {:.1f} to {:.1f}, a reporting effect: above 0.1 percent "
      "of area the change is -0.04.".format(TRB[0]["D0"], TRB[-1]["D0"]),
      "Same {} districts every year, weighted by cropped area. Raw endpoint change in "
      "crops grown +4.11 (SE 0.41); district counts compare 2013-19 with 1998-2004."
      .format(NPAN))

# ================================================== 7. identifying hypotheses
T = "Crop diversity against irrigation"
s = page(T, "Effective number of crops by fifth of the irrigation share, within "
            "state, against the least irrigated fifth", "", "")
place(s, "f5_irrigation_shape", TWOUP_X[0], 1.55, TWOUP_W, 3.30)
_rows7 = [["1 (driest)", str(FIF[0]["n"]), "{:.2f}".format(FIF[0]["irr"]), "reference",
           "", ""]]
for r in FIF[1:]:
    _rows7.append([str(r["q"]) + (" (wettest)" if r["q"] == 5 else ""), str(r["n"]),
                   "{:.2f}".format(r["irr"]), sg(r["b"]), "{:.2f}".format(r["se"]),
                   pf(r["p"])])
table(s, TWOUP_X[1], 1.42, [1.25, 0.85, 0.95, 1.15, 0.75, 1.025],
      ["Fifth", "Districts", "Irrigation", "Difference", "SE", "Wild p"],
      _rows7, row_h=0.30, align="lrrrrr", row_fill={4: TINT}, where="s7 fifths")
bare_lines(s, TWOUP_X[1], 3.55, TWOUP_W, 1.70, [
    ("A line with a knot at 0.30 of sown area has slope {} per unit below the knot "
     "(p {}) and {} above it (p {}), about {} effective crops for each 10 points of "
     "irrigation. {:.0f} percent of districts lie above the knot.".format(
         sg(SPL["below"]), pf(SPL["below_p"]), sg(SPL["above"]), pf(SPL["above_p"]),
         sg(SPL["above"] / 10, 1), IRRJ["share_above_030"] * 100),
     12.0, False, INK),
    ("In tenths of the irrigation share none of the first seven differs "
     "significantly from the driest tenth.", 12.0, False, INK)],
    gap_pt=5, where="s7 line note")
close(s, T, "Within a state the effective number of crops is flat across the less "
            "irrigated districts and lower in the most irrigated ones",
      "The wettest fifth averages {:.2f} fewer effective crops than the driest fifth "
      "of the same state (wild p {}), and the second fifth differs by {}.".format(
          -FIF[4]["b"], pf(FIF[4]["p"]), sg(FIF[1]["b"])),
      "Differences carry state fixed effects. Intervals cluster on the state ({} "
      "states) and p values come from a restricted wild cluster bootstrap by "
      "state.".format(IRRJ["fifths"]["full"]["G"]))

# ================================================== 8. identifying hypotheses
T = "Thirteen candidate explanations"
s = page(T, "Association with the effective number of crops, before and after "
            "adjustment. One star is p below 0.10, two below 0.05, three below 0.01",
         "", "")
_C8 = [("irrigation share of sown area", "irrigation share", "the slope at zero, with the square held"),
       ("canal village share", "canal village share", "command-area rotations"),
       ("mean holding size, log hectares", "mean holding (log ha)", "small farms spreading risk"),
       ("cultivator share of agricultural workers", "cultivator share of ag workers", "labour supply and crop choice"),
       ("landless share", "landless share (SECC)", "rural deprivation records"),
       ("mandi village share", "mandi village share", "assured procurement"),
       ("weekly haat village share", "weekly haat village share", "smallholder selling points"),
       ("regular market village share", "regular market village share", "daily retail demand"),
       ("producer organisation village share", "FPO village share", "collective aggregation"),
       ("cold storage village share", "cold storage village share", "perishables need cold"),
       ("SC population share", "SC population share", "caste and land access"),
       ("ST population share", "ST population share", "forest-fringe cropping"),
       ("cropping intensity", "cropping intensity", "an accounting identity as much as a lever")]
_SIG8 = [i for i, (_, k, _) in enumerate(_C8) if DIM.loc[k, "adj_p"] < .05]
assert _SIG8 == [0, 8, 12], _SIG8
table(s, STRIP_X, 1.42, [2.95, 0.75, 1.05, 1.00, 5.10],
      ["Candidate", "Districts", "Unadjusted", "Adjusted", "Where it comes from"],
      [[lab, str(int(DIM.loc[k, "n"])),
        "{}{}".format(sg(DIM.loc[k, "raw_b"], 3), stars(DIM.loc[k, "raw_p"])),
        "{}{}".format(sg(DIM.loc[k, "adj_b"], 3), stars(DIM.loc[k, "adj_p"])), why]
       for lab, k, why in _C8],
      row_h=0.265, align="lrrrl", pt=9.0, head_pt=9.0, bold_rows=(0, 8, 12),
      rail=[("Water", 2), ("Agrarian structure", 3),
            ("Markets and infrastructure", 5), ("Population and land use", 3)],
      rail_w=1.60, where="s8 screen")
close(s, T, "Three of the thirteen candidates keep a significant coefficient after adjustment",
      "The irrigation share of sown area, producer organisation coverage and "
      "cropping intensity.",
      "Adjusted models add state fixed effects, log mean holding size and the "
      "irrigation share with its square. Facility shares are village shares. "
      "The irrigation row is the linear term.")

# ========================================================= 9. the hypotheses
T = "The five hypotheses and what would falsify each"
s = page(T, "The test each one gets and what would falsify it", "", "")
table(s, STRIP_X, 1.44, [1.75, 3.75, 3.60, 3.35],
      ["Hypothesis", "Mechanism", "What would confirm it", "What would falsify it"],
      [["H1 Water quantity",
        "At high levels of irrigation water pays for one assured package, so other "
        "crops give way to it",
        "A lower effective number of crops in the wettest fifth than in the driest "
        "fifth of the same state, and a negative slope above the knot",
        "Flat fifths all the way up, or a fall that depends on which index or "
        "sample is used"],
       ["H2 Water type",
        "Canal water arrives on a rotation the system sets and groundwater arrives "
        "when the cultivator chooses, so fewer crops fit the canal calendar",
        "Source coefficients that stay large with the level of irrigation held "
        "constant",
        "Null source coefficients once the level of irrigation is in"],
       ["H3 Somewhere to sell",
        "A weekly haat lets a smallholder sell small volumes of many things, so crops "
        "with no other outlet get planted",
        "Haat coverage raising crops grown in an average year, surviving development "
        "controls",
        "Null haat coefficients, or coefficients that vanish under those controls"],
       ["H4 Collective marketing",
        "A producer organisation aggregates volume for crops a single farm cannot "
        "market, so land shifts between crops already grown",
        "Producer organisation coverage raising the effective number of crops with "
        "crops grown flat",
        "Both institutions moving both measures together"],
       ["H5 Assured procurement",
        "Dense procurement and input supply make the cereal package the only paying "
        "choice once water is plentiful, which is what pulls diversity down at "
        "high irrigation",
        "A steeper fall at high irrigation where mandi and input coverage is dense, "
        "and a higher cereal share where mandis are dense",
        "A positive interaction, or a negative mandi coefficient on cereal share"]],
      row_h=0.66, head_h=0.30, align="llll", pt=9.5, wrap=True, where="s9 hypotheses")
close(s, T, "Each hypothesis names the coefficient that decides it",
      "The outcome is the effective number of crops throughout, with state fixed "
      "effects on every estimate.",
      "Adjusted models carry the irrigation share of sown area and its square, log "
      "mean holding size, ST population share, log night lights, non-farm "
      "establishment density and a connectivity index.")

# =============================================================== 10. findings
T = "The fall at high irrigation in three windows"
s = page(T, "H1, fifths of the irrigation share against the driest fifth, and the "
            "squared term across indices", "", "")
verdict(s, STRIP_X, 1.32, 2.60, "H1 holds for the fall")
_WIN = [("full", "1998-99 to 2019-20"), ("w1517", "2015-16 to 2019-20"),
        ("w1721", "2017-18 to 2021-22")]
_rows10 = []
for k, lab in _WIN:
    blk = IRRJ["fifths"][k]
    rr = blk["rows"]
    _rows10.append([lab, str(blk["n"])] +
                   ["{} ({})".format(sg(r["b"]), pf(r["p"])) for r in rr[1:]] +
                   [sg(IRRJ["spline"][k]["above"]) if k in IRRJ["spline"] else ""])
table(s, STRIP_X, 1.70, [2.60, 1.00, 1.55, 1.55, 1.55, 1.55, 1.50],
      ["Window", "Districts", "Fifth 2", "Fifth 3", "Fifth 4", "Fifth 5",
       "Slope above 0.30"],
      _rows10, row_h=0.34, head_h=0.28, pt=10.0, head_pt=9.5, align="lrrrrrr",
      row_fill={0: TINT}, where="s10 windows")
_ROB = pd.read_csv(COV / "final_robustness.csv").set_index("spec")
_d1 = _ROB.iloc[1:8]
_rows10b = [["Effective number of crops, state fixed effects",
             sg(_ROB.loc["D1: state fixed effects", "b2"]),
             pf(_ROB.loc["D1: state fixed effects", "p2"])],
            ["Effective number of crops, seven samples and weights",
             "{} to {}".format(sg(_d1["b2"].max()), sg(_d1["b2"].min())),
             "all below 0.0001"]]
for k, lab in [("D0 richness", "Crops grown in an average year"),
               ("D2 inverse Simpson", "Effective number weighted to the dominant crops"),
               ("evenness D1/D0", "Evenness")]:
    _rows10b.append([lab, sg(_ROB.loc[k, "b2"], 3), pf(_ROB.loc[k, "p2"])])
block_head(s, STRIP_X, 3.18, 8.0, "Squared term on the irrigation share, state fixed effects")
table(s, STRIP_X, 3.46, [5.10, 2.20, 2.00], ["Outcome", "Squared term", "p"],
      _rows10b, row_h=0.28, head_h=0.26, pt=9.5, align="lrr", where="s10 squares")
close(s, T, "The fall in the effective number of crops at high irrigation holds in "
            "every window",
      "The wettest fifth sits {:.2f} to {:.2f} effective crops below the driest "
      "fifth, and the squared term is negative for every index tried.".format(
          -max(IRRJ["fifths"][k]["rows"][4]["b"] for k, _ in _WIN),
          -min(IRRJ["fifths"][k]["rows"][4]["b"] for k, _ in _WIN)),
      "Fifth columns give the difference against the driest fifth with the wild "
      "bootstrap p in brackets, state fixed effects, errors clustered on the state. "
      "The 2017-18 to 2021-22 window drops six crops with reporting breaks.")

# =============================================================== 11. findings
T = "Irrigation source and what a district grows"
s = page(T, "H2, canal and surface water against groundwater at the same irrigation "
            "share", "", "")
place(s, "f7_irrigation_source", TWOUP_X[0], 1.60, TWOUP_W, 3.20)
verdict(s, TWOUP_X[1], 1.36, 2.10, "H2 holds")
table(s, TWOUP_X[1], 1.74, [2.85, 0.88, 0.62, 1.02, 0.685],
      ["Outcome", "Canal", "p", "Surface water", "p"],
      [[lab, sg(SRCR.loc[k, "b_canal"], 3), pf(SRCR.loc[k, "p_canal"]),
        sg(SRCR.loc[k, "b_surface"], 3), pf(SRCR.loc[k, "p_surface"])]
       for lab, k in [("Effective number of crops", "D1_exp_shannon"),
                      ("Crops grown in an average year", "D0_richness"),
                      ("Effective number weighted to the dominant crops",
                       "D2_inv_simpson"),
                      ("Evenness", "evenness_D1_D0"), ("Cereal share", "share_cereals"),
                      ("Pulse share", "share_pulses"),
                      ("Oilseed share", "share_oilseeds")]],
      row_h=0.29, pt=9.0, where="s11 source")
stack_panel(s, TWOUP_X[1], 4.11, TWOUP_W, 1.14, [
    ("The irrigation share of sown area and its square are held in every row, so "
     "these are differences between water sources at the same amount of water.",
     12.0, False, INK),
    ("Groundwater is the omitted source and covers 50.5 percent of villages, other "
     "sources 21.1, canal 17.2 and surface water 11.3.", 12.0, False, INK)],
    where="s11 note")
close(s, T, "Surface-water districts average two fewer effective crops at the same "
            "irrigation share",
      "Canal dependence takes {:.1f} crops off the list, adds {:.3f} to the pulse "
      "share and leaves the effective number alone.".format(
          -SRCR.loc["D0_richness", "b_canal"], SRCR.loc["share_pulses", "b_canal"]),
      "Coefficients are on the share of a district's villages naming that source, so "
      "they read from none to all. State fixed effects, errors clustered on the "
      "pre-2011 parent district.")

# ================================================== 12. identifying hypotheses
T = "How the rural facilities sit together"
s = page(T, "Correlation across districts between the share of villages carrying "
            "each facility", "", "")
place(s, "f15_facility_matrix", 0.43, 1.22, 5.05, 4.02)
RX, RW = 5.85, 7.03
block_head(s, RX, 1.30, RW, "The weekly haat against the other nine")
table(s, RX, 1.58, [2.55, 1.15],
      ["Facility", "Correlation"],
      [["Regular market", "-0.21"], ["Mandi", "-0.15"], ["Seed centre", "-0.10"],
       ["Custom hiring", "-0.09"], ["Farm-gate processing", "-0.09"],
       ["Soil testing", "-0.04"], ["Fertiliser shop", "-0.02"],
       ["Producer organisation", "+0.05"], ["Cold storage", "+0.06"]],
      row_h=0.222, head_h=0.24, pt=10.0, where="s12 haat row")
stack_panel(s, RX + 4.05, 1.58, RW - 4.05, 2.28, [
    ("What the block means", 12.5, True, INK),
    ("Nine of the ten facilities correlate between 0.24 and 0.83 with each other. "
     "A district that has one tends to have the rest, so they carry one piece of "
     "information between them and not nine.", 12.0, False, INK)],
    where="s12 block note")
stack_panel(s, RX, 4.02, RW, 1.22, [
    ("Why the weekly haat is treated on its own", 12.5, True, INK),
    ("It correlates at -0.21 with regular markets and -0.15 with mandis, at 0.04 "
     "with night-time lights and -0.16 with the density of non-farm establishments. "
     "A periodic market sits where the rest of this infrastructure does not, which "
     "is what lets H3 be asked separately from H4.", 12.0, False, INK)],
    where="s12 haat note")
close(s, T, "Nine of the ten facilities move together across districts",
      "The weekly haat is the only one that does not, and it runs mildly negative "
      "against the output markets.",
      "Each cell is the correlation across 606 districts between the share of "
      "villages carrying one facility and the share carrying the other.")

# =============================================================== 13. findings
T = "Rural facilities and cropping"
s = page(T, "H3 and H4, each facility entered on its own with the full control set",
         "", "")
place(s, "f10_market_solo", TWOUP_X[0], 1.55, TWOUP_W, 2.70)
_adj = HAATW.iloc[1:7]
stack_panel(s, TWOUP_X[0], 4.35, TWOUP_W, 0.90, [
    ("A one standard deviation rise in haat coverage, 0.150 of a district's rural "
     "population, goes with about 0.9 more crops against a mean of {:.1f}, and the "
     "coefficient runs {} to {} across the adjusted specifications.".format(
         MEAN["D0_richness"], sg(_adj["b"].min()), sg(_adj["b"].max())),
     12.0, False, INK)], where="s12 scale")
verdict(s, TWOUP_X[1], 1.36, 4.30, "H3 holds on crops grown only; H4 holds at parent level")
_F13 = [("Weekly haat", "weekly_haat"), ("Cold storage", "storage"),
        ("Seed centre", "seed_centre"), ("Farm-gate processing", "farmgate_proc"),
        ("Soil testing", "soil_test"), ("Producer organisation", "fpo"),
        ("Mandi", "mandi"), ("Custom hiring", "custom_hire"),
        ("Fertiliser shop", "fert_shop"), ("Regular market", "regular_market")]
_F13 = sorted(_F13, key=lambda t: -PW("solo", t[1], "D0_richness")["b"])


def _c13(fac, out, d=2):
    r = PW("solo", fac, out)
    return "{} ({})".format(sg(r["b"], d), pf(r["p"]))


table(s, TWOUP_X[1], 1.74, [1.95, 1.15, 1.10, 0.95, 0.905],
      ["Facility", "Effective crops", "Crops grown", "Cereal share", "Pulse share"],
      [[lab, _c13(k, "D1_exp_shannon"), _c13(k, "D0_richness"),
        _c13(k, "share_cereals", 3), _c13(k, "share_pulses", 3)] for lab, k in _F13],
      row_h=0.29, pt=8.5, row_fill={0: TINT, [k for _, k in _F13].index("fpo"): TINT},
      where="s12 solo")
_h = PW("solo", "weekly_haat", "D0_richness")
_f = PW("solo", "fpo", "D1_exp_shannon")
close(s, T, "Weekly haat coverage goes with {:.1f} more crops grown in an average "
            "year".format(_h["b"]),
      "Producer organisation coverage adds {:.1f} to the effective number of crops, "
      "with p = 0.156 under state clustering.".format(_f["b"]),
      "Facility shares are weighted by Census 2011 rural population. On village "
      "shares nine facilities correlate 0.24 to 0.83 and the haat -0.21 to +0.06 "
      "with them. Errors cluster on the parent district.")

# =============================================================== 14. findings
T = "Market density and the fall at high irrigation"
s = page(T, "H5, curvature of the irrigation relationship where infrastructure is "
            "thin and where it is dense", "", "")
place(s, "f14_interaction_wide", 0.43, 1.42, 8.20, 1.85)
_mc = PW("solo", "mandi", "share_cereals")
_mb = PW("output", "mandi", "share_cereals")
stack_panel(s, 8.90, 1.42, 3.97, 1.85, [
    ("The other half of the same hypothesis", 12.5, True, INK),
    ("Mandi coverage was expected to raise the cereal share. Entered alone with the "
     "full control set it goes with a cereal share {:.3f} lower (p = {:.3f}), and "
     "entered with the other output markets, {:.3f} lower.".format(
         -_mc["b"], _mc["p"], -_mb["b"]), 12.0, False, INK)],
    where="s13 callout")
verdict(s, STRIP_X, 3.36, 2.15, "H5 does not hold")
_tyd = []
for _ln in MKT_MD.splitlines():
    _m = re.match(r"^\| output (high|low), input (high|low) \| \d+ \| [\d.]+ \| ([\d.]+) \|", _ln)
    if _m:
        _tyd.append(float(_m.group(3)))
bare_lines(s, 2.90, 3.40, 9.98, 0.30, [
    ("Districts split at the median on market density run {:.2f} to {:.2f} effective "
     "crops raw, and all three adjusted contrasts are null.".format(
         min(_tyd), max(_tyd)), 10.5, False, MUTED)],
    where="s13 typology")
_I = INTW.sort_values("inter", ascending=False)
_IL = {"mandi": "Mandi population share", "fertiliser shop": "Fertiliser shop population share",
       "output-market index": "Output-market density",
       "input-supply index": "Input-supply density",
       "post-harvest index": "Post-harvest density"}
table(s, STRIP_X, 3.78, [3.05, 2.35, 2.40, 2.05, 2.60],
      ["Infrastructure", "Curvature where thin", "Curvature where dense",
       "Interaction", "p"],
      [[_IL[r["measure"]], sg(r["curv_lo"]), sg(r["curv_hi"]), sg(r["inter"]),
        pf(r["p"])] for _, r in _I.iterrows()],
      row_h=0.24, head_h=0.25, pt=9.5, row_fill={0: TINT}, where="s13 interaction")
_npos = int((INTW["inter"] > 0).sum())
close(s, T, "{} of the five interaction terms run the opposite way to the "
            "hypothesis".format({4: "Four", 5: "All five"}[_npos]),
      "A positive term means the fall at high irrigation is gentler where "
      "infrastructure is dense, not sharper, and none is below 5 percent; the mandi "
      "term is {:.3f}.".format(INTW.loc[INTW["measure"] == "mandi", "p"].iloc[0]),
      "Curvature is the coefficient on the square of the irrigation share of sown "
      "area, read one standard deviation below and above the mean of each measure. "
      "Facility shares are weighted by Census 2011 rural population.")

# =============================================================== 15. findings
T = "The five hypotheses and their verdicts"
s = page(T, "Five hypotheses, the test applied and the number that decides each",
         "", "")
table(s, STRIP_X, 1.42, [1.85, 3.10, 5.20, 2.30],
      ["Hypothesis", "The test", "The deciding number", "Verdict"],
      [["H1 Water quantity",
        "Fifths of the irrigation share against the driest fifth, within state, and a "
        "line with a knot at 0.30",
        "Wettest fifth {} effective crops (wild p {}), slope {} per 10 points above "
        "the knot, and the squared term negative for every index and sample".format(
            sg(FIF[4]["b"]), pf(FIF[4]["p"]), sg(SPL["above"] / 10, 1)),
        "Holds for the fall"],
       ["H2 Water type",
        "Source coefficients with the irrigation share and its square held",
        "Surface water {} effective crops (p = {}) and cereal share {} (p = {}); "
        "canal {} crops grown (p = {})".format(
            sg(SRCR.loc["D1_exp_shannon", "b_surface"]),
            pf(SRCR.loc["D1_exp_shannon", "p_surface"]),
            sg(SRCR.loc["share_cereals", "b_surface"], 3),
            pf(SRCR.loc["share_cereals", "p_surface"]),
            sg(SRCR.loc["D0_richness", "b_canal"]),
            pf(SRCR.loc["D0_richness", "p_canal"])), "Holds"],
       ["H3 Somewhere to sell",
        "Haat coverage on crops grown under seven specifications",
        "{} to {} with controls and {} without, p below 0.001 throughout, with the "
        "effective number at {} (p = {}) and evenness at {}".format(
            sg(HAATW.iloc[1:]["b"].min()), sg(HAATW.iloc[1:]["b"].max()),
            sg(HAATW.iloc[0]["b"]), sg(PW("solo", "weekly_haat", "D1_exp_shannon")["b"]),
            pf(PW("solo", "weekly_haat", "D1_exp_shannon")["p"]),
            sg(PW("solo", "weekly_haat", "evenness_D1_D0")["b"], 3)),
        "Holds on crops grown only"],
       ["H4 Collective marketing",
        "Producer organisation coverage read on both measures at once",
        "{} on the effective number (p = {}) and {} on crops grown (p = {}); under "
        "state clustering p = 0.156".format(
            sg(_f["b"]), pf(_f["p"]), sg(PW("solo", "fpo", "D0_richness")["b"]),
            pf(PW("solo", "fpo", "D0_richness")["p"])),
        "Holds at parent level"],
       ["H5 Assured procurement",
        "A steeper fall at high irrigation where coverage is dense, and a higher "
        "cereal share where mandis are dense",
        "{} of five interactions positive, mandi {} (p = {}), and mandi coverage on "
        "cereal share {}".format(
            {4: "Four", 5: "All five"}[_npos],
            sg(INTW.loc[INTW["measure"] == "mandi", "inter"].iloc[0]),
            pf(INTW.loc[INTW["measure"] == "mandi", "p"].iloc[0]), sg(_mc["b"], 3)),
        "Does not hold"]],
      row_h=0.62, head_h=0.28, align="llll", pt=9.5, wrap=True,
      cell_fill={(0, 3): TINT, (1, 3): TINT, (2, 3): TINT, (3, 3): TINT},
      where="s14 scorecard")
bare_lines(s, STRIP_X, 4.92, STRIP_W, 0.32, [
    ("Agrarian structure was screened out before the hypotheses: mean holding size "
     "{} (p = {}), cultivator share {} (p = {}), landless share {} (p = {}).".format(
         sg(DIM.loc["mean holding (log ha)", "adj_b"]),
         pf(DIM.loc["mean holding (log ha)", "adj_p"]),
         sg(DIM.loc["cultivator share of ag workers", "adj_b"]),
         pf(DIM.loc["cultivator share of ag workers", "adj_p"]),
         sg(DIM.loc["landless share (SECC)", "adj_b"]),
         pf(DIM.loc["landless share (SECC)", "adj_p"])), 11.0, False, MUTED)],
    where="s14 tail")
close(s, T, "Water quantity, water source and producer organisation coverage move "
            "the effective number of crops",
      "Mandi density, market type and agrarian structure do not.",
      "606 districts across 30 states. Irrigation systems and market facilities sit "
      "where terrain and aquifers already favoured them, and the rural record is a "
      "2019 cross-section. Market rows are population weighted.")


os.makedirs(str(OUT.parent), exist_ok=True)
prs.save(str(OUT))
print("WROTE {}".format(OUT))
print("\nRunning order, {} slides".format(len(RUNNING_ORDER)))
for i, (t, h) in enumerate(RUNNING_ORDER, 1):
    print("  {:2d}  {:46s} {}".format(i, t[:46], h))
