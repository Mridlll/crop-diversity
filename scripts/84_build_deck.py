"""
84_build_deck.py

The crop diversity deck, CEEW house style.

Imports the house furniture rather than restating it. Figures come from
83_deck_figures.py and carry data only: the slide title names the page, the
strip carries the finding, the footnote carries definitions.

Output: deck/crop_diversity.pptx
"""
import os
import sys
from pathlib import Path

from PIL import Image
from pptx.util import Inches

FURN = r"D:/Alternative Proteins/demand_pathways/deck"
sys.path.insert(0, FURN)

from deck_furniture import (  # noqa: E402
    plain_slide, chrome, source, footnote, highlight_strip, titled_panel,
    prose_panel, prs, RUNNING_ORDER,
    TWOUP_X, TWOUP_W, STRIP_X, STRIP_W, INK)

REPO = Path(r"D:/crop-diversity")
FIGS = REPO / "deck" / "figs"
OUT = REPO / "deck" / "crop_diversity.pptx"

FIG_TOP, FIG_BOT = 1.42, 5.12          # the band a figure may occupy
FIG_H = FIG_BOT - FIG_TOP


def place(slide, name, x, y, max_w, max_h, centre=False):
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


def note(slide, x, y, w, h, name, body, where):
    titled_panel(slide, x, y, w, h, name, body, where=where)


def line(text):
    """A strip body: one line, ink."""
    return [[(text, {"color": INK})]]


# ===================================================================== 1
s = plain_slide("Crop diversity across Indian districts",
                "What the cropping pattern looks like, and what it is arranged around")
chrome(s)
prose_panel(s, STRIP_X, 1.70, STRIP_W, 1.15,
            "District area and production for 54 crops across 23 agricultural years, read "
            "against village-level rural structure for the same districts. The question is "
            "what a district's crop mix is arranged around: the water it has, and the "
            "markets it sells into.", where="s1 lede")
note(s, TWOUP_X[0], 3.10, TWOUP_W, 1.70, "Cropping",
     "Ministry of Agriculture district statistics, 1997-98 to 2019-20. Area, production "
     "and yield by crop and season, covering 725 districts.", "s1 left")
note(s, TWOUP_X[1], 3.10, TWOUP_W, 1.70, "Rural structure",
     "Village records on irrigation, markets, input supply and agrarian structure, "
     "aggregated to district. Roughly 600 districts across 30 states carry both.",
     "s1 right")
source(s, "Ministry of Agriculture and Farmers Welfare; SHRUG 2.1, Development Data Lab")
RUNNING_ORDER.append(("Crop diversity across Indian districts", ""))

# ===================================================================== 2
s = plain_slide("Counting crops and weighting them",
                "Effective number of crops against the plain count, by district")
chrome(s)
place(s, "f1_count_vs_effective", TWOUP_X[0], FIG_TOP, TWOUP_W, FIG_H)
note(s, TWOUP_X[1], FIG_TOP + 0.15, TWOUP_W, 1.60, "What the measure is",
     "The effective number of crops is how many equally sized crops would give the "
     "diversity observed. A district growing 20 crops with four fifths of its land under "
     "paddy farms like a district growing three, and the measure says three.", "s2 upper")
note(s, TWOUP_X[1], FIG_TOP + 1.95, TWOUP_W, 1.60, "Why it is used",
     "Both axes are counts of crops. The dashed line is where a district would sit if it "
     "split its land equally between every crop it grows, so the drop below that line is "
     "how concentrated its cropping is.", "s2 lower")
highlight_strip(s, "The average district has an effective number of about five crops",
                line("It grows about twenty. Most of its land sits under two or three of "
                     "them, which is what pulls the effective number down."))
footnote(s, "Each index is measured within a year and averaged across the years a "
            "district reports, so it does not scale with length of observation.")
RUNNING_ORDER.append(("Counting crops and weighting them", "twenty grown, five effective"))

# ===================================================================== 3
s = plain_slide("Effective number of crops by district",
                "Averaged over each district's years, 1997-98 to 2019-20")
chrome(s)
place(s, "f2_map_effective_crops", TWOUP_X[0], FIG_TOP, TWOUP_W, FIG_H, centre=True)
note(s, TWOUP_X[1], FIG_TOP + 0.15, TWOUP_W, 1.60, "Where the mix is wide",
     "The peninsular and central belt runs widest: Karnataka, interior Andhra Pradesh, "
     "Madhya Pradesh and Rajasthan.", "s3 upper")
note(s, TWOUP_X[1], FIG_TOP + 1.95, TWOUP_W, 1.60, "Where it is narrow",
     "The eastern rice belt and the intensively irrigated northwest sit at the bottom. "
     "Punjab and Odisha arrive there from opposite directions, one through wheat and "
     "one through paddy.", "s3 lower")
highlight_strip(s, "Districts at the bottom of the range put most of their land under one crop",
                line("They grow about as many crops as everywhere else. The land is "
                     "what is concentrated."))
footnote(s, "Grey districts carry no agricultural record, which is mostly urban territory.")
RUNNING_ORDER.append(("Effective number of crops by district", "land under one crop"))

# ===================================================================== 4
s = plain_slide("The crop that takes the most land",
                "Districts by their largest single land use")
chrome(s)
place(s, "f3_dominant_crop", TWOUP_X[0], FIG_TOP, TWOUP_W, FIG_H)
note(s, TWOUP_X[1], FIG_TOP + 0.15, TWOUP_W, 1.60, "Two cereals",
     "Rice is the largest land use in about half of Indian districts and wheat in a "
     "further fifth. What sits below them is regional specialisation rather than a "
     "national pattern.", "s4 upper")
note(s, TWOUP_X[1], FIG_TOP + 1.95, TWOUP_W, 1.60, "What diversification has meant",
     "Where cereal area gives way, oilseeds and pulses take it up. Fruit and vegetables "
     "stay a small share of land even in the most diverse quarter of districts.",
     "s4 lower")
highlight_strip(s, "Seven in ten districts have rice or wheat as their largest land use",
                line("The dominant crop takes between a third and a half of cropped "
                     "area in those districts."))
RUNNING_ORDER.append(("The crop that takes the most land", "seven in ten rice or wheat"))

# ===================================================================== 5
s = plain_slide("Crop diversity over time",
                "Area-weighted means across districts reporting in every year")
chrome(s)
place(s, "f4_trend", TWOUP_X[0], FIG_TOP, TWOUP_W, FIG_H)
note(s, TWOUP_X[1], FIG_TOP + 0.15, TWOUP_W, 1.60, "The national series",
     "Flat across two decades on every index. Whatever has reshaped Indian agriculture "
     "over this period has not moved the aggregate crop mix.", "s5 upper")
note(s, TWOUP_X[1], FIG_TOP + 1.95, TWOUP_W, 1.60, "Districts underneath it",
     "More districts lost effective diversity than gained it, 264 against 165. The "
     "aggregate holds still because those losses fall in districts carrying less of the "
     "country's cropped area.", "s5 lower")
highlight_strip(s, "264 districts lost effective diversity over the period and 165 gained it",
                line("The area-weighted national series holds still because those losses "
                     "fall in districts carrying less cropped area."))
footnote(s, "Balanced panel of 429 districts. Early period 1998 to 2004 against late "
            "period 2013 to 2019.")
RUNNING_ORDER.append(("Crop diversity over time", "264 down, 165 up"))

# ===================================================================== 6
s = plain_slide("Crop diversity against irrigation",
                "Effective number of crops by share of sown area irrigated")
chrome(s)
place(s, "f5_irrigation_shape", TWOUP_X[0], FIG_TOP, TWOUP_W, FIG_H)
place(s, "f6_map_irrigation", TWOUP_X[1], FIG_TOP, TWOUP_W, FIG_H, centre=True)
highlight_strip(s, "The effective number of crops rises with irrigation and then falls away",
                line("The same rise and fall appears in all fourteen versions of the "
                     "estimate, with the turn between a third and a half of sown area."))
footnote(s, "Irrigation share is irrigated land over irrigated plus unirrigated land, from "
            "village records. The fitted turn shown here carries no state fixed effects; "
            "adding them moves it toward a third.")
RUNNING_ORDER.append(("Crop diversity against irrigation", "rises then falls away"))

# ===================================================================== 7
s = plain_slide("Irrigation source and what a district grows",
                "Coefficients against groundwater, with state fixed effects")
chrome(s)
place(s, "f7_irrigation_source", TWOUP_X[0], FIG_TOP, TWOUP_W, FIG_H)
note(s, TWOUP_X[1], FIG_TOP + 0.15, TWOUP_W, 1.60, "Canal against groundwater",
     "Canal dependence shortens the crop list by about two and a half crops, leaves the "
     "effective count alone, and goes with a larger pulse share.", "s7 upper")
note(s, TWOUP_X[1], FIG_TOP + 1.95, TWOUP_W, 1.60, "Surface water against groundwater",
     "Surface-water dependence is where cereal concentration sits. Effective diversity "
     "falls and cereal share rises by thirty points across the range.", "s7 lower")
highlight_strip(s, "Canal dependence goes with about two and a half fewer crops than groundwater",
                line("Surface water goes with a cereal share thirty points higher across "
                     "its range. How much a district irrigates does not separate these."))
footnote(s, "Dominant irrigation source per village. Groundwater is the omitted category "
            "and covers about half of villages.")
RUNNING_ORDER.append(("Irrigation source and what a district grows", "two and a half fewer crops"))

# ===================================================================== 8
s = plain_slide("Rural facilities and where they sit together",
                "How often each facility turns up in the same districts as the others")
chrome(s)
place(s, "f8_facility_correlation", TWOUP_X[0], FIG_TOP, TWOUP_W, FIG_H, centre=True)
place(s, "f9_map_haat", TWOUP_X[1], FIG_TOP, TWOUP_W, FIG_H, centre=True)
highlight_strip(s, "Nine of the ten rural facilities move together across districts",
                line("The weekly haat runs the other way, falling where mandis, regular "
                     "markets and non-farm activity are dense."))
footnote(s, "Each cell is the correlation across districts between the share of villages "
            "carrying one facility and the share carrying the other. Blue is positive, "
            "orange negative.")
RUNNING_ORDER.append(("Rural facilities and where they sit together", "nine move together"))

# ===================================================================== 9
s = plain_slide("Rural facilities against cropping",
                "One facility at a time, holding wealth, connectivity and state constant")
chrome(s)
place(s, "f10_market_solo", STRIP_X, FIG_TOP, STRIP_W, FIG_H, centre=True)
highlight_strip(s, "Haat coverage goes with about one more crop grown per standard deviation",
                line("Fertiliser shops, cold storage and regular markets each go with a "
                     "smaller share of land under pulses."))
footnote(s, "Each row is a separate estimate carrying night-time lights, non-farm "
            "establishment density, connectivity, irrigation, holding size and state fixed "
            "effects. Errors cluster on the pre-2011 parent district.")
RUNNING_ORDER.append(("Rural facilities against cropping", "one more crop per standard deviation"))

# ==================================================================== 10
s = plain_slide("Haat coverage and producer organisations, by index",
                "What each one moves, and what it leaves alone")
chrome(s)
note(s, TWOUP_X[0], FIG_TOP + 0.05, TWOUP_W, 1.72, "Weekly haat",
     "Raises the count of crops grown by about one crop per standard deviation of "
     "coverage, against a mean of twenty. The effective count does not move and evenness "
     "falls slightly.", "s10 a")
note(s, TWOUP_X[1], FIG_TOP + 0.05, TWOUP_W, 1.72, "Producer organisation",
     "Raises the effective count without raising the plain count, and cuts cereal share. "
     "Land already in cultivation is spread more evenly rather than new crops appearing.",
     "s10 b")
note(s, TWOUP_X[0], FIG_TOP + 1.92, TWOUP_W, 1.72, "Input supply",
     "Fertiliser shops, cold storage and regular markets each go with a smaller share of "
     "land under pulses, at one percent significance and surviving correction across all "
     "ten crop categories.", "s10 c")
note(s, TWOUP_X[1], FIG_TOP + 1.92, TWOUP_W, 1.72, "Why the distinction holds",
     "A single blended diversity score reports both institutions as raising diversity. "
     "Separating the count from the balance shows they act on different margins.",
     "s10 d")
highlight_strip(s, "Haat coverage moves the count of crops and leaves the effective number flat",
                line("Producer organisations do the reverse, so a single blended score "
                     "reports both as more diverse and loses the difference."))
RUNNING_ORDER.append(("Haat coverage and producer organisations, by index", "different margins"))

# ==================================================================== 11
s = plain_slide("Two expectations that did not hold",
                "How sharply diversity falls after the turn, where infrastructure is "
                "thin and where it is dense")
chrome(s)
place(s, "f11_interaction", TWOUP_X[0], FIG_TOP, TWOUP_W, FIG_H)
note(s, TWOUP_X[1], FIG_TOP + 0.15, TWOUP_W, 1.60, "Mandis and cereals",
     "Denser mandi coverage was expected to raise cereal share. Within a state and at a "
     "given level of irrigation it goes with a smaller cereal share.", "s11 upper")
note(s, TWOUP_X[1], FIG_TOP + 1.95, TWOUP_W, 1.60, "Procurement and the downslope",
     "The fall after the turn was expected to be steeper where markets and input supply "
     "are dense. It is gentler there on all five measures.", "s11 lower")
highlight_strip(s, "Assured procurement does not explain the fall in diversity at high irrigation",
                line("Each was written down before the estimates were made, and each "
                     "runs the opposite way to what was expected."))
RUNNING_ORDER.append(("Two expectations that did not hold", "procurement does not explain the downslope"))

# ==================================================================== 12
s = plain_slide("What this establishes",
                "Findings, and the limits of a district cross-section")
chrome(s)
note(s, TWOUP_X[0], FIG_TOP + 0.05, TWOUP_W, 1.72, "Established",
     "The effective number of crops rises with irrigation and then falls, and that shape "
     "holds across every version of the estimate. Which source the water comes from "
     "carries information that how much a district irrigates does not.", "s12 a")
note(s, TWOUP_X[1], FIG_TOP + 0.05, TWOUP_W, 1.72, "Not established",
     "None of this is identified. Irrigation systems and market infrastructure sit where "
     "terrain, aquifers and existing agriculture already favoured them.", "s12 b")
note(s, TWOUP_X[0], FIG_TOP + 1.92, TWOUP_W, 1.72, "Where the data is thin",
     "Rural structure is a single 2019-20 cross-section against cropping averaged over two "
     "decades, which suits slow-moving infrastructure. Holding size is inferred rather "
     "than measured and is null throughout.", "s12 c")
note(s, TWOUP_X[1], FIG_TOP + 1.92, TWOUP_W, 1.72, "What would sharpen it",
     "District procurement volumes, which are not published at that granularity, and an "
     "agricultural census measure of operational holdings in place of the inferred one.",
     "s12 d")
highlight_strip(s, "The cropping pattern is arranged around water type and market type",
                line("Both carry more than the quantities usually used to stand in for "
                     "them."))
RUNNING_ORDER.append(("What this establishes", "water type and market type"))


os.makedirs(str(OUT.parent), exist_ok=True)
prs.save(str(OUT))
print("WROTE {}".format(OUT))
print("\nRunning order, {} slides".format(len(RUNNING_ORDER)))
for i, (t, h) in enumerate(RUNNING_ORDER, 1):
    print("  {:2d}  {:50s} {}".format(i, t[:50], h))
