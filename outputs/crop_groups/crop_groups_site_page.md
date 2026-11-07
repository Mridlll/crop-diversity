# Crop groups site page

Page `docs/crop-groups.html` (with `docs/js/crop-groups.js`, `docs/css/crop-groups.css`),
data `docs/data/site_crop_groups.json` built by `scripts/103_crop_groups_site_export.py`.
Added to the nav and footer section list of every page after Markets, and to the page
template in script 86. No APCNF, PMDS or RySS data.

Only findings that survived the four referee checks are on the page. Refuted claims are
left out: the Andhra Pradesh rice share fall of 7.3 points, Gujarat's pulses collapse, the
trend-adjusted NFSM-Pulses estimate, the OMM effect estimates, "rice area does not give
way", "wheat share rises because wheat is irrigated", the next-kharif millet share effect
and its dose-response, "no evidence of spread" and "maize leans toward spread".

## Scenes
1. 01 · Where each group is grown. Millets lost ground in three districts out of four,
   most heavily across the Deccan. Map, nine groups, change or 2017-19 share.
2. 02 · The national mix. Millets gave up more cropped area than any other group gained.
   Nine small panels, strict balanced panel (352 districts), changes from the 467-unit
   corrected endpoint.
3. 03 · Rice in Andhra Pradesh and Telangana. Telangana added about 0.4 million hectares
   of rice by 2016-18, almost all of it in the rabi season. Kharif and rabi rice area by
   year, 2019-20 outlined.
4. 04 · Missions and prices. Districts that joined the pulses mission in 2007 show no rise
   in pulse share against their 2002-07 level. NFSM event-time coefficients for pulses,
   rice and wheat; OMM and MSP in prose.
5. 05 · Rainfall shocks. After a monsoon 20 percent or more below normal, the following
   rabi is about 6 percent smaller. Dose-response by rainfall band for four outcomes.

Neighbour spread has no scene. Nothing about spread survived, so it is one paragraph in
the method block (same-year co-movement, the distance placebo, and that the design cannot
establish or rule out spread).

## Data sources inside the JSON
- Trends: `referee_fix_*` files (AP niger seed 2015-19 and Gujarat moth 1998-99 removed).
- AP/Telangana: `referee_fix_trends_apts_*`, both end windows.
- Policy: `policy_es_event_*` (no-trend event coefficients, which reproduce), the NFSM
  summary, and the identified 2002-09 window from `referee_policy_es_checks.csv`.
- Rainfall: bins re-estimated in script 103 on the referee's cleaned samples, reusing
  script 100's build and estimator. The deficient/excess dummies reproduce the referee's
  numbers: total rabi -0.063 (0.013), rabi rice -0.255 (0.063), low-irrigation rabi pulses
  -0.221 (0.055), next-kharif millet area after an excess monsoon -0.047 (0.018).
  The two-sided bins are new: total rabi area is -0.118 (0.027) at 40 percent or more
  short and -0.053 (0.013) at 20 to 40 percent short.

## Crop mapping (also in the page's method table)
- rice: Rice. wheat: Wheat. maize: Maize. cotton: Cotton(Lint). sugarcane: Sugarcane.
- millets: Bajra, Jowar, Ragi, Small Millets.
- pulses: Arhar/Tur, Cowpea(Lobia), Gram, Horse-Gram, Khesari, Masoor, Moong(Green Gram),
  Moth, Other Kharif Pulses, Other Rabi Pulses, Other Summer Pulses, Peas & Beans (Pulses), Urad.
- oilseeds: Castor Seed, Groundnut, Linseed, Niger Seed, Other Oilseeds, Rapeseed &Mustard,
  Safflower, Sesamum, Soyabean, Sunflower.
- other: Arecanut, Banana, Barley, Black Pepper, Cardamom, Cashewnut, Coconut, Coriander,
  Dry Chillies, Garlic, Ginger, Guar Seed, Jute, Mesta, Onion, Other Cereals, Potato,
  Sannhamp, Sweet Potato, Tapioca, Tobacco, Turmeric.

## Not done
- Done since this note was written: the page is committed (46bc376) and script 103 runs as a step of `run_all.py` after 102 [updated Nov 2026].
