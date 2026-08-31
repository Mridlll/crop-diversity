"""
86_build_map_pages.py

Writes the four interactive map pages, rebuilt on the corrected indices and in
the same dark house style as the rest of the site.

They replace an earlier set drawn from a first pass at the indices, which
counted the crops a district recorded across the whole period rather than the
crops it grows in a year.

Outputs: docs/{diversity,calorie-diversity,food-nonfood,timeline}.html
"""
import os

DOCS = r"D:/crop-diversity/docs"

HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<meta name="description" content="{desc}">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Crimson+Pro:ital,wght@0,400;0,600;1,400&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<link rel="stylesheet" href="css/analysis.css">
</head>
<body>

<nav class="nav" aria-label="Primary">
  <div class="nav__inner">
    <a class="nav__brand" href="index.html">Crop Diversity &middot; India</a>
    <div class="nav__links">
      <a href="index.html">Overview</a>
      <a href="irrigation.html">Irrigation</a>
      <a href="markets.html">Markets</a>
      <a href="data.html">Data &amp; methods</a>
      <a href="diversity.html"{navcur}>Maps</a>
    </div>
  </div>
</nav>

<header class="mast">
  <div class="prose">
    <p class="mast__label">{kicker}</p>
    <h1 class="mast__title">{h1}</h1>
    <div class="mast__rule"></div>
    <p class="mast__sub">{sub}</p>
  </div>
</header>

<section class="section">
  <div class="prose fade">
{lede}
  </div>

  <div class="figure figure--wide fade">
    <div class="figure__frame">
      <p class="figure__label">{figlabel}</p>
      <p class="figure__title">{figtitle}</p>
      <p class="figure__sub">{figsub}</p>
{controls}
      <div id="themap" class="chart"></div>
      <p class="figure__note">{fignote}</p>
    </div>
  </div>

  <div class="prose prose--flow fade">
{body}
  </div>
</section>

<section class="section">
  <div class="prose fade">
    <p class="section__label">The other views</p>
    <h2 class="section__heading">Same districts, different question</h2>
    <ul style="list-style:none;margin-left:0">
      <li><a href="diversity.html">How varied the cropping is</a></li>
      <li><a href="calorie-diversity.html">How much food energy the land produces</a></li>
      <li><a href="food-nonfood.html">How much of the land feeds people</a></li>
      <li><a href="timeline.html">How it moved, year by year</a></li>
    </ul>
  </div>
</section>

<footer class="foot">
  <div class="foot__grid">
    <div>
      <h4>Sources</h4>
      <p style="font-size:.9rem">Crop area and production at district level from the ISB India
      Data Portal, originally published by the Ministry of Agriculture and Farmers Welfare.
      District characteristics from SHRUG 2.1.</p>
    </div>
    <div>
      <h4>Sections</h4>
      <ul>
        <li><a href="index.html">Overview</a></li>
        <li><a href="irrigation.html">Irrigation</a></li>
        <li><a href="markets.html">Markets</a></li>
        <li><a href="data.html">Data &amp; methods</a></li>
      </ul>
    </div>
    <div>
      <h4>Reference</h4>
      <ul><li><a href="https://github.com/mridlll/crop-diversity">Repository</a></li></ul>
    </div>
  </div>
</footer>

<script src="js/site.js"></script>
<script>
{script}
</script>
</body>
</html>
"""

TIP_CROPS = """
function crops(pr) {
  if (!pr.top) return '';
  return '<br>' + pr.top.map(function(c) {
    return c[0].toLowerCase() + ' ' + Math.round(c[1] * 100) + '%';
  }).join(', ');
}
"""

PAGES = {}

# ------------------------------------------------------------------ diversity
PAGES["diversity.html"] = dict(
    title="How Varied the Cropping Is",
    desc="Crop diversity by Indian district, on four measures, with the crop composition of every district.",
    navcur=' aria-current="page"',
    kicker="District map",
    h1="How varied the cropping is",
    sub="Four ways of counting, and what each one says about the same district.",
    figlabel="Interactive",
    figtitle="Crop diversity by district",
    figsub="Choose a measure. Hover a district for its crop composition.",
    controls='      <div class="legend" id="tabs"></div>',
    fignote="Grey districts carry no agricultural record, which is mostly urban territory. "
            "Values average each district's own years between 1997&ndash;98 and 2019&ndash;20.",
    lede="""    <p class="lead">A district that grows twenty crops and puts four fifths of its land
    under one of them is not a varied district, and a plain count will call it one anyway. These
    four measures pull apart what a single count runs together.</p>""",
    body="""    <h3>What the four measures do</h3>
    <p>The count of crops grown treats every crop alike, so a crop on a single hectare weighs as
    much as the staple. The effective number of crops asks how many equally sized crops would give
    the diversity actually observed, so a district's answer moves toward the crops that hold its
    land. The dominant-weighted count goes further still and is governed almost entirely by the
    largest crops.</p>
    <p>Evenness is the effective number divided by the plain count. It strips the count out and
    leaves only how evenly land is spread, which is why a district can grow many crops and still
    score low.</p>
    <h3>Reading the map</h3>
    <p>Switch between the count and the effective number and watch the eastern rice belt and the
    northwestern wheat belt darken and then fade. Those districts grow an ordinary number of crops.
    What separates them is how much land the leading crop takes.</p>
    <p>Karnataka, interior Andhra Pradesh, Madhya Pradesh and Rajasthan hold their colour across
    all four measures, which is what a genuinely varied cropping pattern looks like.</p>""",
    script=TIP_CROPS + """
const LAYERS = [
  {name:'effective number of crops', field:'D1', fmt:v=>v.toFixed(1),
   label:'effective number of crops'},
  {name:'crops grown', field:'D0', fmt:v=>v.toFixed(0), label:'crops grown in an average year'},
  {name:'dominant-weighted', field:'D2', fmt:v=>v.toFixed(1),
   label:'effective crops, weighted to the largest'},
  {name:'evenness', field:'E', fmt:v=>v.toFixed(2), label:'how evenly land is spread'}
];
function tip(pr, l) {
  if (pr.D1 == null) return '<strong>'+pr.n+'</strong><br>'+pr.s+'<br>no agricultural record';
  return '<strong>'+pr.n+'</strong><br>'+pr.s+
    '<br>'+pr.D0.toFixed(1)+' crops grown'+
    '<br>'+pr.D1.toFixed(1)+' effective crops'+
    '<br>evenness '+pr.E.toFixed(2)+
    '<br>observed '+pr.yrs+' years'+crops(pr);
}
function drawAll(){
  layerTabs('#tabs', LAYERS, l => mapLayer('#themap', {
    field:l.field, fmt:l.fmt, label:l.label, height:620, tipFn:pr=>tip(pr,l)
  }));
}
window.drawAll = drawAll; drawAll();
""")

# ------------------------------------------------------------------- calorie
PAGES["calorie-diversity.html"] = dict(
    title="Food Energy and Crop Diversity",
    desc="Food energy produced per hectare against how varied a district's cropping is.",
    navcur="",
    kicker="District map",
    h1="Food energy against variety",
    sub="Whether growing a wider range of crops costs a district food energy.",
    figlabel="Interactive",
    figtitle="Food energy per hectare, and the four groups",
    figsub="Choose a view. Hover a district for its crops and its energy output.",
    controls='      <div class="legend" id="tabs"></div>',
    fignote="Energy is the food energy in a district's own harvest, per hectare of its cropped "
            "area, averaged across its years. Coconut is counted as edible meat rather than as "
            "whole nuts, which is how it is reported.",
    lede="""    <p class="lead">The worry about diversification is that it costs output. If growing a
    wider range of crops meant producing less food energy from the same land, the diverse districts
    would be the energy-poor ones. They are not.</p>""",
    body="""    <h3>What is being measured</h3>
    <p>Each crop's harvest is converted to food energy using standard energy values per hundred
    grams, summed across the district's crops, and divided by the district's cropped area. The
    result is the food energy a hectare of that district produces in a year. It is what the land
    yields, not what anyone eats, and it says nothing about who gets it.</p>
    <p>The median district produces about 5.9 million kilocalories a hectare. The scale runs across
    two orders of magnitude, so the map uses a logarithmic scale and a linear one would show almost
    nothing.</p>
    <h3>The four groups</h3>
    <p>Splitting districts at the median on both measures gives four groups of roughly equal size,
    which is the first thing worth noticing. If variety came at the cost of energy, the diverse and
    energy-rich group would be close to empty. It holds about the same number of districts as the
    others.</p>
    <p>The concentrated and energy-rich group is the irrigated cereal belt, producing a great deal
    of energy from a narrow rotation. The diverse and energy-poor group is largely rainfed and
    upland. Neither is a failure, and they face different problems.</p>""",
    script=TIP_CROPS + """
function fmtK(v){ return v>=1e6 ? (v/1e6).toFixed(0)+'m' : (v/1e3).toFixed(0)+'k'; }
function tip(pr){
  if (pr.kcal == null) return '<strong>'+pr.n+'</strong><br>'+pr.s+'<br>no agricultural record';
  return '<strong>'+pr.n+'</strong><br>'+pr.s+
    '<br>'+(pr.kcal/1e6).toFixed(1)+' million kcal per hectare'+
    '<br>'+pr.D1.toFixed(1)+' effective crops'+
    '<br>'+pr.quad+crops(pr);
}
const LAYERS = [
  {name:'food energy per hectare', kind:'value'},
  {name:'the four groups', kind:'category'}
];
function drawAll(){
  layerTabs('#tabs', LAYERS, l => {
    if (l.kind === 'category') {
      mapLayer('#themap', {field:'quad', kind:'category', height:620, tipFn:tip});
    } else {
      mapLayer('#themap', {field:'kcal', log:true, fmt:fmtK, height:620,
                           label:'kilocalories per hectare, log scale', tipFn:tip});
    }
  });
}
window.drawAll = drawAll; drawAll();
""")

# ---------------------------------------------------------------- food/nonfood
PAGES["food-nonfood.html"] = dict(
    title="Land That Feeds People",
    desc="The share of a district's cropped area under crops that feed people.",
    navcur="",
    kicker="District map",
    h1="Land that feeds people",
    sub="How much of a district's cropped area grows food, and how much grows everything else.",
    figlabel="Interactive",
    figtitle="Share of cropped area under food crops",
    figsub="Hover a district for its composition.",
    controls="",
    fignote="Food crops are cereals, pulses, oilseeds, fruit, vegetables, sugar and spices. The "
            "remainder is fibre, fodder, and drugs and narcotics.",
    lede="""    <p class="lead">Most Indian districts grow food on most of their land. The exceptions
    are concentrated, and they are concentrated in cotton.</p>""",
    body="""    <h3>What counts as food</h3>
    <p>A crop counts as feeding people if its category does. Cereals, pulses, oilseeds, fruit,
    vegetables, sugar and spices all do. Fibre crops, fodder, and drugs and narcotics do not, which
    is what the remaining share is made of.</p>
    <p>The line is about the crop and not about where the harvest ends up. Maize grown for poultry
    feed counts as food here, and a great deal of it is not eaten by people.</p>
    <h3>Where the land is not growing food</h3>
    <p>The cotton belt across Gujarat, Maharashtra and Telangana is where the food share falls
    hardest. Those are also districts with a low effective number of crops, because a cotton
    district tends to be a cotton district first and something else second.</p>
    <p>Reading this beside the diversity map is the useful comparison. A district can score low on
    food share and high on variety, and several in the northeast do.</p>""",
    script=TIP_CROPS + """
function tip(pr){
  if (pr.food == null) return '<strong>'+pr.n+'</strong><br>'+pr.s+'<br>no agricultural record';
  return '<strong>'+pr.n+'</strong><br>'+pr.s+
    '<br>'+(pr.food*100).toFixed(0)+'% of land under food crops'+
    '<br>'+pr.D1.toFixed(1)+' effective crops'+crops(pr);
}
function drawAll(){
  mapLayer('#themap', {field:'food', fmt:v=>(v*100).toFixed(0)+'%', height:620,
                       label:'share of cropped area under food crops', tipFn:tip});
}
window.drawAll = drawAll; drawAll();
""")

# ------------------------------------------------------------------ timeline
PAGES["timeline.html"] = dict(
    title="Crop Diversity Year by Year",
    desc="District crop diversity animated across 1997-98 to 2019-20.",
    navcur="",
    kicker="District map",
    h1="Year by year",
    sub="Twenty-three years of district cropping, one year at a time.",
    figlabel="Interactive",
    figtitle="Effective number of crops, by year",
    figsub="Drag the slider or press play. Districts with no record that year are grey.",
    controls="""      <div class="legend" style="align-items:center;gap:1rem">
        <button id="play" style="background:none;border:1px solid rgba(255,255,255,.14);
          color:inherit;font:inherit;font-size:.8rem;cursor:pointer;padding:.25rem .8rem;
          border-radius:2px">Play</button>
        <input id="yr" type="range" min="0" max="22" value="22" step="1"
          style="flex:1;min-width:200px;accent-color:#0E9BB5">
        <span id="yrlab" style="font-family:var(--font-mono);font-size:.85rem;
          min-width:5.5em"></span>
      </div>""",
    fignote="A district enters the record when it is formed, so the number of districts shown "
            "climbs across the period. Comparisons across years are made on the districts "
            "reporting in every year.",
    lede="""    <p class="lead">The national picture barely moves across two decades. Individual
    districts move a great deal, and rather more of them lost variety than gained it.</p>""",
    body="""    <h3>What to watch</h3>
    <p>The colour is the effective number of crops, on the same scale in every frame, so a district
    darkening means it genuinely spread its land wider that year rather than the scale shifting
    underneath it.</p>
    <p>The early frames carry fewer districts because districts enter the record as they are
    created. That is why the aggregate figures quoted elsewhere use only the districts reporting in
    every year: comparing 1997 against 2019 on all available districts would measure administrative
    reorganisation as much as farming.</p>
    <h3>The direction of travel</h3>
    <p>Comparing each district's first seven years against its last seven, 264 districts ended with
    a lower effective number of crops and 165 with a higher one. Weighting districts by their
    cropped area, the national series is flat, because the losses fall in districts holding less of
    the country's land.</p>""",
    script="""
let TL = null, HANDLE = null, LO = 0, HI = 1;
async function drawAll(){
  if (!TL) TL = await (await fetch('data/timeline.json')).json();
  const geo = await loadGeo();
  const norm = s => s.toLowerCase().replace(/[^a-z0-9 |]/g,' ').replace(/\s+/g,' ').trim();
  const byNorm = {};
  Object.keys(TL.districts).forEach(k => { byNorm[norm(k)] = TL.districts[k]; });
  let hit = 0;
  geo.features.forEach(f => {
    const t = byNorm[norm(f.properties.s + '|' + f.properties.n)];
    f.properties._tl = t || null;
    if (t) hit++;
  });

  const all = [];
  Object.keys(TL.districts).forEach(k =>
    TL.districts[k].D1.forEach(v => { if (v != null) all.push(v); }));
  all.sort((a,b)=>a-b);
  LO = all[Math.floor(all.length*0.02)]; HI = all[Math.floor(all.length*0.98)];

  const slider = document.getElementById('yr');
  const lab = document.getElementById('yrlab');
  const play = document.getElementById('play');
  slider.max = TL.years.length - 1;
  let i = TL.years.length - 1;

  function label(){
    const y = TL.years[i];
    lab.textContent = y + '\u2013' + String(y+1).slice(2);
  }
  function tip(pr){
    const y = TL.years[i];
    const t = pr._tl, v = t ? t.D1[i] : null, d0 = t ? t.D0[i] : null;
    if (v == null) return '<strong>'+pr.n+'</strong><br>'+pr.s+'<br>no record in '+y;
    return '<strong>'+pr.n+'</strong><br>'+pr.s+'<br>'+y+'\u2013'+String(y+1).slice(2)+
           '<br>'+v.toFixed(1)+' effective crops'+
           (d0 != null ? '<br>'+d0+' crops grown' : '');
  }

  // The geometry is drawn once. A frame only changes the fills, so stepping
  // through the years stays smooth instead of rebuilding 735 polygons.
  HANDLE = await mapLayer('#themap', {
    field:'_none', lo:LO, hi:HI, height:600, fmt:v=>v.toFixed(1),
    label:'effective number of crops', tipFn:tip
  });

  function paint(){
    label();
    HANDLE.recolour(pr => pr._tl ? pr._tl.D1[i] : null, LO, HI);
  }
  slider.value = i; paint();

  slider.oninput = () => { i = +slider.value; paint(); };
  let timer = null;
  play.onclick = () => {
    if (timer) { clearInterval(timer); timer = null; play.textContent = 'Play'; return; }
    if (i >= TL.years.length - 1) i = 0;
    play.textContent = 'Pause';
    timer = setInterval(() => {
      i = i + 1;
      if (i >= TL.years.length) {
        i = TL.years.length - 1;
        clearInterval(timer); timer = null; play.textContent = 'Play';
      }
      slider.value = i; paint();
    }, 620);
  };
  console.log('timeline: matched ' + hit + ' of ' + geo.features.length + ' districts');
}
window.drawAll = drawAll; drawAll();
""")

for name, cfg in PAGES.items():
    with open(os.path.join(DOCS, name), "w", encoding="utf-8") as f:
        f.write(HEAD.format(**cfg))
    print("WROTE {}".format(name))
