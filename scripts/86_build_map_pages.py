"""
86_build_map_pages.py

Writes the four interactive map pages in the dark house style of the rest of the
site. The template below is the live page skeleton; each page fills its fields.

Outputs: docs/{diversity,calorie-diversity,food-nonfood,timeline}.html
"""
import hashlib
import os

DOCS = r"D:/crop-diversity/docs"


def stamp(rel):
    """Content hash for a shared asset, so a deploy cannot serve a cached copy
    of the script or the stylesheet against markup that has moved on."""
    with open(os.path.join(DOCS, rel), "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()[:8]


VCSS, VJS = stamp("css/analysis.css"), stamp("js/site.js")

HEAD = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<meta name="description" content="{desc}">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Crimson+Pro:ital,wght@0,400;0,600;1,400&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<link rel="stylesheet" href="css/analysis.css?v={vcss}">
</head>
<body>

<nav class="nav" aria-label="Primary">
  <div class="nav__inner">
    <a class="nav__brand" href="index.html">Crop Diversity &middot; India</a>
    <div class="nav__links">
      <a href="index.html">Overview</a>
      <a href="measurement.html">Measurement</a>
      <a href="irrigation.html">Irrigation</a>
      <a href="markets.html">Markets</a>
      <a href="crop-groups.html">Crop groups</a>
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
    <p class="section__label">More maps</p>
    <h2 class="section__heading">The other district maps</h2>
    <ul style="list-style:none;margin-left:0">
      <li><a href="diversity.html">Crop diversity in each district, on four measures</a></li>
      <li><a href="calorie-diversity.html">Food energy each district&rsquo;s harvest produces per hectare</a></li>
      <li><a href="food-nonfood.html">Share of each district&rsquo;s cropped area under food crops</a></li>
      <li><a href="timeline.html">Effective number of crops in each district, year by year</a></li>
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
        <li><a href="crop-groups.html">Crop groups</a></li>
        <li><a href="data.html">Data &amp; methods</a></li>
      </ul>
    </div>
    <div>
      <h4>Reference</h4>
      <ul><li><a href="https://github.com/mridlll/crop-diversity">Repository</a></li></ul>
    </div>
  </div>
</footer>

<script src="js/site.js?v={vjs}"></script>
<script>
{script}
</script>
</body>
</html>
"""

PAGES = {}

PAGES['diversity.html'] = dict(
    title='Crop Diversity in Each District, on Four Measures',
    desc='Crop diversity by Indian district, on four measures, with the crop composition of every district.',
    navcur=' aria-current="page"',
    kicker='District map',
    h1='Crop diversity in each district, on four measures',
    sub='Although Chhattisgarh and Assam record long crop lists, rice holds most of their land.',
    lede=r"""    <p class="lead">The rice districts of the east record long crop lists, yet rice holds so much
    of their land that they fall to the bottom once each crop is weighted by its area.</p>""",
    figlabel='Interactive',
    figtitle='Crop diversity by district',
    figsub='Choose a measure. Brighter means more varied cropping. Hover over a district for its largest crops.',
    controls='      <div class="legend" id="tabs"></div>',
    fignote="Unshaded districts have no agricultural record, and most of them are urban. Each value averages a district's years from 1998&ndash;99 to 2019&ndash;20.",
    body=r"""    <h3>Where the measures disagree</h3>
    <p>The count of crops gives a crop on one hectare the same weight as the staple, whereas the
    <a href="measurement.html#s1">effective number of crops</a> and the number of main crops weigh
    each crop by the land it covers. Chhattisgarh and Assam grow many crops, yet rice takes so much
    of their land that they drop near the bottom on the effective number.</p>
    <p>Karnataka, interior Andhra Pradesh, Madhya Pradesh and Rajasthan stay near the top on the
    count of crops, the effective number of crops and the number of main crops, although their
    evenness, the effective number divided by the count, is no better than middling. A district
    can grow many crops and still spread its land unevenly across them.</p>""",
    script=r"""
function crops(pr) {
  if (!pr.top) return '';
  return '<br>' + pr.top.map(function(c) {
    return c[0].toLowerCase() + ' ' + Math.round(c[1] * 100) + '%';
  }).join(', ');
}

const LAYERS = [
  {name:'effective number of crops', field:'D1', fmt:v=>v.toFixed(1),
   label:'effective number of crops'},
  {name:'count of crops', field:'D0', fmt:v=>v.toFixed(0), label:'count of crops'},
  {name:'number of main crops', field:'D2', fmt:v=>v.toFixed(1),
   label:'number of main crops'},
  {name:'evenness', field:'E', fmt:v=>v.toFixed(2), label:'evenness'}
];
function tip(pr, l) {
  if (pr.D1 == null) return '<strong>'+pr.n+'</strong><br>'+pr.s+'<br>no agricultural record';
  return '<strong>'+pr.n+'</strong><br>'+pr.s+
    '<br>count of crops '+pr.D0.toFixed(1)+
    '<br>effective number of crops '+pr.D1.toFixed(1)+
    '<br>evenness '+pr.E.toFixed(2)+
    '<br>observed '+pr.yrs+' years'+crops(pr);
}
function drawAll(){
  layerTabs('#tabs', LAYERS, l => mapLayer('#themap', {
    field:l.field, fmt:l.fmt, label:l.label, height:620, tipFn:pr=>tip(pr,l)
  }));
}
window.drawAll = drawAll; drawAll();
""",
)

PAGES['calorie-diversity.html'] = dict(
    title='Food Energy per Hectare in Each District',
    desc="Food energy produced per hectare against how varied a district's cropping is.",
    navcur='',
    kicker='District map',
    h1='Food energy each district&rsquo;s harvest produces per hectare',
    sub='Set beside each district&rsquo;s effective number of crops.',
    lede=r"""    <p class="lead">Varied cropping and high food energy per hectare go together about as often as
    any other pairing. Split at the median on food energy and again on the effective number of
    crops, each of the four groups holds roughly a quarter of districts.</p>""",
    figlabel='Interactive',
    figtitle='Food energy per hectare, and the four groups',
    figsub='Choose a view. Brighter means more food energy per hectare, on a logarithmic scale. Hover over a district for its crops.',
    controls='      <div class="legend" id="tabs"></div>',
    fignote="Energy is the food energy in a district's own harvest, per hectare of its cropped area, averaged across its years. Coconut is reported in nuts and is converted using the weight of its edible meat.",
    body=r"""    <h3>What is being measured</h3>
    <p>Each crop's harvest is converted to food energy using standard energy values per hundred
    grams. These are summed across the district's crops and divided by its cropped area, which
    gives the food energy one hectare in that district produces in a year. It describes the
    harvest and leaves out what happens to it afterwards, including who eats it.</p>
    <p>The median district produces about 5.9 million kilocalories a hectare. Values across
    districts span two orders of magnitude.</p>
    <h3>The four groups</h3>
    <p>The concentrated, energy-rich districts form the irrigated cereal belt, which gets a lot of
    energy from a narrow rotation, while the diverse but energy-poor districts are mostly rainfed
    and upland.</p>""",
    script=r"""
function crops(pr) {
  if (!pr.top) return '';
  return '<br>' + pr.top.map(function(c) {
    return c[0].toLowerCase() + ' ' + Math.round(c[1] * 100) + '%';
  }).join(', ');
}

function fmtK(v){ return v>=1e6 ? (v/1e6).toFixed(0)+'m' : (v/1e3).toFixed(0)+'k'; }
function tip(pr){
  if (pr.kcal == null) return '<strong>'+pr.n+'</strong><br>'+pr.s+'<br>no agricultural record';
  return '<strong>'+pr.n+'</strong><br>'+pr.s+
    '<br>'+(pr.kcal/1e6).toFixed(1)+' million kcal per hectare'+
    '<br>effective number of crops '+pr.D1.toFixed(1)+
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
""",
)

PAGES['food-nonfood.html'] = dict(
    title='Land That Feeds People',
    desc="The share of a district's cropped area under crops that feed people.",
    navcur='',
    kicker='District map',
    h1='Share of each district&rsquo;s cropped area under food crops',
    sub='The share of each district&rsquo;s cropped land under crops that people eat.',
    lede=r"""    <p class="lead">In five of every six districts, food crops cover at least 90 percent of the
    cropped area. The districts with the lowest shares are mostly cotton districts in Telangana,
    Gujarat and Maharashtra.</p>""",
    figlabel='Interactive',
    figtitle='Share of cropped area under food crops',
    figsub='Brighter means a larger share under food crops. Hover over a district for its largest crops.',
    controls='',
    fignote='Food crops are cereals, pulses, oilseeds, fruit, vegetables, sugar and spices. The remaining land is under fibre crops, fodder crops, and drugs and narcotics.',
    body=r"""    <h3>Where the land is not growing food</h3>
    <p>The food share is lowest in the cotton belt across Gujarat, Maharashtra and Telangana, and
    in the guar districts of western Rajasthan. Because cotton tends to share a district's land
    with several other crops, 42 of the 64 districts with less than 80 percent of their land under
    food crops have an effective number of crops above the median district.</p>
    <p>A crop counts as food by its category, whatever later happens to the harvest, so maize
    counts as food even when it is grown for poultry feed.</p>""",
    script=r"""
function crops(pr) {
  if (!pr.top) return '';
  return '<br>' + pr.top.map(function(c) {
    return c[0].toLowerCase() + ' ' + Math.round(c[1] * 100) + '%';
  }).join(', ');
}

function tip(pr){
  if (pr.food == null) return '<strong>'+pr.n+'</strong><br>'+pr.s+'<br>no agricultural record';
  return '<strong>'+pr.n+'</strong><br>'+pr.s+
    '<br>'+(pr.food*100).toFixed(0)+'% of land under food crops'+
    '<br>effective number of crops '+pr.D1.toFixed(1)+crops(pr);
}
function drawAll(){
  mapLayer('#themap', {field:'food', fmt:v=>(v*100).toFixed(0)+'%', height:620,
                       label:'share under food crops', tipFn:tip});
}
window.drawAll = drawAll; drawAll();
""",
)

PAGES['timeline.html'] = dict(
    title='Crop Diversity Year by Year',
    desc='District crop diversity animated across 1998-99 to 2019-20.',
    navcur='',
    kicker='District map',
    h1='Effective number of crops in each district, year by year',
    sub='Districts move from year to year while the national spread of land across crops stays flat.',
    lede=r"""    <p class="lead">Weighted by cropped area, the national effective number of crops stays flat
    from 1998&ndash;99 to 2019&ndash;20. District by district, 297 ended the period with a lower
    effective number of crops and 203 with a higher one, although the median change is only about
    a sixth of a crop.</p>""",
    figlabel='Interactive',
    figtitle='Effective number of crops, by year',
    figsub='Drag the slider or press Play. Brighter means a higher effective number of crops, on the same scale every year.',
    controls=r"""      <div class="legend" style="align-items:center;gap:1rem">
        <button id="play" style="background:none;border:1px solid rgba(255,255,255,.14);
          color:inherit;font:inherit;font-size:.8rem;cursor:pointer;padding:.25rem .8rem;
          border-radius:2px">Play</button>
        <input id="yr" type="range" min="0" max="21" value="21" step="1"
          style="flex:1;min-width:200px;accent-color:#0E9BB5">
        <span id="yrlab" style="font-family:var(--font-mono);font-size:.85rem;
          min-width:5.5em"></span>
      </div>""",
    fignote='Districts with no record that year are unshaded. The record starts in 1998&ndash;99 because the 1997&ndash;98 returns are broken, with whole seasons missing in some states.',
    body=r"""    <h3>Why the early years show fewer districts</h3>
    <p>Because districts enter the record when they are created, the early years show fewer of
    them. National figures use only the districts that report in every year, which keeps new
    districts from passing for change on the land.</p>
    <h3>Which way districts moved</h3>
    <p>Between 1998&ndash;2004 and 2013&ndash;2019 the median district lost 0.17 of an effective crop.
    With districts weighted by their cropped land the fall is 0.20, which is not statistically
    significant once each state's swings in reporting are allowed for.</p>""",
    script=r"""
let TL = null, HANDLE = null, LO = 0, HI = 1;
async function drawAll(){
  if (!TL) TL = await (await fetch('data/timeline.json')).json();
  const geo = await loadGeo();
  const norm = s => s.toLowerCase().replace(/[^a-z0-9 |]/g,' ').replace(/\s+/g,' ').trim();
  const byNorm = {};
  Object.keys(TL.districts).forEach(k => { byNorm[norm(k)] = TL.districts[k]; });
  let hit = 0;
  // Join on the analysis key the build resolved for each polygon (crosswalk and
  // spelling matches included); the map's own spelling is only a fallback.
  geo.features.forEach(f => {
    const pr = f.properties;
    const t = (pr.k && TL.districts[pr.k]) || byNorm[norm(pr.s + '|' + pr.n)];
    f.properties._tl = t || null;
    if (t) hit++;
  });

  // 1997-98 returns are broken (whole seasons missing), so the map starts in 1998-99
  const Y0 = Math.max(0, TL.years.indexOf(1998));
  const all = [];
  Object.keys(TL.districts).forEach(k =>
    TL.districts[k].D1.forEach((v, j) => { if (v != null && j >= Y0) all.push(v); }));
  all.sort((a,b)=>a-b);
  LO = all[Math.floor(all.length*0.02)]; HI = all[Math.floor(all.length*0.98)];

  const slider = document.getElementById('yr');
  const lab = document.getElementById('yrlab');
  const play = document.getElementById('play');
  slider.min = Y0;
  slider.max = TL.years.length - 1;
  let i = TL.years.length - 1;

  function label(){
    const y = TL.years[i];
    lab.textContent = y + '–' + String(y+1).slice(2);
  }
  function tip(pr){
    const y = TL.years[i];
    const t = pr._tl, v = t ? t.D1[i] : null, d0 = t ? t.D0[i] : null;
    if (v == null) return '<strong>'+pr.n+'</strong><br>'+pr.s+'<br>no record in '+y;
    return '<strong>'+pr.n+'</strong><br>'+pr.s+'<br>'+y+'–'+String(y+1).slice(2)+
           '<br>effective number of crops '+v.toFixed(1)+
           (d0 != null ? '<br>count of crops '+d0 : '');
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
    if (i >= TL.years.length - 1) i = Y0;
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
""",
)

for name, cfg in PAGES.items():
    with open(os.path.join(DOCS, name), "w", encoding="utf-8") as f:
        f.write(HEAD.format(vcss=VCSS, vjs=VJS, **cfg))
    print("WROTE {}".format(name))
