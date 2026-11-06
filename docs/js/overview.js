/* ================================================================
   OVERVIEW PAGE (index.html): scene visuals.
   Reads data/site_overview.json, data/site_indices.json,
   data/site_overview_examples.json and data/districts.geojson.
   Depends on site.js for svgEl, lin, showTip, hideTip, loadJSON,
   loadGeo and rampCyan. Everything is wrapped so no name collides
   with site.js globals.

   Pattern for every scene (as on the Markets page):
     render()  draws the visual for the scene's current state
     prep()    sets the pre-entrance state (only before first play)
     play()    runs the entrance once the visual is in view
   drawAll() re-renders every scene on resize, so render() is idempotent.
   Under prefers-reduced-motion every tween resolves at once.
   ================================================================ */
(function () {
'use strict';

const RM = !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);

/* one colour per role, validated on the #0B0B0C ground in this order */
const CY = '#0E9BB5', VI = '#7C5CE0', GR = '#1FA35A', MG = '#C63FD8';
const NEU = '#B3B3B0', NEU2 = '#8A8A87', NEU3 = '#5E5E5C';
const OFF = 'rgba(255,255,255,0.075)';
const TX = '#EDEDEB', TX2 = '#B3B3B0', TX3 = '#7E7E7C', BG = '#0B0B0C', HALO = '#121214';
const GRID = 'rgba(255,255,255,0.075)', ZERO = 'rgba(255,255,255,0.28)';
const MONO = 'JetBrains Mono, monospace';

const $ = s => document.querySelector(s);
function E(tag, attrs, parent) { const e = svgEl(tag, attrs || {}); if (parent) parent.appendChild(e); return e; }
function T(parent, attrs, s) { const t = E('text', Object.assign({ 'font-family': MONO }, attrs), parent); t.textContent = s; return t; }
const lerp = (a, b, t) => a + (b - a) * t;
const easeOut = t => 1 - Math.pow(1 - t, 3);
const easeIO = t => t < .5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
const sgn = (v, d) => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(d);
const pct = (v, d) => (v * 100).toFixed(d == null ? 0 : d) + '%';
const tc = s => s.toLowerCase().replace(/\b([a-z])/g, c => c.toUpperCase()).replace(/\b(And|Of)\b/g, w => w.toLowerCase());

function svgFor(host, W, H) {
    host.innerHTML = '';
    const s = E('svg', { viewBox: `0 0 ${W} ${H}`, width: '100%', height: H, class: 'chart' });
    host.appendChild(s);
    return s;
}
function niceTicks(a, b, n) {
    const span = b - a, raw = span / Math.max(1, n);
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 2.5, 5, 10].map(k => k * mag).find(s => s >= raw) || raw;
    const out = [];
    for (let v = Math.ceil(a / step) * step; v <= b + 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
}

/* ---------- motion ---------- */
function tween(sc, dur, fn, done) {
    if (RM || dur <= 0) { fn(1); if (done) done(); return; }
    const token = sc.run, t0 = performance.now();
    (function step(now) {
        if (sc.run !== token) return;
        const t = Math.min(1, (now - t0) / dur);
        fn(t);
        if (t < 1) requestAnimationFrame(step); else if (done) done();
    })(t0);
}
function cancel(sc) { sc.run++; sc.timers.forEach(clearTimeout); sc.timers = []; }
const scene = o => Object.assign({ run: 0, timers: [], played: false }, o);

function segs(host, items, onPick, cur) {
    if (!host) return;
    host.innerHTML = '';
    items.forEach(it => {
        const b = document.createElement('button');
        b.type = 'button'; b.className = 'seg';
        b.setAttribute('aria-pressed', it.key === cur ? 'true' : 'false');
        if (it.color) {
            const sw = document.createElement('span'); sw.className = 'seg__sw';
            sw.style.background = it.color; b.appendChild(sw);
        }
        b.appendChild(document.createTextNode(it.name));
        b.dataset.key = it.key;
        b.addEventListener('click', () => onPick(it.key));
        host.appendChild(b);
    });
}
function pressSeg(host, key) {
    if (!host) return;
    [...host.querySelectorAll('.seg')].forEach(b => b.setAttribute('aria-pressed', b.dataset.key === key ? 'true' : 'false'));
}
/* a tap on a phone fires click without mousemove; both open the card */
function tipOn(el, html) {
    const f = ev => showTip(html(), ev);
    el.addEventListener('mousemove', f);
    el.addEventListener('click', f);
    el.addEventListener('mouseleave', hideTip);
}
/* nearest point under the pointer, for dense dot charts */
function nearestTip(svg, pts, maxD, html, onHi) {
    const pick = ev => {
        const r = svg.getBoundingClientRect(), vb = svg.viewBox.baseVal;
        const k = vb.width / r.width;
        const x = (ev.clientX - r.left) * k, y = (ev.clientY - r.top) * k;
        let best = -1, bd = maxD * maxD;
        pts().forEach((p, i) => { const d = (p[0] - x) ** 2 + (p[1] - y) ** 2; if (d < bd) { bd = d; best = i; } });
        onHi(best);
        if (best < 0) { hideTip(); return; }
        showTip(html(best), ev);
    };
    svg.addEventListener('mousemove', pick);
    svg.addEventListener('click', pick);
    svg.addEventListener('mouseleave', () => { onHi(-1); hideTip(); });
}

let OV = null, IX = null, EX = null;
const SC = {};

/* ================================================================
   01 a district's land as 100 plots
   ================================================================ */
const PLOT_COL = [CY, VI, GR, MG];
SC[1] = scene({
    ex: 0,
    controls() {
        segs($('#v1-pick'), EX.examples.map((e, i) => ({ key: String(i), name: e.d })),
            k => { cancel(this); this.played = true; this.ex = +k; pressSeg($('#v1-pick'), k); this.paint(true); this.readout(); }, String(this.ex));
    },
    alloc(x) {
        /* largest remainder, so the squares add to exactly 100 */
        const raw = x.shares.map(s => s[1] * 100), fl = raw.map(Math.floor);
        let left = 100 - fl.reduce((a, b) => a + b, 0);
        raw.map((v, i) => [v - fl[i], i]).sort((a, b) => b[0] - a[0]).slice(0, left).forEach(r => { fl[r[1]]++; });
        const cells = [];
        fl.forEach((n, i) => { for (let k = 0; k < n; k++) cells.push(i); });
        return cells;
    },
    colOf(i) { return i < 4 ? PLOT_COL[i] : (i % 2 ? NEU2 : NEU3); },
    render() {
        const host = $('#v1'); if (!host) return;
        const W = Math.min(host.clientWidth || 340, 380), cell = W / 10;
        const svg = svgFor(host, W, W);
        this.sq = [];
        for (let i = 0; i < 100; i++) {
            const r = E('rect', { x: (i % 10) * cell + 1.5, y: Math.floor(i / 10) * cell + 1.5, width: cell - 3, height: cell - 3, rx: 2 }, svg);
            r.style.fill = OFF;
            tipOn(r, () => {
                const x = EX.examples[this.ex], c = this.cells[i];
                if (c == null) return '';
                return `<strong>${x.shares[c][0]}</strong><br>${pct(x.shares[c][1], 1)} of cropped land<br>${x.d}`;
            });
            this.sq.push(r);
        }
        this.paint(false);
        this.readout();
    },
    paint(anim) {
        const x = EX.examples[this.ex];
        this.cells = this.alloc(x);
        this.sq.forEach((r, i) => {
            const c = this.cells[i];
            r.style.transition = anim && !RM ? `fill 180ms ease ${i * 14}ms` : 'none';
            r.style.fill = c == null ? OFF : this.colOf(c);
        });
    },
    readout() {
        const r = $('#v1-read'); if (!r) return;
        const x = EX.examples[this.ex];
        const top = x.shares.slice(0, 4), rest = x.shares.slice(4);
        const restSh = rest.reduce((a, s) => a + s[1], 0);
        r.innerHTML = `<span class="readout__name">${x.d}<small>${x.s}</small></span>` +
            `<div class="rd-tbl"><span>count of crops</span><b>${x.D0.toFixed(1)}</b>` +
            `<span>effective number of crops</span><b>${x.D1.toFixed(1)}</b>` +
            `<span>evenness</span><b>${x.E.toFixed(2)}</b>` +
            `<span>years reported</span><b>${x.y}</b></div>` +
            `<ul class="rd-crops">` + top.map((s, i) => `<li><i style="background:${PLOT_COL[i]}"></i><span>${pct(s[1])}</span>${s[0]}</li>`).join('') + `</ul>` +
            (rest.length ? `<p class="rd-more">and ${rest.length} more crops on the remaining ${pct(restSh, restSh < .1 ? 1 : 0)} of the land. ` +
            `${x.shares.length} crops appear in at least one year, ${x.D0.toFixed(0)} in an average year.</p>` : '');
    },
    prep() { this.sq.forEach(r => { r.style.transition = 'none'; r.style.fill = OFF; }); },
    play() { this.paint(true); this.readout(); },
    reset() { this.prep(); void (this.sq[0] && this.sq[0].getBoundingClientRect()); },
});

/* ================================================================
   02 count of crops against the effective number, 725 districts
   ================================================================ */
const EXK = ['TRIPURA|SEPAHIJALA', 'CHHATTISGARH|RAIPUR', 'PUNJAB|LUDHIANA', 'KARNATAKA|BELAGAVI', 'KARNATAKA|CHAMARAJANAGARA'];
const EXLAB = { 'TRIPURA|SEPAHIJALA': ['Sipahijala', 'below'], 'CHHATTISGARH|RAIPUR': ['Raipur', 'below'],
                'PUNJAB|LUDHIANA': ['Ludhiana', 'below'], 'KARNATAKA|BELAGAVI': ['Belagavi', 'left'],
                'KARNATAKA|CHAMARAJANAGARA': ['Chamarajanagara', 'left'] };
SC[2] = scene({
    key: 'D1', u: 1, hi: -1,
    controls() {
        segs($('#v2-ctl'), [{ key: 'D1', name: 'effective number of crops' }, { key: 'D2', name: 'number of main crops' }],
            k => { if (k === this.key) return; cancel(this); this.played = true; const from = this.key; this.key = k; pressSeg($('#v2-ctl'), k); this.swap(from); }, this.key);
    },
    render() {
        const host = $('#v2'); if (!host) return;
        const pts = this.pts = IX.districts.filter(r => r.D0 != null && r.D1 != null);
        const W = host.clientWidth || 800, nar = W < 560;
        const m = { t: 16, r: nar ? 12 : 20, b: 44, l: nar ? 34 : 44 };
        const iw = W - m.l - m.r;
        const xmax = Math.max(...pts.map(p => p.D0)) * 1.03;
        const ih = Math.max(nar ? 210 : 230, Math.round(iw * 16.5 / xmax));
        const ymax = ih * xmax / iw; /* same crops-per-pixel on both axes */
        const H = ih + m.t + m.b;
        const svg = this.svg = svgFor(host, W, H);
        const X = this.X = lin(0, xmax, m.l, m.l + iw), Y = this.Y = lin(0, ymax, m.t + ih, m.t);
        niceTicks(0, ymax, 4).forEach(v => {
            E('line', { x1: m.l, x2: m.l + iw, y1: Y(v), y2: Y(v), stroke: GRID }, svg);
            T(svg, { x: m.l - 8, y: Y(v) + 4, 'text-anchor': 'end', fill: TX3, 'font-size': 10.5 }, v.toFixed(0));
        });
        niceTicks(0, xmax, nar ? 4 : 7).forEach(v => T(svg, { x: X(v), y: m.t + ih + 18, 'text-anchor': 'middle', fill: TX3, 'font-size': 10.5 }, v.toFixed(0)));
        E('line', { x1: m.l, x2: m.l + iw, y1: m.t + ih, y2: m.t + ih, stroke: ZERO }, svg);
        const dm = Math.min(ymax, xmax);
        E('line', { x1: X(0), y1: Y(0), x2: X(dm), y2: Y(dm), stroke: 'rgba(255,255,255,0.34)', 'stroke-width': 1.3, 'stroke-dasharray': '5 4' }, svg);
        T(svg, { x: X(dm) + 8, y: Y(dm) + 12, fill: TX2, 'font-size': 10.5 }, 'every crop on equal land');
        T(svg, { x: m.l + iw, y: H - 6, 'text-anchor': 'end', fill: TX3, 'font-size': 10.5 }, 'count of crops in an average year');
        this.yl = T(svg, { x: m.l + 4, y: m.t + 10, fill: TX3, 'font-size': 10.5 }, '');
        const gD = E('g', {}, svg), gX = E('g', {}, svg), gL = E('g', {}, svg);
        const rr = nar ? 2.3 : 2.7;
        this.dots = pts.map(p => {
            const ex = EXK.indexOf(p.k) >= 0;
            return E('circle', { cx: X(p.D0), r: ex ? rr + 1.6 : rr, fill: ex ? CY : NEU, 'fill-opacity': ex ? 1 : .32,
                                 stroke: ex ? BG : 'none', 'stroke-width': ex ? 1.5 : 0 }, ex ? gX : gD);
        });
        this.labs = pts.map(p => {
            const L = EXLAB[p.k]; if (!L) return null;
            return T(gL, { 'font-size': nar ? 10 : 11, fill: TX, stroke: HALO, 'stroke-width': 4, 'paint-order': 'stroke',
                           'text-anchor': L[1] === 'left' ? 'end' : 'middle' }, L[0]);
        });
        this.ring = E('circle', { r: 6, fill: 'none', stroke: TX, 'stroke-width': 1.4, visibility: 'hidden' }, svg);
        nearestTip(svg, () => this.pts.map((p, i) => [X(p.D0), this.yOf(i)]), 14, i => {
            const p = this.pts[i];
            return `<strong>${tc(p.d)}</strong><br>${tc(p.s)}<br>count of crops ${p.D0.toFixed(1)}<br>effective number of crops ${p.D1.toFixed(1)}<br>number of main crops ${p.D2.toFixed(1)}<br>evenness ${p.E.toFixed(2)}`;
        }, i => {
            if (i < 0) { this.ring.setAttribute('visibility', 'hidden'); return; }
            this.ring.setAttribute('cx', X(this.pts[i].D0)); this.ring.setAttribute('cy', this.yOf(i));
            this.ring.setAttribute('visibility', 'visible');
        });
        this.from = this.key; this.v = 1;
        this.place();
    },
    yOf(i) {
        const p = this.pts[i], a = p[this.from], b = p[this.key];
        return this.Y(lerp(0, lerp(a, b, easeIO(this.v)), easeOut(this.u)));
    },
    place() {
        if (!this.dots) return;
        this.yl.textContent = this.key === 'D1' ? 'effective number of crops' : 'number of main crops';
        this.dots.forEach((c, i) => c.setAttribute('cy', this.yOf(i).toFixed(1)));
        this.labs.forEach((t, i) => {
            if (!t) return;
            const p = this.pts[i], x = this.X(p.D0), y = this.yOf(i);
            const left = EXLAB[p.k][1] === 'left';
            t.setAttribute('x', (left ? x - 9 : x).toFixed(1));
            t.setAttribute('y', (left ? y + 4 : y + 16).toFixed(1));
            t.setAttribute('opacity', this.u < 1 ? Math.max(0, (this.u - .7) / .3).toFixed(2) : 1);
        });
    },
    swap(from) {
        this.from = from; this.v = 0;
        tween(this, 900, t => { this.v = t; this.place(); }, () => { this.from = this.key; this.v = 1; });
    },
    prep() { this.u = 0; this.place(); },
    play() { this.u = 0; this.place(); tween(this, 1200, t => { this.u = t; this.place(); }); },
});

/* ================================================================
   03 district map
   ================================================================ */
const NODATA = 'rgba(255,255,255,0.035)';
const RAMP0 = 0.14;
const ramp = t => rampCyan(RAMP0 + (1 - RAMP0) * t);
function projectGeo(geo, W, H, keyH) {
    let x0 = 180, x1 = -180, y0 = 90, y1 = -90;
    const walk = c => {
        if (typeof c[0] === 'number') { x0 = Math.min(x0, c[0]); x1 = Math.max(x1, c[0]); y0 = Math.min(y0, c[1]); y1 = Math.max(y1, c[1]); }
        else c.forEach(walk);
    };
    geo.features.forEach(f => walk(f.geometry.coordinates));
    const pad = 6;
    const kx = Math.cos((y0 + y1) / 2 * Math.PI / 180);
    const s = Math.min((W - pad * 2) / ((x1 - x0) * kx), (H - pad * 2 - keyH) / (y1 - y0));
    const ox = pad + ((W - pad * 2) - (x1 - x0) * kx * s) / 2, oy = pad + ((H - pad * 2 - keyH) - (y1 - y0) * s) / 2;
    const PX = v => ox + (v - x0) * kx * s, PY = v => oy + (y1 - v) * s;
    const ring = c => c.map((p, i) => (i ? 'L' : 'M') + PX(p[0]).toFixed(1) + ',' + PY(p[1]).toFixed(1)).join('') + 'Z';
    const path = (c, d) => d === 1 ? ring(c) : c.map(k => path(k, d - 1)).join('');
    return { ring, fpath: f => path(f.geometry.coordinates, f.geometry.type === 'Polygon' ? 2 : 3) };
}
function hatchDef(svg, id, stroke, w, gap) {
    const pat = E('pattern', { id, width: gap, height: gap, patternUnits: 'userSpaceOnUse', patternTransform: 'rotate(45)' }, E('defs', {}, svg));
    E('rect', { width: gap, height: gap, fill: 'rgba(255,255,255,0.03)' }, pat);
    E('line', { x1: 0, y1: 0, x2: 0, y2: gap, stroke, 'stroke-width': w }, pat);
    return `url(#${id})`;
}
const ML = [
    { key: 'D1', name: 'effective number of crops', fmt: v => v.toFixed(1) },
    { key: 'D0', name: 'count of crops', fmt: v => v.toFixed(0) },
    { key: 'E', name: 'evenness', fmt: v => v.toFixed(2) },
];
/* a polygon's values: from the geojson, or from the crop record joined by name */
function rec(pr) {
    if (pr.D1 != null) return pr;
    return EX.name_fix[pr.s + '|' + pr.n] || null;
}
SC[3] = scene({
    layer: 'D1', filled: 1,
    controls() {
        segs($('#v3-tabs'), ML.map(l => ({ key: l.key, name: l.name })),
            k => { this.layer = k; pressSeg($('#v3-tabs'), k); cancel(this); this.played = true; this.filled = 1; this.recolour(true); }, this.layer);
    },
    range(key) {
        this.rng = this.rng || {};
        if (!this.rng[key]) {
            const v = this.paths.map(p => p.r && p.r[key]).filter(x => x != null).sort((a, b) => a - b);
            this.rng[key] = [v[Math.floor(v.length * .02)], v[Math.floor(v.length * .98)]];
        }
        return this.rng[key];
    },
    async render() {
        const host = $('#v3'); if (!host) return;
        const geo = await loadGeo();
        const W = host.clientWidth || 900, nar = W < 560;
        const H = Math.round(Math.min(nar ? 460 : 640, W * 1.08));
        const svg = svgFor(host, W, H);
        this.nod = hatchDef(svg, 'ovh' + Math.random().toString(36).slice(2, 7), 'rgba(255,255,255,0.3)', 1.1, 5);
        const P = projectGeo(geo, W, H, 46);
        this.paths = [];
        const gD = E('g', {}, svg);
        geo.features.forEach(f => {
            const pr = f.properties, r = rec(pr);
            const el = E('path', { d: P.fpath(f), stroke: 'rgba(11,11,12,0.8)', 'stroke-width': 0.35 }, gD);
            el.style.fill = NODATA;
            tipOn(el, () => {
                if (!r) return `<strong>${pr.n}</strong><br>${pr.s}<br>no crop area in the record`;
                return `<strong>${pr.n}</strong><br>${pr.s}<br>count of crops ${r.D0.toFixed(1)}<br>effective number of crops ${r.D1.toFixed(1)}<br>evenness ${r.E.toFixed(2)}` +
                    (r.dom ? `<br>mostly ${r.dom.toLowerCase().replace('(lint)', '').replace('soyabean', 'soybean')} (${pct(r.domsh)} of land)` : '');
            });
            el.addEventListener('mouseenter', () => { el.setAttribute('stroke', TX); el.setAttribute('stroke-width', 1.1); });
            el.addEventListener('mouseleave', () => { el.setAttribute('stroke', 'rgba(11,11,12,0.8)'); el.setAttribute('stroke-width', 0.35); });
            this.paths.push({ el, r });
        });
        const gS = E('g', { 'pointer-events': 'none' }, svg);
        EX.state_rings.forEach(r => E('path', { d: P.ring(r), fill: 'none', stroke: 'rgba(237,237,235,0.42)', 'stroke-width': 0.7, 'stroke-linejoin': 'round' }, gS));
        const kw = Math.min(260, W * .55), kx0 = W - kw - 8, ky = H - 30;
        const gid = 'ovg' + Math.random().toString(36).slice(2, 7);
        const lg = E('linearGradient', { id: gid, x1: 0, x2: 1, y1: 0, y2: 0 }, E('defs', {}, svg));
        for (let i = 0; i <= 10; i++) E('stop', { offset: i * 10 + '%', 'stop-color': ramp(i / 10) }, lg);
        E('rect', { x: kx0, y: ky, width: kw, height: 9, fill: `url(#${gid})`, stroke: 'rgba(255,255,255,0.18)', 'stroke-width': .6 }, svg);
        const fm = { 'font-size': 10.5, fill: TX3 };
        this.kLo = T(svg, Object.assign({ x: kx0, y: ky - 6, 'text-anchor': 'start' }, fm), '');
        this.kHi = T(svg, Object.assign({ x: kx0 + kw, y: ky - 6, 'text-anchor': 'end' }, fm), '');
        this.kLab = T(svg, Object.assign({ x: kx0 + kw, y: ky + 23, 'text-anchor': 'end' }, fm), '');
        this.recolour(false);
    },
    fillOf(r, key) {
        const v = r ? r[key] : null; if (v == null) return this.nod;
        const g = this.range(key);
        return ramp(Math.max(0, Math.min(1, (v - g[0]) / (g[1] - g[0] || 1))));
    },
    recolour(anim) {
        if (!this.paths) return;
        const L = ML.find(l => l.key === this.layer), g = this.range(L.key);
        this.kLo.textContent = L.fmt(g[0]); this.kHi.textContent = L.fmt(g[1]);
        this.kLab.textContent = L.name;
        const lim = this.filled >= 1 ? Infinity : g[0] + (g[1] - g[0]) * this.filled;
        this.paths.forEach(p => {
            p.el.style.transition = anim && !RM ? 'fill .45s ease' : 'none';
            const v = p.r ? p.r[L.key] : null;
            p.el.style.fill = v == null ? this.nod : (v <= lim ? this.fillOf(p.r, L.key) : NODATA);
        });
    },
    prep() { this.filled = 0; this.recolour(false); },
    play() { this.filled = 0; this.recolour(false); tween(this, 1400, t => { this.filled = t; this.recolour(false); }); },
});

/* ================================================================
   04 states: count of crops and effective number, one row each
   ================================================================ */
const SORTS = [{ key: 'D1', name: 'sort by effective number of crops' }, { key: 'D0', name: 'sort by count of crops' }, { key: 'E', name: 'sort by evenness' }];
SC[4] = scene({
    sort: 'D1', u: 1,
    controls() {
        segs($('#v4-ctl'), SORTS, k => { if (k === this.sort) return; cancel(this); this.played = true; this.u = 1; this.sort = k; pressSeg($('#v4-ctl'), k); this.reorder(true); }, this.sort);
    },
    order() { return OV.states.slice().sort((a, b) => b[this.sort] - a[this.sort]).map(s => s.state_name); },
    render() {
        const host = $('#v4'); if (!host) return;
        const W = host.clientWidth || 800, nar = W < 560;
        const rowH = nar ? 31 : 21, lw = nar ? 0 : 150;
        const m = { t: 8, b: 34, l: nar ? 6 : lw + 6, r: nar ? 10 : 18 };
        const n = OV.states.length, H = m.t + n * rowH + m.b;
        const svg = svgFor(host, W, H);
        const X = this.X = lin(0, 32, m.l, W - m.r);
        this.m = m; this.rowH = rowH; this.nar = nar;
        [0, 5, 10, 15, 20, 25, 30].forEach(v => {
            E('line', { x1: X(v), x2: X(v), y1: m.t, y2: m.t + n * rowH, stroke: GRID }, svg);
            T(svg, { x: X(v), y: m.t + n * rowH + 16, 'text-anchor': 'middle', fill: TX3, 'font-size': 10.5 }, String(v));
        });
        T(svg, { x: W - m.r, y: H - 3, 'text-anchor': 'end', fill: TX3, 'font-size': 10.5 }, 'number of crops');
        const gR = E('g', {}, svg);
        this.rows = {};
        OV.states.forEach(s => {
            const g = E('g', {}, gR);
            const nm = tc(s.state_name), key = s.state_name === 'CHHATTISGARH' || s.state_name === 'PUNJAB';
            E('rect', { x: 0, y: -rowH / 2, width: W, height: rowH, fill: 'transparent' }, g);
            if (nar) T(g, { x: m.l, y: -rowH / 2 + 10, fill: key ? TX : TX2, 'font-size': 10 }, nm);
            else T(g, { x: lw - 4, y: 4, 'text-anchor': 'end', fill: key ? TX : TX2, 'font-size': 11 }, nm);
            const cy = nar ? 6 : 0;
            const ln = E('line', { y1: cy, y2: cy, stroke: 'rgba(255,255,255,0.3)', 'stroke-width': 1 }, g);
            const o = E('circle', { cy, r: 4.5, fill: BG, stroke: NEU, 'stroke-width': 1.6 }, g);
            const f = E('circle', { cy, r: 5, fill: CY, stroke: BG, 'stroke-width': 1.2 }, g);
            tipOn(g, () => `<strong>${nm}</strong><br>${s.n} districts<br>count of crops ${s.D0.toFixed(1)}<br>effective number of crops ${s.D1.toFixed(1)}<br>evenness ${s.E.toFixed(2)}`);
            this.rows[s.state_name] = { g, ln, o, f, s, y: 0 };
        });
        this.reorder(false);
        this.place();
    },
    yAt(i) { return this.m.t + i * this.rowH + this.rowH / 2; },
    reorder(anim) {
        const ord = this.order();
        const from = {}; Object.keys(this.rows).forEach(k => { from[k] = this.rows[k].y; });
        const to = {}; ord.forEach((k, i) => { to[k] = this.yAt(i); });
        const set = t => Object.keys(this.rows).forEach(k => {
            const r = this.rows[k]; r.y = anim ? lerp(from[k], to[k], easeIO(t)) : to[k];
            r.g.setAttribute('transform', `translate(0,${r.y.toFixed(1)})`);
        });
        if (anim) tween(this, 750, set); else set(1);
    },
    place() {
        const e = easeOut(this.u), X = this.X;
        Object.values(this.rows).forEach(r => {
            const a = X(r.s.D1 * e), b = X(r.s.D0 * e);
            r.ln.setAttribute('x1', a); r.ln.setAttribute('x2', b);
            r.o.setAttribute('cx', b); r.f.setAttribute('cx', a);
        });
    },
    prep() { this.u = 0; this.place(); },
    play() { this.u = 0; this.place(); tween(this, 1100, t => { this.u = t; this.place(); }); },
});

/* ================================================================
   05 the largest crop: one dot per district
   ================================================================ */
const CROPN = { 'Cotton(Lint)': 'cotton', 'Soyabean': 'soybean', 'Rapeseed &Mustard': 'mustard' };
SC[5] = scene({
    u: 1,
    groups() {
        const g = OV.dominant.map(d => ({ name: CROPN[d.dominant_crop] || d.dominant_crop.toLowerCase(), n: d.n, share: d.share }));
        const rest = OV.national.n - g.reduce((a, x) => a + x.n, 0);
        g.push({ name: 'other crops', n: rest, share: null });
        return g;
    },
    render() {
        const host = $('#v5'); if (!host) return;
        const W = host.clientWidth || 800, nar = W < 560;
        const lw = nar ? 78 : 112, rw = nar ? 62 : 84, sp = nar ? 6.4 : 7.6, r = nar ? 2.4 : 2.9;
        const cols = Math.floor((W - lw - rw - 8) / sp);
        const G = this.groups();
        let y = 26;
        const svg0 = []; /* rows first, to size the svg */
        G.forEach(g => { const lines = Math.ceil(g.n / cols), h = Math.max(20, lines * sp + 8); svg0.push({ g, y, h, lines }); y += h; });
        const H = y + 8;
        const svg = svgFor(host, W, H);
        T(svg, { x: W - 4, y: 12, 'text-anchor': 'end', fill: TX3, 'font-size': 10 }, nar ? 'districts · share' : 'districts · share of land');
        this.dots = [];
        svg0.forEach((row, gi) => {
            const g = row.g, col = gi === 0 ? CY : gi === 1 ? GR : NEU2;
            const grp = E('g', {}, svg);
            E('rect', { x: 0, y: row.y, width: W, height: row.h, fill: 'transparent' }, grp);
            T(grp, { x: lw - 10, y: row.y + 4 + Math.min(row.h, 20) / 2, 'text-anchor': 'end', fill: gi < 2 ? TX : TX2, 'font-size': nar ? 10 : 11 }, g.name);
            T(grp, { x: W - 4, y: row.y + 4 + Math.min(row.h, 20) / 2, 'text-anchor': 'end', fill: gi < 2 ? TX : TX2, 'font-size': nar ? 10 : 11 },
              g.n + (g.share != null ? ' · ' + pct(g.share) : ''));
            const top = row.y + (row.h - row.lines * sp) / 2 + sp / 2;
            for (let k = 0; k < g.n; k++) {
                const cx = lw + (k % cols) * sp + sp / 2, cy = top + Math.floor(k / cols) * sp;
                const c = E('circle', { cx, cy, r, fill: col }, grp);
                this.dots.push({ c, cy });
            }
            tipOn(grp, () => g.share != null
                ? `<strong>${g.name}</strong><br>largest crop in ${g.n} districts<br>takes ${pct(g.share)} of their cropped land on average`
                : `<strong>other crops</strong><br>largest crop in ${g.n} districts<br>no one of these crops leads in more than 6`);
        });
        this.place();
    },
    place() {
        const N = this.dots.length;
        this.dots.forEach((d, i) => {
            const t0 = i / N * .7, t = Math.max(0, Math.min(1, (this.u - t0) / .3)), e = easeOut(t);
            d.c.setAttribute('opacity', e.toFixed(2));
            d.c.setAttribute('cy', (d.cy - (1 - e) * 10).toFixed(1));
        });
    },
    prep() { this.u = 0; this.place(); },
    play() { this.u = 0; this.place(); tween(this, 1800, t => { this.u = t; this.place(); }); },
});

/* ================================================================
   06 what land is used for, by quarter of districts
   ================================================================ */
const CATS = [
    { key: 'cereals', name: 'cereals', col: CY, v: r => r.share_cereals },
    { key: 'oilseeds', name: 'oilseeds', col: VI, v: r => r.share_oilseeds },
    { key: 'pulses', name: 'pulses', col: GR, v: r => r.share_pulses },
    { key: 'fibre', name: 'fibre crops', col: MG, v: r => r.share_fiber_crops },
    { key: 'other', name: 'other', col: NEU3, v: r => 1 - r.share_cereals - r.share_oilseeds - r.share_pulses - r.share_fiber_crops },
];
const QN = ['lowest quarter', 'second quarter', 'third quarter', 'highest quarter'];
SC[6] = scene({
    sel: 'all', u: 1,
    controls() {
        segs($('#v6-ctl'), [{ key: 'all', name: 'all' }].concat(CATS.map(c => ({ key: c.key, name: c.name, color: c.col }))),
            k => { this.sel = k; pressSeg($('#v6-ctl'), k); this.dress(); }, this.sel);
    },
    render() {
        const host = $('#v6'); if (!host) return;
        const W = host.clientWidth || 800, nar = W < 560;
        const lw = nar ? 0 : 118, bh = 34, gap = nar ? 34 : 20;
        const m = { t: nar ? 18 : 8, l: lw, r: 8 };
        const rows = OV.cat_by_quartile;
        const H = m.t + rows.length * (bh + gap) + 18;
        const svg = svgFor(host, W, H);
        const X = lin(0, 1, m.l, W - m.r);
        [0, .25, .5, .75, 1].forEach(v => {
            E('line', { x1: X(v), x2: X(v), y1: m.t - 4, y2: H - 22, stroke: GRID }, svg);
            T(svg, { x: X(v), y: H - 6, 'text-anchor': v === 0 ? 'start' : v === 1 ? 'end' : 'middle', fill: TX3, 'font-size': 10.5 }, pct(v));
        });
        this.bars = rows.map((r, i) => {
            const y = m.t + i * (bh + gap) + (nar ? 4 : 0);
            if (nar) T(svg, { x: 0, y: y - 6, fill: TX2, 'font-size': 10 }, QN[i] + (i === 0 ? ', least diverse' : i === 3 ? ', most diverse' : ''));
            else {
                T(svg, { x: lw - 10, y: y + bh / 2 - 1, 'text-anchor': 'end', fill: TX2, 'font-size': 11 }, QN[i]);
                if (i === 0 || i === 3) T(svg, { x: lw - 10, y: y + bh / 2 + 12, 'text-anchor': 'end', fill: TX3, 'font-size': 9.5 }, i === 0 ? 'least diverse' : 'most diverse');
            }
            let acc = 0;
            const segsL = CATS.map(c => {
                const v = Math.max(0, c.v(r)), x0 = acc; acc += v;
                const rect = E('rect', { y, height: bh, fill: c.col, rx: 1.5 }, svg);
                const lab = T(svg, { y: y + bh / 2 + 4, 'text-anchor': 'middle', fill: TX, 'font-size': 10.5, stroke: 'rgba(11,11,12,0.55)', 'stroke-width': 3, 'paint-order': 'stroke' }, '');
                tipOn(rect, () => `<strong>${c.name}</strong><br>${pct(v, 1)} of cropped land<br>${QN[i]} of districts`);
                return { c, v, x0, rect, lab, y };
            });
            return segsL;
        });
        this.X = X; this.bh = bh;
        this.place(); this.dress();
    },
    place() {
        const X = this.X;
        this.bars.forEach((segsL, i) => {
            const t = Math.max(0, Math.min(1, (this.u - i * .12) / .64)), e = easeOut(t);
            segsL.forEach(s => {
                const x = X(s.x0 * e), w = Math.max(0, X((s.x0 + s.v) * e) - x - 2);
                s.rect.setAttribute('x', x.toFixed(1)); s.rect.setAttribute('width', w.toFixed(1));
                s.lab.setAttribute('x', (x + w / 2).toFixed(1));
                s.w = w;
            });
        });
        this.dress();
    },
    dress() {
        if (!this.bars) return;
        const sel = this.sel;
        this.bars.forEach(segsL => segsL.forEach(s => {
            const on = sel === 'all' || sel === s.c.key;
            s.rect.setAttribute('opacity', on ? 1 : .16);
            const show = this.u >= 1 && (sel === s.c.key || (sel === 'all' && s.w > 34));
            s.lab.textContent = show ? (s.v < .1 ? pct(s.v, 1) : pct(s.v)) : '';
            /* a narrow segment carries its label above the bar */
            const above = show && s.w < 34;
            s.lab.setAttribute('y', (above ? s.y - 5 : s.y + this.bh / 2 + 4).toFixed(1));
        }));
    },
    prep() { this.u = 0; this.place(); },
    play() { this.u = 0; this.place(); tween(this, 1300, t => { this.u = t; this.place(); }); },
});

/* ================================================================
   07 national series, 429 districts, area-weighted
   ================================================================ */
SC[7] = scene({
    u: 1,
    render() {
        const host = $('#v7'); if (!host) return;
        const tr = OV.trend_balanced;
        const W = host.clientWidth || 800, nar = W < 560;
        const m = { t: 16, r: nar ? 38 : 50, b: 32, l: nar ? 28 : 40 };
        const H = nar ? 300 : 340, iw = W - m.l - m.r, ih = H - m.t - m.b;
        const svg = this.svg = svgFor(host, W, H);
        const X = lin(tr[0].year, tr[tr.length - 1].year, m.l + (nar ? 12 : 18), m.l + iw), Y = lin(0, 28, m.t + ih, m.t);
        [0, 5, 10, 15, 20, 25].forEach(v => {
            E('line', { x1: m.l, x2: m.l + iw, y1: Y(v), y2: Y(v), stroke: v ? GRID : ZERO }, svg);
            T(svg, { x: m.l - 8, y: Y(v) + 4, 'text-anchor': 'end', fill: TX3, 'font-size': 10.5 }, String(v));
        });
        (nar ? [1997, 2005, 2012, 2019] : [1997, 2001, 2005, 2009, 2013, 2017, 2019]).forEach(v =>
            T(svg, { x: X(v), y: m.t + ih + 18, 'text-anchor': 'middle', fill: TX3, 'font-size': 10.5 }, String(v)));
        T(svg, { x: m.l, y: m.t - 4, fill: TX3, 'font-size': 10.5 }, 'number of crops');
        const body = tr.filter(t => t.year >= 1998);
        this.lines = [['D0', NEU], ['D1', CY]].map(([k, col]) => {
            const d = body.map((t, i) => (i ? 'L' : 'M') + X(t.year).toFixed(1) + ',' + Y(t[k]).toFixed(1)).join('');
            const p = E('path', { d, fill: 'none', stroke: col, 'stroke-width': 2, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' }, svg);
            const first = tr[0];
            const h = E('circle', { cx: X(first.year), cy: Y(first[k]), r: 4, fill: BG, stroke: col, 'stroke-width': 1.6 }, svg);
            const last = tr[tr.length - 1];
            const lab = T(svg, { x: X(last.year) + 7, y: Y(last[k]) + 4, fill: TX, 'font-size': 11 }, last[k].toFixed(1));
            return { p, h, lab, len: 0 };
        });
        const f = tr[0];
        this.note = T(svg, { x: X(f.year) + 9, y: Y(f.D1) + 18, fill: TX2, 'font-size': 10.5 }, 'first year of the record');
        const cross = E('line', { y1: m.t, y2: m.t + ih, stroke: 'rgba(255,255,255,0.35)', visibility: 'hidden' }, svg);
        const hit = E('rect', { x: m.l, y: m.t, width: iw, height: ih, fill: 'transparent' }, svg);
        const pick = ev => {
            const r = svg.getBoundingClientRect(), k = W / r.width, x = (ev.clientX - r.left) * k;
            let best = tr[0], bd = 1e9;
            tr.forEach(t => { const d = Math.abs(X(t.year) - x); if (d < bd) { bd = d; best = t; } });
            cross.setAttribute('x1', X(best.year)); cross.setAttribute('x2', X(best.year)); cross.setAttribute('visibility', 'visible');
            showTip(`<strong>${best.year}&ndash;${String(best.year + 1).slice(2)}</strong><br>count of crops ${best.D0.toFixed(1)}<br>effective number of crops ${best.D1.toFixed(1)}<br>evenness ${best.E.toFixed(2)}<br>${best.n} districts`, ev);
        };
        hit.addEventListener('mousemove', pick); hit.addEventListener('click', pick);
        hit.addEventListener('mouseleave', () => { cross.setAttribute('visibility', 'hidden'); hideTip(); });
        this.lines.forEach(l => { l.len = l.p.getTotalLength ? l.p.getTotalLength() : 1000; });
        this.place();
    },
    place() {
        const e = easeIO(this.u);
        this.lines.forEach(l => {
            l.p.setAttribute('stroke-dasharray', `${l.len} ${l.len}`);
            l.p.setAttribute('stroke-dashoffset', (l.len * (1 - e)).toFixed(1));
            const o = this.u >= 1 ? 1 : Math.max(0, (this.u - .85) / .15);
            l.lab.setAttribute('opacity', o.toFixed(2));
            l.h.setAttribute('opacity', Math.min(1, this.u * 4).toFixed(2));
        });
        this.note.setAttribute('opacity', Math.min(1, this.u * 4).toFixed(2));
    },
    prep() { this.u = 0; this.place(); },
    play() { this.u = 0; this.place(); tween(this, 1600, t => { this.u = t; this.place(); }); },
});

/* ================================================================
   08 district change, 429 districts as a beeswarm
   ================================================================ */
const CH = [
    { key: 'dD1', i: 2, name: 'effective number of crops', d: 2, med: () => OV.change.dD1_median, up: () => OV.change.D1_up, dn: () => OV.change.D1_down },
    { key: 'dE', i: 3, name: 'evenness', d: 3, med: () => OV.change.dE_median, up: () => OV.change.E_up, dn: () => OV.change.E_down },
    { key: 'dD0', i: 4, name: 'count of crops', d: 2, med: () => OV.change.dD0_median, up: () => OV.change.D0_up, dn: () => OV.change.D0_down },
];
function swarm(xs, r) {
    /* place each dot at the smallest offset from the centre line that clears earlier dots */
    const idx = xs.map((x, i) => i).sort((a, b) => xs[a] - xs[b]);
    const ys = new Array(xs.length), placed = [], d2 = (2 * r + .6) ** 2;
    idx.forEach(i => {
        const x = xs[i], near = placed.filter(j => Math.abs(xs[j] - x) < 2 * r + .6);
        let best = 0;
        for (let k = 0; k < 400; k++) {
            const y = (k % 2 ? 1 : -1) * Math.ceil(k / 2) * (r * .55);
            if (near.every(j => (xs[j] - x) ** 2 + (ys[j] - y) ** 2 >= d2)) { best = y; break; }
        }
        ys[i] = best; placed.push(i);
    });
    return ys;
}
SC[8] = scene({
    key: 'dD1', u: 1, v: 1,
    controls() {
        segs($('#v8-ctl'), CH.map(c => ({ key: c.key, name: c.name })),
            k => { if (k === this.key) return; cancel(this); this.played = true; this.u = 1; this.prev = this.layout; this.key = k; pressSeg($('#v8-ctl'), k); this.build(); this.v = 0; tween(this, 800, t => { this.v = t; this.place(); }); }, this.key);
    },
    render() {
        const host = $('#v8'); if (!host) return;
        const W = this.W = host.clientWidth || 800; this.nar = W < 560;
        this.prev = null; this.v = 1;
        this.build();
    },
    build() {
        const host = $('#v8'), W = this.W, nar = this.nar;
        const L = CH.find(c => c.key === this.key), rows = EX.change;
        const vals = rows.map(r => r[L.i]);
        const lo = Math.min(0, ...vals), hi = Math.max(0, ...vals), pad = (hi - lo) * .04;
        const m = { l: 14, r: 14, t: 58, b: 42 };
        const X = lin(lo - pad, hi + pad, m.l, W - m.r);
        const rad = nar ? 2.6 : 3.1;
        const ys = swarm(vals.map(X), rad);
        const half = Math.max(60, Math.max(...ys.map(Math.abs)) + rad + 4);
        const H = Math.round(m.t + 2 * half + m.b), cy = m.t + half;
        const keep = this.svg && this.svg.parentNode === host && this.svg.viewBox.baseVal.width === W;
        const svg = this.svg = svgFor(host, W, H);
        niceTicks(lo - pad, hi + pad, nar ? 4 : 8).forEach(v => {
            E('line', { x1: X(v), x2: X(v), y1: m.t - 6, y2: H - m.b + 4, stroke: v === 0 ? ZERO : GRID, 'stroke-width': v === 0 ? 1.2 : 1 }, svg);
            T(svg, { x: X(v), y: H - m.b + 18, 'text-anchor': 'middle', fill: TX3, 'font-size': 10.5 }, v === 0 ? '0' : sgn(v, L.key === 'dE' ? 2 : (Number.isInteger(v) ? 0 : 1)));
        });
        T(svg, { x: X(0) - 8, y: 14, 'text-anchor': 'end', fill: TX, 'font-size': 11 }, `${L.dn()} fell`);
        T(svg, { x: X(0) + 8, y: 14, fill: TX, 'font-size': 11 }, `${L.up()} rose`);
        const med = L.med();
        E('line', { x1: X(med), x2: X(med), y1: m.t - 14, y2: m.t - 4, stroke: TX, 'stroke-width': 1.6 }, svg);
        T(svg, { x: X(med) + (med < 0 ? -5 : 5), y: m.t - 18, 'text-anchor': med < 0 ? 'end' : 'start', fill: TX2, 'font-size': 10.5 }, `median ${sgn(med, L.d)}`);
        T(svg, { x: W - m.r, y: H - 3, 'text-anchor': 'end', fill: TX3, 'font-size': 10.5 }, 'change in ' + L.name + ', 1998–2004 to 2013–2019');
        const g = E('g', {}, svg);
        this.layout = rows.map((r, i) => [X(vals[i]), cy + ys[i]]);
        this.cy = cy; this.cx0 = X(0);
        this.dots = rows.map((r, i) => E('circle', { r: rad, fill: vals[i] > 0 ? CY : NEU, 'fill-opacity': vals[i] > 0 ? .95 : .55 }, g));
        this.ring = E('circle', { r: rad + 3, fill: 'none', stroke: TX, 'stroke-width': 1.4, visibility: 'hidden' }, svg);
        nearestTip(svg, () => this.layout, 12, i => {
            const r = rows[i];
            return `<strong>${r[0]}</strong><br>${r[1]}<br>effective number of crops ${sgn(r[2], 2)}<br>evenness ${sgn(r[3], 3)}<br>count of crops ${sgn(r[4], 1)}`;
        }, i => {
            if (i < 0) { this.ring.setAttribute('visibility', 'hidden'); return; }
            this.ring.setAttribute('cx', this.layout[i][0]); this.ring.setAttribute('cy', this.layout[i][1]); this.ring.setAttribute('visibility', 'visible');
        });
        if (!keep) this.prev = null;
        this.place();
    },
    place() {
        if (!this.dots) return;
        const eu = easeOut(this.u), ev = easeIO(this.v);
        this.dots.forEach((c, i) => {
            let [x, y] = this.layout[i];
            if (this.prev && this.v < 1) { const p = this.prev[i]; x = lerp(p[0], x, ev); y = lerp(p[1], y, ev); }
            x = lerp(this.cx0, x, eu); y = lerp(this.cy, y, eu);
            c.setAttribute('cx', x.toFixed(1)); c.setAttribute('cy', y.toFixed(1));
        });
    },
    prep() { this.u = 0; this.place(); },
    play() { this.u = 0; this.v = 1; this.place(); tween(this, 1300, t => { this.u = t; this.place(); }); },
});

/* ================================================================
   wiring
   ================================================================ */
const HOSTS = { 1: '#v1', 2: '#v2', 3: '#v3', 4: '#v4', 5: '#v5', 6: '#v6', 7: '#v7', 8: '#v8' };
let inited = false;
function startPlay(sc) { if (sc.played) return; sc.played = true; sc.play(); }

async function init() {
    [OV, IX, EX] = await Promise.all([loadJSON('data/site_overview.json'), loadJSON('data/site_indices.json'), loadJSON('data/site_overview_examples.json')]);
    Object.values(SC).forEach(sc => { if (sc.controls) sc.controls(); });
    for (const k of Object.keys(SC)) {
        const sc = SC[k];
        try { await sc.render(); } catch (e) { console.error('scene ' + k, e); }
        if (!RM && sc.prep) sc.prep();
    }
    if (RM) {
        Object.values(SC).forEach(sc => { sc.played = true; sc.play(); });
    } else if ('IntersectionObserver' in window) {
        const io = new IntersectionObserver(es => es.forEach(en => {
            if (!en.isIntersecting) return;
            io.unobserve(en.target);
            const sc = SC[en.target.dataset.scene];
            if (sc) startPlay(sc);
        }), { threshold: 0.3 });
        Object.keys(HOSTS).forEach(k => { const h = $(HOSTS[k]); if (!h) return; h.dataset.scene = k; io.observe(h); });
    } else {
        Object.values(SC).forEach(startPlay);
    }
    document.querySelectorAll('.replay').forEach(b => b.addEventListener('click', () => {
        const sc = SC[b.dataset.scene]; if (!sc) return;
        cancel(sc); sc.played = true;
        if (sc.reset) sc.reset();
        sc.play();
    }));
    inited = true;
}

async function drawAll() {
    if (!inited) { await init(); return; }
    for (const k of Object.keys(SC)) {
        try { await SC[k].render(); } catch (e) { console.error('scene ' + k, e); }
        if (!SC[k].played && !RM && SC[k].prep) SC[k].prep();
    }
}
window.drawAll = drawAll;
drawAll();
})();
