/* ================================================================
   MARKETS PAGE: scene visuals.
   Reads data/site_markets.json, data/site_markets_pop.json,
   data/site_market_map.json, data/site_typology.json and
   data/districts.geojson. Facilities are population shares: the share
   of a district's rural people living in a village with the facility.
   Depends on site.js for svgEl, lin, showTip, hideTip, safeTip,
   loadJSON, loadGeo, rampCyan and corrMatrix. Everything here is
   wrapped so no name collides with site.js globals.

   Pattern for every scene:
     render()  draws the visual for the scene's current state
     prep()    sets the pre-entrance state (only before first play)
     play()    runs the entrance sequence once the visual is in view
   drawAll() re-renders every scene on resize, so render() is idempotent.
   Under prefers-reduced-motion every tween resolves at once.
   ================================================================ */
(function () {
'use strict';

const RM = !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);

/* ---------- identity: one colour and one name per facility ---------- */
const NEU = '#B3B3B0';
const ID = { 'weekly haat': '#0E9BB5', 'FPO': '#1FA35A', 'fertiliser shop': '#7C5CE0', 'mandi': '#C63FD8' };
const colOf = f => ID[f] || NEU;
const LBL = { 'FPO': 'producer organisation', 'custom hiring': 'custom hiring of machinery',
              'cold storage': 'food storage warehouse' };
const lab = f => LBL[f] || f;
/* what a village with the facility "has", for readouts */
const HAS = { 'mandi': 'a mandi', 'regular market': 'a regular market', 'weekly haat': 'a weekly haat',
              'fertiliser shop': 'a fertiliser shop', 'seed centre': 'a seed centre', 'soil testing': 'soil testing',
              'custom hiring': 'a custom hiring centre for machinery', 'cold storage': 'a food storage warehouse',
              'farm-gate processing': 'farm-gate processing', 'FPO': 'a producer organisation' };
/* standard deviation of each facility's population share, filled from the data at init */
let SD = {}, SDV = {};
/* outcome means not carried in the export */
const MEAN = { D0: 21.4, D1: 4.9, pulses: 10.2, cereals: 66 };

const OFF = 'rgba(255,255,255,0.075)';
const TX = '#EDEDEB', TX2 = '#B3B3B0', TX3 = '#7E7E7C', BG = '#0B0B0C';
const GRID = 'rgba(255,255,255,0.075)', ZERO = 'rgba(255,255,255,0.28)';
const MONO = 'JetBrains Mono, monospace';

const $ = s => document.querySelector(s);
function E(tag, attrs, parent) { const e = svgEl(tag, attrs || {}); if (parent) parent.appendChild(e); return e; }
function T(parent, attrs, s) { const t = E('text', attrs, parent); t.textContent = s; return t; }
const sgn = (v, d) => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(d);
const fmtP = p => p < 0.0001 ? '< 0.0001' : (p < 0.001 ? p.toFixed(4) : p.toFixed(3));
const lerp = (a, b, t) => a + (b - a) * t;
const easeOut = t => 1 - Math.pow(1 - t, 3);

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
/* a row label right-aligned at x, wrapped onto up to three lines if it will not fit */
function rowLabel(g, x, y, s, maxW, fill, size) {
    const fs = size || 12.5, cw = fs * 0.5;
    const t = E('text', { x: x, y: y + 4, 'text-anchor': 'end', fill: fill || TX2, 'font-size': fs }, g);
    if (s.length * cw <= maxW || s.indexOf(' ') < 0) { t.textContent = s; return t; }
    const words = s.split(' '), lines = [];
    let cur = '';
    words.forEach(w => {
        const nx = cur ? cur + ' ' + w : w;
        if (nx.length * cw > maxW && cur) { lines.push(cur); cur = w; } else cur = nx;
    });
    if (cur) lines.push(cur);
    /* two lines: balance them rather than leave a short tail */
    if (lines.length === 2) {
        let best = 1, bestD = 1e9;
        for (let i = 1; i < words.length; i++) {
            const d = Math.abs(words.slice(0, i).join(' ').length - words.slice(i).join(' ').length);
            if (d < bestD) { bestD = d; best = i; }
        }
        lines[0] = words.slice(0, best).join(' '); lines[1] = words.slice(best).join(' ');
    }
    const L = lines.slice(0, 3), lh = 1.15 * fs;
    t.setAttribute('y', (y + 4 - (L.length - 1) * lh / 2).toFixed(1));
    L.forEach((ln, i) => { const sp = E('tspan', i ? { x: x, dy: lh.toFixed(1) } : { x: x }, t); sp.textContent = ln; });
    return t;
}

/* inverse standard normal (Acklam), for approximate intervals from b and p */
function qnorm(p) {
    const a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
    const b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01];
    const c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
    const d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00];
    const pl = 0.02425;
    let q, r;
    if (p < pl) { q = Math.sqrt(-2 * Math.log(p)); return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1); }
    if (p > 1 - pl) { q = Math.sqrt(-2 * Math.log(1 - p)); return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1); }
    q = p - 0.5; r = q * q;
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1);
}
function ci(b, p) {
    const z = qnorm(1 - Math.max(1e-12, Math.min(0.999, p)) / 2);
    const se = Math.abs(b) / Math.max(z, 1e-6);
    return [b - 1.96 * se, b + 1.96 * se];
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
function later(sc, ms, fn) {
    const token = sc.run;
    sc.timers.push(setTimeout(() => { if (sc.run === token) fn(); }, RM ? 0 : ms));
}
function cancel(sc) { sc.run++; sc.timers.forEach(clearTimeout); sc.timers = []; }
const scene = o => Object.assign({ run: 0, timers: [], played: false }, o);

/* segmented buttons */
function segs(host, items, onPick, cur) {
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
        b.addEventListener('click', () => onPick(it.key, true));
        host.appendChild(b);
    });
}
function pressSeg(host, key) {
    if (!host) return;
    [...host.querySelectorAll('.seg')].forEach(b => b.setAttribute('aria-pressed', b.dataset.key === key ? 'true' : 'false'));
}

let MK = null;
const SC = {};

/* ================================================================
   01 village grid
   ================================================================ */
function mulberry(seed) {
    return function () {
        seed |= 0; seed = seed + 0x6D2B79F5 | 0;
        let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
        t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
        return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
}
const RANK = (() => {
    const a = [...Array(100).keys()], r = mulberry(11);
    for (let i = 99; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
    const rank = []; a.forEach((cell, k) => { rank[cell] = k; });
    return rank;
})();
const LEVELS = [
    { key: 'p10', name: 'district with a low share', desc: 'in a district with a low share (one district in ten has less)' },
    { key: 'mean', name: 'average district', desc: 'in the average district' },
    { key: 'p90', name: 'district with a high share', desc: 'in a district with a high share (one district in ten has more)' },
];
SC[1] = scene({
    fac: null, lvl: 'mean',
    share() { const d = MK.descriptives.find(x => x.name === this.fac); return d ? d[this.lvl] : 0; },
    controls() {
        const list = MK.descriptives.slice().sort((a, b) => a.mean - b.mean)
            .map(d => ({ key: d.name, name: lab(d.name), color: colOf(d.name) }));
        segs($('#v1-pick'), list, k => { cancel(this); this.played = true; this.set(k, this.lvl, true); }, this.fac);
        segs($('#v1-level'), LEVELS, k => { cancel(this); this.played = true; this.set(this.fac || 'weekly haat', k, true); }, this.lvl);
    },
    render() {
        const host = $('#v1'); if (!host) return;
        const W = Math.min(host.clientWidth || 340, 380), cell = W / 10;
        const svg = svgFor(host, W, W);
        this.dots = [];
        for (let i = 0; i < 100; i++) {
            const c = E('circle', { cx: (i % 10 + .5) * cell, cy: (Math.floor(i / 10) + .5) * cell, r: cell * .34 }, svg);
            this.dots.push(c);
        }
        this.paint(false);
        this.readout();
    },
    paint(anim) {
        const n = this.fac ? Math.round(this.share() * 100) : 0, col = colOf(this.fac);
        const changed = [];
        this.dots.forEach((c, i) => {
            const on = RANK[i] < n, was = c.dataset.on === '1';
            if (on !== was) changed.push(i);
            c.dataset.on = on ? '1' : '0';
        });
        changed.sort((a, b) => RANK[a] - RANK[b]);
        const pos = {}; changed.forEach((i, k) => { pos[i] = k; });
        this.dots.forEach((c, i) => {
            const on = c.dataset.on === '1';
            if (!anim || RM) c.style.transition = 'none';
            else if (i in pos) c.style.transition = `fill 200ms ease ${Math.round(pos[i] / Math.max(1, changed.length) * 420)}ms`;
            else c.style.transition = 'fill 300ms ease';
            c.style.fill = on ? col : OFF;
        });
    },
    readout() {
        const r = $('#v1-read'); if (!r) return;
        if (!this.fac) { r.innerHTML = '&nbsp;'; return; }
        const d = MK.descriptives.find(x => x.name === this.fac);
        const lv = LEVELS.find(l => l.key === this.lvl);
        r.innerHTML = `<span class="readout__name">${(d[this.lvl] * 100).toFixed(1)} percent of rural people live in a village with ${HAS[this.fac] || lab(this.fac)}</span>` +
            `${lv.desc}<br>${d.zero ? d.zero + (d.zero === 1 ? ' district has' : ' districts have') + ' none' : 'every district has at least one'}`;
    },
    set(fac, lvl, anim) {
        this.fac = fac; this.lvl = lvl;
        pressSeg($('#v1-pick'), fac); pressSeg($('#v1-level'), lvl);
        this.paint(anim); this.readout();
    },
    prep() { this.fac = null; },
    play() {
        if (RM) { this.set('weekly haat', 'mean', false); return; }
        this.set('mandi', 'mean', true);
        later(this, 1500, () => this.set('weekly haat', 'mean', true));
        later(this, 3000, () => this.set('FPO', 'mean', true));
        later(this, 4500, () => this.set('weekly haat', 'mean', true));
    },
    reset() { this.fac = null; this.lvl = 'mean'; this.render(); },
});

/* ================================================================
   02 cluster diagram from classical MDS of 1 - r
   ================================================================ */
function mds2(M) {
    const n = M.length;
    const D2 = M.map(r => r.map(v => Math.pow(1 - v, 2)));
    const rm = D2.map(r => r.reduce((a, b) => a + b, 0) / n);
    const gm = rm.reduce((a, b) => a + b, 0) / n;
    let B = D2.map((r, i) => r.map((v, j) => -0.5 * (v - rm[i] - rm[j] + gm)));
    const out = [];
    for (let k = 0; k < 2; k++) {
        let v = Array.from({ length: n }, (_, i) => Math.sin(i * 1.7 + k + 1));
        let lam = 0;
        for (let it = 0; it < 400; it++) {
            const w = B.map(r => r.reduce((a, x, j) => a + x * v[j], 0));
            const nr = Math.hypot(...w) || 1;
            lam = w.reduce((a, x, j) => a + x * v[j], 0);
            v = w.map(x => x / nr);
        }
        out.push(v.map(x => x * Math.sqrt(Math.max(lam, 1e-9))));
        B = B.map((r, i) => r.map((x, j) => x - lam * v[i] * v[j]));
    }
    return out[0].map((x, i) => [x, out[1][i]]);
}
const EDGE_MIN = 0.4;
SC[2] = scene({
    mode: 'net', focus: -1, u: 1,
    controls() {
        segs($('#v2-ctl'), [{ key: 'net', name: 'Show as network' }, { key: 'mat', name: 'Show as table' }],
            k => { this.mode = k; pressSeg($('#v2-ctl'), k); cancel(this); this.u = 1; this.render(); }, this.mode);
    },
    render() {
        const host = $('#v2'); if (!host) return;
        const labels = MK.corr.labels, M = MK.corr.matrix, n = labels.length;
        if (this.mode === 'mat') {
            corrMatrix('#v2', labels.map(lab), M, {});
            return;
        }
        const W = host.clientWidth || 800, nar = W < 560;
        const H = nar ? 460 : 460;
        const svg = svgFor(host, W, H);
        this.svg = svg;
        /* target positions */
        if (!this.raw) this.raw = mds2(M);
        /* the nine are scaled to fill the left of the frame; the weekly haat is
           far from all of them, so its distance is shortened to fit at the right */
        const hi = labels.indexOf('weekly haat');
        const flip = this.raw[hi][0] < 0 ? -1 : 1;
        let R0 = this.raw.map(p => [p[0] * flip, p[1]]);
        /* open up the dense centre with a monotone radial stretch, so the
           ordering of distances from the centre is kept */
        const cc = [0, 1].map(k => R0.filter((_, i) => i !== hi).reduce((a, p) => a + p[k], 0) / (n - 1));
        R0 = R0.map((p, i) => {
            if (i === hi) return p;
            const dx = p[0] - cc[0], dy = p[1] - cc[1], r = Math.hypot(dx, dy) || 1e-9, r2 = Math.pow(r, 0.55);
            return [cc[0] + dx / r * r2, cc[1] + dy / r * r2];
        });
        const core = R0.filter((_, i) => i !== hi);
        const mx = { l: nar ? 16 : 40, t: 30, b: 30 };
        const xr = nar ? W * 0.64 : W * 0.66;
        const X = lin(Math.min(...core.map(p => p[0])), Math.max(...core.map(p => p[0])), mx.l + (nar ? 60 : 120), xr);
        const Y = lin(Math.min(...core.map(p => p[1])), Math.max(...core.map(p => p[1])), mx.t, H - mx.b);
        let P = R0.map((p, i) => i === hi ? [W - (nar ? 36 : 130), H / 2] : [X(p[0]), Y(p[1])]);
        /* deterministic relaxation: keep nodes at least minD apart */
        const minD = nar ? 48 : 56;
        for (let it = 0; it < 300; it++) {
            for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
                const dx = P[j][0] - P[i][0], dy = P[j][1] - P[i][1], d = Math.hypot(dx, dy) || .01;
                if (d < minD) {
                    const k = (minD - d) / 2 / d;
                    P[i][0] -= dx * k; P[i][1] -= dy * k; P[j][0] += dx * k; P[j][1] += dy * k;
                }
            }
            P = P.map(p => [Math.max(mx.l, Math.min(W - 20, p[0])), Math.max(mx.t, Math.min(H - mx.b, p[1]))]);
        }
        this.P = P;
        const cx = W / 2, cy = H / 2, R = Math.min(W, H) * .36;
        this.C = labels.map((_, i) => [cx + R * Math.cos(i / n * 2 * Math.PI - Math.PI / 2), cy + R * Math.sin(i / n * 2 * Math.PI - Math.PI / 2)]);

        const gE = E('g', {}, svg), gN = E('g', {}, svg), gL = E('g', {}, svg);
        this.edges = [];
        for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
            const r = M[i][j];
            if (r < EDGE_MIN) continue;
            const t = (r - EDGE_MIN) / (0.83 - EDGE_MIN);
            const l = E('line', { stroke: '#B3B3B0', 'stroke-width': (0.5 + 3.5 * t).toFixed(2), 'stroke-linecap': 'round' }, gE);
            this.edges.push({ i, j, r, l, op: 0.25 + 0.55 * t });
        }
        this.nodes = labels.map((f, i) => {
            const g = E('g', { style: 'cursor:pointer' }, gN);
            const hit = E('circle', { r: 20, fill: 'transparent' }, g);
            const c = E('circle', { r: 9, fill: colOf(f), stroke: BG, 'stroke-width': 2 }, g);
            const on = () => { this.focus = i; this.dress(); };
            g.addEventListener('mouseenter', on);
            g.addEventListener('mouseleave', () => { this.focus = -1; this.dress(); });
            g.addEventListener('click', ev => { ev.stopPropagation(); this.focus = this.focus === i ? -1 : i; this.dress(); });
            return { g, c, hit, f };
        });
        svg.addEventListener('click', () => { this.focus = -1; this.dress(); });
        /* labels: right of node if node is right of centre, else left */
        this.labs = labels.map((f, i) => T(gL, {
            'font-size': nar ? 10.5 : 11.5, 'font-family': MONO,
            fill: f === 'weekly haat' ? '#3FC2DA' : TX2,
            stroke: '#111113', 'stroke-width': 5, 'paint-order': 'stroke', 'stroke-linejoin': 'round'
        }, lab(f)));
        this.place(this.u);
        this.dress();
    },
    pos(i, u) { const e = easeOut(u); return [lerp(this.C[i][0], this.P[i][0], e), lerp(this.C[i][1], this.P[i][1], e)]; },
    place(u) {
        if (!this.nodes) return;
        const W = this.svg.viewBox.baseVal.width;
        const pts = this.nodes.map((_, i) => this.pos(i, u));
        this.nodes.forEach((nd, i) => { nd.g.setAttribute('transform', `translate(${pts[i][0].toFixed(1)},${pts[i][1].toFixed(1)})`); });
        this.edges.forEach(e => {
            e.l.setAttribute('x1', pts[e.i][0]); e.l.setAttribute('y1', pts[e.i][1]);
            e.l.setAttribute('x2', pts[e.j][0]); e.l.setAttribute('y2', pts[e.j][1]);
            e.l.setAttribute('opacity', (e.op * Math.max(0, (u - .6) / .4)).toFixed(3));
        });
        /* greedy label placement against nodes and earlier labels */
        const fs = W < 560 ? 10.5 : 11.5, cw = fs * 0.6;
        const boxes = [];
        const nodeBoxes = pts.map(p => [p[0] - 11, p[1] - 11, p[0] + 11, p[1] + 11]);
        /* points along the visible edges, so labels avoid sitting on a line */
        const ePts = [];
        this.edges.forEach(e => {
            if (e.l.style.visibility === 'hidden') return;
            for (let k = 1; k < 24; k++) {
                const t = k / 24;
                ePts.push([lerp(pts[e.i][0], pts[e.j][0], t), lerp(pts[e.i][1], pts[e.j][1], t)]);
            }
        });
        const hit = (b) => boxes.concat(nodeBoxes).reduce((a, o) => {
            const ox = Math.max(0, Math.min(b[2], o[2]) - Math.max(b[0], o[0]));
            const oy = Math.max(0, Math.min(b[3], o[3]) - Math.max(b[1], o[1]));
            return a + ox * oy;
        }, 0) + 60 * ePts.reduce((a, q) => a + (q[0] > b[0] - 2 && q[0] < b[2] + 2 && q[1] > b[1] - 2 && q[1] < b[3] + 2 ? 1 : 0), 0)
          + (b[0] < 2 ? 1e5 : 0) + (b[2] > W - 2 ? 1e5 : 0) + (b[1] < 0 ? 1e5 : 0) + (b[3] > this.svg.viewBox.baseVal.height ? 1e5 : 0);
        const order = pts.map((_, i) => i).sort((a, b) => pts[a][0] - pts[b][0]);
        order.forEach(i => {
            const t = this.labs[i], s = t.textContent, w = s.length * cw, p = pts[i];
            const cand = [
                [p[0] + 14, p[1] + 4, 'start'], [p[0] - 14, p[1] + 4, 'end'],
                [p[0], p[1] - 15, 'middle'], [p[0], p[1] + 24, 'middle'],
                [p[0] + 10, p[1] - 12, 'start'], [p[0] - 10, p[1] - 12, 'end'],
                [p[0] + 10, p[1] + 20, 'start'], [p[0] - 10, p[1] + 20, 'end'],
                [p[0], p[1] - 27, 'middle'], [p[0], p[1] + 36, 'middle'],
                [p[0] + 14, p[1] - 22, 'start'], [p[0] - 14, p[1] - 22, 'end'],
                [p[0] + 14, p[1] + 30, 'start'], [p[0] - 14, p[1] + 30, 'end'],
            ];
            let best = null, bs = 1e12;
            cand.forEach(c => {
                const x0 = c[2] === 'start' ? c[0] : (c[2] === 'end' ? c[0] - w : c[0] - w / 2);
                const b = [x0, c[1] - fs + 2, x0 + w, c[1] + 3];
                const sc = hit(b);
                if (sc < bs) { bs = sc; best = [c, b]; }
            });
            t.setAttribute('x', best[0][0].toFixed(1)); t.setAttribute('y', best[0][1].toFixed(1));
            t.setAttribute('text-anchor', best[0][2]);
            boxes.push(best[1]);
        });
    },
    dress() {
        if (!this.nodes || this.mode !== 'net') return;
        const M = MK.corr.matrix, f = this.focus;
        this.nodes.forEach((nd, i) => {
            nd.g.setAttribute('opacity', f < 0 || f === i ? 1 : 0.55);
            const t = this.labs[i];
            t.textContent = lab(nd.f) + (f >= 0 && f !== i ? '  ' + sgn(M[f][i], 2) : '');
            t.setAttribute('fill', f >= 0 && f !== i ? TX : (nd.f === 'weekly haat' ? '#3FC2DA' : TX2));
            t.setAttribute('font-weight', f === i ? 600 : 400);
        });
        this.edges.forEach(e => {
            const on = f < 0 || e.i === f || e.j === f;
            e.l.style.visibility = on ? 'visible' : 'hidden';
        });
        if (f >= 0) this.place(1);
    },
    prep() { this.u = 0; this.place(0); },
    play() {
        if (this.mode !== 'net') { this.mode = 'net'; pressSeg($('#v2-ctl'), 'net'); this.render(); }
        this.u = 0; this.place(0);
        tween(this, 1600, t => { this.u = t; this.place(t); });
    },
    reset() { this.u = 0; this.focus = -1; },
});

/* ================================================================
   03 district map with layer tabs
   ================================================================ */
const LAYERS = [
    { key: 'haat', f: 'weekly haat', name: 'weekly haat' },
    { key: 'fert', f: 'fertiliser shop', name: 'fertiliser shop' },
    { key: 'fpo', f: 'FPO', name: 'producer organisation' },
    { key: 'mandi', f: 'mandi', name: 'mandi' },
];
const NODATA = 'rgba(255,255,255,0.035)';
const RAMP0 = 0.14;
const ramp = t => rampCyan(RAMP0 + (1 - RAMP0) * t);
SC[3] = scene({
    layer: 'haat', filled: 1,
    controls() {
        segs($('#v3-tabs'), LAYERS.map(l => ({ key: l.key, name: l.name, color: colOf(l.f) })),
            k => { this.layer = k; pressSeg($('#v3-tabs'), k); if (this.filled < 1) { cancel(this); this.filled = 1; } this.recolour(true); }, this.layer);
    },
    range(key) {
        if (!this.rng) this.rng = {};
        if (!this.rng[key]) {
            const v = this.geo.features.map(f => f.properties[key]).filter(x => x !== null && x !== undefined).sort((a, b) => a - b);
            this.rng[key] = [v[Math.floor(v.length * .02)], v[Math.floor(v.length * .98)]];
        }
        return this.rng[key];
    },
    async render() {
        const host = $('#v3'); if (!host) return;
        this.geo = await loadGeo();
        const geo = this.geo;
        const W = host.clientWidth || 900, nar = W < 560;
        const H = Math.round(Math.min(nar ? 440 : 600, W * 1.08));
        const svg = svgFor(host, W, H);
        const pid = 'mkh' + Math.random().toString(36).slice(2, 7);
        const pat = E('pattern', { id: pid, width: 5, height: 5, patternUnits: 'userSpaceOnUse', patternTransform: 'rotate(45)' }, E('defs', {}, svg));
        E('rect', { width: 5, height: 5, fill: 'rgba(255,255,255,0.03)' }, pat);
        E('line', { x1: 0, y1: 0, x2: 0, y2: 5, stroke: 'rgba(255,255,255,0.32)', 'stroke-width': 1.2 }, pat);
        this.nod = `url(#${pid})`;
        let x0 = 180, x1 = -180, y0 = 90, y1 = -90;
        const walk = c => {
            if (typeof c[0] === 'number') { x0 = Math.min(x0, c[0]); x1 = Math.max(x1, c[0]); y0 = Math.min(y0, c[1]); y1 = Math.max(y1, c[1]); }
            else c.forEach(walk);
        };
        geo.features.forEach(f => walk(f.geometry.coordinates));
        const pad = 6, keyH = 44;
        const kx = Math.cos((y0 + y1) / 2 * Math.PI / 180);
        const s = Math.min((W - pad * 2) / ((x1 - x0) * kx), (H - pad * 2 - keyH) / (y1 - y0));
        const ox = pad + ((W - pad * 2) - (x1 - x0) * kx * s) / 2, oy = pad + ((H - pad * 2 - keyH) - (y1 - y0) * s) / 2;
        const PX = v => ox + (v - x0) * kx * s, PY = v => oy + (y1 - v) * s;
        const path = (c, d) => d === 1
            ? c.map((p, i) => (i ? 'L' : 'M') + PX(p[0]).toFixed(1) + ',' + PY(p[1]).toFixed(1)).join('') + 'Z'
            : c.map(k => path(k, d - 1)).join('');
        this.paths = [];
        geo.features.forEach(f => {
            const pr = f.properties;
            const d = path(f.geometry.coordinates, f.geometry.type === 'Polygon' ? 2 : 3);
            if (!d) return;
            const el = E('path', { d, stroke: 'rgba(11,11,12,0.85)', 'stroke-width': 0.4 }, svg);
            el.style.fill = NODATA;
            el.addEventListener('mousemove', ev => {
                el.setAttribute('stroke', TX); el.setAttribute('stroke-width', 1.1);
                const L = LAYERS.find(l => l.key === this.layer);
                showTip(safeTip(p => {
                    const v = p[L.key];
                    return `<strong>${p.n}</strong><br>${p.s}<br>` +
                        (v == null ? 'no facility record' : `${L.name}: ${(v * 100).toFixed(0)}% of rural people`) +
                        (p.D0 != null ? `<br>count of crops ${p.D0.toFixed(1)}` : '');
                }, pr), ev);
            });
            el.addEventListener('mouseleave', () => { el.setAttribute('stroke', 'rgba(11,11,12,0.85)'); el.setAttribute('stroke-width', 0.4); hideTip(); });
            this.paths.push({ el, pr });
        });
        this.order = this.paths.filter(p => p.pr.haat != null).sort((a, b) => a.pr.haat - b.pr.haat);
        /* key */
        const kw = Math.min(260, W * .5), kx0 = W - kw - 8, ky = H - 30;
        const gid = 'mkg' + Math.random().toString(36).slice(2, 7);
        const defs = E('defs', {}, svg), lg = E('linearGradient', { id: gid, x1: 0, x2: 1, y1: 0, y2: 0 }, defs);
        for (let i = 0; i <= 10; i++) E('stop', { offset: i * 10 + '%', 'stop-color': ramp(i / 10) }, lg);
        E('rect', { x: kx0, y: ky, width: kw, height: 9, fill: `url(#${gid})`, stroke: 'rgba(255,255,255,0.18)', 'stroke-width': .6 }, svg);
        const fm = { 'font-size': 10.5, 'font-family': MONO, fill: TX3 };
        this.kLo = T(svg, Object.assign({ x: kx0, y: ky - 6, 'text-anchor': 'start' }, fm), '');
        this.kHi = T(svg, Object.assign({ x: kx0 + kw, y: ky - 6, 'text-anchor': 'end' }, fm), '');
        this.kLab = T(svg, Object.assign({ x: kx0 + kw / 2, y: ky + 23, 'text-anchor': 'middle' }, fm), '');
        this.recolour(false);
    },
    fillOf(pr, key) {
        const v = pr[key]; if (v == null) return this.nod || NODATA;
        const r = this.range(key);
        return ramp(Math.max(0, Math.min(1, (v - r[0]) / (r[1] - r[0] || 1))));
    },
    recolour(anim) {
        if (!this.paths) return;
        const key = this.layer, L = LAYERS.find(l => l.key === key), r = this.range(key);
        this.kLo.textContent = (r[0] * 100).toFixed(0) + '%';
        this.kHi.textContent = (r[1] * 100).toFixed(0) + '%';
        this.kLab.textContent = 'share of rural people with a ' + L.name + ' in their village';
        const lim = this.filled >= 1 ? Infinity : Math.floor(this.filled * this.order.length);
        const rank = new Map(this.order.map((p, i) => [p, i]));
        this.paths.forEach(p => {
            p.el.style.transition = anim && !RM ? 'fill .4s ease' : 'none';
            const k = rank.has(p) ? rank.get(p) : -1;
            p.el.style.fill = (this.filled >= 1 || (k >= 0 && k < lim)) ? this.fillOf(p.pr, key) : NODATA;
        });
    },
    prep() { this.filled = 0; this.recolour(false); },
    play() {
        this.layer = 'haat'; pressSeg($('#v3-tabs'), 'haat');
        this.filled = 0; this.recolour(false);
        let done = 0;
        const N = this.order.length;
        tween(this, 1200, t => {
            const upto = Math.floor(t * N);
            for (let k = done; k < upto; k++) { const p = this.order[k]; p.el.style.transition = 'none'; p.el.style.fill = this.fillOf(p.pr, 'haat'); }
            done = upto; this.filled = t;
            if (t >= 1) { this.filled = 1; this.recolour(false); }
        });
    },
    reset() {},
});

/* ================================================================
   04 correlation strip against development
   ================================================================ */
SC[4] = scene({
    u: 1,
    render() {
        const host = $('#v4'); if (!host) return;
        const rows = MK.descriptives.slice().sort((a, b) => b.estab - a.estab);
        const W = host.clientWidth || 800, nar = W < 560;
        const lw = nar ? 124 : 210, rowH = nar ? 32 : 30;
        const m = { t: 10, r: nar ? 16 : 40, b: 44, l: lw };
        const H = m.t + rows.length * rowH + m.b;
        const svg = svgFor(host, W, H);
        const X = lin(-0.25, 0.4, m.l, W - m.r);
        [-0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4].forEach(v => {
            if (nar && Math.round(v * 10) % 2 !== 0) return;
            E('line', { x1: X(v), x2: X(v), y1: m.t, y2: H - m.b, stroke: v === 0 ? ZERO : GRID, 'stroke-width': v === 0 ? 1.2 : 1, 'shape-rendering': 'crispEdges' }, svg);
            T(svg, { x: X(v), y: H - m.b + 17, 'text-anchor': 'middle', fill: TX3, 'font-size': 10.5, 'font-family': MONO }, sgn(v, 1).replace('+0.0', '0').replace('−0.0', '0'));
        });
        T(svg, { x: (m.l + W - m.r) / 2, y: H - 6, 'text-anchor': 'middle', fill: TX3, 'font-size': 11.5 }, 'correlation with local development, across districts');
        this.X = X; this.items = rows.map((d, i) => {
            const y = m.t + rowH * (i + .5), col = d.name === 'weekly haat' ? '#0E9BB5' : NEU;
            rowLabel(svg, m.l - 12, y, lab(d.name), lw - 16, d.name === 'weekly haat' ? '#3FC2DA' : TX2, nar ? 11.5 : 12.5);
            const con = E('line', { y1: y, y2: y, stroke: col, 'stroke-opacity': .45, 'stroke-width': 1.2 }, svg);
            const dia = E('path', { fill: BG, stroke: col, 'stroke-width': 1.6 }, svg);
            const dot = E('circle', { cy: y, r: 5.5, fill: col }, svg);
            let val = null;
            if (d.name === 'weekly haat') val = T(svg, { y: y + 4, 'text-anchor': 'end', fill: '#3FC2DA', 'font-size': 11, 'font-family': MONO }, sgn(d.estab, 2));
            const hit = E('rect', { x: 0, y: y - rowH / 2, width: W, height: rowH, fill: 'transparent' }, svg);
            hit.addEventListener('mousemove', ev => showTip(`<strong>${lab(d.name)}</strong><br>correlation with non-farm businesses: ${sgn(d.estab, 2)}<br>correlation with night lights: ${sgn(d.viirs, 2)}`, ev));
            hit.addEventListener('mouseleave', hideTip);
            return { d, y, con, dia, dot, val, i };
        });
        this.place(this.u);
    },
    place(u) {
        if (!this.items) return;
        const X = this.X, n = this.items.length, tot = 700 + 60 * n;
        this.items.forEach(it => {
            const local = Math.max(0, Math.min(1, (u * tot - it.i * 60) / 700)), e = easeOut(local);
            const xe = X(it.d.estab * e), xv = X(it.d.viirs * e);
            it.dot.setAttribute('cx', xe);
            it.dia.setAttribute('d', `M${xv} ${it.y - 6}L${xv + 6} ${it.y}L${xv} ${it.y + 6}L${xv - 6} ${it.y}Z`);
            it.con.setAttribute('x1', Math.min(xe, xv)); it.con.setAttribute('x2', Math.max(xe, xv));
            if (it.val) { it.val.setAttribute('x', Math.min(xe, xv) - 10); it.val.setAttribute('opacity', e); }
        });
    },
    prep() { this.u = 0; this.place(0); },
    play() { this.u = 0; tween(this, 700 + 60 * 10, t => { this.u = t; this.place(t); }); },
    reset() {},
});

/* ================================================================
   05 toy land bar
   ================================================================ */
const BASE = [62, 9, 6, 5, 3.5, 2.5, 2, 1.6, 1.4, 1.2, 1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.4, 0.3, 0.2];
const TOY = [
    { key: 1, name: 'Starting district', shares: BASE.concat([0]), note: '20 crops, one cereal on 62% of the land' },
    { key: 2, name: 'Add a small crop', shares: BASE.map(s => s * .99).concat([1]), note: 'a 21st crop on 1% of the land' },
    { key: 3, name: 'Spread land from the cereal', shares: BASE.map((s, i) => i < 5 ? s + [-10, 4, 3, 2, 1][i] : s).concat([0]), note: '10% of the land moved from the cereal to the next four crops' },
    { key: 4, name: 'Twenty equal crops', shares: Array(20).fill(5).concat([0]), note: '20 crops on 5% of the land each' },
];
const effN = sh => Math.exp(-sh.reduce((a, s) => s > 0 ? a + (s / 100) * Math.log(s / 100) : a, 0));
const cntN = sh => sh.filter(s => s > 0).length;
function toyColour(i, key) {
    /* the crops that change are drawn in plain white, since hue means a facility here */
    if (i === 20) return TX;
    if (key === 3 && i >= 1 && i <= 4) return TX;
    if (i === 0) return NEU;
    const a = .55 - (i - 1) / 18 * (.55 - .18);
    return `rgba(255,255,255,${a.toFixed(3)})`;
}
SC[5] = scene({
    st: 1, cur: TOY[0].shares.slice(), eff: effN(TOY[0].shares), cnt: 20,
    controls() {
        segs($('#v5-ctl'), TOY.map(t => ({ key: String(t.key), name: t.name })),
            k => { cancel(this); this.played = true; this.go(+k); }, String(this.st));
    },
    render() {
        const host = $('#v5'); if (!host) return;
        const W = host.clientWidth || 800;
        const H = 136;
        const svg = svgFor(host, W, H);
        this.W = W;
        this.lab0 = T(svg, { x: 0, y: 16, fill: TX2, 'font-size': 11.5, 'font-family': MONO }, '');
        this.labN = T(svg, { x: W, y: 16, 'text-anchor': 'end', fill: TX, 'font-size': 11.5, 'font-family': MONO }, '');
        this.segs = this.cur.map((_, i) => E('rect', { y: 28, height: 56 }, svg));
        this.note = T(svg, { x: 0, y: 106, fill: TX3, 'font-size': 11.5, 'font-family': MONO }, '');
        this.cntT = T(svg, { x: 0, y: 130, fill: TX, 'font-size': 12, 'font-family': MONO }, '');
        this.cntE = T(svg, { x: W, y: 130, 'text-anchor': 'end', fill: TX, 'font-size': 12, 'font-family': MONO }, '');
        this.draw(this.cur, false);
        this.counters();
    },
    draw(sh, anim) {
        if (!this.segs) return;
        const W = this.W; let x = 0;
        const key = this.st;
        sh.forEach((s, i) => {
            const w = s / 100 * W, r = this.segs[i];
            r.setAttribute('x', x.toFixed(2)); r.setAttribute('width', Math.max(0, w - 1).toFixed(2));
            r.style.transition = anim && !RM ? 'fill .5s ease' : 'none';
            r.style.fill = toyColour(i, key);
            x += w;
        });
        this.lab0.textContent = `cereal ${sh[0].toFixed(sh[0] % 1 ? 1 : 0)}% of land`;
        this.labN.textContent = sh[20] > 0.05 ? 'new crop, 1% ↓' : '';
        this.note.textContent = TOY[key - 1].note;
    },
    counters() {
        if (!this.cntT) return;
        this.cntT.textContent = `count of crops ${this.cnt}`;
        this.cntE.textContent = `effective number of crops ${this.eff.toFixed(1)}`;
    },
    go(k) {
        const from = this.cur.slice(), to = TOY[k - 1].shares, e0 = this.eff, e1 = effN(to);
        this.st = k; pressSeg($('#v5-ctl'), String(k));
        this.cnt = cntN(to);
        this.note && (this.note.textContent = TOY[k - 1].note);
        tween(this, 700, t => {
            const e = easeOut(t);
            this.cur = from.map((s, i) => lerp(s, to[i], e));
            this.draw(this.cur, true);
            this.eff = lerp(e0, e1, Math.min(1, t * 700 / 600));
            this.counters();
        }, () => { this.cur = to.slice(); this.eff = e1; this.draw(this.cur, true); this.counters(); });
    },
    prep() {},
    play() {
        if (RM) { this.go(1); return; }
        this.go(1);
        later(this, 2000, () => this.go(2));
        later(this, 4000, () => this.go(3));
        later(this, 6000, () => this.go(1));
    },
    reset() {},
});

/* ================================================================
   shared coefficient chart for scenes 06, 08 and 09
   ================================================================ */
function coefChart(host, rows, o) {
    const W = host.clientWidth || 700, nar = W < 560;
    const lw = o.labelW ? o.labelW(nar) : (nar ? 120 : 200);
    const rowH = nar ? 32 : 32;
    const m = { t: o.top || 12, r: nar ? 18 : 30, b: 48, l: lw };
    const H = m.t + rows.length * rowH + m.b;
    const svg = svgFor(host, W, H);
    const gAx = E('g', {}, svg);
    const api = { svg, W, H, m, nar, X: null };
    api.axis = (d0, d1, xLab, dec) => {
        gAx.innerHTML = '';
        const X = lin(d0, d1, m.l, W - m.r);
        niceTicks(d0, d1, nar ? 4 : 6).forEach(v => {
            E('line', { x1: X(v), x2: X(v), y1: m.t, y2: H - m.b, stroke: Math.abs(v) < 1e-9 ? ZERO : GRID, 'stroke-width': Math.abs(v) < 1e-9 ? 1.2 : 1, 'shape-rendering': 'crispEdges' }, gAx);
            T(gAx, { x: X(v), y: H - m.b + 17, 'text-anchor': 'middle', fill: TX3, 'font-size': 10.5, 'font-family': MONO }, Math.abs(v) < 1e-9 ? '0' : sgn(v, Number.isInteger(v) ? 0 : 1));
        });
        T(gAx, { x: (m.l + W - m.r) / 2, y: H - 8, 'text-anchor': 'middle', fill: TX3, 'font-size': 11.5 }, xLab);
        if (!RM && o.fadeAxis) { gAx.style.opacity = 0; gAx.style.transition = 'opacity .5s'; requestAnimationFrame(() => { gAx.style.opacity = 1; }); }
        api.X = X; return X;
    };
    api.items = rows.map((r, i) => {
        const y = m.t + rowH * (i + .5), col = r.color || NEU;
        rowLabel(svg, m.l - 12, y, r.label, lw - 16, r.color && r.color !== NEU ? TX : TX2, nar ? 11.5 : 12.5);
        const wh = E('line', { y1: y, y2: y, stroke: col, 'stroke-opacity': r.color && r.color !== NEU ? .7 : .35, 'stroke-width': 1.5 }, svg);
        const dot = E('circle', { cy: y, r: 6, 'stroke-width': 1.7 }, svg);
        const val = T(svg, { y: y - 10, 'text-anchor': 'middle', fill: TX, 'font-size': 10.5, 'font-family': MONO }, '');
        const hit = E('rect', { x: 0, y: y - rowH / 2, width: W, height: rowH, fill: 'transparent' }, svg);
        const it = { r, y, wh, dot, val, col, i };
        hit.addEventListener('mousemove', ev => showTip(o.tip(it.cur || r), ev));
        hit.addEventListener('mouseleave', hideTip);
        return it;
    });
    /* place one row: b, lo, hi in data units; sig decides filled or hollow */
    api.place = (it, b, lo, hi, sig, showVal, dec) => {
        const X = api.X;
        it.dot.setAttribute('cx', X(b));
        it.dot.style.fill = sig ? it.col : BG;
        it.dot.style.stroke = sig ? 'none' : (it.col === NEU ? 'rgba(255,255,255,0.45)' : it.col);
        it.wh.setAttribute('x1', X(lo)); it.wh.setAttribute('x2', X(hi));
        it.val.setAttribute('x', X(b));
        it.val.textContent = showVal ? sgn(b, dec) : '';
    };
    return api;
}
const OUT = {
    D0: { key: 'D0_richness', name: 'Count of crops', axis: 'change in count of crops if every rural resident had one in the village (mean 21.4)', axisN: 'change in count of crops (mean 21.4)', tipName: 'count of crops', unit: '', k: 1, dec: 2 },
    D1: { key: 'D1_exp_shannon', name: 'Effective number of crops', axis: 'change in effective number of crops if every rural resident had one in the village (mean 4.9)', axisN: 'change in effective number of crops (mean 4.9)', tipName: 'effective number of crops', unit: '', k: 1, dec: 2 },
    pul: { key: 'share_pulses', axis: 'percentage points of cropped area', axisN: 'percentage points of cropped area', tipName: 'pulse share', unit: ' percentage points', k: 100, dec: 1 },
    cer: { key: 'share_cereals', axis: 'change in cereal share, percentage points, if every rural resident had one in the village', axisN: 'change in cereal share, percentage points', tipName: 'cereal share', unit: ' percentage points', k: 100, dec: 1 },
};
function rowsFor(out, src) {
    const O = OUT[out];
    return (src === 'village' ? MK.solo_village : MK.solo).filter(r => r.outcome === O.key).map(r => {
        const b = r.b * O.k, c = ci(b, r.p);
        return { fac: r.facility, label: lab(r.facility), b, p: r.p, lo: c[0], hi: c[1], color: colOf(r.facility), out, src };
    });
}
function coefTip(r) {
    const O = OUT[r.out], v = r.src === 'village', sd = v ? SDV[r.fac] : SD[r.fac];
    const who = v ? 'villages' : 'rural people';
    return `<strong>${r.label}</strong><br>${O.tipName}: ${sgn(r.b, O.dec)}${O.unit} going from no ${v ? 'village' : 'resident'} to all with one, share of ${who}<br>p = ${fmtP(r.p)}` +
        (sd ? `<br>one standard deviation more (${(sd * 100).toFixed(1)} percent of ${who}): ${sgn(r.b * sd, O.dec)}${O.unit}` : '');
}
const axLab = (O, nar) => nar ? O.axisN : O.axis;
function domainOf(rows) {
    const lo = Math.min(0, ...rows.map(r => r.lo)), hi = Math.max(0, ...rows.map(r => r.hi));
    const pad = (hi - lo) * .05;
    return [lo - pad, hi + pad];
}

/* ================================================================
   06 coefficient plot with outcome toggle
   ================================================================ */
SC[6] = scene({
    u: 1, src: 'pop',
    controls() {
        segs($('#v6-ctl'), [{ key: 'pop', name: 'share of rural people' }, { key: 'village', name: 'share of villages' }],
            k => { this.src = k; cancel(this); this.played = true; this.u = 1; this.render(); }, this.src);
    },
    order() {
        if (!this._ord) this._ord = rowsFor('D0').sort((a, b) => b.b - a.b).map(r => r.fac);
        return this._ord;
    },
    rowsOf(out) { const R = rowsFor(out, this.src); return this.order().map(f => R.find(r => r.fac === f)); },
    render() {
        const ha = $('#v6a'), hb = $('#v6b'); if (!ha || !hb) return;
        this.panels = [['D0', ha], ['D1', hb]].map(([out, host]) => {
            const rows = this.rowsOf(out);
            const ch = coefChart(host, rows, { tip: coefTip, labelW: nar => nar ? 124 : 176 });
            const d = domainOf(rows);
            ch.axis(d[0], d[1], axLab(OUT[out], ch.nar || ch.W < 520));
            return { out, rows, ch };
        });
        this.drawAt(this.u);
    },
    drawAt(u) {
        if (!this.panels) return;
        const e = easeOut(u);
        this.panels.forEach(P => P.ch.items.forEach((it, i) => {
            const r = P.rows[i]; it.cur = r;
            const hl = r.fac === 'weekly haat' || r.fac === 'FPO';
            P.ch.place(it, r.b * e, lerp(0, r.lo, e), lerp(0, r.hi, e), r.p < 0.05, hl && u >= 1, OUT[r.out].dec);
        }));
    },
    prep() { this.u = 0; this.drawAt(0); },
    play() { this.u = 0; tween(this, 900, t => { this.u = t; this.drawAt(t); }); },
    reset() {},
});

/* ================================================================
   07 specification ladder, read from site_markets_pop.json at init
   ================================================================ */
let LADDER = [];
SC[7] = scene({
    u: 1,
    render() {
        const host = $('#v7'); if (!host) return;
        const W = host.clientWidth || 800, nar = W < 560;
        const lw = nar ? 150 : 290, rowH = nar ? 46 : 38;
        const m = { t: 44, r: nar ? 44 : 60, b: 46, l: lw };
        const H = m.t + LADDER.length * rowH + m.b;
        const svg = svgFor(host, W, H);
        const xmax = Math.ceil(Math.max(...LADDER.map(r => r.b)) / 2) * 2 + 2;
        const X = lin(0, xmax, m.l, W - m.r);
        const adj = LADDER.slice(1).map(r => r.b), a0 = Math.min(...adj), a1 = Math.max(...adj);
        const band = E('rect', { x: X(a0), width: X(a1) - X(a0), y: m.t, height: LADDER.length * rowH, fill: 'rgba(14,155,181,0.10)' }, svg);
        const bandT = T(svg, { x: (X(a0) + X(a1)) / 2, y: 13, 'text-anchor': 'middle', fill: '#3FC2DA', 'font-size': 10.5, 'font-family': MONO }, sgn(a0, 1) + ' to ' + sgn(a1, 1));
        niceTicks(0, xmax, nar ? 4 : 6).forEach(v => {
            E('line', { x1: X(v), x2: X(v), y1: m.t, y2: H - m.b, stroke: v === 0 ? ZERO : GRID, 'shape-rendering': 'crispEdges' }, svg);
            T(svg, { x: X(v), y: H - m.b + 17, 'text-anchor': 'middle', fill: TX3, 'font-size': 10.5, 'font-family': MONO }, v === 0 ? '0' : '+' + v);
        });
        T(svg, { x: (m.l + W - m.r) / 2, y: H - 8, 'text-anchor': 'middle', fill: TX3, 'font-size': 11.5 }, nar ? 'change in count of crops' : 'change in count of crops if every rural resident had a weekly haat in the village');
        const yOf = i => m.t + rowH * (i + .5);
        this.rows = LADDER.map((r, i) => {
            const y = yOf(i);
            const g = E('g', {}, svg);
            rowLabel(g, m.l - 12, y, r.label + (r.n !== 606 ? ` (${r.n} districts)` : ''), lw - 16, TX2, nar ? 11 : 12.5);
            const dot = E('circle', { cx: X(r.b), cy: y, r: 6, fill: '#0E9BB5' }, g);
            T(g, { x: X(r.b) + 11, y: y + 4, fill: TX, 'font-size': 11, 'font-family': MONO }, '+' + r.b.toFixed(2));
            return { g, dot };
        });
        /* connector from the raw estimate to the within-state estimate */
        const cx0 = X(LADDER[0].b), cy0 = yOf(0), cx1 = X(LADDER[1].b), cy1 = yOf(1);
        const con = E('path', { d: `M${cx0 - 7} ${cy0 + 3}L${cx1 + 64} ${cy1}`, fill: 'none', stroke: 'rgba(255,255,255,0.35)', 'stroke-width': 1.2, 'stroke-dasharray': '2 4' }, svg);
        const conT = T(svg, { x: cx0 - 12, y: cy0 - 9, 'text-anchor': 'end', fill: TX2, 'font-size': 10.5, 'font-family': MONO }, 'comparing within state');
        const ghost = E('circle', { r: 6, fill: '#0E9BB5', opacity: 0 }, svg);
        Object.assign(this, { band, bandT, con, conT, ghost, cx0, cy0, cx1, cy1 });
        this.show(this.u);
    },
    show(u) {
        if (!this.rows) return;
        /* timeline in ms: 0-300 first dot, 300-1200 ghost, then rows 250ms apart, band last */
        const ms = u * 3200;
        const vis = (a, b) => Math.max(0, Math.min(1, (ms - a) / (b - a)));
        this.rows[0].g.setAttribute('opacity', vis(0, 300));
        const g = easeOut(vis(300, 1200));
        const on = ms > 300 && ms < 1200;
        this.ghost.setAttribute('cx', lerp(this.cx0, this.cx1, g)); this.ghost.setAttribute('cy', lerp(this.cy0, this.cy1, g));
        this.ghost.setAttribute('opacity', on ? .8 : 0);
        this.con.setAttribute('opacity', vis(300, 1200)); this.conT.setAttribute('opacity', vis(800, 1200));
        this.rows[1].g.setAttribute('opacity', vis(1150, 1250));
        for (let i = 2; i < this.rows.length; i++) this.rows[i].g.setAttribute('opacity', vis(1250 + (i - 2) * 250, 1450 + (i - 2) * 250));
        const bv = vis(2700, 3200);
        this.band.setAttribute('opacity', bv); this.bandT.setAttribute('opacity', bv);
    },
    prep() { this.u = 0; this.show(0); },
    play() { this.u = 0; tween(this, 3200, t => { this.u = t; this.show(t); }); },
    reset() {},
});

/* ================================================================
   08 cropland grid plus pulse coefficients
   ================================================================ */
SC[8] = scene({
    st: 'avg', u: 1,
    controls() {
        segs($('#v8-ctl'), [{ key: 'avg', name: 'Average district' }, { key: 'fert', name: 'One standard deviation more fertiliser shops' }],
            k => { cancel(this); this.played = true; this.u = 1; this.placeCoef(1); this.setGrid(k, true); }, this.st);
    },
    fert() {
        const r = MK.solo.find(x => x.facility === 'fertiliser shop' && x.outcome === 'share_pulses');
        const drop = -r.b * 100 * SD['fertiliser shop'];
        return { drop, sq: Math.round(drop), after: MEAN.pulses - drop };
    },
    render() {
        const ha = $('#v8a'), hb = $('#v8b'); if (!ha || !hb) return;
        const W = Math.min(ha.clientWidth || 320, 340), cell = W / 10, H = W + 30;
        const svg = svgFor(ha, W, H);
        const F = this.fert();
        this.cells = [];
        for (let i = 0; i < 100; i++) {
            const kind = i < MEAN.cereals ? 'c' : (i < MEAN.cereals + Math.round(MEAN.pulses) ? 'p' : 'o');
            const r = E('rect', { x: (i % 10) * cell + 1, y: Math.floor(i / 10) * cell + 1, width: cell - 2, height: cell - 2, rx: 1.5 }, svg);
            this.cells.push({ r, kind, i });
        }
        const pul = this.cells.filter(c => c.kind === 'p');
        this.lose = new Set(pul.slice(pul.length - F.sq).map(c => c.i));
        const lg = [['cereals', NEU], ['pulses', '#7C5CE0'], ['other crops', 'rgba(255,255,255,0.18)']];
        let lx = 0;
        lg.forEach(([n, c]) => {
            E('rect', { x: lx, y: W + 12, width: 10, height: 10, fill: c }, svg);
            T(svg, { x: lx + 15, y: W + 21, fill: TX2, 'font-size': 11, 'font-family': MONO }, n);
            lx += 15 + n.length * 6.8 + 16;
        });
        this.setGrid(this.st, false);

        const rows = rowsFor('pul').sort((a, b) => a.b - b.b);
        rows.forEach(r => { r.color = r.fac === 'fertiliser shop' ? '#7C5CE0' : NEU; });
        this.rowsB = rows;
        this.ch = coefChart(hb, rows, { tip: coefTip, labelW: nar => nar ? 124 : 190 });
        const d = domainOf(rows);
        this.ch.axis(d[0], d[1], axLab(OUT.pul, this.ch.nar), 0);
        this.placeCoef(this.u);
    },
    placeCoef(u) {
        if (!this.ch) return;
        const n = this.rowsB.length, tot = 700 + 60 * n;
        this.ch.items.forEach((it, i) => {
            const r = this.rowsB[i]; it.cur = r;
            const e = easeOut(Math.max(0, Math.min(1, (u * tot - i * 60) / 700)));
            this.ch.place(it, r.b * e, r.lo * e, r.hi * e, r.p < 0.05, r.fac === 'fertiliser shop' && u >= 1, 1);
        });
    },
    setGrid(k, anim) {
        this.st = k; pressSeg($('#v8-ctl'), k);
        const F = this.fert();
        (this.cells || []).forEach(c => {
            let col = c.kind === 'c' ? NEU : (c.kind === 'p' ? '#7C5CE0' : 'rgba(255,255,255,0.18)');
            if (k === 'fert' && this.lose.has(c.i)) col = 'rgba(255,255,255,0.18)';
            c.r.style.transition = anim && !RM ? 'fill .7s ease' : 'none';
            c.r.style.fill = col;
        });
        const rd = $('#v8-read');
        if (rd) rd.innerHTML = k === 'fert'
            ? `pulses ${MEAN.pulses.toFixed(1)}% → ${F.after.toFixed(1)}% of cropped area`
            : `pulses ${MEAN.pulses.toFixed(1)}% of cropped area, cereals ${MEAN.cereals}%`;
    },
    prep() { this.u = 0; this.placeCoef(0); },
    play() {
        this.setGrid('avg', false); this.u = 0;
        tween(this, 1300, t => { this.u = t; this.placeCoef(t); });
        later(this, 1500, () => this.setGrid('fert', true));
    },
    reset() {},
});

/* ================================================================
   09 cereal-share coefficients
   ================================================================ */
SC[9] = scene({
    u: 1,
    render() {
        const host = $('#v9'); if (!host) return;
        let rows = rowsFor('cer').sort((a, b) => a.b - b.b);
        rows.forEach(r => { r.color = r.fac === 'mandi' ? '#C63FD8' : (r.fac === 'fertiliser shop' ? '#7C5CE0' : NEU); });
        this.rows = rows;
        this.ch = coefChart(host, rows, { tip: coefTip, labelW: nar => nar ? 124 : 210, top: 34 });
        const d = domainOf(rows);
        const X = this.ch.axis(d[0], d[1], axLab(OUT.cer, this.ch.nar), 0);
        /* expected direction, right of zero */
        const x0 = X(0) + 8, x1 = Math.min(X(0) + 120, this.ch.W - this.ch.m.r - 4), y = 16;
        const svg = this.ch.svg;
        E('line', { x1: x0, x2: x1 - 6, y1: y, y2: y, stroke: 'rgba(255,255,255,0.4)', 'stroke-width': 1 }, svg);
        E('path', { d: `M${x1} ${y}L${x1 - 7} ${y - 4}L${x1 - 7} ${y + 4}Z`, fill: 'rgba(255,255,255,0.4)' }, svg);
        T(svg, { x: x0, y: y - 6, fill: TX3, 'font-size': 10.5, 'font-family': MONO }, 'expected direction');
        this.place(this.u);
    },
    place(u) {
        if (!this.ch) return;
        const n = this.rows.length, ms = u * 2400;
        const mi = this.rows.findIndex(r => r.fac === 'mandi');
        let k = 0;
        this.ch.items.forEach((it, i) => {
            const r = this.rows[i]; it.cur = r;
            const start = i === mi ? 60 * n + 300 : 60 * (k++);
            const e = easeOut(Math.max(0, Math.min(1, (ms - start) / 700)));
            let lo = r.lo * e, hi = r.hi * e;
            if (i === mi) {
                const w = easeOut(Math.max(0, Math.min(1, (ms - start - 700) / 600)));
                lo = lerp(r.b, r.lo, w); hi = lerp(r.b, r.hi, w);
                if (ms - start < 700) { lo = r.b * e; hi = r.b * e; }
            }
            this.ch.place(it, r.b * e, lo, hi, r.p < 0.05, (r.fac === 'mandi' || r.fac === 'fertiliser shop') && u >= 1, 1);
        });
    },
    prep() { this.u = 0; this.place(0); },
    play() { this.u = 0; tween(this, 2400, t => { this.u = t; this.place(t); }); },
    reset() {},
});

/* ================================================================
   10 irrigation curves at low and high facility shares
   ================================================================ */
const MEAS = [
    { key: 'mandi village share', name: 'mandi', color: '#C63FD8' },
    { key: 'fertiliser shop village share', name: 'fertiliser shop', color: '#7C5CE0' },
    { key: 'output-market index', name: 'output markets', color: TX },
    { key: 'input-supply index', name: 'input supply', color: TX },
    { key: 'post-harvest index', name: 'post-harvest', color: TX },
];
const curve = (c, tp) => x => c * (x * x - 2 * tp * x);
SC[10] = scene({
    m: 'mandi village share', u: 1, par: null,
    controls() {
        segs($('#v10-tabs'), MEAS.map(m => ({ key: m.key, name: m.name, color: m.color === TX ? null : m.color })),
            k => { this.played = true; if (this.u < 1) { cancel(this); this.u = 1; this.reveal(1); } this.morph(k); }, this.m);
    },
    rec(k) { return MK.interaction.find(r => r.measure === k); },
    render() {
        const host = $('#v10'); if (!host) return;
        const W = host.clientWidth || 800, nar = W < 560;
        const H = nar ? 320 : 380;
        const m = { t: 30, r: nar ? 14 : 24, b: 44, l: nar ? 40 : 52 };
        const svg = svgFor(host, W, H);
        /* one y range across all five measures so tabs compare */
        let ymin = 0, ymax = 0;
        MK.interaction.forEach(r => [[r.curv_lo, r.tp_lo], [r.curv_hi, r.tp_hi]].forEach(([c, tp]) => {
            const f = curve(c, tp);
            for (let i = 0; i <= 50; i++) { const v = f(i / 50); ymin = Math.min(ymin, v); ymax = Math.max(ymax, v); }
        }));
        ymin -= .2; ymax += .45;
        const X = lin(0, 1, m.l, W - m.r), Y = lin(ymin, ymax, H - m.b, m.t);
        Object.assign(this, { X, Y, mg: m, W, H });
        niceTicks(ymin, ymax, 5).forEach(v => {
            E('line', { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), stroke: Math.abs(v) < 1e-9 ? ZERO : GRID, 'stroke-width': Math.abs(v) < 1e-9 ? 1.2 : 1, 'shape-rendering': 'crispEdges' }, svg);
            T(svg, { x: m.l - 8, y: Y(v) + 4, 'text-anchor': 'end', fill: TX3, 'font-size': 10.5, 'font-family': MONO }, Math.abs(v) < 1e-9 ? '0' : sgn(v, 1));
        });
        [0, .2, .4, .6, .8, 1].forEach(v => T(svg, { x: X(v), y: H - m.b + 17, 'text-anchor': 'middle', fill: TX3, 'font-size': 10.5, 'font-family': MONO }, Math.round(v * 100) + '%'));
        T(svg, { x: (m.l + W - m.r) / 2, y: H - 8, 'text-anchor': 'middle', fill: TX3, 'font-size': 11.5 }, 'share of sown area irrigated');
        T(svg, { x: m.l - (nar ? 34 : 44), y: 14, fill: TX3, 'font-size': 11.5 }, nar ? 'change in effective number of crops vs no irrigation' : 'change in effective number of crops relative to an unirrigated district');
        const cid = 'mkc' + Math.random().toString(36).slice(2, 7);
        const defs = E('defs', {}, svg);
        const mkClip = id => { const c = E('clipPath', { id }, defs); return E('rect', { x: 0, y: 0, width: W, height: H }, c); };
        this.clipLo = mkClip(cid + 'a'); this.clipHi = mkClip(cid + 'b');
        this.pLo = E('path', { fill: 'none', stroke: '#9A9A98', 'stroke-width': 2, 'stroke-dasharray': '5 4', 'clip-path': `url(#${cid}a)` }, svg);
        this.pHi = E('path', { fill: 'none', 'stroke-width': 2.6, 'clip-path': `url(#${cid}b)` }, svg);
        this.kLo = E('circle', { r: 4.5, fill: BG, stroke: '#9A9A98', 'stroke-width': 1.6 }, svg);
        this.kHi = E('circle', { r: 4.5 }, svg);
        this.tLo = T(svg, { 'text-anchor': 'middle', fill: TX2, 'font-size': 10.5, 'font-family': MONO }, '');
        this.tHi = T(svg, { 'text-anchor': 'middle', fill: TX, 'font-size': 10.5, 'font-family': MONO }, '');
        const r = this.rec(this.m);
        this.par = this.par || { cl: r.curv_lo, tl: r.tp_lo, ch: r.curv_hi, th: r.tp_hi };
        this.drawCurves(this.par);
        this.reveal(this.u);
        this.readout();
    },
    pathOf(c, tp) {
        const f = curve(c, tp); let d = '';
        for (let i = 0; i <= 80; i++) { const x = i / 80; d += (i ? 'L' : 'M') + this.X(x).toFixed(1) + ',' + this.Y(f(x)).toFixed(1); }
        return d;
    },
    drawCurves(p) {
        if (!this.pLo) return;
        const M = MEAS.find(x => x.key === this.m);
        this.pLo.setAttribute('d', this.pathOf(p.cl, p.tl));
        this.pHi.setAttribute('d', this.pathOf(p.ch, p.th));
        this.pHi.setAttribute('stroke', M.color);
        this.kHi.setAttribute('fill', M.color);
        const pk = (c, tp) => [this.X(tp), this.Y(-c * tp * tp)];
        const a = pk(p.cl, p.tl), b = pk(p.ch, p.th);
        this.kLo.setAttribute('cx', a[0]); this.kLo.setAttribute('cy', a[1]);
        this.kHi.setAttribute('cx', b[0]); this.kHi.setAttribute('cy', b[1]);
        this.tLo.setAttribute('x', a[0]); this.tLo.setAttribute('y', a[1] - 11); this.tLo.textContent = 'peak ' + Math.round(p.tl * 100) + '%';
        /* the high-share label goes below its peak so the two never collide */
        this.tHi.setAttribute('x', b[0]); this.tHi.setAttribute('y', b[1] + 20); this.tHi.textContent = 'peak ' + Math.round(p.th * 100) + '%';
        const sw = $('#v10-sw'); if (sw) sw.setAttribute('stroke', M.color);
    },
    reveal(u) {
        if (!this.clipLo) return;
        const W = this.W, ms = u * 1800;
        const a = easeOut(Math.max(0, Math.min(1, ms / 900))), b = easeOut(Math.max(0, Math.min(1, (ms - 900) / 900)));
        this.clipLo.setAttribute('width', (this.mg.l + (W - this.mg.l) * a + (a >= 1 ? 20 : 0)).toFixed(1));
        this.clipHi.setAttribute('width', (this.mg.l + (W - this.mg.l) * b + (b >= 1 ? 20 : 0)).toFixed(1));
        this.kLo.setAttribute('opacity', a >= 1 ? 1 : 0); this.tLo.setAttribute('opacity', a >= 1 ? 1 : 0);
        this.kHi.setAttribute('opacity', b >= 1 ? 1 : 0); this.tHi.setAttribute('opacity', b >= 1 ? 1 : 0);
    },
    readout() {
        const r = this.rec(this.m), el = $('#v10-read'); if (!el) return;
        el.textContent = `How sharply the curve bends down: ${sgn(r.curv_lo, 2)} one standard deviation below average, ${sgn(r.curv_hi, 2)} one standard deviation above (p = ${r.p_inter.toFixed(3)} for the difference).`;
    },
    morph(k) {
        const from = Object.assign({}, this.par), r = this.rec(k);
        const to = { cl: r.curv_lo, tl: r.tp_lo, ch: r.curv_hi, th: r.tp_hi };
        this.m = k; pressSeg($('#v10-tabs'), k); this.readout();
        tween(this, 600, t => {
            const e = easeOut(t);
            this.par = { cl: lerp(from.cl, to.cl, e), tl: lerp(from.tl, to.tl, e), ch: lerp(from.ch, to.ch, e), th: lerp(from.th, to.th, e) };
            this.drawCurves(this.par);
        });
    },
    prep() { this.u = 0; this.reveal(0); },
    play() {
        if (this.m !== MEAS[0].key) { this.m = MEAS[0].key; pressSeg($('#v10-tabs'), this.m); const r = this.rec(this.m); this.par = { cl: r.curv_lo, tl: r.tp_lo, ch: r.curv_hi, th: r.tp_hi }; this.drawCurves(this.par); this.readout(); }
        this.u = 0; tween(this, 1800, t => { this.u = t; this.reveal(t); });
    },
    reset() {},
});

/* ================================================================
   11 timeline
   ================================================================ */
SC[11] = scene({
    u: 1,
    render() {
        const host = $('#v11'); if (!host) return;
        const W = host.clientWidth || 800, nar = W < 560;
        const H = 150, m = { l: 14, r: 14 };
        const svg = svgFor(host, W, H);
        const X = lin(1998, 2020, m.l, W - m.r);
        const yA = 92;
        E('line', { x1: X(1998), x2: X(2020), y1: yA, y2: yA, stroke: ZERO }, svg);
        [2000, 2005, 2010, 2015, 2020].forEach(v => {
            if (nar && (v === 2005 || v === 2015)) return;
            E('line', { x1: X(v), x2: X(v), y1: yA, y2: yA + 5, stroke: ZERO }, svg);
            T(svg, { x: X(v), y: yA + 20, 'text-anchor': 'middle', fill: TX3, 'font-size': 10.5, 'font-family': MONO }, v);
        });
        const tw = Math.max(3, (X(1999) - X(1998)) * .32);
        this.ticks = [];
        for (let y = 1998; y < 2020; y++) {
            this.ticks.push(E('rect', { x: X(y + .5) - tw / 2, y: yA - 30, width: tw, height: 24, rx: 1 }, svg));
        }
        E('path', { d: `M${X(1998.1)} ${yA - 38}V${yA - 44}H${X(2019.9)}V${yA - 38}`, fill: 'none', stroke: 'rgba(255,255,255,0.25)' }, svg);
        this.tA = T(svg, { x: X(1998.1), y: yA - 52, fill: TX2, 'font-size': 11, 'font-family': MONO }, nar ? '22 years averaged for cropping' : 'cropping: 22 agricultural years averaged, 1998–99 to 2019–20');
        this.dot = E('circle', { cx: X(2019.5), cy: yA + 40, r: 6, fill: '#0E9BB5' }, svg);
        this.dotT = T(svg, { x: X(2019.5) - 12, y: yA + 44, 'text-anchor': 'end', fill: '#3FC2DA', 'font-size': 11, 'font-family': MONO }, 'facility survey, 2019–20');
        this.dotL = E('line', { x1: X(2019.5), x2: X(2019.5), y1: yA + 2, y2: yA + 33, stroke: 'rgba(14,155,181,0.6)', 'stroke-dasharray': '2 3' }, svg);
        this.show(this.u);
    },
    show(u) {
        if (!this.ticks) return;
        const ms = u * (22 * 40 + 600);
        this.ticks.forEach((t, i) => { t.style.fill = ms >= i * 40 + 40 ? NEU : OFF; });
        const d = Math.max(0, Math.min(1, (ms - 22 * 40) / 600));
        this.dot.setAttribute('opacity', d); this.dotT.setAttribute('opacity', d); this.dotL.setAttribute('opacity', d);
        this.tA.setAttribute('opacity', Math.min(1, ms / 400));
    },
    prep() { this.u = 0; this.show(0); },
    play() { this.u = 0; tween(this, 22 * 40 + 600, t => { this.u = t; this.show(t); }); },
    reset() {},
});

/* ================================================================
   method tables
   ================================================================ */
const GROUPS = [
    ['Output markets', ['mandi', 'regular market', 'weekly haat']],
    ['Input supply', ['fertiliser shop', 'seed centre', 'soil testing', 'custom hiring']],
    ['Post-harvest facilities and producer organisations', ['cold storage', 'farm-gate processing', 'FPO']],
];
const stars = p => p < 0.01 ? ' ***' : p < 0.05 ? ' **' : p < 0.10 ? ' *' : '';
const pct = v => (v * 100).toFixed(1) + '%';
function tables() {
    const ta = $('#tbl-a tbody'), tb = $('#tbl-b tbody');
    if (ta) {
        let h = '';
        GROUPS.forEach(([g, fs]) => {
            h += `<tr><th colspan="7" scope="rowgroup">${g}</th></tr>`;
            fs.forEach(f => {
                const d = MK.descriptives.find(x => x.name === f); if (!d) return;
                h += `<tr><td>${lab(f)}</td><td class="num">${pct(d.mean)}</td><td class="num">${pct(d.p10)}</td><td class="num">${pct(d.p90)}</td>` +
                    `<td class="num">${d.zero}</td><td class="num">${sgn(d.viirs, 2)}</td><td class="num${d.estab < 0 ? ' neg' : ''}">${sgn(d.estab, 2)}</td></tr>`;
            });
        });
        ta.innerHTML = h;
    }
    if (tb) {
        const cell = (f, o, k, dec) => {
            const r = MK.solo.find(x => x.facility === f && x.outcome === o); if (!r) return '<td class="num">&ndash;</td>';
            const b = r.b * k, cls = r.p < 0.10 ? (b > 0 ? ' pos' : ' neg') : '';
            return `<td class="num${cls}">${sgn(b, dec)}${stars(r.p)}</td>`;
        };
        let h = '';
        GROUPS.forEach(([g, fs]) => {
            h += `<tr><th colspan="5" scope="rowgroup">${g}</th></tr>`;
            fs.forEach(f => {
                h += `<tr><td>${lab(f)}</td>${cell(f, 'D0_richness', 1, 2)}${cell(f, 'D1_exp_shannon', 1, 2)}${cell(f, 'share_cereals', 100, 1)}${cell(f, 'share_pulses', 100, 1)}</tr>`;
            });
        });
        tb.innerHTML = h;
    }
}

/* ================================================================
   shared: district map projection for the new map scenes
   ================================================================ */
let MM = null, TY = null;
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
    return { PX, PY, ring, path, fpath: f => path(f.geometry.coordinates, f.geometry.type === 'Polygon' ? 2 : 3) };
}
function hatchDef(svg, id, stroke, w, gap) {
    const pat = E('pattern', { id, width: gap, height: gap, patternUnits: 'userSpaceOnUse', patternTransform: 'rotate(45)' }, E('defs', {}, svg));
    E('line', { x1: 0, y1: 0, x2: 0, y2: gap, stroke, 'stroke-width': w }, pat);
    return `url(#${id})`;
}
function stateOutlines(svg, P) {
    const g = E('g', { 'pointer-events': 'none' }, svg);
    MM.state_rings.forEach(r => E('path', { d: P.ring(r), fill: 'none', stroke: 'rgba(237,237,235,0.42)', 'stroke-width': 0.7, 'stroke-linejoin': 'round' }, g));
    return g;
}
/* a tap on a phone fires click without mousemove; both open the card */
function tipOn(el, html) {
    const f = ev => showTip(html(), ev);
    el.addEventListener('mousemove', f);
    el.addEventListener('click', f);
}
const pct0 = v => (v * 100).toFixed(0) + '%';

/* ================================================================
   new scene: district map, twelve layers
   ================================================================ */
const ML = [
    { key: 'D0', name: 'count of crops', fmt: v => v.toFixed(1) + ' crops', kf: v => v.toFixed(0), kl: 'count of crops' },
    { key: 'D1', name: 'effective number of crops', fmt: v => v.toFixed(1), kf: v => v.toFixed(1), kl: 'effective number of crops' },
].concat(['mandi', 'regular market', 'weekly haat', 'fertiliser shop', 'seed centre', 'soil testing',
          'custom hiring', 'cold storage', 'farm-gate processing', 'FPO'].map((f, i) => ({
    key: 'f' + i, i, f, name: lab(f), fmt: v => pct0(v) + ' of rural people', kf: pct0, kl: 'share of rural people with ' + HAS[f] + ' in the village'
})));
const mmVal = (d, L) => d == null ? null : (L.i != null ? d.f[L.i] : d[L.key]);
SC[12] = scene({
    layer: 'D0', filled: 1,
    controls() {
        segs($('#v12-pick'), ML.map(l => ({ key: l.key, name: l.name })),
            k => { this.layer = k; pressSeg($('#v12-pick'), k); cancel(this); this.filled = 1; this.recolour(true); }, this.layer);
    },
    range(L) {
        this.rng = this.rng || {};
        if (!this.rng[L.key]) {
            const v = Object.values(MM.districts).map(d => mmVal(d, L)).filter(x => x != null).sort((a, b) => a - b);
            this.rng[L.key] = [v[Math.floor(v.length * .02)], v[Math.floor(v.length * .98)]];
        }
        return this.rng[L.key];
    },
    async render() {
        const host = $('#v12'); if (!host || !MM) return;
        const geo = await loadGeo();
        const W = host.clientWidth || 900, nar = W < 560;
        const H = Math.round(Math.min(nar ? 440 : 620, W * 1.08));
        const svg = svgFor(host, W, H);
        this.nod = hatchDef(svg, 'h12' + Math.random().toString(36).slice(2, 7), 'rgba(255,255,255,0.22)', 1, 5);
        const P = projectGeo(geo, W, H, 46);
        this.paths = [];
        const gD = E('g', {}, svg);
        geo.features.forEach(f => {
            const pr = f.properties, d = MM.districts[pr.k];
            const el = E('path', { d: P.fpath(f), stroke: 'rgba(11,11,12,0.8)', 'stroke-width': 0.35 }, gD);
            el.style.fill = NODATA;
            tipOn(el, () => {
                const L = ML.find(l => l.key === this.layer), v = mmVal(d, L);
                return `<strong>${pr.n}</strong><br>${pr.s}<br>` + (v == null ? 'no usable record' : `${L.name}: ${L.fmt(v)}`);
            });
            el.addEventListener('mouseenter', () => { el.setAttribute('stroke', TX); el.setAttribute('stroke-width', 1.1); });
            el.addEventListener('mouseleave', () => { el.setAttribute('stroke', 'rgba(11,11,12,0.8)'); el.setAttribute('stroke-width', 0.35); hideTip(); });
            this.paths.push({ el, d });
        });
        stateOutlines(svg, P);
        const kw = Math.min(260, W * .55), kx0 = W - kw - 8, ky = H - 30;
        const gid = 'g12' + Math.random().toString(36).slice(2, 7);
        const lg = E('linearGradient', { id: gid, x1: 0, x2: 1, y1: 0, y2: 0 }, E('defs', {}, svg));
        for (let i = 0; i <= 10; i++) E('stop', { offset: i * 10 + '%', 'stop-color': ramp(i / 10) }, lg);
        E('rect', { x: kx0, y: ky, width: kw, height: 9, fill: `url(#${gid})`, stroke: 'rgba(255,255,255,0.18)', 'stroke-width': .6 }, svg);
        const fm = { 'font-size': 10.5, 'font-family': MONO, fill: TX3 };
        this.kLo = T(svg, Object.assign({ x: kx0, y: ky - 6, 'text-anchor': 'start' }, fm), '');
        this.kHi = T(svg, Object.assign({ x: kx0 + kw, y: ky - 6, 'text-anchor': 'end' }, fm), '');
        this.kLab = T(svg, Object.assign({ x: kx0 + kw, y: ky + 23, 'text-anchor': 'end' }, fm), '');
        this.recolour(false);
    },
    fillOf(d, L) {
        const v = mmVal(d, L); if (v == null) return this.nod;
        const r = this.range(L);
        return ramp(Math.max(0, Math.min(1, (v - r[0]) / (r[1] - r[0] || 1))));
    },
    recolour(anim) {
        if (!this.paths) return;
        const L = ML.find(l => l.key === this.layer), r = this.range(L);
        this.kLo.textContent = L.kf(r[0]); this.kHi.textContent = L.kf(r[1]);
        this.kLab.textContent = L.kl;
        const lim = this.filled >= 1 ? Infinity : r[0] + (r[1] - r[0]) * this.filled;
        this.paths.forEach(p => {
            p.el.style.transition = anim && !RM ? 'fill .4s ease' : 'none';
            const v = mmVal(p.d, L);
            p.el.style.fill = v == null ? this.nod : (v <= lim ? this.fillOf(p.d, L) : NODATA);
        });
    },
    prep() { this.filled = 0; this.recolour(false); },
    play() {
        this.filled = 0; this.recolour(false);
        tween(this, 1400, t => { this.filled = t; this.recolour(false); });
    },
});

/* ================================================================
   new scene: states, every district as a dot in its state's row
   ================================================================ */
SC[13] = scene({
    m: 'D1', u: 1,
    controls() {
        segs($('#v13-ctl'), [{ key: 'D1', name: 'effective number of crops' }, { key: 'D0', name: 'count of crops' }],
            k => { this.m = k; pressSeg($('#v13-ctl'), k); cancel(this); this.u = 1; this.render(); }, this.m);
    },
    render() {
        const host = $('#v13'); if (!host || !TY || !MM) return;
        const m = this.m, names = new Set(TY.states.map(s => s.state));
        const ds = Object.values(MM.districts).filter(d => names.has(d.s));
        const rows = TY.states.slice().sort((a, b) => b[m].mean - a[m].mean);
        const W = host.clientWidth || 800, nar = W < 560;
        const lw = nar ? 124 : 150, rowH = nar ? 19 : 21;
        const mg = { t: 8, r: nar ? 12 : 24, b: 40, l: lw };
        const H = mg.t + rows.length * rowH + mg.b;
        const svg = svgFor(host, W, H);
        const xm = Math.ceil(Math.max(...ds.map(d => d[m])) / 5) * 5;
        const X = lin(0, xm, mg.l, W - mg.r);
        niceTicks(0, xm, nar ? 4 : 6).forEach(v => {
            E('line', { x1: X(v), x2: X(v), y1: mg.t, y2: H - mg.b, stroke: GRID }, svg);
            T(svg, { x: X(v), y: H - mg.b + 15, 'text-anchor': 'middle', 'font-size': 10.5, 'font-family': MONO, fill: TX3 }, v);
        });
        T(svg, { x: W - mg.r, y: H - 6, 'text-anchor': 'end', 'font-size': 10.5, 'font-family': MONO, fill: TX3 },
            m === 'D1' ? 'effective number of crops, per district' : 'count of crops, per district');
        const allMean = ds.reduce((a, d) => a + d[m], 0) / ds.length;
        E('line', { x1: X(allMean), x2: X(allMean), y1: mg.t - 2, y2: H - mg.b, stroke: ZERO, 'stroke-dasharray': '3 3' }, svg);
        const Y = {};
        rows.forEach((r, i) => {
            const y = mg.t + i * rowH + rowH / 2; Y[r.state] = y;
            if (i % 2 === 0) E('rect', { x: mg.l, y: y - rowH / 2, width: W - mg.r - mg.l, height: rowH, fill: 'rgba(255,255,255,0.022)' }, svg);
            T(svg, { x: lw - 10, y: y + 3.5, 'text-anchor': 'end', 'font-size': nar ? 9.5 : 11, 'font-family': MONO, fill: TX2 }, r.state);
        });
        const rnd = mulberry(7);
        this.dots = ds.map(d => {
            const y = Y[d.s] + (rnd() - .5) * rowH * .5;
            const c = E('circle', { cx: X(d[m]), cy: y, r: nar ? 2.6 : 3, fill: 'rgba(64,214,240,0.55)' }, svg);
            tipOn(c, () => `<strong>${d.n}</strong><br>${d.s}<br>${m === 'D1' ? 'effective number of crops ' + d.D1.toFixed(1) : 'count of crops ' + d.D0.toFixed(1)}`);
            c.addEventListener('mouseleave', hideTip);
            return { c, y, y0: mg.t + rnd() * (H - mg.t - mg.b) };
        });
        this.ticks = rows.map(r => E('line', { x1: X(r[m].mean), x2: X(r[m].mean), y1: Y[r.state] - rowH * .42, y2: Y[r.state] + rowH * .42, stroke: TX, 'stroke-width': 2, 'stroke-linecap': 'round' }, svg));
        this.split(m);
        this.draw(this.u);
    },
    split(m) {
        const host = $('#v13-split'); if (!host) return;
        const v = TY.variance_split.vars.find(x => x.var === m);
        const W = host.clientWidth || 600, H = 44, svg = svgFor(host, W, H);
        const X = lin(0, 1, 0, W);
        this.bar = [
            E('rect', { x: 0, y: 4, width: 0, height: 14, rx: 2, fill: '#40D6F0' }, svg),
            E('rect', { x: X(v.between) + 2, y: 4, width: 0, height: 14, rx: 2, fill: 'rgba(237,237,235,0.24)' }, svg),
        ];
        this.barW = [X(v.between), W - X(v.between) - 2];
        T(svg, { x: 0, y: 36, 'font-size': 11, 'font-family': MONO, fill: TX }, `between states ${pct0(v.between)}`);
        T(svg, { x: W, y: 36, 'text-anchor': 'end', 'font-size': 11, 'font-family': MONO, fill: TX2 }, `within state ${pct0(v.within)}`);
    },
    draw(u) {
        if (!this.dots) return;
        const a = easeOut(Math.min(1, u * 1.25));
        this.dots.forEach(p => p.c.setAttribute('cy', lerp(p.y0, p.y, a)));
        const b = Math.max(0, (u - .55) / .45);
        this.ticks.forEach(t => t.setAttribute('opacity', b));
        if (this.bar) { this.bar[0].setAttribute('width', this.barW[0] * b); this.bar[1].setAttribute('width', Math.max(0, this.barW[1] * b)); }
    },
    prep() { this.u = 0; this.draw(0); },
    play() { this.u = 0; this.draw(0); tween(this, 1800, t => { this.u = t; this.draw(t); }); },
});

/* ================================================================
   new scene: six groups of district, map plus profile cards
   ================================================================ */
/* facility supply third: one hue, three lightness steps validated on the dark ground;
   many weekly haats: a dark hatch on top, so the two axes never rest on hue alone */
const BAND = { low: '#4A4282', mid: '#7A6AD6', high: '#BDB3F7' };
const TYPES = [
    { code: 'few', band: 'low', haat: false, name: 'Bottom third, few weekly haats' },
    { code: 'haat_only', band: 'low', haat: true, name: 'Bottom third, many weekly haats' },
    { code: 'mid', band: 'mid', haat: false, name: 'Middle third, few weekly haats' },
    { code: 'mid_haat', band: 'mid', haat: true, name: 'Middle third, many weekly haats' },
    { code: 'served', band: 'high', haat: false, name: 'Top third, few weekly haats' },
    { code: 'served_haat', band: 'high', haat: true, name: 'Top third, many weekly haats' },
];
const typeOf = c => TYPES.find(t => t.code === c);
SC[14] = scene({
    sel: null, u: 1,
    cards() {
        const host = $('#v14-cards'); if (!host || !TY) return;
        host.innerHTML = '';
        const P = TY.types;
        /* each row is a track across the six group averages: faint ticks for the
           other groups, a bright dot for this one, so the cards compare at a glance */
        const ROWS = [['D0', 'count of crops', v => v.toFixed(1)], ['D1', 'effective number of crops', v => v.toFixed(1)],
                      ['cereal_share', 'cereal share', pct0], ['pulse_share', 'pulse share', pct0]];
        const DOM = {};
        ROWS.forEach(([k]) => { const v = P.map(t => t[k]), lo = Math.min(...v), hi = Math.max(...v), pd = (hi - lo) * .12; DOM[k] = [lo - pd, hi + pd]; });
        const at = (k, v) => ((v - DOM[k][0]) / (DOM[k][1] - DOM[k][0]) * 100).toFixed(1) + '%';
        TYPES.forEach(T0 => {
            const t = P.find(x => x.code === T0.code); if (!t) return;
            const b = document.createElement('button');
            b.type = 'button'; b.className = 'tcard'; b.dataset.code = T0.code;
            b.setAttribute('aria-pressed', 'false');
            const sw = `<svg class="tcard__sw" width="22" height="22" aria-hidden="true"><defs><pattern id="sw-${T0.code}" width="4.5" height="4.5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="4.5" stroke="#0B0B0C" stroke-width="1.6"/></pattern></defs><rect width="22" height="22" rx="2" fill="${BAND[T0.band]}"/>${T0.haat ? `<rect width="22" height="22" rx="2" fill="url(#sw-${T0.code})"/>` : ''}</svg>`;
            const bars = ROWS.map(([k, l, f]) => `<span class="tcard__row"><span class="tcard__l">${l}</span><span class="tcard__trk">${P.filter(o => o !== t).map(o => `<b style="left:${at(k, o[k])}"></b>`).join('')}<i style="left:${at(k, t[k])}"></i></span><span class="tcard__v">${f(t[k])}</span></span>`).join('');
            const st = t.top_states.slice(0, 3).map(s => `${s.state} (${s.n})`).join(', ');
            b.innerHTML = `<span class="tcard__h">${sw}<span><span class="tcard__t">${T0.name}</span><span class="tcard__n">${t.n} districts, most in ${st}</span></span></span>${bars}`;
            b.addEventListener('click', () => this.pick(this.sel === T0.code ? null : T0.code));
            host.appendChild(b);
        });
    },
    pick(code) {
        this.sel = code;
        document.querySelectorAll('#v14-cards .tcard').forEach(b => b.setAttribute('aria-pressed', b.dataset.code === code ? 'true' : 'false'));
        this.draw(this.u);
    },
    async render() {
        const host = $('#v14'); if (!host || !TY || !MM) return;
        if (!$('#v14-cards').children.length) this.cards();
        const geo = await loadGeo();
        const W = host.clientWidth || 900, nar = W < 560;
        const H = Math.round(Math.min(nar ? 420 : 600, W * 1.05));
        const svg = svgFor(host, W, H);
        const sfx = Math.random().toString(36).slice(2, 7);
        this.hat = hatchDef(svg, 'h14' + sfx, 'rgba(11,11,12,0.9)', 1.6, 4.5);
        this.nod = hatchDef(svg, 'n14' + sfx, 'rgba(255,255,255,0.16)', 1, 5);
        const P = projectGeo(geo, W, H, 0);
        this.paths = [];
        const g1 = E('g', {}, svg), g2 = E('g', { 'pointer-events': 'none' }, svg);
        geo.features.forEach(f => {
            const pr = f.properties, d = TY.districts[pr.k], ty = d ? typeOf(d.t) : null;
            const dd = P.fpath(f);
            const el = E('path', { d: dd, stroke: 'rgba(11,11,12,0.8)', 'stroke-width': 0.35 }, g1);
            const ov = ty && ty.haat ? E('path', { d: dd, fill: this.hat, opacity: 0 }, g2) : null;
            tipOn(el, () => `<strong>${pr.n}</strong><br>${pr.s}<br>` + (ty ? `${ty.name}<br>${pct0(d.haat)} of rural people have a weekly haat in the village` : 'not among the districts compared'));
            el.addEventListener('mouseleave', hideTip);
            this.paths.push({ el, ov, ty });
        });
        stateOutlines(svg, P);
        this.draw(this.u);
    },
    draw(u) {
        if (!this.paths) return;
        const a = Math.min(1, u / .5), h = Math.max(0, (u - .55) / .45);
        this.paths.forEach(p => {
            if (!p.ty) { p.el.style.fill = this.nod; return; }
            const on = !this.sel || this.sel === p.ty.code;
            p.el.style.fill = BAND[p.ty.band];
            p.el.style.opacity = (on ? 1 : .12) * a;
            if (p.ov) p.ov.setAttribute('opacity', on ? h : 0);
        });
    },
    prep() { this.u = 0; this.draw(0); },
    play() { this.u = 0; this.draw(0); tween(this, 1800, t => { this.u = t; this.draw(t); }); },
});

/* ================================================================
   wiring
   ================================================================ */
const HOSTS = { 1: '#v1', 2: '#v2', 3: '#v3', 4: '#v4', 5: '#v5', 6: '#v6a', 7: '#v7', 8: '#v8b', 9: '#v9', 10: '#v10', 11: '#v11', 12: '#v12', 13: '#v13', 14: '#v14' };
let inited = false;

function startPlay(sc) {
    if (sc.played) return;
    sc.played = true;
    sc.play();
}

async function init() {
    let MP;
    [MK, MP, MM, TY] = await Promise.all([loadJSON('data/site_markets.json'), loadJSON('data/site_markets_pop.json'),
                                         loadJSON('data/site_market_map.json'), loadJSON('data/site_typology.json')]);
    /* the page reads population shares; village shares stay for the comparison toggle */
    ['descriptives', 'corr', 'solo', 'solo_village', 'interaction', 'scatter', 'n'].forEach(k => { MK[k] = MP[k]; });
    LADDER = MP.ladder;
    MP.descriptives.forEach(d => { SD[d.name] = d.sd; SDV[d.name] = d.sd_village; });
    Object.values(SC).forEach(sc => { if (sc.controls) sc.controls(); });
    tables();
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
        }), { threshold: 0.35 });
        Object.keys(HOSTS).forEach(k => {
            const h = $(HOSTS[k]); if (!h) return;
            h.dataset.scene = k; io.observe(h);
        });
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
