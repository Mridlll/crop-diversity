/* ================================================================
   IRRIGATION PAGE: scene visuals.
   Reads data/site_irrigation_story.json, data/site_irrigation_panel.json,
   data/site_irrigation_v2.json and data/districts.geojson.
   Depends on site.js for svgEl, showTip, hideTip, loadJSON, loadGeo.
   Helper code follows the markets template; everything is wrapped so
   no name collides with site.js globals.

   Each scene: render() draws the current state, prep() sets the
   pre-entrance state, play() runs the entrance once in view.
   drawAll() re-renders on resize. Under prefers-reduced-motion every
   tween resolves at once.
   ================================================================ */
(function () {
'use strict';

const RM = !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);

const CY = '#0E9BB5', VI = '#7C5CE0', GR = '#1FA35A', LT = '#EDEDEB', GY = '#7E7E7C';
const TX = '#EDEDEB', TX2 = '#B3B3B0', TX3 = '#7E7E7C', BG = '#0B0B0C';
const GRID = 'rgba(255,255,255,0.075)', ZERO = 'rgba(255,255,255,0.28)';
const OUTSIDE = '#2A2A2B';
const REG = ['#1F6E7E', '#0E9BB5', '#A9E4EE'];
const REGN = ['under 40% irrigated', '40 to 60% irrigated', 'over 60% irrigated'];

const $ = s => document.querySelector(s);
function E(tag, attrs, parent) { const e = svgEl(tag, attrs || {}); if (parent) parent.appendChild(e); return e; }
function T(parent, attrs, s) { const t = E('text', attrs, parent); t.textContent = s; return t; }
const sgn = (v, d) => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(d);
const fmtP = p => p < 0.0001 ? '< 0.0001' : (p < 0.001 ? p.toFixed(4) : p.toFixed(3));
const pct = (v, d) => (v * 100).toFixed(d || 0) + '%';
const lerp = (a, b, t) => a + (b - a) * t;
const ease = t => t < .5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
const easeOut = t => 1 - Math.pow(1 - t, 3);
const lin = (d0, d1, r0, r1) => v => r0 + (v - d0) / (d1 - d0 || 1) * (r1 - r0);

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
    for (let v = Math.ceil(a / step - 1e-9) * step; v <= b + 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
}
function rowLabel(g, x, y, s, maxW, fill, size) {
    const fs = size || 12, cw = fs * 0.62;
    const t = E('text', { x: x, y: y + 4, 'text-anchor': 'end', fill: fill || TX2, 'font-size': fs, class: 'mono' }, g);
    if (s.length * cw <= maxW || s.indexOf(' ') < 0) { t.textContent = s; return t; }
    const words = s.split(' '), lines = [];
    let cur = '';
    words.forEach(w => {
        const nx = cur ? cur + ' ' + w : w;
        if (nx.length * cw > maxW && cur) { lines.push(cur); cur = w; } else cur = nx;
    });
    if (cur) lines.push(cur);
    const L = lines.slice(0, 3), lh = 1.15 * fs;
    t.setAttribute('y', (y + 4 - (L.length - 1) * lh / 2).toFixed(1));
    L.forEach((ln, i) => { const sp = E('tspan', i ? { x: x, dy: lh.toFixed(1) } : { x: x }, t); sp.textContent = ln; });
    return t;
}
/* x and y gridlines with mono tick labels */
function grid(svg, X, Y, xt, yt, fx, fy, m, W, H, o) {
    o = o || {};
    const g = E('g', { class: 'mono' }, svg);
    yt.forEach(v => {
        E('line', { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), stroke: GRID }, g);
        T(g, { x: m.l - 7, y: Y(v) + 4, 'text-anchor': 'end', fill: TX3, 'font-size': 11 }, fy(v));
    });
    if (xt) xt.forEach(v => {
        if (o.xgrid !== false) E('line', { x1: X(v), x2: X(v), y1: m.t, y2: H - m.b, stroke: GRID }, g);
        if (o.xlab !== false) T(g, { x: X(v), y: H - m.b + 17, 'text-anchor': 'middle', fill: TX3, 'font-size': 11 }, fx(v));
    });
    return g;
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
function cancel(sc) { sc.run++; }
const scene = o => Object.assign({ run: 0, played: false }, o);

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
        b.addEventListener('click', () => { pressSeg(host, it.key); onPick(it.key); });
        host.appendChild(b);
    });
}
function pressSeg(host, key) {
    if (!host) return;
    [...host.querySelectorAll('.seg')].forEach(b => b.setAttribute('aria-pressed', b.dataset.key === key ? 'true' : 'false'));
}
/* pointer position in svg user units */
function svgPt(svg, ev) {
    const r = svg.getBoundingClientRect(), vb = svg.viewBox.baseVal;
    return [(ev.clientX - r.left) * vb.width / r.width, (ev.clientY - r.top) * vb.height / r.height];
}
function setRead(sel, html) { const e = $(sel); if (e) e.innerHTML = html; }

let D = null;
const SC = {};

/* ================================================================
   shared: within-state gaps by irrigation group (quintiles, deciles)
   rows: {q, lo, hi, irr, n, b, ci:[lo,hi], p}; the first row is the
   reference group at zero. Each group is a step across its irrigation
   range with a whisker at its mean share.
   ================================================================ */
let V = null;
const clamp01 = v => Math.max(0, Math.min(1, v));
const fmtPw = p => p == null ? '' : (p < 0.001 ? '< 0.001' : p.toFixed(3));
const ORD = ['least irrigated', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'most irrigated'];
function grpName(r, nb) {
    const word = nb === 5 ? 'quintile' : 'decile';
    if (r.q === 1) return 'least irrigated ' + word;
    if (r.q === nb) return 'most irrigated ' + word;
    return ORD[r.q - 1] + ' ' + word;
}
function binChart(sc, host, rows, o) {
    const W = host.clientWidth || 900, nar = W < 560;
    const H = nar ? 330 : 390;
    const m = { t: 26, r: 14, b: 44, l: nar ? 40 : 52 };
    const svg = sc.svg = svgFor(host, W, H);
    let lo = Math.min(0, ...rows.map(r => r.ci[0])), hi = Math.max(0, ...rows.map(r => r.ci[1]));
    if (o.dots) { lo = Math.min(lo, ...o.dots.map(d => d.y)); hi = Math.max(hi, ...o.dots.map(d => d.y)); }
    const pad = (hi - lo) * .07; lo -= pad; hi += pad;
    const X = lin(0, 1, m.l, W - m.r), Y = lin(lo, hi, H - m.b, m.t);
    const yt = niceTicks(lo, hi, nar ? 4 : 6), st = yt.length > 1 ? yt[1] - yt[0] : 1;
    rows.forEach((r, i) => { if (i % 2) E('rect', { x: X(r.lo), width: X(r.hi) - X(r.lo), y: m.t, height: H - m.t - m.b, fill: 'rgba(255,255,255,0.03)' }, svg); });
    grid(svg, X, Y, [0, .2, .4, .6, .8, 1], yt, v => pct(v), v => Math.abs(v) < 1e-9 ? '0' : sgn(v, st < 1 ? 1 : 0), m, W, H, { xgrid: false });
    E('line', { x1: m.l, x2: W - m.r, y1: Y(0), y2: Y(0), stroke: ZERO, 'stroke-width': 1.2 }, svg);
    T(svg, { class: 'mono', x: (m.l + W - m.r) / 2, y: H - 6, 'text-anchor': 'middle', fill: TX3, 'font-size': 11 }, o.xlab);
    T(svg, { class: 'mono', x: 4, y: 12, fill: TX3, 'font-size': 11 }, nar ? o.ylabN : o.ylab);
    Object.assign(sc, { X, Y, m, W, H, nar });
    sc.dotEls = null;
    if (o.dots) {
        const g = E('g', {}, svg);
        sc.dotEls = o.dots.map(d => E('circle', { cx: X(d.x).toFixed(1), cy: Y(d.y).toFixed(1), r: nar ? 2.3 : 2.7, fill: '#5E8F99', 'fill-opacity': .55 }, g));
        sc.ring = E('circle', { r: 6.5, fill: 'none', stroke: TX, 'stroke-width': 1.5, opacity: 0 }, svg);
    }
    sc.items = rows.map((r, i) => {
        const ref = i === 0, sig = r.p != null && r.p < .05;
        const g = E('g', {}, svg);
        const seg = E('line', { x1: X(r.lo), x2: X(r.hi), stroke: ref ? TX2 : CY, 'stroke-width': ref ? 1.6 : 2.6, 'stroke-linecap': 'round' }, g);
        const wh = ref ? null : E('line', { x1: X(r.irr), x2: X(r.irr), stroke: CY, 'stroke-width': 1.5, opacity: .85 }, g);
        const dot = E('circle', { cx: X(r.irr), r: nar ? 4.6 : 5.4, fill: sig ? CY : BG, stroke: ref ? TX2 : CY, 'stroke-width': 1.7 }, g);
        const lab = ref ? T(svg, { class: 'mono', x: X(r.irr), y: Y(0) - 11, 'text-anchor': 'middle', fill: TX3, 'font-size': 10.5 }, 'reference') : null;
        return { r, seg, wh, dot, lab, i };
    });
    /* hover: nearest district in the district view, else the group under the pointer */
    const pick = ev => {
        const [px, py] = svgPt(svg, ev);
        const xv = (px - m.l) / (W - m.l - m.r);
        if (o.dots) {
            let best = -1, bd = 1e9;
            o.dots.forEach((d, i) => { const dx = X(d.x) - px, dy = Y(d.y) - py, dd = dx * dx + dy * dy; if (dd < bd) { bd = dd; best = i; } });
            if (best >= 0 && bd < 400) {
                const d = o.dots[best];
                sc.ring.setAttribute('cx', X(d.x)); sc.ring.setAttribute('cy', Y(d.y)); sc.ring.setAttribute('opacity', 1);
                o.onDot(d); return;
            }
            sc.ring.setAttribute('opacity', 0);
        }
        if (xv < -0.02 || xv > 1.02) { o.onRead(null); return; }
        const r = rows.find(z => xv <= z.hi + 0.002) || rows[rows.length - 1];
        o.onRead(r);
    };
    svg.addEventListener('mousemove', pick);
    svg.addEventListener('click', pick);
    svg.addEventListener('mouseleave', () => { if (sc.ring) sc.ring.setAttribute('opacity', 0); o.onRead(null); });
}
function placeBins(sc, t) {
    if (!sc.items) return;
    const Y = sc.Y, n = sc.items.length;
    if (sc.dotEls) sc.dotEls.forEach(el => el.setAttribute('opacity', clamp01(t * 2)));
    sc.items.forEach(it => {
        const e = easeOut(clamp01(t * 1.5 - it.i * (0.5 / n)));
        const r = it.r, y = lerp(Y(0), Y(r.b), e);
        it.seg.setAttribute('y1', y.toFixed(1)); it.seg.setAttribute('y2', y.toFixed(1));
        it.dot.setAttribute('cy', y.toFixed(1));
        it.seg.setAttribute('opacity', e > 0 ? 1 : 0); it.dot.setAttribute('opacity', e > 0 ? 1 : 0);
        if (it.wh) {
            it.wh.setAttribute('y1', lerp(Y(0), Y(r.ci[0]), e).toFixed(1)); it.wh.setAttribute('y2', lerp(Y(0), Y(r.ci[1]), e).toFixed(1));
            it.wh.setAttribute('opacity', e > 0 ? .85 : 0);
        }
    });
}
function binText(r, nb, unit) {
    return `<b>${grpName(r, nb)}</b>, ${pct(r.lo)} to ${pct(r.hi)} irrigated <span class="dim">·</span> ${r.n} districts` +
        (r.q === 1 ? ' <span class="dim">· the reference</span>'
            : ` <span class="dim">·</span> <b>${sgn(r.b, 2)}</b> ${unit} <span class="dim">(95% interval ${sgn(r.ci[0], 2)} to ${sgn(r.ci[1], 2)}; p ${fmtPw(r.p)})</span>`);
}

/* ================================================================
   01 the shape: quintiles, deciles, every district
   ================================================================ */
SC[1] = scene({
    view: 'q5', u: 1,
    controls() {
        segs($('#v1-view'), [{ key: 'q5', name: 'quintiles' }, { key: 'q10', name: 'deciles' }, { key: 'dist', name: 'every district' }],
            k => { this.view = k; cancel(this); this.played = true; this.render(); this.play(); }, this.view);
    },
    rows() { return this.view === 'q10' ? V.tenths.rows : V.fifths.full.rows; },
    render() {
        const host = $('#v1'); if (!host || !V) return;
        const nb = this.view === 'q10' ? 10 : 5;
        binChart(this, host, this.rows(), {
            xlab: 'share of sown area irrigated, 2019–20',
            ylab: 'effective number of crops, against the least irrigated group in the same state',
            ylabN: 'effective crops vs least irrigated, same state',
            dots: this.view === 'dist' ? V.districts : null,
            onRead: r => r ? setRead('#v1-read', binText(r, nb, 'effective crops')) : this.readDefault(),
            onDot: d => setRead('#v1-read', `<b>${d.n}</b>, ${d.s} <span class="dim">·</span> ${pct(d.x)} irrigated <span class="dim">·</span> ` +
                `${sgn(d.y, 2)} effective crops against the least irrigated quintile of its state`),
        });
        this.readDefault();
        placeBins(this, this.u);
    },
    readDefault() {
        const rs = this.rows(), top = rs[rs.length - 1];
        setRead('#v1-read', `<span class="dim">Hover or tap a group.</span> Most irrigated ${this.view === 'q10' ? 'decile' : 'quintile'}, ` +
            `above ${pct(top.lo)} irrigated: <b>${sgn(top.b, 2)}</b> effective crops <span class="dim">(p ${fmtPw(top.p)})</span>`);
    },
    prep() { this.u = 0; placeBins(this, 0); },
    play() { this.u = 0; placeBins(this, 0); tween(this, 1300, t => { this.u = t; placeBins(this, t); }, () => { this.u = 1; placeBins(this, 1); }); },
});

/* ================================================================
   02 ten groups: two stacked panels
   ================================================================ */
const TOPS = { D1: { name: 'effective number of crops', d: 2 }, D2: { name: 'number of main crops', d: 2 }, E: { name: 'evenness', d: 3 } };
function panelDom(rows, k) {
    const lo = Math.min(...rows.map(r => r[k] - r[k + 'se'])), hi = Math.max(...rows.map(r => r[k] + r[k + 'se']));
    const pad = (hi - lo) * .12;
    return [lo - pad, hi + pad];
}
SC[2] = scene({
    top: 'D1',
    controls() {
        segs($('#v2-tab'), Object.keys(TOPS).map(k => ({ key: k, name: TOPS[k].name })), k => { this.top = k; cancel(this); this.played = true; this.render(); }, this.top);
    },
    render() {
        const host = $('#v2'); if (!host || !D) return;
        const W = host.clientWidth || 900, nar = W < 560;
        const m = { l: nar ? 38 : 50, r: 14 };
        const hT = nar ? 190 : 220, hB = nar ? 150 : 170, gap = 46, top0 = 26;
        const H = top0 + hT + gap + hB + 40;
        const svg = this.svg = svgFor(host, W, H);
        const P = D.profile;
        const X = lin(0, 1, m.l, W - m.r);
        this.marks = [];
        const panels = [
            { k: this.top, y0: top0, h: hT, col: CY, name: TOPS[this.top].name, d: TOPS[this.top].d },
            { k: 'D0', y0: top0 + hT + gap, h: hB, col: LT, name: 'count of crops', d: 1 },
        ];
        panels.forEach((pn, pi) => {
            const dom = panelDom(P, pn.k);
            const Y = lin(dom[0], dom[1], pn.y0 + pn.h, pn.y0);
            const mm = { l: m.l, r: m.r, t: pn.y0, b: H - (pn.y0 + pn.h) };
            grid(svg, X, Y, [0, .2, .4, .6, .8, 1], niceTicks(dom[0], dom[1], 4), v => pct(v),
                v => pn.d === 3 ? v.toFixed(2) : (Math.abs(v - Math.round(v)) < 1e-9 ? v.toFixed(0) : v.toFixed(1)), mm, W, H, { xlab: pi === 1 });
            T(svg, { class: 'mono', x: m.l, y: pn.y0 - 9, fill: pn.col, 'font-size': 11.5 }, pn.name);
            const path = E('path', { fill: 'none', stroke: pn.col, 'stroke-width': 1.6, opacity: .75,
                d: P.map((r, i) => (i ? 'L' : 'M') + X(r.irr).toFixed(1) + ',' + Y(r[pn.k]).toFixed(1)).join('') }, svg);
            const len = path.getTotalLength ? path.getTotalLength() : 0;
            path.setAttribute('stroke-dasharray', len); path.setAttribute('stroke-dashoffset', 0);
            const pts = P.map(r => {
                const g = E('g', {}, svg);
                E('line', { x1: X(r.irr), x2: X(r.irr), y1: Y(r[pn.k] - r[pn.k + 'se']), y2: Y(r[pn.k] + r[pn.k + 'se']), stroke: pn.col, 'stroke-width': 1.2, opacity: .7 }, g);
                E('circle', { cx: X(r.irr), cy: Y(r[pn.k]), r: nar ? 3.6 : 4.4, fill: pn.col, stroke: BG, 'stroke-width': 1.2 }, g);
                return g;
            });
            this.marks.push({ path, len, pts });
        });
        T(svg, { class: 'mono', x: (m.l + W - m.r) / 2, y: H - 4, 'text-anchor': 'middle', fill: TX3, 'font-size': 11 }, 'mean share of sown area irrigated in the decile');
        /* hover bands */
        this.hl = E('rect', { y: top0 - 4, height: H - 40 - top0 + 4, fill: 'rgba(255,255,255,0.05)', opacity: 0 }, svg);
        svg.insertBefore(this.hl, svg.firstChild);
        const edges = P.map((r, i) => i ? (P[i - 1].irr + r.irr) / 2 : 0).concat([1]);
        P.forEach((r, i) => {
            const x0 = X(edges[i]), x1 = X(edges[i + 1]);
            const hit = E('rect', { x: x0, width: x1 - x0, y: top0 - 4, height: H - 40 - top0 + 4, fill: 'transparent' }, svg);
            const f = () => { this.hl.setAttribute('x', x0); this.hl.setAttribute('width', x1 - x0); this.hl.setAttribute('opacity', 1); this.read(r); };
            hit.addEventListener('mousemove', f); hit.addEventListener('click', f);
            hit.addEventListener('mouseleave', () => { this.hl.setAttribute('opacity', 0); this.read(null); });
        });
        this.read(null);
        this.show(1);
    },
    read(r) {
        if (!r) { setRead('#v2-read', '<span class="dim">Hover or tap a decile.</span> Highest effective number of crops in the raw averages: <b>5.89</b> in decile 4, 44% irrigated.'); return; }
        setRead('#v2-read', `<b>decile ${r.dec}</b> <span class="dim">·</span> ${r.n} districts <span class="dim">·</span> ${pct(r.irr, 1)} irrigated ` +
            `<span class="dim">(${pct(r.lo, 1)} to ${pct(r.hi, 1)})</span><br>count of crops ${r.D0.toFixed(1)} <span class="dim">·</span> effective number ${r.D1.toFixed(2)} ` +
            `<span class="dim">·</span> number of main crops ${r.D2.toFixed(2)} <span class="dim">·</span> evenness ${r.E.toFixed(3)} <span class="dim">·</span> cereal share ${r.cereal.toFixed(3)}`);
    },
    show(t) {
        (this.marks || []).forEach(mk => {
            mk.path.setAttribute('stroke-dashoffset', (mk.len * (1 - t)).toFixed(1));
            mk.pts.forEach((g, i) => g.setAttribute('opacity', Math.max(0, Math.min(1, t * 10 - i * 0.9))));
        });
    },
    prep() { this.show(0); },
    play() { this.show(0); tween(this, 1300, t => this.show(t)); },
});

/* ================================================================
   03 map
   ================================================================ */
function projectGeo(geo, W, H) {
    let x0 = 180, x1 = -180, y0 = 90, y1 = -90;
    const walk = c => {
        if (typeof c[0] === 'number') { x0 = Math.min(x0, c[0]); x1 = Math.max(x1, c[0]); y0 = Math.min(y0, c[1]); y1 = Math.max(y1, c[1]); }
        else c.forEach(walk);
    };
    geo.features.forEach(f => walk(f.geometry.coordinates));
    const pad = 6, kx = Math.cos((y0 + y1) / 2 * Math.PI / 180);
    const s = Math.min((W - pad * 2) / ((x1 - x0) * kx), (H - pad * 2) / (y1 - y0));
    const ox = pad + ((W - pad * 2) - (x1 - x0) * kx * s) / 2, oy = pad + ((H - pad * 2) - (y1 - y0) * s) / 2;
    const PX = v => ox + (v - x0) * kx * s, PY = v => oy + (y1 - v) * s;
    const ring = c => c.map((p, i) => (i ? 'L' : 'M') + PX(p[0]).toFixed(1) + ',' + PY(p[1]).toFixed(1)).join('') + 'Z';
    const path = (c, dd) => dd === 1 ? ring(c) : c.map(k => path(k, dd - 1)).join('');
    return { ring, fpath: f => path(f.geometry.coordinates, f.geometry.type === 'Polygon' ? 2 : 3) };
}
const hex = h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16));
const RA = [16, 52, 60], RB = hex('#A9E4EE');
const ramp = t => { const c = RA.map((v, i) => Math.round(v + (RB[i] - v) * Math.max(0, Math.min(1, t)))); return `rgb(${c[0]},${c[1]},${c[2]})`; };
SC[3] = scene({
    mode: 'group', pick: null, u: 1,
    controls() {
        segs($('#v3-mode'), [{ key: 'group', name: 'three irrigation groups' }, { key: 'cont', name: 'irrigated share' }],
            k => { this.mode = k; this.keyHTML(); this.paint(); }, this.mode);
        this.keyHTML();
    },
    keyHTML() {
        const host = $('#v3-key'); if (!host || !D) return;
        host.innerHTML = '';
        D.regimes.ay.forEach(r => {
            const b = document.createElement('button');
            b.type = 'button'; b.className = 'irr-key__b';
            b.setAttribute('aria-pressed', this.pick === r.g ? 'true' : 'false');
            b.innerHTML = `<span class="irr-key__sw" style="background:${REG[r.g]}"></span><span class="irr-key__t">${REGN[r.g]}</span>` +
                `<span class="irr-key__n">${r.n} districts, ${pct(r.irr)} on average</span>` +
                `<span class="irr-key__v">effective number ${r.D1.toFixed(2)} · count ${r.D0.toFixed(1)}</span>`;
            const on = () => { this.hl = r.g; this.paint(); };
            b.addEventListener('mouseenter', on);
            b.addEventListener('focus', on);
            b.addEventListener('mouseleave', () => { this.hl = null; this.paint(); });
            b.addEventListener('blur', () => { this.hl = null; this.paint(); });
            b.addEventListener('click', () => {
                this.pick = this.pick === r.g ? null : r.g;
                [...host.querySelectorAll('.irr-key__b')].forEach((x, i) => x.setAttribute('aria-pressed', this.pick === i ? 'true' : 'false'));
                this.paint();
            });
            host.appendChild(b);
        });
        const n = document.createElement('p'); n.className = 'irr-key__ramp';
        n.innerHTML = this.mode === 'cont' ? 'Shading runs from 4% irrigated (darkest) to 98% (lightest). The groups above still pick out districts.'
                                           : `<span style="display:inline-block;width:10px;height:10px;background:${OUTSIDE};border:1px solid rgba(255,255,255,.25);vertical-align:-1px;margin-right:.4rem"></span>no usable record`;
        host.appendChild(n);
    },
    async render() {
        const host = $('#v3'); if (!host || !D) return;
        const geo = await loadGeo();
        const W = host.clientWidth || 900, nar = W < 560;
        const H = Math.round(Math.min(nar ? 470 : 660, W * 1.1));
        const svg = svgFor(host, W, H);
        const P = projectGeo(geo, W, H);
        const byK = {}; D.districts.forEach(d => { byK[d.k] = d; });
        this.paths = [];
        const gD = E('g', {}, svg);
        geo.features.forEach(f => {
            const pr = f.properties, ok = D.geo_by_name[pr.n + '|' + pr.s], d = ok ? byK[ok] : byK[pr.k];
            const el = E('path', { d: P.fpath(f), stroke: d ? 'rgba(11,11,12,0.75)' : 'rgba(255,255,255,0.16)', 'stroke-width': d ? .35 : .4 }, gD);
            el.style.fill = OUTSIDE;
            const tip = ev => showTip(d ? `<strong>${d.n}</strong><br>${d.s}<br>irrigated ${pct(d.x)} of sown area<br>canal main source in ${pct(d.can)} of villages` +
                `<br>effective number of crops ${d.D1.toFixed(2)}<br>count of crops ${d.D0.toFixed(1)}`
                : `<strong>${pr.n}</strong><br>${pr.s}<br>no usable record`, ev);
            el.addEventListener('mousemove', tip); el.addEventListener('click', tip);
            el.addEventListener('mouseenter', () => { el.setAttribute('stroke', TX); el.setAttribute('stroke-width', 1.1); });
            el.addEventListener('mouseleave', () => { el.setAttribute('stroke', d ? 'rgba(11,11,12,0.75)' : 'rgba(255,255,255,0.16)'); el.setAttribute('stroke-width', d ? .35 : .4); hideTip(); });
            this.paths.push({ el, d });
        });
        const gS = E('g', { 'pointer-events': 'none' }, svg);
        D.state_rings.forEach(r => E('path', { d: P.ring(r), fill: 'none', stroke: 'rgba(237,237,235,0.42)', 'stroke-width': .7, 'stroke-linejoin': 'round' }, gS));
        this.paint();
    },
    paint() {
        const u = this.u, act = this.hl != null ? this.hl : this.pick;
        (this.paths || []).forEach(p => {
            if (!p.d) return;
            const on = p.d.x <= u * 1.02 + 0.001;
            p.el.style.fill = on ? (this.mode === 'group' ? REG[p.d.g] : ramp((p.d.x - 0.04) / 0.94)) : OUTSIDE;
            p.el.style.opacity = act == null || act === p.d.g ? 1 : .18;
        });
    },
    prep() { this.u = 0; this.paint(); },
    play() { this.u = 0; this.paint(); tween(this, 1600, t => { this.u = t; this.paint(); }, () => { this.u = 1; this.paint(); }); },
});

/* ================================================================
   04 crop-group shares, three panels
   ================================================================ */
const SHARES = [
    { k: 'cereal', name: 'cereal share', col: CY, dom: [0, 1], ticks: [0, .25, .5, .75, 1] },
    { k: 'pulse', name: 'pulse share', col: VI, dom: [0, .25], ticks: [0, .05, .1, .15, .2, .25] },
    { k: 'oilseed', name: 'oilseed share', col: GR, dom: [0, .25], ticks: [0, .05, .1, .15, .2, .25] },
];
SC[4] = scene({
    render() {
        const host = $('#v4'); if (!host || !D) return;
        const W = host.clientWidth || 900, nar = W < 700;
        const P = D.profile;
        const ph = nar ? 170 : 250, gapY = 52, gapX = 34;
        const cols = nar ? 1 : 3, rows = nar ? 3 : 1;
        const H = 24 + rows * ph + (rows - 1) * gapY + 42;
        const svg = svgFor(host, W, H);
        const pw = (W - (cols - 1) * gapX) / cols;
        this.marks = [];
        this.hls = [];
        SHARES.forEach((S, si) => {
            const cx = nar ? 0 : si * (pw + gapX), cy = 24 + (nar ? si * (ph + gapY) : 0);
            const m = { l: cx + 40, r: W - (cx + pw) + 18, t: cy, b: H - (cy + ph) };
            const X = lin(0, 1, m.l, cx + pw - 18), Y = lin(S.dom[0], S.dom[1], cy + ph, cy);
            const x8 = X((P[6].irr + P[7].irr) / 2);
            const hl = E('rect', { x: X(0), width: 0, y: cy, height: ph, fill: 'rgba(255,255,255,0.06)', opacity: 0 }, svg);
            this.hls.push(hl);
            E('rect', { x: x8, width: X(1) - x8, y: cy, height: ph, fill: 'rgba(237,237,235,0.05)' }, svg);
            grid(svg, X, Y, [0, .5, 1], S.ticks, v => pct(v), v => v.toFixed(2), m, W, H, { xlab: true });
            T(svg, { class: 'mono', x: m.l, y: cy - 9, fill: S.col, 'font-size': 11.5 }, S.name);
            const path = E('path', { fill: 'none', stroke: S.col, 'stroke-width': 1.6, opacity: .75,
                d: P.map((r, i) => (i ? 'L' : 'M') + X(r.irr).toFixed(1) + ',' + Y(r[S.k]).toFixed(1)).join('') }, svg);
            const len = path.getTotalLength ? path.getTotalLength() : 0;
            path.setAttribute('stroke-dasharray', len);
            const pts = P.map(r => {
                const g = E('g', {}, svg);
                E('line', { x1: X(r.irr), x2: X(r.irr), y1: Y(r[S.k] - r[S.k + 'se']), y2: Y(r[S.k] + r[S.k + 'se']), stroke: S.col, 'stroke-width': 1.2, opacity: .7 }, g);
                E('circle', { cx: X(r.irr), cy: Y(r[S.k]), r: 4, fill: S.col, stroke: BG, 'stroke-width': 1.2 }, g);
                return g;
            });
            this.marks.push({ path, len, pts });
            const edges = P.map((r, i) => i ? (P[i - 1].irr + r.irr) / 2 : 0).concat([1]);
            P.forEach((r, i) => {
                const hit = E('rect', { x: X(edges[i]), width: X(edges[i + 1]) - X(edges[i]), y: cy, height: ph, fill: 'transparent' }, svg);
                const f = () => { this.hls.forEach(h => { h.setAttribute('opacity', 0); }); this.focus(i); };
                hit.addEventListener('mousemove', f); hit.addEventListener('click', f);
                hit.addEventListener('mouseleave', () => { this.unfocus(); });
            });
            this.hls[si].dataset.edges = JSON.stringify(edges.map(e => X(e)));
        });
        T(svg, { class: 'mono', x: W / 2, y: H - 4, 'text-anchor': 'middle', fill: TX3, 'font-size': 11 }, 'mean share of sown area irrigated in the decile');
        this.unfocus();
        this.show(1);
    },
    focus(i) {
        const r = D.profile[i];
        this.hls.forEach(h => { const e = JSON.parse(h.dataset.edges); h.setAttribute('x', e[i]); h.setAttribute('width', e[i + 1] - e[i]); h.setAttribute('opacity', 1); });
        setRead('#v4-read', `<b>decile ${r.dec}</b>, ${pct(r.irr)} irrigated <span class="dim">·</span> cereal ${r.cereal.toFixed(3)} <span class="dim">·</span> pulse ${r.pulse.toFixed(3)} ` +
            `<span class="dim">·</span> oilseed ${r.oilseed.toFixed(3)} <span class="dim">·</span> effective number of crops ${r.D1.toFixed(2)}`);
    },
    unfocus() {
        (this.hls || []).forEach(h => h.setAttribute('opacity', 0));
        setRead('#v4-read', '<span class="dim">Hover or tap a decile.</span> Cereal share is lowest in decile 7 at <b>0.583</b> and highest in decile 10 at <b>0.794</b>.');
    },
    show(t) {
        (this.marks || []).forEach(mk => {
            mk.path.setAttribute('stroke-dashoffset', (mk.len * (1 - t)).toFixed(1));
            mk.pts.forEach((g, i) => g.setAttribute('opacity', Math.max(0, Math.min(1, t * 10 - i * 0.9))));
        });
    },
    prep() { this.show(0); },
    play() { this.show(0); tween(this, 1300, t => this.show(t)); },
});

/* ================================================================
   05 robustness strip
   ================================================================ */
SC[5] = scene({
    mode: 'b2', u: 1,
    render() {
        const host = $('#v5'); if (!host || !D) return;
        const W = host.clientWidth || 900, nar = W < 560;
        const rows = D.robustness.filter(r => r.y === 'D1');
        const rh = nar ? 34 : 30, lw = nar ? 150 : 360;
        const groups = [['effective number of crops', 0]];
        const H = 16 + rows.length * rh + groups.length * 24 + 40;
        const m = { l: lw + 14, r: 18, t: 10, b: 40 };
        const svg = svgFor(host, W, H);
        const X = lin(-16, 0, m.l, W - m.r), xt = niceTicks(-16, 0, nar ? 4 : 8), fx = v => v.toFixed(0);
        this.X = X;
        const g0 = E('g', { class: 'mono' }, svg);
        xt.forEach(v => {
            E('line', { x1: X(v), x2: X(v), y1: m.t, y2: H - m.b, stroke: v === 0 && this.mode === 'b2' ? ZERO : GRID }, g0);
            T(g0, { x: X(v), y: H - m.b + 17, 'text-anchor': 'middle', fill: TX3, 'font-size': 11 }, fx(v));
        });
        T(svg, { class: 'mono', x: (m.l + W - m.r) / 2, y: H - 4, 'text-anchor': 'middle', fill: TX3, 'font-size': 11 },
            nar ? 'squared term, 95% interval' : 'squared term on irrigation, with 95% interval');
        this.items = [];
        let y = m.t + 6;
        rows.forEach((r, i) => {
            const gh = groups.find(g => g[1] === i);
            if (gh) { y += 18; T(svg, { class: 'mono', x: 0, y: y - 4, fill: TX3, 'font-size': 10.5, 'letter-spacing': '.08em' }, gh[0].toUpperCase()); y += 6; }
            const cy = y + rh / 2;
            const col = r.y === 'D1' ? CY : VI;
            const lab = r.lab.replace(' (headline)', ', main estimate') + (r.n !== 606 ? ` (${r.n})` : '');
            rowLabel(svg, lw, cy, lab, lw - 4, i === 1 ? TX : TX2, nar ? 10.5 : 11.5);
            E('line', { x1: m.l, x2: W - m.r, y1: cy, y2: cy, stroke: 'rgba(255,255,255,0.04)' }, svg);
            const g = E('g', {}, svg);
            let wl = null;
            if (this.mode === 'b2') wl = E('line', { y1: cy, y2: cy, stroke: col, 'stroke-width': 1.6, opacity: .8 }, g);
            const v = r.b2;
            const dot = E('circle', { cy, r: 5, fill: r.p2 < .05 ? col : 'none', stroke: col, 'stroke-width': 1.5 }, g);
            let ring = null;
            if (i === 1) ring = E('circle', { cy, r: 8.5, fill: 'none', stroke: TX, 'stroke-width': 1.2 }, g);
            const hit = E('rect', { x: 0, y: y, width: W, height: rh, fill: 'transparent' }, svg);
            const f = () => setRead('#v5-read', `<b>${r.lab}</b> <span class="dim">·</span> ${r.n} districts <span class="dim">·</span> squared term ${sgn(r.b2, 3)} ` +
                `<span class="dim">(95% interval ${sgn(r.b2 - 1.96 * r.se2, 2)} to ${sgn(r.b2 + 1.96 * r.se2, 2)}; p ${fmtP(r.p2)})</span>`);
            hit.addEventListener('mousemove', f); hit.addEventListener('click', f);
            hit.addEventListener('mouseleave', () => this.readDefault());
            this.items.push({ r, v, dot, ring, wl, i });
            y += rh;
        });
        this.readDefault();
        this.place(this.u);
    },
    readDefault() {
        setRead('#v5-read', '<span class="dim">Hover or tap a row.</span> Main estimate <b>−6.84</b>, from −5.44 to −10.42 across the eight versions.');
    },
    place(t) {
        const X = this.X;
        (this.items || []).forEach(it => {
            const k = Math.max(0, Math.min(1, t * 1.6 - it.i * 0.045));
            const e = easeOut(k);
            const x0 = X(0);
            const x = lerp(x0, X(it.v), e);
            it.dot.setAttribute('cx', x.toFixed(1));
            it.dot.setAttribute('opacity', k > 0 ? 1 : 0);
            if (it.ring) { it.ring.setAttribute('cx', x.toFixed(1)); it.ring.setAttribute('opacity', k >= 1 ? 1 : 0); }
            if (it.wl) {
                const lo = it.r.b2 - 1.96 * it.r.se2, hi = it.r.b2 + 1.96 * it.r.se2;
                it.wl.setAttribute('x1', lerp(x, X(Math.max(lo, -16)), e)); it.wl.setAttribute('x2', lerp(x, X(Math.min(hi, 0)), e));
                it.wl.setAttribute('opacity', k > 0 ? .8 : 0);
            }
        });
    },
    prep() { this.u = 0; this.place(0); },
    play() { this.u = 0; this.place(0); tween(this, 1300, t => { this.u = t; this.place(t); }, () => { this.u = 1; this.place(1); }); },
});

/* ================================================================
   06 source mix bars
   ================================================================ */
const SRCS = [
    { k: 'ground', name: 'groundwater', col: CY, ink: BG },
    { k: 'canal', name: 'canal', col: VI, ink: TX },
    { k: 'surface', name: 'surface water', col: GR, ink: BG },
    { k: 'other', name: 'other', col: GY, ink: BG },
];
SC[6] = scene({
    u: 1,
    controls() {
        const l = $('#v6-leg'); if (!l) return;
        l.innerHTML = SRCS.map(s => `<span class="legend__item"><span class="legend__swatch" style="background:${s.col}"></span>${s.name}</span>`).join('');
    },
    render() {
        const host = $('#v6'); if (!host || !D) return;
        const W = host.clientWidth || 900, nar = W < 560;
        const lw = nar ? 92 : 150, bh = nar ? 22 : 24, gap = nar ? 9 : 10;
        const bars = [{ lab: 'all districts', d: D.source_mix, all: true }].concat(D.profile.map(r => ({ lab: nar ? `${r.dec} · ${pct(r.irr)}` : `decile ${r.dec} · ${pct(r.irr)} irr.`, d: r, r })));
        const H = 8 + bars.length * (bh + gap) + 18 + 34;
        const m = { l: lw + 10, r: 12 };
        const svg = svgFor(host, W, H);
        const X = lin(0, 1, m.l, W - m.r);
        [0, .25, .5, .75, 1].forEach(v => {
            E('line', { x1: X(v), x2: X(v), y1: 4, y2: H - 34, stroke: GRID }, svg);
            T(svg, { class: 'mono', x: X(v), y: H - 34 + 16, 'text-anchor': 'middle', fill: TX3, 'font-size': 11 }, pct(v));
        });
        T(svg, { class: 'mono', x: (m.l + W - m.r) / 2, y: H - 4, 'text-anchor': 'middle', fill: TX3, 'font-size': 11 }, 'share of villages naming each source as main');
        this.segsEl = [];
        let y = 8;
        bars.forEach((b, bi) => {
            if (bi === 1) y += 18;
            if (bi === 1) T(svg, { class: 'mono', x: 0, y: y - 8, fill: TX3, 'font-size': 10.5, 'letter-spacing': '.08em' }, nar ? 'DECILES BY IRRIGATION' : 'DECILES, LEAST TO MOST IRRIGATED');
            rowLabel(svg, lw, y + bh / 2, b.lab, lw, b.all ? TX : TX2, nar ? 10.5 : 11.5);
            const tot = SRCS.reduce((s, k) => s + b.d[k.k], 0);
            let acc = 0;
            SRCS.forEach(s => {
                const v = b.d[s.k] / tot;
                const rect = E('rect', { y, height: bh, fill: s.col, stroke: BG, 'stroke-width': 1 }, svg);
                const lab = v > .08 ? T(svg, { class: 'mono', y: y + bh / 2 + 4, 'text-anchor': 'middle', fill: s.ink, 'font-size': nar ? 10 : 11 }, pct(b.d[s.k])) : null;
                this.segsEl.push({ rect, lab, a: acc, v, X, bi });
                acc += v;
            });
            const hit = E('rect', { x: 0, y: y - gap / 2, width: W, height: bh + gap, fill: 'transparent' }, svg);
            const f = () => setRead('#v6-read', `<b>${b.all ? 'all districts' : 'decile ' + b.r.dec + ', ' + pct(b.r.irr) + ' irrigated'}</b> <span class="dim">·</span> ` +
                SRCS.map(s => `${s.name} ${pct(b.d[s.k], 1)}`).join(' <span class="dim">·</span> '));
            hit.addEventListener('mousemove', f); hit.addEventListener('click', f);
            hit.addEventListener('mouseleave', () => this.readDefault());
            y += bh + gap;
        });
        this.readDefault();
        this.grow(this.u);
    },
    readDefault() { setRead('#v6-read', '<span class="dim">Hover or tap a bar.</span> Groundwater rises from <b>23%</b> of villages in decile 1 to <b>78%</b> in decile 10.'); },
    grow(t) {
        (this.segsEl || []).forEach(s => {
            const k = easeOut(Math.max(0, Math.min(1, t * 1.5 - s.bi * 0.045)));
            const x0 = s.X(s.a * k), w = s.X((s.a + s.v) * k) - x0;
            s.rect.setAttribute('x', x0.toFixed(1)); s.rect.setAttribute('width', Math.max(0, w).toFixed(1));
            if (s.lab) { s.lab.setAttribute('x', (x0 + w / 2).toFixed(1)); s.lab.setAttribute('opacity', k >= .98 ? 1 : 0); }
        });
    },
    prep() { this.u = 0; this.grow(0); },
    play() { this.u = 0; this.grow(0); tween(this, 1300, t => { this.u = t; this.grow(t); }, () => { this.u = 1; this.grow(1); }); },
});

/* ================================================================
   07 source coefficients
   ================================================================ */
const OUTN = { D1: 'effective number of crops', D0: 'count of crops', D2: 'number of main crops', E: 'evenness',
               cereal: 'cereal share', pulse: 'pulse share', oilseed: 'oilseed share' };
const TABS7 = [{ key: 'n', name: 'count and effective numbers', ys: ['D1', 'D0', 'D2'] }, { key: 'e', name: 'evenness', ys: ['E'] },
               { key: 's', name: 'area shares', ys: ['cereal', 'pulse', 'oilseed'] }];
SC[7] = scene({
    tab: 'n', scale: 1, u: 1,
    controls() {
        segs($('#v7-tab'), TABS7.map(t => ({ key: t.key, name: t.name })), k => { this.tab = k; cancel(this); this.played = true; this.render(); this.play(); }, this.tab);
        segs($('#v7-scale'), [{ key: '1', name: 'every village' }, { key: '10', name: '10 points of villages' }],
            k => { this.scale = +k; cancel(this); this.played = true; this.render(); this.play(); }, '1');
    },
    render() {
        const host = $('#v7'); if (!host || !D) return;
        const W = host.clientWidth || 900, nar = W < 560;
        const tab = TABS7.find(t => t.key === this.tab);
        const rows = D.source.filter(r => tab.ys.includes(r.y));
        const rh = nar ? 64 : 58, lw = nar ? 120 : 260;
        const H = 10 + rows.length * rh + 44;
        const m = { l: lw + 14, r: 18, t: 10, b: 44 };
        const svg = svgFor(host, W, H);
        const sc = this.scale === 10 ? 0.1 : 1;
        const all = [];
        rows.forEach(r => ['canal', 'surface'].forEach(s => { const c = r[s]; all.push((c.b - 1.96 * c.se) * sc, (c.b + 1.96 * c.se) * sc); }));
        let lo = Math.min(0, ...all), hi = Math.max(0, ...all);
        const pad = (hi - lo) * .06; lo -= pad; hi += pad;
        const X = lin(lo, hi, m.l, W - m.r);
        const xt = niceTicks(lo, hi, nar ? 4 : 7);
        const dp = (hi - lo) < .2 ? 3 : ((hi - lo) < 2 ? 2 : 1);
        const st = xt.length > 1 ? xt[1] - xt[0] : 1, tdp = Math.max(0, -Math.floor(Math.log10(st) + 1e-9)) + (Math.abs(st / Math.pow(10, Math.floor(Math.log10(st))) - 2.5) < 1e-6 ? 1 : 0);
        xt.forEach(v => {
            E('line', { x1: X(v), x2: X(v), y1: m.t, y2: H - m.b, stroke: Math.abs(v) < 1e-9 ? ZERO : GRID }, svg);
            T(svg, { class: 'mono', x: X(v), y: H - m.b + 17, 'text-anchor': 'middle', fill: TX3, 'font-size': 11 }, Math.abs(v) < 1e-9 ? '0' : sgn(v, tdp));
        });
        E('line', { x1: X(0), x2: X(0), y1: m.t, y2: H - m.b, stroke: ZERO }, svg);
        T(svg, { class: 'mono', x: (m.l + W - m.r) / 2, y: H - 6, 'text-anchor': 'middle', fill: TX3, 'font-size': 11 },
            this.scale === 10 ? (nar ? 'per 10 points of villages' : 'difference for 10 more percentage points of villages') : (nar ? 'every village against none' : 'difference, every village against none'));
        this.items = [];
        rows.forEach((r, i) => {
            const y0 = m.t + i * rh, cy = y0 + rh / 2;
            rowLabel(svg, lw, cy, OUTN[r.y], lw - 4, TX2, nar ? 10.5 : 11.5);
            if (i) E('line', { x1: 0, x2: W - m.r, y1: y0, y2: y0, stroke: 'rgba(255,255,255,0.05)' }, svg);
            [['canal', VI, -10], ['surface', GR, 10]].forEach(([s, col, off]) => {
                const c = r[s], yy = cy + off;
                const wl = E('line', { y1: yy, y2: yy, stroke: col, 'stroke-width': 1.8 }, svg);
                const dot = E('circle', { cy: yy, r: 5.2, fill: c.p < .05 ? col : BG, stroke: col, 'stroke-width': 1.6 }, svg);
                this.items.push({ wl, dot, b: c.b * sc, lo: (c.b - 1.96 * c.se) * sc, hi: (c.b + 1.96 * c.se) * sc, X, i });
                const hit = E('rect', { x: 0, y: yy - 10, width: W, height: 20, fill: 'transparent' }, svg);
                const f = () => setRead('#v7-read', `<b>${OUTN[r.y]}</b>, ${s === 'canal' ? 'canal' : 'surface water'} <span class="dim">·</span> ${sgn(c.b * sc, dp + 1)} ` +
                    `<span class="dim">(95% interval ${sgn((c.b - 1.96 * c.se) * sc, dp + 1)} to ${sgn((c.b + 1.96 * c.se) * sc, dp + 1)})</span> <span class="dim">·</span> p ${fmtP(c.p)}`);
                hit.addEventListener('mousemove', f); hit.addEventListener('click', f);
                hit.addEventListener('mouseleave', () => this.readDefault());
            });
        });
        this.readDefault();
        this.place(this.u);
    },
    readDefault() {
        const s = D.source.find(r => r.y === 'D1').surface.b, c = D.source.find(r => r.y === 'cereal').surface.b;
        setRead('#v7-read', this.scale === 10
            ? `<span class="dim">Hover or tap a marker.</span> Surface water, 10 more percentage points of villages: <b>${sgn(s / 10, 2)}</b> on the effective number of crops, cereal share <b>${sgn(c / 10, 3)}</b>.`
            : `<span class="dim">Hover or tap a marker.</span> Surface water, every village against none: <b>${sgn(s, 2)}</b> on the effective number of crops, cereal share <b>${sgn(c, 2)}</b>.`);
    },
    place(t) {
        (this.items || []).forEach(it => {
            const e = easeOut(Math.max(0, Math.min(1, t * 1.4 - it.i * 0.1)));
            const X = it.X, x = lerp(X(0), X(it.b), e);
            it.dot.setAttribute('cx', x.toFixed(1));
            it.wl.setAttribute('x1', lerp(X(0), X(it.lo), e).toFixed(1)); it.wl.setAttribute('x2', lerp(X(0), X(it.hi), e).toFixed(1));
        });
    },
    prep() { this.u = 0; this.place(0); },
    play() { this.u = 0; this.place(0); tween(this, 1000, t => { this.u = t; this.place(t); }, () => { this.u = 1; this.place(1); }); },
});

/* ================================================================
   08 district characteristics dumbbell
   ================================================================ */
const DIMN = { canal: 'share of villages relying on canals', hold: 'mean holding size (log)', cult: 'cultivator share of agricultural workers',
               landless: 'landless share of rural households', mandi: 'share of villages with a mandi', haat: 'share of villages with a weekly haat',
               regular: 'share of villages with a regular market', fpo: 'share of villages with a producer organisation', cold: 'share of villages with a food storage warehouse',
               sc: 'Scheduled Caste population share', st: 'Scheduled Tribe population share', ci: 'cropping intensity' };
SC[8] = scene({
    mode: 'sd', u: 1,
    controls() {
        segs($('#v8-mode'), [{ key: 'sd', name: 'per standard deviation' }, { key: 'full', name: 'full range' }],
            k => { this.mode = k; cancel(this); this.played = true; this.render(); this.play(); }, this.mode);
    },
    render() {
        const host = $('#v8'); if (!host || !D) return;
        const W = host.clientWidth || 900, nar = W < 560;
        const sd = this.mode === 'sd';
        const rows = D.dims.filter(r => r.key !== 'irr').map(r => {
            const k = sd ? r.sd : 1;
            return Object.assign({}, r, { a: r.raw.b * k, j: r.adj.b * k, k });
        }).sort((p, q) => q.j - p.j);
        const rh = nar ? 36 : 30, lw = nar ? 150 : 300;
        const H = 30 + rows.length * rh + 44;
        const m = { l: lw + 14, r: 18, t: 30, b: 44 };
        const svg = svgFor(host, W, H);
        const vals = rows.flatMap(r => [r.a, r.j]);
        let lo = Math.min(0, ...vals), hi = Math.max(0, ...vals);
        const pad = (hi - lo) * .06; lo -= pad; hi += pad;
        const X = lin(lo, hi, m.l, W - m.r);
        const xt8 = niceTicks(lo, hi, nar ? 4 : 7), st8 = xt8.length > 1 ? xt8[1] - xt8[0] : 1;
        const d8 = Math.max(0, -Math.floor(Math.log10(st8) + 1e-9)) + (Math.abs(st8 / Math.pow(10, Math.floor(Math.log10(st8))) - 2.5) < 1e-6 ? 1 : 0);
        xt8.forEach(v => {
            E('line', { x1: X(v), x2: X(v), y1: m.t, y2: H - m.b, stroke: Math.abs(v) < 1e-9 ? ZERO : GRID }, svg);
            T(svg, { class: 'mono', x: X(v), y: H - m.b + 17, 'text-anchor': 'middle', fill: TX3, 'font-size': 11 }, Math.abs(v) < 1e-9 ? '0' : sgn(v, d8));
        });
        E('line', { x1: X(0), x2: X(0), y1: m.t, y2: H - m.b, stroke: ZERO }, svg);
        T(svg, { class: 'mono', x: (m.l + W - m.r) / 2, y: H - 6, 'text-anchor': 'middle', fill: TX3, 'font-size': 11 },
            nar ? (sd ? 'change per standard deviation' : 'change over the full range') : (sd ? 'change in the effective number of crops for one standard deviation' : 'change in the effective number of crops over the full range'));
        if (!sd) {
            const t = T(svg, { class: 'mono', x: W - m.r, y: 14, 'text-anchor': 'end', fill: TX2, 'font-size': 11 }, 'scales differ between rows');
            const bb = { w: 'scales differ between rows'.length * 6.6 + 14 };
            E('rect', { x: W - m.r - bb.w + 7, y: 1, width: bb.w, height: 19, fill: 'none', stroke: 'rgba(255,255,255,0.3)', 'stroke-dasharray': '3 2', rx: 2 }, svg);
            void t;
        }
        this.items = [];
        rows.forEach((r, i) => {
            const cy = m.t + i * rh + rh / 2;
            rowLabel(svg, lw, cy, DIMN[r.key] + (r.n !== 606 ? ` (${r.n})` : ''), lw - 4, TX2, nar ? 10.5 : 11.5);
            E('line', { x1: m.l, x2: W - m.r, y1: cy, y2: cy, stroke: 'rgba(255,255,255,0.035)' }, svg);
            const ln = E('line', { y1: cy, y2: cy, stroke: 'rgba(237,237,235,0.35)', 'stroke-width': 1.4 }, svg);
            const a = E('circle', { cy, cx: X(r.a), r: 4.6, fill: BG, stroke: '#9A9A98', 'stroke-width': 1.5 }, svg);
            const sig = r.adj.p < .05;
            const j = E('circle', { cy, r: 5.4, fill: sig ? CY : BG, stroke: CY, 'stroke-width': 1.7 }, svg);
            this.items.push({ ln, a, j, xa: X(r.a), xj: X(r.j), i });
            const hit = E('rect', { x: 0, y: cy - rh / 2, width: W, height: rh, fill: 'transparent' }, svg);
            const f = () => setRead('#v8-read', `<b>${DIMN[r.key]}</b> <span class="dim">·</span> ${r.n} districts <span class="dim">·</span> standard deviation ${r.sd.toFixed(3)}<br>` +
                `on its own ${sgn(r.raw.b, 3)} <span class="dim">(p ${fmtP(r.raw.p)}; ${sgn(r.raw.b * r.sd, 2)} per SD)</span> <span class="dim">·</span> ` +
                `adjusted ${sgn(r.adj.b, 3)} <span class="dim">(p ${fmtP(r.adj.p)}; ${sgn(r.adj.b * r.sd, 2)} per SD)</span>`);
            hit.addEventListener('mousemove', f); hit.addEventListener('click', f);
            hit.addEventListener('mouseleave', () => this.readDefault());
        });
        this.readDefault();
        this.place(this.u);
    },
    readDefault() {
        setRead('#v8-read', '<span class="dim">Hover or tap a row.</span> Adjusted, per standard deviation: producer organisations <b>+0.36</b>, cropping intensity <b>+0.56</b>.');
    },
    place(t) {
        (this.items || []).forEach(it => {
            const e = ease(Math.max(0, Math.min(1, t * 1.5 - it.i * 0.04)));
            const x = lerp(it.xa, it.xj, e);
            it.j.setAttribute('cx', x.toFixed(1));
            it.ln.setAttribute('x1', Math.min(it.xa, x).toFixed(1)); it.ln.setAttribute('x2', Math.max(it.xa, x).toFixed(1));
        });
    },
    prep() { this.u = 0; this.place(0); },
    play() { this.u = 0; this.place(0); tween(this, 1200, t => { this.u = t; this.place(t); }, () => { this.u = 1; this.place(1); }); },
});

/* ================================================================
   shared: horizontal dot-and-whisker rows
   rows: {name, b, ci:[lo,hi], p, col}
   ================================================================ */
function rowChart(sc, host, rows, o) {
    const W = host.clientWidth || 800, nar = W < 560;
    const lw = nar ? (o.lwN || 128) : (o.lw || 250), rowH = nar ? 50 : 46;
    const m = { t: 14, r: nar ? 14 : 26, b: 46, l: lw };
    const H = m.t + rows.length * rowH + m.b;
    const svg = sc.svg = svgFor(host, W, H);
    let lo = Math.min(0, ...rows.map(r => r.ci[0])), hi = Math.max(0, ...rows.map(r => r.ci[1]));
    const pad = (hi - lo) * .08; lo -= pad; hi += pad;
    const X = lin(lo, hi, m.l, W - m.r);
    const xt = niceTicks(lo, hi, nar ? 4 : 6), st = xt.length > 1 ? xt[1] - xt[0] : 1;
    const dp = Math.max(0, -Math.floor(Math.log10(st) + 1e-9)) + (Math.abs(st / Math.pow(10, Math.floor(Math.log10(st))) - 2.5) < 1e-6 ? 1 : 0);
    xt.forEach(v => {
        E('line', { x1: X(v), x2: X(v), y1: m.t, y2: H - m.b, stroke: Math.abs(v) < 1e-9 ? ZERO : GRID }, svg);
        T(svg, { class: 'mono', x: X(v), y: H - m.b + 17, 'text-anchor': 'middle', fill: TX3, 'font-size': 10.5 }, Math.abs(v) < 1e-9 ? '0' : sgn(v, dp));
    });
    E('line', { x1: X(0), x2: X(0), y1: m.t, y2: H - m.b, stroke: ZERO, 'stroke-width': 1.2 }, svg);
    T(svg, { class: 'mono', x: (m.l + W - m.r) / 2, y: H - 6, 'text-anchor': 'middle', fill: TX3, 'font-size': 11 }, nar ? o.xlabN : o.xlab);
    sc.X = X;
    sc.items = rows.map((r, i) => {
        const y = m.t + rowH * (i + .5), col = r.col || CY, sig = r.p != null && r.p < .05;
        if (i) E('line', { x1: 0, x2: W - m.r, y1: y - rowH / 2, y2: y - rowH / 2, stroke: 'rgba(255,255,255,0.05)' }, svg);
        rowLabel(svg, m.l - 12, y, r.name, lw - 16, TX2, nar ? 10.5 : 11.5);
        const wh = E('line', { x1: X(0), x2: X(0), y1: y, y2: y, stroke: col, 'stroke-width': 1.7, opacity: .85 }, svg);
        const dot = E('circle', { cx: X(0), cy: y, r: 5.5, fill: sig ? col : BG, stroke: col, 'stroke-width': 1.7 }, svg);
        const val = T(svg, { class: 'mono', x: X(r.b), y: y - 11, 'text-anchor': 'middle', fill: TX, 'font-size': 10.5 }, '');
        const hit = E('rect', { x: 0, y: y - rowH / 2, width: W, height: rowH, fill: 'transparent' }, svg);
        const f = () => o.onRead(r);
        hit.addEventListener('mousemove', f); hit.addEventListener('click', f);
        hit.addEventListener('mouseleave', () => o.onRead(null));
        return { r, wh, dot, val, i, dp };
    });
}
function placeRows(sc, u) {
    if (!sc.items) return;
    const X = sc.X, n = sc.items.length;
    sc.items.forEach((it, i) => {
        const e = easeOut(clamp01((u * (700 + 120 * n) - i * 120) / 700));
        const r = it.r;
        it.dot.setAttribute('cx', X(r.b * e).toFixed(1));
        it.wh.setAttribute('x1', X(r.ci[0] * e).toFixed(1)); it.wh.setAttribute('x2', X(r.ci[1] * e).toFixed(1));
        it.val.setAttribute('x', X(r.b * e).toFixed(1)); it.val.textContent = e >= 1 ? sgn(r.b, 2) : '';
    });
}
const rowScene = o => scene(Object.assign({
    u: 1,
    render() {
        const host = $(o.host); if (!host || !V) return;
        rowChart(this, host, this.rows(), { xlab: o.xlab, xlabN: o.xlabN, lw: o.lw, lwN: o.lwN,
            onRead: r => r ? setRead(o.read, this.text(r)) : this.readDefault() });
        this.readDefault();
        placeRows(this, this.u);
    },
    prep() { this.u = 0; placeRows(this, 0); },
    play() { this.u = 0; placeRows(this, 0); tween(this, 1500, t => { this.u = t; placeRows(this, t); }, () => { this.u = 1; placeRows(this, 1); }); },
}, o));

/* ================================================================
   05 inside each district: the slope across districts above 30 percent
   irrigated against the slope inside a district over time.
   Reads data/site_irrigation_panel.json and site_irrigation_v2.json.
   ================================================================ */
let DP = null;
SC[9] = rowScene({
    host: '#v9', read: '#v9-read', lw: 270, lwN: 132,
    xlab: 'change in effective number of crops for 10 more points of sown area irrigated',
    xlabN: 'per 10 points more irrigated',
    rows() {
        const s = V.spline.full, w = DP.within;
        return [
            { name: 'across districts in a state, above 30% irrigated', b: s.above / 10, ci: [s.above_ci[0] / 10, s.above_ci[1] / 10], p: s.above_p, col: LT },
            { name: 'same district over time', b: w.b / 10, ci: [(w.b - 1.96 * w.se) / 10, (w.b + 1.96 * w.se) / 10], p: w.p, col: CY },
        ];
    },
    text(r) {
        return `<b>${r.name}</b> <span class="dim">·</span> <b>${sgn(r.b, 3)}</b> effective crops per 10 points ` +
            `<span class="dim">(95% interval ${sgn(r.ci[0], 2)} to ${sgn(r.ci[1], 2)}; p ${fmtPw(r.p)})</span>`;
    },
    readDefault() {
        const w = DP.within;
        setRead('#v9-read', `<span class="dim">same district over time:</span> 10 more points irrigated goes with ` +
            `<b>${sgn(w.b / 10, 3)}</b> effective crops <span class="dim">(95% interval ${sgn((w.b - 1.96 * w.se) / 10, 2)} to ${sgn((w.b + 1.96 * w.se) / 10, 2)})</span>`);
    },
});

/* ================================================================
   02 recent years: quintile gaps in four sets of years
   ================================================================ */
const WINS = [{ k: 'full', col: LT }, { k: 'w1517', col: CY }, { k: 'w1721', col: VI }, { k: 'ext', col: GR }];
SC[11] = scene({
    u: 1,
    controls() {
        const l = $('#v11-leg'); if (!l || !V) return;
        l.innerHTML = WINS.map(w => `<span class="legend__item"><span class="legend__swatch" style="background:${w.col};border-radius:50%"></span>${V.fifths[w.k].label}</span>`).join('');
    },
    render() {
        const host = $('#v11'); if (!host || !V) return;
        const W = host.clientWidth || 900, nar = W < 560;
        const H = nar ? 320 : 360;
        const m = { t: 24, r: 10, b: 52, l: nar ? 36 : 50 };
        const svg = svgFor(host, W, H);
        const all = WINS.flatMap(w => V.fifths[w.k].rows.slice(1).flatMap(r => r.ci));
        let lo = Math.min(0, ...all), hi = Math.max(0, ...all);
        const pad = (hi - lo) * .07; lo -= pad; hi += pad;
        const Y = lin(lo, hi, H - m.b, m.t);
        const gw = (W - m.l - m.r) / 4;
        const yt = niceTicks(lo, hi, 5);
        yt.forEach(v => {
            E('line', { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), stroke: Math.abs(v) < 1e-9 ? ZERO : GRID, 'stroke-width': Math.abs(v) < 1e-9 ? 1.2 : 1 }, svg);
            T(svg, { class: 'mono', x: m.l - 7, y: Y(v) + 4, 'text-anchor': 'end', fill: TX3, 'font-size': 11 }, Math.abs(v) < 1e-9 ? '0' : sgn(v, 0));
        });
        T(svg, { class: 'mono', x: 4, y: 12, fill: TX3, 'font-size': 11 }, nar ? 'effective crops vs least irrigated quintile' : 'effective number of crops, against the least irrigated quintile in the same state');
        const ref = V.fifths.full.rows;
        this.items = [];
        for (let q = 1; q < 5; q++) {
            const x0 = m.l + gw * (q - 1);
            if (q % 2 === 0) { const bg = E('rect', { x: x0, width: gw, y: m.t, height: H - m.t - m.b, fill: 'rgba(255,255,255,0.03)' }, svg); svg.insertBefore(bg, svg.firstChild); }
            const r0 = ref[q];
            T(svg, { class: 'mono', x: x0 + gw / 2, y: H - m.b + 18, 'text-anchor': 'middle', fill: TX2, 'font-size': nar ? 10.5 : 11.5 }, (nar ? 'Q' : 'quintile ') + (q + 1));
            T(svg, { class: 'mono', x: x0 + gw / 2, y: H - m.b + 34, 'text-anchor': 'middle', fill: TX3, 'font-size': 10.5 }, `${pct(r0.lo)}–${pct(r0.hi)}`);
            WINS.forEach((w, j) => {
                const r = V.fifths[w.k].rows[q], x = x0 + gw * (0.2 + 0.2 * j), sig = r.p < .05;
                const wh = E('line', { x1: x, x2: x, stroke: w.col, 'stroke-width': 1.6, opacity: .85 }, svg);
                const dot = E('circle', { cx: x, r: nar ? 4.2 : 5, fill: sig ? w.col : BG, stroke: w.col, 'stroke-width': 1.6 }, svg);
                this.items.push({ r, wh, dot, w, q, Y });
            });
            const hit = E('rect', { x: x0, width: gw, y: m.t, height: H - m.t - m.b, fill: 'transparent' }, svg);
            const f = () => setRead('#v11-read', `<b>quintile ${q + 1}</b>, ${pct(r0.lo)} to ${pct(r0.hi)} irrigated<br>` +
                WINS.map(w => { const r = V.fifths[w.k].rows[q]; return `${V.fifths[w.k].label} <b>${sgn(r.b, 2)}</b> <span class="dim">(p ${fmtPw(r.p)})</span>`; }).join(' <span class="dim">·</span> '));
            hit.addEventListener('mousemove', f); hit.addEventListener('click', f);
            hit.addEventListener('mouseleave', () => this.readDefault());
        }
        this.readDefault();
        this.place(this.u);
    },
    readDefault() {
        const v = WINS.map(w => V.fifths[w.k].rows[4].b);
        setRead('#v11-read', `<span class="dim">Hover or tap a quintile.</span> Most irrigated quintile: between <b>${sgn(Math.max(...v), 2)}</b> and <b>${sgn(Math.min(...v), 2)}</b> effective crops in every set of years.`);
    },
    place(t) {
        (this.items || []).forEach(it => {
            const e = easeOut(clamp01(t * 1.6 - (it.q - 1) * 0.15));
            const Y = it.Y, r = it.r;
            it.dot.setAttribute('cy', lerp(Y(0), Y(r.b), e).toFixed(1));
            it.wh.setAttribute('y1', lerp(Y(0), Y(r.ci[0]), e).toFixed(1)); it.wh.setAttribute('y2', lerp(Y(0), Y(r.ci[1]), e).toFixed(1));
            it.dot.setAttribute('opacity', e > 0 ? 1 : 0); it.wh.setAttribute('opacity', e > 0 ? .85 : 0);
        });
    },
    prep() { this.u = 0; this.place(0); },
    play() { this.u = 0; this.place(0); tween(this, 1400, t => { this.u = t; this.place(t); }, () => { this.u = 1; this.place(1); }); },
});

/* ================================================================
   03 irrigation in 1991: quintiles of the 1991 share
   ================================================================ */
SC[12] = scene({
    win: 'w1721', u: 1,
    controls() {
        if (!V) return;
        segs($('#v12-win'), ['w1721', 'full'].map(k => ({ key: k, name: 'crops in ' + V.y1991[k].label })),
            k => { this.win = k; cancel(this); this.played = true; this.render(); this.play(); }, this.win);
    },
    render() {
        const host = $('#v12'); if (!host || !V) return;
        binChart(this, host, V.y1991[this.win].rows, {
            xlab: 'share of sown area irrigated, 1991',
            ylab: 'effective number of crops, against the least irrigated 1991 quintile in the same state',
            ylabN: 'effective crops vs least irrigated, same state',
            onRead: r => r ? setRead('#v12-read', binText(r, 5, 'effective crops').replace('irrigated <span', 'irrigated in 1991 <span')) : this.readDefault(),
        });
        this.readDefault();
        placeBins(this, this.u);
    },
    readDefault() {
        const top = V.y1991[this.win].rows[4];
        setRead('#v12-read', `<span class="dim">Hover or tap a quintile.</span> Most irrigated in 1991, above ${pct(top.lo)}: ` +
            `<b>${sgn(top.b, 2)}</b> effective crops in ${V.y1991[this.win].label} <span class="dim">(p ${fmtPw(top.p)})</span>`);
    },
    prep() { this.u = 0; placeBins(this, 0); },
    play() { this.u = 0; placeBins(this, 0); tween(this, 1300, t => { this.u = t; placeBins(this, t); }, () => { this.u = 1; placeBins(this, 1); }); },
});

/* ================================================================
   04 where irrigation spread, 1991 to 2011
   ================================================================ */
SC[13] = rowScene({
    host: '#v13', read: '#v13-read', lw: 290, lwN: 140,
    xlab: 'change in effective number of crops for 10 more points of sown area irrigated gained',
    xlabN: 'per 10 points of irrigation gained',
    rows() {
        return V.change.rows.map(r => ({ name: `${r.sample}, crops to ${r.to}`, b: r.b / 10, ci: [r.ci[0] / 10, r.ci[1] / 10], p: r.p, n: r.n, col: CY }));
    },
    text(r) {
        return `<b>${r.name}</b> <span class="dim">·</span> ${r.n} districts <span class="dim">·</span> <b>${sgn(r.b, 3)}</b> per 10 points ` +
            `<span class="dim">(95% interval ${sgn(r.ci[0], 2)} to ${sgn(r.ci[1], 2)}; p ${fmtPw(r.p)})</span>`;
    },
    readDefault() {
        setRead('#v13-read', `<span class="dim">Hover or tap a row.</span> Irrigation rose by <b>${(V.change.mean_change * 100).toFixed(0)} points</b> of sown area in the average district between 1991 and 2011.`);
    },
});

/* ================================================================
   12 canals, wells and tanks: per within-state standard deviation
   ================================================================ */
const SRCOL = { sh_tk: GR, sh_can: VI, sh_gw: CY, sh_ro: GY };
SC[14] = rowScene({
    host: '#v14', read: '#v14-read', lw: 200, lwN: 118,
    xlab: 'change in effective number of crops for one within-state standard deviation',
    xlabN: 'per within-state standard deviation',
    rows() { return V.sources.rows.map(r => Object.assign({}, r, { col: SRCOL[r.key] })); },
    text(r) {
        return `<b>${r.name}</b> <span class="dim">·</span> one standard deviation is ${(r.sd * 100).toFixed(1)} points of sown area ` +
            `<span class="dim">·</span> <b>${sgn(r.b, 2)}</b> effective crops <span class="dim">(95% interval ${sgn(r.ci[0], 2)} to ${sgn(r.ci[1], 2)}; p ${fmtPw(r.p)})</span>`;
    },
    readDefault() {
        setRead('#v14-read', `<span class="dim">Hover or tap a row.</span> Whether the curve differs between canal and well water: <b>p ${V.sources.gw_vs_canal_curve_p.toFixed(2)}</b>.`);
    },
});

/* ================================================================
   method tables
   ================================================================ */
function tables() {
    const put = (id, html) => { const t = document.querySelector(id + ' tbody'); if (t) t.innerHTML = html; };
    const n = (v, d) => `<td class="num">${v.toFixed(d)}</td>`;
    put('#tbl-1', D.profile.map(r => `<tr><td>${r.dec}${r.dec === 1 ? ', least irrigated' : (r.dec === 10 ? ', most irrigated' : '')}</td><td class="num">${r.n}</td>` +
        n(r.irr, 3) + n(r.D0, 1) + n(r.D1, 2) + n(r.D2, 2) + n(r.E, 3) + n(r.cereal, 3) + n(r.pulse, 3) + n(r.oilseed, 3) + '</tr>').join(''));
    const REGS = ['under 40%', '40 to 60%', 'over 60%'];
    const reg = (lab, rs) => rs.map(r => `<tr><td>${REGS[r.g]}, ${lab}</td><td class="num">${r.n}</td>` + n(r.irr, 3) + n(r.D0, 1) + n(r.D1, 2) + n(r.D2, 2) + n(r.E, 3) + n(r.cereal, 3) + '</tr>').join('');
    put('#tbl-r', reg('village records', D.regimes.ay) + reg('2011 directory', D.regimes.vd));
    put('#tbl-2', D.robustness.map((r, i) => `<tr><td>${OUTN[r.y] && r.y !== 'D1' ? r.lab : (r.y === 'D1' ? 'effective number: ' + r.lab : r.lab)}</td><td class="num">${r.n}</td>` +
        `<td class="num">${sgn(r.b1, 3)}</td><td class="num neg">${sgn(r.b2, 3)}</td><td class="num">${fmtP(r.p2)}</td>` + n(r.r2, 3) + '</tr>').join(''));
    if (V) put('#tbl-q', ['full', 'w1517', 'w1721', 'ext'].map(k => { const f = V.fifths[k];
        return `<tr><td>${f.label}</td><td class="num">${f.n}</td>` + f.rows.slice(1).map(r => `<td class="num">${sgn(r.b, 2)} <span class="dim">(${fmtPw(r.p)})</span></td>`).join('') + '</tr>'; }).join(''));
    put('#tbl-3', D.source.map(r => {
        const dp = r.y.length < 3 ? 2 : 3;
        const c = s => `<td class="num">${sgn(r[s].b, 3)} <span class="dim">(${sgn(r[s].b / 10, dp + 1)})</span></td><td class="num">${fmtP(r[s].p)}</td>`;
        return `<tr><td>${OUTN[r.y]}</td>${c('canal')}${c('surface')}</tr>`;
    }).join(''));
    put('#tbl-4', D.dims.map(r => `<tr><td>${r.key === 'irr' ? 'irrigated share of sown area' : DIMN[r.key]}</td><td class="num">${r.n}</td>` + n(r.sd, 3) +
        `<td class="num">${sgn(r.raw.b, 3)} <span class="dim">(${sgn(r.raw.b * r.sd, 2)})</span></td><td class="num">${fmtP(r.raw.p)}</td>` +
        `<td class="num">${sgn(r.adj.b, 3)} <span class="dim">(${sgn(r.adj.b * r.sd, 2)})</span></td><td class="num">${fmtP(r.adj.p)}</td></tr>`).join(''));
}

/* ================================================================
   wiring
   ================================================================ */
const HOSTS = { 1: '#v1', 2: '#v2', 3: '#v3', 4: '#v4', 5: '#v5', 6: '#v6', 7: '#v7', 8: '#v8', 9: '#v9', 11: '#v11', 12: '#v12', 13: '#v13', 14: '#v14' };
let inited = false;
function startPlay(sc) { if (sc.played) return; sc.played = true; sc.play(); }

async function init() {
    [D, DP, V] = await Promise.all([loadJSON('data/site_irrigation_story.json'), loadJSON('data/site_irrigation_panel.json'),
                                    loadJSON('data/site_irrigation_v2.json')]);
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
        }), { threshold: 0.3 });
        Object.keys(HOSTS).forEach(k => { const h = $(HOSTS[k]); if (!h) return; h.dataset.scene = k; io.observe(h); });
    } else {
        Object.values(SC).forEach(startPlay);
    }
    document.querySelectorAll('.replay').forEach(b => b.addEventListener('click', () => {
        const sc = SC[b.dataset.scene]; if (!sc) return;
        cancel(sc); sc.played = true; sc.play();
    }));
    inited = true;
}

async function drawAll() {
    if (!inited) { await init(); return; }
    for (const k of Object.keys(SC)) {
        const sc = SC[k];
        cancel(sc);
        if (sc.played) sc.u = 1;
        try { await sc.render(); } catch (e) { console.error('scene ' + k, e); }
        if (!sc.played && !RM && sc.prep) sc.prep();
    }
}
window.drawAll = drawAll;
drawAll();
})();
