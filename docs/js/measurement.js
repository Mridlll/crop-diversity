/* ================================================================
   MEASUREMENT PAGE: scene visuals.
   Reads data/site_measurement_data.json, data/site_indices.json,
   data/site_audit.json, data/site_markets_pop.json,
   data/site_monoculture.json, data/site_measurement_rev.json
   and data/districts.geojson.
   Depends on site.js for svgEl, lin, showTip, hideTip and loadJSON.
   Everything is wrapped so no name collides with site.js globals.

   Each scene: render() draws the current state, prep() sets the
   pre-entrance state, play() runs the entrance once in view.
   drawAll() re-renders on resize. Under prefers-reduced-motion
   every tween resolves at once.
   ================================================================ */
(function () {
'use strict';

const RM = !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);

/* one colour per measure */
const C = { D0: '#EDEDEB', D1: '#0E9BB5', D2: '#7C5CE0', E: '#1FA35A' };
const NAME = { D0: 'count of crops', D1: 'effective number of crops', D2: 'number of main crops', E: 'evenness' };
const NEU = '#B3B3B0';
const TX = '#EDEDEB', TX2 = '#B3B3B0', TX3 = '#7E7E7C', BG = '#0B0B0C';
const GRID = 'rgba(255,255,255,0.075)', ZERO = 'rgba(255,255,255,0.28)';
const MONO = 'JetBrains Mono, monospace';

const $ = s => document.querySelector(s);
function E(tag, attrs, parent) { const e = svgEl(tag, attrs || {}); if (parent) parent.appendChild(e); return e; }
function T(parent, attrs, s) { const t = E('text', attrs, parent); t.textContent = s; return t; }
const sgn = (v, d) => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(d);
const lerp = (a, b, t) => a + (b - a) * t;
const easeOut = t => 1 - Math.pow(1 - t, 3);
const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const title = s => s.toLowerCase().replace(/(^|[\s(-])([a-z])/g, (m, a, b) => a + b.toUpperCase()).replace(/ And /g, ' and ').replace(/ Of /g, ' of ');

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
function mulberry(seed) {
    return function () {
        seed |= 0; seed = seed + 0x6D2B79F5 | 0;
        let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
        t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
        return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
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

function segs(host, items, onPick, cur) {
    if (!host) return;
    host.innerHTML = '';
    items.forEach(it => {
        const b = document.createElement('button');
        b.type = 'button'; b.className = 'seg' + (it.act ? ' seg--act' : '');
        if (!it.act) b.setAttribute('aria-pressed', it.key === cur ? 'true' : 'false');
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
    [...host.querySelectorAll('.seg')].forEach(b => { if (b.hasAttribute('aria-pressed')) b.setAttribute('aria-pressed', b.dataset.key === key ? 'true' : 'false'); });
}

/* ---------- diversity of a share vector (percent) ---------- */
function hill(sh) {
    const p = sh.filter(s => s > 1e-9), tot = p.reduce((a, b) => a + b, 0);
    const q = p.map(s => s / tot);
    const H = -q.reduce((a, x) => a + x * Math.log(x), 0);
    const S = q.reduce((a, x) => a + x * x, 0);
    const D0 = q.length, D1 = Math.exp(H), D2 = 1 / S;
    return { D0, D1, D2, E: D1 / D0, H, simp: 1 - S };
}
function hillQ(sh, qq) {
    const p = sh.filter(s => s > 1e-9), tot = p.reduce((a, b) => a + b, 0);
    const q = p.map(s => s / tot);
    if (Math.abs(qq - 1) < 1e-6) return Math.exp(-q.reduce((a, x) => a + x * Math.log(x), 0));
    return Math.pow(q.reduce((a, x) => a + Math.pow(x, qq), 0), 1 / (1 - qq));
}
/* largest-remainder allocation of 100 plots */
function plots(sh) {
    const tot = sh.reduce((a, b) => a + b, 0) || 1;
    const raw = sh.map(s => s / tot * 100), fl = raw.map(Math.floor);
    let left = 100 - fl.reduce((a, b) => a + b, 0);
    const ord = raw.map((r, i) => [r - fl[i], i]).sort((a, b) => b[0] - a[0]);
    for (let k = 0; k < ord.length && left > 0; k++, left--) fl[ord[k][1]]++;
    const out = [];
    fl.forEach((n, i) => { for (let j = 0; j < n; j++) out.push(i); });
    return out;
}
function cropFill(i) {
    if (i === 0) return NEU;
    const a = (i % 2 ? .50 : .27) * Math.max(.45, 1 - i / 50);
    return `rgba(255,255,255,${a.toFixed(3)})`;
}

let MD = null, IND = null, AUD = null, MKT = null;
const SC = {};

/* ================================================================
   toy state shared by scenes 01 and 02
   ================================================================ */
const BASE = [62, 9, 6, 5, 3.5, 2.5, 2, 1.6, 1.4, 1.2, 1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.4, 0.3, 0.2];
const gen = (first, n) => [first].concat(Array.from({ length: n - 1 }, (_, i) => 'crop ' + (i + 2)));
let PRESETS = [];
function buildPresets() {
    PRESETS = [
        { key: 'equal', name: 'Twenty equal crops', shares: Array(20).fill(5), crops: gen('crop 1', 20) },
        { key: 'base', name: 'One cereal on 62%', shares: BASE.slice(), crops: gen('cereal', 20) },
        { key: 'rice', name: 'Rice on 80%', shares: [80].concat(Array(19).fill(20 / 19)), crops: gen('rice', 20) },
    ];
    (MD.examples || []).forEach((ex, i) => PRESETS.push({
        key: 'ex' + i, name: ex.name.split(',')[0] + ' ' + ex.year, shares: ex.shares.slice(),
        crops: ex.crops.slice(), real: ex.name + ', ' + ex.year,
    }));
}
const TOY = {
    key: 'base', shares: BASE.slice(), crops: gen('cereal', 20), prev: null, subs: [],
    set(key, opts) {
        const p = PRESETS.find(x => x.key === key); if (!p) return;
        this.prev = this.shares.slice();
        this.key = key; this.shares = p.shares.slice(); this.crops = p.crops.slice();
        this.emit(opts || {});
    },
    addCrop() {
        if (this.shares.length >= 60) return;
        this.prev = this.shares.slice();
        const tot = this.shares.reduce((a, b) => a + b, 0);
        this.shares = this.shares.map(s => s * (tot - 1) / tot).concat([1]);
        this.crops = this.crops.concat(['added crop']);
        this.key = null; this.emit({});
    },
    emit(o) { this.subs.forEach(f => f(o)); },
};
const presetItems = () => PRESETS.map(p => ({ key: p.key, name: p.name }))
    .concat([{ key: 'add', name: 'Add a crop on 1%', act: true }, { key: 'reset', name: 'Reset', act: true }]);
function pickPreset(k) {
    if (k === 'add') TOY.addCrop();
    else if (k === 'reset') TOY.set('base');
    else TOY.set(k);
}

/* ================================================================
   01 the district field
   ================================================================ */
SC[1] = scene({
    controls() {
        segs($('#v1-ctl'), presetItems(), k => { cancel(this); this.played = true; pickPreset(k); }, TOY.key);
    },
    render() {
        const host = $('#v1'); if (!host) return;
        const W = Math.min(host.clientWidth || 340, 360), cell = W / 10;
        const svg = svgFor(host, W, W);
        this.cells = [];
        for (let i = 0; i < 100; i++) {
            const r = E('rect', { x: (i % 10) * cell + 1, y: Math.floor(i / 10) * cell + 1, width: cell - 2, height: cell - 2, rx: 1.5 }, svg);
            r.addEventListener('mousemove', ev => {
                const c = this.plot[i]; if (c === undefined) return;
                showTip(`<strong>${esc(TOY.crops[c])}</strong><br>${TOY.shares[c].toFixed(TOY.shares[c] < 1 ? 2 : 1)}% of the land`, ev);
            });
            r.addEventListener('mouseleave', hideTip);
            this.cells.push(r);
        }
        this.bar();
        this.paint(false, TOY.shares);
    },
    paint(anim, sh) {
        if (!this.cells) return;
        this.plot = plots(sh);
        this.cells.forEach((r, i) => {
            r.style.transition = anim && !RM ? 'fill .25s ease' : 'none';
            const c = this.plot[i];
            r.style.fill = c === undefined ? 'rgba(255,255,255,.04)' : cropFill(c);
        });
        this.drawBar(sh);
        this.readout(sh);
    },
    /* stacked bar of shares with draggable dividers */
    bar() {
        const host = $('#v1-bar'); if (!host) return;
        const W = host.clientWidth || 500, H = 78;
        const svg = svgFor(host, W, H);
        this.bW = W; this.bsvg = svg;
        this.gSeg = E('g', {}, svg);
        this.lab0 = T(svg, { x: 0, y: 12, fill: TX2, 'font-size': 11, 'font-family': MONO }, '');
        this.gKnob = E('g', {}, svg);
        this.knobs = [];
    },
    knobSet(sh) {
        /* the dividers offered: among the first seven crops, spaced far enough apart to grab */
        const W = this.bW, out = []; let cum = 0, last = -1e9;
        for (let i = 0; i < Math.min(sh.length - 1, 7); i++) {
            cum += sh[i];
            const x = cum / 100 * W;
            if (x - last >= 22 && W - x >= 10) { out.push(i); last = x; }
        }
        return out;
    },
    drawBar(sh) {
        if (!this.gSeg) return;
        const W = this.bW, y0 = 20, h = 30;
        this.gSeg.innerHTML = '';
        let x = 0;
        sh.forEach((s, i) => {
            const w = s / 100 * W;
            if (w > 0.05) {
                const r = E('rect', { x: x.toFixed(2), y: y0, width: Math.max(0.5, w - (w > 3 ? 1.5 : 0)).toFixed(2), height: h, fill: cropFill(i) }, this.gSeg);
                r.addEventListener('mousemove', ev => showTip(`<strong>${esc(TOY.crops[i])}</strong><br>${s.toFixed(s < 1 ? 2 : 1)}% of the land`, ev));
                r.addEventListener('mouseleave', hideTip);
            }
            x += w;
        });
        this.lab0.textContent = `${TOY.crops[0]} ${sh[0].toFixed(sh[0] % 1 ? 1 : 0)}% of the land`;
        if (!this.dragging) this.knobSetCur = this.knobSet(sh);
        this.gKnob.innerHTML = '';
        let cum = 0; const ks = this.knobSetCur || [];
        const cums = []; sh.forEach((s, i) => { cum += s; cums[i] = cum; });
        ks.forEach(i => {
            const kx = cums[i] / 100 * W;
            const g = E('g', { class: 'knob', tabindex: 0, role: 'slider', 'aria-label': `Land between ${TOY.crops[i]} and ${TOY.crops[i + 1]}`,
                'aria-valuemin': 0, 'aria-valuemax': (sh[i] + sh[i + 1]).toFixed(0), 'aria-valuenow': sh[i].toFixed(0),
                'aria-valuetext': `${TOY.crops[i]} ${sh[i].toFixed(1)} percent` }, this.gKnob);
            E('line', { x1: kx, x2: kx, y1: y0 - 3, y2: y0 + h + 3, stroke: TX, 'stroke-width': 1.5 }, g);
            E('circle', { class: 'knob__c', cx: kx, cy: y0 + h + 12, r: 7, fill: BG, stroke: TX, 'stroke-width': 1.5 }, g);
            E('rect', { x: kx - 12, y: y0 - 4, width: 24, height: h + 28, fill: 'transparent' }, g);
            g.addEventListener('pointerdown', ev => this.drag(ev, i));
            g.addEventListener('keydown', ev => {
                const d = ev.key === 'ArrowRight' || ev.key === 'ArrowUp' ? 1 : (ev.key === 'ArrowLeft' || ev.key === 'ArrowDown' ? -1 : 0);
                if (!d) return;
                ev.preventDefault();
                this.shift(i, d);
                const nk = this.gKnob.querySelector(`[data-i="${i}"]`); if (nk) nk.focus();
            });
            g.dataset.i = i;
        });
    },
    shift(i, d) {
        const sh = TOY.shares, pair = sh[i] + sh[i + 1];
        const v = Math.max(0, Math.min(pair, sh[i] + d));
        sh[i] = v; sh[i + 1] = pair - v; TOY.key = null;
        pressSeg($('#v1-ctl'), ''); pressSeg($('#v2-ctl'), '');
        this.paint(true, sh);
        if (SC[2].redraw) SC[2].redraw(false);
    },
    drag(ev, i) {
        ev.preventDefault();
        const svg = this.bsvg, W = this.bW, sh = TOY.shares;
        const before = sh.slice(0, i).reduce((a, b) => a + b, 0), pair = sh[i] + sh[i + 1];
        this.dragging = true;
        const move = e => {
            const r = svg.getBoundingClientRect();
            const pct = (e.clientX - r.left) / r.width * 100;
            const v = Math.max(0, Math.min(pair, pct - before));
            sh[i] = v; sh[i + 1] = pair - v; TOY.key = null;
            this.paint(true, sh);
            if (SC[2].redraw) SC[2].redraw(false);
        };
        const up = () => {
            this.dragging = false; window.removeEventListener('pointermove', move); window.removeEventListener('pointerup', up);
            pressSeg($('#v1-ctl'), ''); pressSeg($('#v2-ctl'), '');
            this.drawBar(TOY.shares);
        };
        window.addEventListener('pointermove', move); window.addEventListener('pointerup', up);
    },
    readout(sh) {
        const host = $('#v1-read'); if (!host) return;
        const h = hill(sh);
        const rows = [['D0', h.D0, 0, 'count of crops'], ['D1', h.D1, 1, 'effective number of crops'],
                      ['D2', h.D2, 1, 'number of main crops'], ['E', h.E, 3, 'evenness, D1 / D0']];
        if (!host.children.length) {
            host.innerHTML = rows.map(r => `<div class="gauge"><span class="gauge__k"><svg width="9" height="9"><circle cx="4.5" cy="4.5" r="4.5" fill="${C[r[0]]}"/></svg>${r[0]}</span>` +
                `<span class="gauge__n">${r[3]}</span><span class="gauge__trk"><span class="gauge__fill" style="background:${C[r[0]]}"></span></span><span class="gauge__v"></span></div>`).join('');
        }
        const G = host.querySelectorAll('.gauge');
        rows.forEach((r, i) => {
            const frac = r[0] === 'E' ? r[1] : (h.D0 > 1 ? (r[1] - 1) / (h.D0 - 1) : 1);
            G[i].querySelector('.gauge__fill').style.width = (Math.max(0.01, Math.min(1, frac)) * 100).toFixed(1) + '%';
            G[i].querySelector('.gauge__v').textContent = r[1].toFixed(r[2]);
        });
        const p = PRESETS.find(x => x.key === TOY.key);
        const hint = $('#v1-hint');
        if (hint) hint.textContent = (p && p.real ? p.real + '. ' : '') + 'Gauges run from 1 to the count of crops. Drag a divider, or focus it and use the arrow keys, to move land between the two crops beside it.';
    },
    animateTo(from, to, dur) {
        const n = Math.max(from.length, to.length);
        const a = from.concat(Array(n - from.length).fill(0)), b = to.concat(Array(n - to.length).fill(0));
        tween(this, dur, t => {
            const e = easeOut(t);
            this.paint(false, a.map((s, i) => lerp(s, b[i], e)));
        }, () => this.paint(false, to));
    },
    prep() { this.paint(false, Array(20).fill(5)); },
    play() {
        /* replay: from twenty equal crops to the current example */
        const target = TOY.shares.slice();
        if (RM) { this.paint(false, target); return; }
        this.paint(false, Array(target.length).fill(100 / target.length));
        later(this, 500, () => this.animateTo(Array(target.length).fill(100 / target.length), target, 1400));
    },
});

/* ================================================================
   02 the Hill profile
   ================================================================ */
const QS = Array.from({ length: 61 }, (_, i) => i * 0.05);
const profile = sh => QS.map(q => hillQ(sh, q));
SC[2] = scene({
    controls() {
        segs($('#v2-ctl'), presetItems(), k => { cancel(this); this.played = true; pickPreset(k); }, TOY.key);
    },
    render() {
        const host = $('#v2'); if (!host) return;
        const W = host.clientWidth || 800, nar = W < 560, H = nar ? 300 : 340;
        const m = { t: 32, r: nar ? 74 : 120, b: 46, l: 46 };
        const svg = svgFor(host, W, H);
        this.W = W; this.H = H; this.m = m;
        const ymax = 50;
        const X = lin(0, 3, m.l, W - m.r);
        const Y = v => m.t + (1 - Math.log(v) / Math.log(ymax)) * (H - m.t - m.b);
        this.X = X; this.Y = Y;
        [1, 2, 5, 10, 20, 50].forEach(v => {
            E('line', { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), stroke: GRID, 'shape-rendering': 'crispEdges' }, svg);
            T(svg, { x: m.l - 8, y: Y(v) + 4, 'text-anchor': 'end', fill: TX3, 'font-size': 10.5, 'font-family': MONO }, v);
        });
        [0, 1, 2, 3].forEach(q => {
            E('line', { x1: X(q), x2: X(q), y1: m.t, y2: H - m.b, stroke: q === 3 ? GRID : 'rgba(255,255,255,0.12)', 'stroke-dasharray': q === 3 ? '' : '2 3' }, svg);
            T(svg, { x: X(q), y: H - m.b + 17, 'text-anchor': 'middle', fill: TX3, 'font-size': 10.5, 'font-family': MONO }, 'q = ' + q);
        });
        T(svg, { x: (m.l + W - m.r) / 2, y: H - 6, 'text-anchor': 'middle', fill: TX3, 'font-size': 11.5 }, 'weight given to the largest crops →');
        T(svg, { x: m.l - 8, y: m.t - 16, 'text-anchor': 'end', fill: TX3, 'font-size': 10.5, 'font-family': MONO }, 'crops');
        this.ghost = E('path', { fill: 'none', stroke: 'rgba(255,255,255,0.22)', 'stroke-width': 1.5, 'stroke-dasharray': '4 3' }, svg);
        this.line = E('path', { fill: 'none', stroke: TX2, 'stroke-width': 2 }, svg);
        this.dots = ['D0', 'D1', 'D2'].map((k, i) => {
            const c = E('circle', { r: 6, fill: C[k], stroke: BG, 'stroke-width': 2 }, svg);
            const t = T(svg, { fill: TX, 'font-size': nar ? 11 : 12, 'font-family': MONO }, '');
            return { k, q: i, c, t };
        });
        const hit = E('rect', { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: 'transparent' }, svg);
        this.cross = E('line', { y1: m.t, y2: H - m.b, stroke: 'rgba(255,255,255,0.3)', opacity: 0 }, svg);
        hit.addEventListener('mousemove', ev => {
            const r = svg.getBoundingClientRect(), x = (ev.clientX - r.left) / r.width * W;
            const q = Math.max(0, Math.min(3, (x - m.l) / (W - m.l - m.r) * 3));
            const v = hillQ(TOY.shares, q);
            this.cross.setAttribute('x1', X(q)); this.cross.setAttribute('x2', X(q)); this.cross.setAttribute('opacity', 1);
            showTip(`<strong>q = ${q.toFixed(2)}</strong><br>${v.toFixed(2)} crops`, ev);
        });
        hit.addEventListener('mouseleave', () => { hideTip(); this.cross.setAttribute('opacity', 0); });
        this.cur = profile(TOY.shares);
        this.place(this.cur, null);
        this.legend();
    },
    legend() {
        const L = $('#v2-leg'); if (!L) return;
        L.innerHTML = ['D0', 'D1', 'D2'].map(k => `<span class="legend__item"><svg width="12" height="12"><circle cx="6" cy="6" r="5" fill="${C[k]}"/></svg>${k} ${NAME[k]}</span>`).join('') +
            '<span class="legend__item"><svg width="26" height="10"><line x1="1" y1="5" x2="25" y2="5" stroke="rgba(255,255,255,0.4)" stroke-width="1.5" stroke-dasharray="4 3"/></svg>previous example</span>';
    },
    path(vals) { return vals.map((v, i) => (i ? 'L' : 'M') + this.X(QS[i]).toFixed(1) + ',' + this.Y(Math.max(1, v)).toFixed(1)).join(''); },
    place(vals, ghost) {
        if (!this.line) return;
        this.line.setAttribute('d', this.path(vals));
        if (ghost) this.ghost.setAttribute('d', this.path(ghost)); else this.ghost.setAttribute('d', '');
        const nar = this.W < 560;
        this.dots.forEach(d => {
            const v = vals[d.q * 20], x = this.X(d.q), y = this.Y(Math.max(1, v));
            d.c.setAttribute('cx', x); d.c.setAttribute('cy', y);
            d.t.setAttribute('x', x + 10); d.t.setAttribute('y', y - 9 - (d.q === 0 ? 0 : 0));
            d.t.textContent = `${d.k} ${v.toFixed(d.q === 0 ? 0 : 1)}`;
        });
    },
    redraw(anim) {
        if (!this.line) return;
        const to = profile(TOY.shares), from = this.cur || to;
        const ghost = TOY.prev ? profile(TOY.prev) : null;
        this.ghostVals = ghost;
        if (!anim) { this.cur = to; this.place(to, ghost); return; }
        cancel(this);
        tween(this, 600, t => { const e = easeOut(t); this.cur = from.map((v, i) => Math.exp(lerp(Math.log(v), Math.log(to[i]), e))); this.place(this.cur, ghost); },
            () => { this.cur = to; this.place(to, ghost); });
    },
    prep() { const f = Array(QS.length).fill(1); this.cur = f; this.place(f, null); },
    play() {
        const to = profile(TOY.shares);
        if (RM) { this.cur = to; this.place(to, this.ghostVals); return; }
        const flat = Array(QS.length).fill(TOY.shares.filter(s => s > 0).length);
        this.cur = flat; this.place(flat, null);
        later(this, 300, () => tween(this, 1400, t => {
            const e = easeOut(t); this.cur = flat.map((v, i) => Math.exp(lerp(Math.log(v), Math.log(to[i]), e))); this.place(this.cur, null);
        }, () => { this.cur = to; this.place(to, this.ghostVals); }));
    },
});

TOY.subs.push(o => {
    SC[1].paint(true, TOY.shares);
    pressSeg($('#v1-ctl'), TOY.key || ''); pressSeg($('#v2-ctl'), TOY.key || '');
    SC[2].redraw(true);
});

/* ================================================================
   03 strips across districts
   ================================================================ */
SC[3] = scene({
    mode: 'hill', sel: null,
    controls() {
        segs($('#v3-ctl'), [{ key: 'hill', name: 'D0, D1 and D2' }, { key: 'E', name: 'Evenness', color: C.E }],
            k => { this.mode = k; pressSeg($('#v3-ctl'), k); this.render(); this.u = 1; this.place(1); }, this.mode);
        const dl = $('#v3-list'), inp = $('#v3-find');
        if (!dl || !inp) return;
        this.byName = {};
        this.rows.forEach(r => { this.byName[r.label.toLowerCase()] = r; });
        dl.innerHTML = this.rows.map(r => `<option value="${esc(r.label)}"></option>`).join('');
        inp.addEventListener('input', () => {
            const r = this.byName[inp.value.trim().toLowerCase()];
            if (r) this.pick(r);
        });
    },
    data() {
        const S = new Set(MD.in_sample);
        this.rows = IND.districts.filter(d => S.has(d.k)).map(d => {
            const [s, n] = d.k.split('|');
            return Object.assign({}, d, { label: title(n) + ', ' + title(s) });
        });
        const r = mulberry(7);
        this.rows.forEach(d => { d.j = r(); });
    },
    render() {
        const host = $('#v3'); if (!host) return;
        const W = host.clientWidth || 900, nar = W < 560;
        const keys = this.mode === 'E' ? ['E'] : ['D0', 'D1', 'D2'];
        const sH = nar ? 66 : 76, gap = nar ? 34 : 38;
        const m = { t: 8, r: 14, b: 56, l: nar ? 34 : 150 };
        const H = m.t + keys.length * sH + (keys.length - 1) * gap + m.b;
        const svg = svgFor(host, W, H);
        this.svg = svg; this.W = W; this.H = H; this.keys = keys;
        const X = this.mode === 'E' ? lin(0, 0.8, m.l, W - m.r) : lin(0, 38, m.l, W - m.r);
        this.X = X;
        const ticks = this.mode === 'E' ? [0, .2, .4, .6, .8] : [0, 5, 10, 15, 20, 25, 30, 35];
        ticks.forEach(v => {
            if (nar && this.mode !== 'E' && v % 10 !== 0) return;
            E('line', { x1: X(v), x2: X(v), y1: m.t, y2: H - m.b + 18, stroke: GRID, 'shape-rendering': 'crispEdges' }, svg);
            T(svg, { x: X(v), y: H - m.b + 33, 'text-anchor': 'middle', fill: TX3, 'font-size': 10.5, 'font-family': MONO }, this.mode === 'E' ? v.toFixed(1) : v);
        });
        T(svg, { x: (m.l + W - m.r) / 2, y: H - 6, 'text-anchor': 'middle', fill: TX3, 'font-size': 11.5 }, this.mode === 'E' ? 'evenness, effective number of crops ÷ count of crops' : 'crops');
        const st = MD.stats.sample;
        this.strips = keys.map((k, si) => {
            const y0 = m.t + si * (sH + gap);
            const g = E('g', {}, svg);
            if (nar) T(g, { x: m.l, y: y0 - 4 + (si ? 0 : 10), fill: TX, 'font-size': 11, 'font-family': MONO }, k + '  ' + NAME[k]);
            else {
                T(g, { x: m.l - 14, y: y0 + sH / 2 - 2, 'text-anchor': 'end', fill: TX, 'font-size': 12.5, 'font-family': MONO }, k);
                T(g, { x: m.l - 14, y: y0 + sH / 2 + 13, 'text-anchor': 'end', fill: TX3, 'font-size': 10.5 }, NAME[k]);
            }
            const top = nar ? y0 + (si ? 4 : 14) : y0;
            const h = nar ? sH - (si ? 4 : 14) : sH;
            const dots = this.rows.map(d => E('circle', { cx: X(d[k]), cy: top + 4 + d.j * (h - 8), r: nar ? 1.8 : 2.2, fill: C[k], 'fill-opacity': .55 }, g));
            const med = st[k].median;
            E('line', { x1: X(med), x2: X(med), y1: top - 3, y2: top + h + 16, stroke: TX, 'stroke-width': 2 }, g);
            T(g, { x: X(med) + 5, y: top + h + 14, fill: TX2, 'font-size': 10, 'font-family': MONO }, 'median ' + med.toFixed(k === 'E' ? 3 : 1));
            return { k, top, h, dots };
        });
        this.gSel = E('g', { 'pointer-events': 'none' }, svg);
        const hit = E('rect', { x: m.l - 6, y: 0, width: W - m.l - m.r + 12, height: H - m.b, fill: 'transparent' }, svg);
        hit.addEventListener('mousemove', ev => {
            const r = svg.getBoundingClientRect(), x = (ev.clientX - r.left) / r.width * W, y = (ev.clientY - r.top) / r.height * H;
            const s = this.strips.reduce((b, s) => Math.abs(y - (s.top + s.h / 2)) < Math.abs(y - (b.top + b.h / 2)) ? s : b, this.strips[0]);
            let best = null, bd = 1e9;
            this.rows.forEach((d, i) => {
                const dx = X(d[s.k]) - x, dy = (s.top + 4 + d.j * (s.h - 8)) - y, dd = dx * dx + dy * dy;
                if (dd < bd) { bd = dd; best = d; }
            });
            if (best && bd < 400) { this.pick(best); showTip(this.tip(best), ev); } else hideTip();
        });
        hit.addEventListener('mouseleave', hideTip);
        hit.addEventListener('click', () => {});
        this.place(this.u === undefined ? 1 : this.u);
        if (this.sel) this.pick(this.sel);
    },
    tip(d) {
        return `<strong>${esc(d.label)}</strong><br>count of crops ${d.D0.toFixed(1)}<br>effective number of crops ${d.D1.toFixed(2)}<br>number of main crops ${d.D2.toFixed(2)}<br>evenness ${d.E.toFixed(3)}<br>${d.y} years of data`;
    },
    pick(d) {
        this.sel = d;
        if (!this.gSel) return;
        this.gSel.innerHTML = '';
        const pts = this.strips.map(s => [this.X(d[s.k]), s.top + 4 + d.j * (s.h - 8)]);
        if (pts.length > 1) E('path', { d: pts.map((p, i) => (i ? 'L' : 'M') + p[0].toFixed(1) + ',' + p[1].toFixed(1)).join(''), fill: 'none', stroke: TX, 'stroke-width': 1, 'stroke-opacity': .7 }, this.gSel);
        pts.forEach((p, i) => E('circle', { cx: p[0], cy: p[1], r: 5.5, fill: C[this.strips[i].k], stroke: BG, 'stroke-width': 2 }, this.gSel));
        const rd = $('#v3-read');
        if (rd) rd.textContent = `${d.label}: count of crops ${d.D0.toFixed(1)}, effective number of crops ${d.D1.toFixed(2)}, number of main crops ${d.D2.toFixed(2)}, evenness ${d.E.toFixed(3)}, ${d.y} years of data.`;
    },
    place(u) {
        if (!this.strips) return;
        this.strips.forEach((s, si) => {
            const local = Math.max(0, Math.min(1, u * 1.6 - si * .3));
            const e = easeOut(local);
            s.dots.forEach((c, i) => {
                const d = this.rows[i];
                c.setAttribute('cx', this.X(lerp(0, d[s.k], e)));
                c.setAttribute('fill-opacity', (.55 * Math.min(1, local * 3)).toFixed(3));
            });
        });
    },
    prep() { this.u = 0; this.place(0); },
    play() { this.u = 0; tween(this, 1600, t => { this.u = t; this.place(t); }); },
});

/* ================================================================
   04 a rescaled score
   ================================================================ */
const TOY4 = [2.1, 2.8, 3.3, 3.9, 4.4, 4.9, 5.6, 6.2, 7.0, 8.1, 9.5, 12.0];
const HL = 5;
SC[4] = scene({
    mode: 'toy', samp: 'base',
    controls() {
        segs($('#v4-mode'), [{ key: 'toy', name: 'Toy example' }, { key: 'data', name: '200 random draws' }],
            k => { cancel(this); this.mode = k; pressSeg($('#v4-mode'), k); this.controls(); this.render(); this.play(); }, this.mode);
        if (this.mode === 'toy') {
            segs($('#v4-ctl'), [{ key: 'add', name: 'Add three single-crop districts' }, { key: 'drop', name: 'Drop the least diverse district' }, { key: 'base', name: 'Reset' }],
                k => { cancel(this); this.played = true; this.samp = k; pressSeg($('#v4-ctl'), k); this.go(true); }, this.samp);
        } else $('#v4-ctl').innerHTML = '';
    },
    set4() {
        if (this.samp === 'add') return [1, 1, 1].concat(TOY4);
        if (this.samp === 'drop') return TOY4.slice(1);
        return TOY4.slice();
    },
    render() {
        const host = $('#v4'); if (!host) return;
        const W = host.clientWidth || 800, nar = W < 560;
        this.W = W; this.nar = nar;
        if (this.mode === 'toy') this.renderToy(host); else this.renderData(host);
        this.weights();
    },
    renderToy(host) {
        const W = this.W, nar = this.nar, H = 190;
        const m = { l: 18, r: 18 };
        const svg = svgFor(host, W, H);
        const X = lin(0, 1, m.l, W - m.r); this.X = X;
        const yA = 96;
        E('line', { x1: X(0), x2: X(1), y1: yA, y2: yA, stroke: ZERO }, svg);
        [0, .25, .5, .75, 1].forEach(v => {
            E('line', { x1: X(v), x2: X(v), y1: yA - 4, y2: yA + 4, stroke: ZERO }, svg);
            T(svg, { x: X(v), y: yA + 20, 'text-anchor': 'middle', fill: TX3, 'font-size': 10.5, 'font-family': MONO }, v.toFixed(2).replace(/0$/, '').replace(/\.0$/, ''));
        });
        T(svg, { x: (X(0) + X(1)) / 2, y: yA + 40, 'text-anchor': 'middle', fill: TX3, 'font-size': 11.5 }, 'score rescaled between the least and most diverse district in the sample');
        this.gD = E('g', {}, svg);
        this.hl = E('circle', { r: 8, fill: C.D1, stroke: BG, 'stroke-width': 2 }, svg);
        this.hlT = T(svg, { y: yA - 46, 'text-anchor': 'middle', fill: TX, 'font-size': nar ? 11 : 12, 'font-family': MONO }, '');
        this.hlT2 = T(svg, { y: yA - 30, 'text-anchor': 'middle', fill: TX2, 'font-size': nar ? 10.5 : 11, 'font-family': MONO }, '');
        this.hlL = E('line', { y1: yA - 24, y2: yA - 10, stroke: TX2 }, svg);
        this.foot = T(svg, { x: X(0), y: H - 8, fill: TX3, 'font-size': 10.5, 'font-family': MONO }, '');
        this.pos = this.scores(this.set4());
        this.drawToy(this.pos, false);
    },
    scores(vals) {
        const mn = Math.min(...vals), mx = Math.max(...vals);
        return { vals, s: vals.map(v => (v - mn) / (mx - mn)), mn, mx, hs: (4.9 - mn) / (mx - mn) };
    },
    drawToy(P, anim) {
        if (!this.gD) return;
        const X = this.X, yA = 96;
        this.gD.innerHTML = '';
        P.vals.forEach((v, i) => {
            const isHL = Math.abs(v - 4.9) < 1e-9;
            if (isHL) return;
            const single = v === 1;
            const stack = single ? P.vals.slice(0, i).filter(x => x === 1).length : 0;
            const c = E('circle', { cx: X(P.s[i]), cy: yA - stack * 11, r: 5, fill: single ? 'none' : NEU, stroke: single ? TX : 'none', 'stroke-width': 1.5 }, this.gD);
            c.addEventListener('mousemove', ev => showTip(`effective number of crops ${v.toFixed(1)}<br>score ${P.s[i].toFixed(3)}`, ev));
            c.addEventListener('mouseleave', hideTip);
        });
        const hx = X(P.hs);
        this.hl.setAttribute('cx', hx); this.hl.setAttribute('cy', yA);
        this.hlL.setAttribute('x1', hx); this.hlL.setAttribute('x2', hx);
        const ax = Math.max(90, Math.min(this.W - 90, hx));
        this.hlT.setAttribute('x', ax); this.hlT2.setAttribute('x', ax);
        this.hlT.textContent = `score ${P.hs.toFixed(3)}`;
        this.hlT2.textContent = 'effective number of crops 4.9';
        this.foot.textContent = `${P.vals.length} districts · least diverse ${P.mn.toFixed(1)} · most diverse ${P.mx.toFixed(1)}`;
    },
    go(anim) {
        if (this.mode !== 'toy') return;
        const from = this.pos, to = this.scores(this.set4());
        /* tween only the shared districts' scores and the highlighted one */
        tween(this, anim ? 700 : 0, t => {
            const e = easeOut(t);
            const P = { vals: to.vals, mn: to.mn, mx: to.mx, hs: lerp(from.hs, to.hs, e),
                s: to.vals.map((v, i) => { const j = from.vals.indexOf(v); return j >= 0 && v !== 1 ? lerp(from.s[j], to.s[i], e) : to.s[i]; }) };
            this.drawToy(P, true);
        }, () => { this.pos = to; this.drawToy(to, false); });
    },
    renderData(host) {
        const W = this.W, nar = this.nar, H = nar ? 300 : 320;
        const m = { t: 18, r: 18, b: 120, l: 44 };
        const svg = svgFor(host, W, H);
        const mv = MD.subsample_moves, comp = MD.composite;
        const bins = 16, lo = 0, hi = 0.016, bw = (hi - lo) / bins;
        const cnt = Array(bins).fill(0);
        mv.forEach(v => { cnt[Math.min(bins - 1, Math.floor((v - lo) / bw))]++; });
        const cmax = Math.max(...cnt);
        const X = lin(lo, hi, m.l, W - m.r), Y = lin(0, Math.ceil(cmax / 10) * 10, H - m.b, m.t);
        niceTicks(0, Math.ceil(cmax / 10) * 10, 4).forEach(v => {
            E('line', { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), stroke: GRID, 'shape-rendering': 'crispEdges' }, svg);
            T(svg, { x: m.l - 8, y: Y(v) + 4, 'text-anchor': 'end', fill: TX3, 'font-size': 10.5, 'font-family': MONO }, v);
        });
        [0, .004, .008, .012, .016].forEach(v => T(svg, { x: X(v), y: H - m.b + 17, 'text-anchor': 'middle', fill: TX3, 'font-size': 10.5, 'font-family': MONO }, v === 0 ? '0' : v.toFixed(3)));
        T(svg, { x: (m.l + W - m.r) / 2, y: H - m.b + 36, 'text-anchor': 'middle', fill: TX3, 'font-size': 11.5 }, 'average shift in a district\'s score, one random draw');
        T(svg, { x: m.l - 8, y: m.t - 6, 'text-anchor': 'end', fill: TX3, 'font-size': 10.5, 'font-family': MONO }, 'draws');
        this.bars = cnt.map((c, i) => {
            const x = X(lo + i * bw) + 1, w = Math.max(1, X(lo + (i + 1) * bw) - X(lo + i * bw) - 2);
            const r = E('rect', { x, width: w, y: Y(0), height: 0, fill: NEU, rx: 1.5 }, svg);
            E('rect', { x: x - 1, width: w + 2, y: m.t, height: H - m.b - m.t, fill: 'transparent' }, svg)
                .addEventListener('mousemove', ev => showTip(`${c} of 200 draws shifted scores by ${(lo + i * bw).toFixed(3)} to ${(lo + (i + 1) * bw).toFixed(3)} on average`, ev));
            return { r, c };
        });
        this.dY = Y;
        svg.addEventListener('mouseleave', hideTip);
        E('line', { x1: X(comp.move_mean), x2: X(comp.move_mean), y1: m.t, y2: H - m.b, stroke: TX, 'stroke-dasharray': '3 3' }, svg);
        T(svg, { x: X(comp.move_mean) + 6, y: m.t + 10, fill: TX, 'font-size': 11, 'font-family': MONO }, 'mean ' + comp.move_mean.toFixed(4));
        /* the same shifts against one standard deviation of the score */
        const y2 = H - 48, X2 = lin(0, 0.19, m.l, W - m.r);
        T(svg, { x: m.l, y: y2 - 16, fill: TX2, 'font-size': 11, 'font-family': MONO }, nar ? 'largest shift against one standard deviation' : 'the same shifts beside one standard deviation of the score');
        E('line', { x1: X2(0), x2: X2(comp.sd), y1: y2, y2: y2, stroke: 'rgba(255,255,255,0.35)', 'stroke-width': 8, 'stroke-linecap': 'butt' }, svg);
        E('line', { x1: X2(0), x2: X2(comp.move_max), y1: y2, y2: y2, stroke: C.D1, 'stroke-width': 8 }, svg);
        T(svg, { x: X2(comp.sd), y: y2 + 22, 'text-anchor': 'end', fill: TX2, 'font-size': 10.5, 'font-family': MONO }, 'standard deviation ' + comp.sd.toFixed(2));
        T(svg, { x: X2(0), y: y2 + 22, fill: '#3FC2DA', 'font-size': 10.5, 'font-family': MONO }, 'largest shift ' + comp.move_max.toFixed(4));
    },
    weights() {
        const host = $('#v4-w'); if (!host) return;
        const W = host.clientWidth || 800, nar = W < 560, H = nar ? 116 : 104;
        const svg = svgFor(host, W, H);
        const lw = nar ? 0 : 130, X = lin(0, 1, lw, W - 4);
        const r = MD.composite.shannon_simpson_r;
        const rows = [
            { y: 14, lab: 'stated weights', parts: [[1 / 3, 'count', C.D0], [1 / 3, 'Shannon', C.D1], [1 / 3, 'Simpson', C.D1]] },
            { y: 58, lab: 'in effect', parts: [[1 / 3, 'count of crops', C.D0], [2 / 3, 'land-weighted, nearly one measure', C.D1]] },
        ];
        rows.forEach(rw => {
            const yb = nar ? rw.y + 12 : rw.y;
            T(svg, nar ? { x: 0, y: rw.y + 6, fill: TX3, 'font-size': 10.5, 'font-family': MONO } : { x: lw - 10, y: rw.y + 15, 'text-anchor': 'end', fill: TX3, 'font-size': 10.5, 'font-family': MONO }, rw.lab);
            let x0 = 0;
            rw.parts.forEach(p => {
                E('rect', { x: X(x0), y: yb, width: X(x0 + p[0]) - X(x0) - 2, height: 22, fill: p[2], 'fill-opacity': p[2] === C.D0 ? .8 : .75, rx: 1.5 }, svg);
                T(svg, { x: X(x0) + 8, y: yb + 15, fill: BG, 'font-size': 10.5, 'font-family': MONO }, p[1]);
                x0 += p[0];
            });
        });
        T(svg, { x: X(1), y: H - 2, 'text-anchor': 'end', fill: TX3, 'font-size': 10.5, 'font-family': MONO }, `Shannon and Simpson correlate at ${r.toFixed(3)} across districts`);
    },
    prep() {
        if (this.mode === 'data' && this.bars) this.bars.forEach(b => { b.r.setAttribute('height', 0); b.r.setAttribute('y', this.dY(0)); });
    },
    play() {
        if (this.mode === 'data') {
            if (!this.bars) return;
            tween(this, 900, t => {
                const e = easeOut(t);
                this.bars.forEach(b => { const y = this.dY(b.c * e); b.r.setAttribute('y', y); b.r.setAttribute('height', Math.max(0, this.dY(0) - y)); });
            });
            return;
        }
        this.samp = 'base'; pressSeg($('#v4-ctl'), 'base');
        this.pos = this.scores(this.set4()); this.drawToy(this.pos, false);
        later(this, 1200, () => { this.samp = 'add'; pressSeg($('#v4-ctl'), 'add'); this.go(true); });
        later(this, 3600, () => { this.samp = 'drop'; pressSeg($('#v4-ctl'), 'drop'); this.go(true); });
        later(this, 6000, () => { this.samp = 'base'; pressSeg($('#v4-ctl'), 'base'); this.go(true); });
    },
});

/* ================================================================
   05 two kinds of change
   ================================================================ */
const ADD = BASE.map(s => s * .99).concat([1]);
const SPREAD = BASE.map((s, i) => i < 5 ? s + [-10, 4, 3, 2, 1][i] : s);
function composite(h, R) {
    return ((h.D0 - R.D0[0]) / (R.D0[1] - R.D0[0]) + (h.H - R.H[0]) / (R.H[1] - R.H[0]) + (h.simp - R.S[0]) / (R.S[1] - R.S[0])) / 3;
}
SC[5] = scene({
    render() {
        const st = MD.stats.sample;
        this.R = { D0: [st.D0.min, st.D0.max], H: [Math.log(st.D1.min), Math.log(st.D1.max)], S: [1 - 1 / st.D2.min, 1 - 1 / st.D2.max] };
        const h0 = hill(BASE.concat([0]));
        this.panels = [['#v5a', ADD, i => i === 20], ['#v5b', SPREAD.concat([0]), (i, p) => i >= 1 && i <= 4 && p]].map(([sel, after, changed]) => {
            const host = $(sel); if (!host) return null;
            const W = Math.min(host.clientWidth || 220, 220), cell = W / 10;
            const svg = svgFor(host, W, W);
            const cells = [];
            for (let i = 0; i < 100; i++) cells.push(E('rect', { x: (i % 10) * cell + 1, y: Math.floor(i / 10) * cell + 1, width: cell - 2, height: cell - 2, rx: 1.2 }, svg));
            const h1 = hill(after);
            const d = { D0: h1.D0 - h0.D0, D1: h1.D1 - h0.D1, C: composite(h1, this.R) - composite(h0, this.R) };
            return { sel, cells, after, changed, d };
        }).filter(Boolean);
        const mx = { D0: 1.2, D1: 1.2, C: Math.max(...this.panels.map(p => Math.abs(p.d.C))) * 1.15 };
        this.panels.forEach(p => {
            const host = $(p.sel + '-d');
            host.innerHTML = [['D0', 'count of crops', 0, C.D0], ['D1', 'effective number of crops', 2, C.D1], ['C', 'composite score', 3, NEU]].map(r =>
                `<div class="delta"><span>${r[1]}</span><span class="delta__trk"><span class="delta__fill" data-k="${r[0]}" style="background:${r[3]};width:0"></span></span><span class="delta__v" data-k="${r[0]}"></span></div>`).join('');
            p.host = host; p.mx = mx;
        });
        this.coef();
        this.paint(this.u === undefined ? 1 : this.u);
    },
    paint(u) {
        if (!this.panels) return;
        const e = easeOut(Math.max(0, Math.min(1, u)));
        const base = BASE.concat([0]);
        this.panels.forEach(p => {
            const sh = base.map((s, i) => lerp(s, p.after[i], e));
            const pl = plots(sh), pb = plots(base);
            /* the plots a crop gained sit at the end of its block */
            const nB = {}, nA = {}, seen = {};
            pb.forEach(c => { nB[c] = (nB[c] || 0) + 1; });
            pl.forEach(c => { nA[c] = (nA[c] || 0) + 1; });
            p.cells.forEach((r, i) => {
                const c = pl[i];
                seen[c] = (seen[c] || 0) + 1;
                const ch = c !== undefined && seen[c] > (nB[c] || 0);
                r.style.fill = c === undefined ? 'rgba(255,255,255,.04)' : (ch ? TX : cropFill(c === 20 ? 3 : c));
            });
            ['D0', 'D1', 'C'].forEach(k => {
                const v = k === 'D0' ? (u >= .5 ? p.d.D0 : 0) : p.d[k] * e;
                p.host.querySelector(`.delta__fill[data-k="${k}"]`).style.width = (Math.abs(v) / p.mx[k] * 100).toFixed(1) + '%';
                p.host.querySelector(`.delta__v[data-k="${k}"]`).textContent = Math.abs(v) < 1e-9 ? '0' : sgn(v, k === 'D0' ? 0 : (k === 'C' ? 3 : 2));
            });
        });
    },
    coef() {
        if (!MKT) return;
        const pick = (f, o) => MKT.solo.find(r => r.facility === f && r.outcome === o);
        [['#v5c', 'D0_richness', [-4, 12], C.D0], ['#v5d', 'D1_exp_shannon', [-2, 4.5], C.D1]].forEach(([sel, out, dom, col]) => {
            const host = $(sel); if (!host) return;
            const W = host.clientWidth || 400, nar = W < 420;
            const m = { t: 12, r: 16, b: 34, l: nar ? 112 : 150 }, rowH = 34;
            const rows = [['weekly haat', 'weekly haat'], ['FPO', 'farmer collective']];
            const H = m.t + rows.length * rowH + m.b;
            const svg = svgFor(host, W, H);
            const X = lin(dom[0], dom[1], m.l, W - m.r);
            niceTicks(dom[0], dom[1], nar ? 4 : 6).forEach(v => {
                E('line', { x1: X(v), x2: X(v), y1: m.t, y2: H - m.b, stroke: v === 0 ? ZERO : GRID, 'stroke-width': v === 0 ? 1.2 : 1 }, svg);
                T(svg, { x: X(v), y: H - m.b + 16, 'text-anchor': 'middle', fill: TX3, 'font-size': 10.5, 'font-family': MONO }, v === 0 ? '0' : sgn(v, 0));
            });
            rows.forEach(([f, lab], i) => {
                const r = pick(f, out); if (!r) return;
                const y = m.t + rowH * (i + .5), [lo, hi] = ci(r.b, r.p), sig = r.p < .05;
                T(svg, { x: m.l - 12, y: y + 4, 'text-anchor': 'end', fill: TX2, 'font-size': nar ? 11 : 12 }, lab);
                E('line', { x1: X(Math.max(dom[0], lo)), x2: X(Math.min(dom[1], hi)), y1: y, y2: y, stroke: col, 'stroke-opacity': .6, 'stroke-width': 1.6 }, svg);
                E('circle', { cx: X(r.b), cy: y, r: 6, fill: sig ? col : BG, stroke: col, 'stroke-width': 1.7 }, svg);
                T(svg, { x: X(r.b), y: y - 10, 'text-anchor': 'middle', fill: TX, 'font-size': 10.5, 'font-family': MONO }, sgn(r.b, 2));
                E('rect', { x: 0, y: y - rowH / 2, width: W, height: rowH, fill: 'transparent' }, svg)
                    .addEventListener('mousemove', ev => showTip(`<strong>${lab}</strong><br>${sgn(r.b, 2)} (p = ${r.p < .001 ? '< 0.001' : r.p.toFixed(3)})<br>interval ${sgn(lo, 2)} to ${sgn(hi, 2)}`, ev));
            });
            svg.addEventListener('mouseleave', hideTip);
        });
    },
    prep() { this.u = 0; this.paint(0); },
    play() { this.u = 0; this.paint(0); later(this, 400, () => tween(this, 1400, t => { this.u = t; this.paint(t); })); },
});

/* ================================================================
   06 within a year
   ================================================================ */
const YRS = 22, NCROP = 30;
const GROWN = (() => {
    /* 12 crops every year, 18 occasional ones; fixed seed */
    const r = mulberry(23), g = [];
    for (let c = 0; c < NCROP; c++) {
        const p = c < 12 ? 1 : Math.max(.2, .75 - (c - 12) * .03);
        g.push(Array.from({ length: YRS }, () => r() < p));
    }
    /* every occasional crop appears at least once */
    g.forEach((row, c) => { if (!row.some(Boolean)) row[(c * 7) % YRS] = true; });
    return g;
})();
SC[6] = scene({
    mode: 'year',
    controls() {
        segs($('#v6-ctl'), [{ key: 'year', name: 'Count each year, then average' }, { key: 'all', name: 'Count across all years' }],
            k => { cancel(this); this.played = true; this.mode = k; pressSeg($('#v6-ctl'), k); this.draw(1); }, this.mode);
    },
    render() { this.renderToy(); this.renderData(); },
    renderToy() {
        const host = $('#v6a'); if (!host) return;
        const W = host.clientWidth || 500, nar = W < 420;
        const m = { t: 6, r: 6, b: 40, l: 6 };
        const cols = YRS + 2, cw = (W - m.l - m.r) / cols, ch = Math.min(cw, 9);
        const H = m.t + NCROP * ch + m.b;
        const svg = svgFor(host, W, H);
        this.cells = [];
        GROWN.forEach((row, c) => row.forEach((on, y) => {
            const r = E('rect', { x: m.l + y * cw + .5, y: m.t + c * ch + .5, width: cw - 1, height: ch - 1, rx: .8, fill: on ? 'rgba(255,255,255,0.55)' : 'rgba(255,255,255,0.05)' }, svg);
            this.cells.push({ r, on, y, c });
        }));
        const xu = m.l + (YRS + 1) * cw;
        this.union = GROWN.map((row, c) => E('rect', { x: xu + .5, y: m.t + c * ch + .5, width: cw - 1, height: ch - 1, rx: .8, fill: TX, opacity: 0 }, svg));
        this.uLab = T(svg, { x: xu + cw / 2, y: H - m.b + 14, 'text-anchor': 'middle', fill: TX2, 'font-size': 10, 'font-family': MONO }, 'all');
        this.counts = Array.from({ length: YRS }, (_, y) => GROWN.filter(r => r[y]).length);
        this.cT = this.counts.map((n, y) => T(svg, { x: m.l + y * cw + cw / 2, y: H - m.b + 14, 'text-anchor': 'middle', fill: C.D1, 'font-size': nar ? 7.5 : 9, 'font-family': MONO }, n));
        T(svg, { x: m.l, y: H - 6, fill: TX3, 'font-size': 10.5, 'font-family': MONO }, '22 years →');
        T(svg, { x: W - m.r, y: H - 6, 'text-anchor': 'end', fill: TX3, 'font-size': 10.5, 'font-family': MONO }, 'rows: 30 crops');
        this.draw(this.u === undefined ? 1 : this.u);
    },
    draw(u) {
        if (!this.cells) return;
        const all = this.mode === 'all';
        const mean = this.counts.reduce((a, b) => a + b, 0) / YRS;
        this.cells.forEach(o => {
            const vis = o.y <= u * YRS;
            o.r.setAttribute('fill', o.on && vis ? (all ? 'rgba(255,255,255,0.32)' : 'rgba(255,255,255,0.55)') : 'rgba(255,255,255,0.05)');
        });
        this.union.forEach((r, c) => r.setAttribute('opacity', all && u >= 1 ? 1 : 0));
        this.uLab.setAttribute('opacity', all ? 1 : 0);
        this.cT.forEach((t, y) => t.setAttribute('opacity', !all && y <= u * YRS ? 1 : 0));
        const rd = $('#v6-read');
        if (rd) rd.textContent = all ? `Counted across all years: ${NCROP} crops.` : `Counted each year: ${Math.min(...this.counts)} to ${Math.max(...this.counts)} crops, ${mean.toFixed(1)} on average.`;
    },
    renderData() {
        const host = $('#v6b'); if (!host) return;
        const cov = AUD.coverage_richness.filter(c => c.n >= 5);
        const W = host.clientWidth || 500, nar = W < 420, H = nar ? 280 : 320;
        const m = { t: 28, r: 46, b: 44, l: 40 };
        const svg = svgFor(host, W, H);
        const ym = Math.ceil(Math.max(...cov.map(c => c.pooled)) / 10) * 10;
        const X = lin(1, 23, m.l, W - m.r), Y = lin(0, ym, H - m.b, m.t);
        niceTicks(0, ym, 4).forEach(v => {
            E('line', { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), stroke: v ? GRID : ZERO, 'shape-rendering': 'crispEdges' }, svg);
            T(svg, { x: m.l - 8, y: Y(v) + 4, 'text-anchor': 'end', fill: TX3, 'font-size': 10.5, 'font-family': MONO }, v);
        });
        [1, 5, 10, 15, 20, 23].forEach(v => T(svg, { x: X(v), y: H - m.b + 17, 'text-anchor': 'middle', fill: TX3, 'font-size': 10.5, 'font-family': MONO }, v));
        T(svg, { x: (m.l + W - m.r) / 2, y: H - 6, 'text-anchor': 'middle', fill: TX3, 'font-size': 11.5 }, 'years of data the district has');
        T(svg, { x: m.l - 8, y: m.t - 14, 'text-anchor': 'end', fill: TX3, 'font-size': 10.5, 'font-family': MONO }, 'crops');
        const nmax = Math.max(...cov.map(c => c.n));
        this.series = [['pooled', C.D0, 2], ['annual', C.D1, 2.5]].map(([k, col, sw]) => {
            const d = cov.map((c, i) => (i ? 'L' : 'M') + X(c.n_years).toFixed(1) + ',' + Y(c[k]).toFixed(1)).join('');
            const p = E('path', { d, fill: 'none', stroke: col, 'stroke-width': sw }, svg);
            const len = p.getTotalLength ? p.getTotalLength() : 1000;
            p.setAttribute('stroke-dasharray', len); p.setAttribute('stroke-dashoffset', 0);
            const dots = cov.map(c => E('circle', { cx: X(c.n_years), cy: Y(c[k]), r: (2.5 + 5 * Math.sqrt(c.n / nmax)).toFixed(1), fill: col, stroke: BG, 'stroke-width': 1.5 }, svg));
            return { p, len, dots };
        });
        const last = cov[cov.length - 1];
        T(svg, { x: X(last.n_years) + 12, y: Y(last.pooled) + 4, fill: TX, 'font-size': 11, 'font-family': MONO }, last.pooled.toFixed(1));
        T(svg, { x: X(last.n_years) + 12, y: Y(last.annual) + 4, fill: TX, 'font-size': 11, 'font-family': MONO }, last.annual.toFixed(1));
        cov.forEach(c => {
            E('rect', { x: X(c.n_years) - 8, y: m.t, width: 16, height: H - m.t - m.b, fill: 'transparent' }, svg)
                .addEventListener('mousemove', ev => showTip(`<strong>${c.n_years} years of data</strong><br>across all years ${c.pooled.toFixed(1)} crops<br>each year, averaged ${c.annual.toFixed(1)} crops<br>${c.n} districts`, ev));
        });
        svg.addEventListener('mouseleave', hideTip);
        this.drawData(this.ud === undefined ? 1 : this.ud);
    },
    drawData(u) {
        if (!this.series) return;
        this.series.forEach(s => {
            s.p.setAttribute('stroke-dashoffset', (s.len * (1 - u)).toFixed(1));
            s.dots.forEach((d, i) => d.setAttribute('opacity', i / s.dots.length <= u ? 1 : 0));
        });
    },
    prep() { this.u = 0; this.draw(0); this.ud = 0; this.drawData(0); },
    play() {
        this.mode = 'year'; pressSeg($('#v6-ctl'), 'year');
        this.u = 0; this.ud = 0;
        tween(this, 1500, t => { this.u = t; this.draw(t); this.ud = t; this.drawData(t); }, () => {
            later(this, 1400, () => { this.mode = 'all'; pressSeg($('#v6-ctl'), 'all'); this.draw(1); });
        });
    },
});

/* ================================================================
   03 reading the scale, 05 monoculture, 06 cropping systems.
   Read data/site_monoculture.json (MN) and data/districts.geojson.
   Class and system hues: cyan, violet and olive green, checked as a
   set on the dark ground for every pair (colour-blind and normal
   vision). Everything else is grey, so no more than three hues share
   a map.
   ================================================================ */
let MN = null, DIST = [];
const HUE = { cy: '#0E9BB5', vi: '#7C5CE0', ol: '#5E9F1E' };
const pct = v => Math.round(v * 100) + '%';
const CROPN2 = { 'soyabean': 'soybean', 'cotton(lint)': 'cotton', 'rapeseed &mustard': 'mustard', 'arhar/tur': 'tur',
    'moong(green gram)': 'moong', 'castor seed': 'castor', 'small millets': 'small millets', 'other cereals': 'other cereals' };
const crop = s => { const l = String(s || '').toLowerCase(); return CROPN2[l] || l; };
const cap = s => s.charAt(0).toUpperCase() + s.slice(1);
function svgXY(svg, ev) {
    const r = svg.getBoundingClientRect(), vb = svg.viewBox.baseVal;
    return [(ev.clientX - r.left) * vb.width / r.width, (ev.clientY - r.top) * vb.height / r.height];
}
function measLine(r) {
    return `count of crops ${r.D0.toFixed(1)}<br>effective number of crops ${r.D1.toFixed(2)}<br>number of main crops ${r.D2.toFixed(2)}`;
}
/* dots packed along x without overlap; returns a y offset per value */
function swarmY(xs, r) {
    const idx = xs.map((x, i) => i).sort((a, b) => xs[a] - xs[b]);
    const placed = [], out = new Array(xs.length), d2 = (2 * r + .4) * (2 * r + .4), step = r * .8;
    idx.forEach(i => {
        const x = xs[i];
        const near = [];
        for (let j = placed.length - 1; j >= 0 && placed[j][0] > x - 2 * r - .5; j--) near.push(placed[j]);
        let y = 0;
        for (let k = 0; k < 400; k++) {
            const c = k === 0 ? 0 : (k % 2 ? 1 : -1) * Math.ceil(k / 2) * step;
            if (!near.some(p => (p[0] - x) * (p[0] - x) + (p[1] - c) * (p[1] - c) < d2)) { y = c; break; }
        }
        placed.push([x, y]); out[i] = y;
    });
    return out;
}
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
    const path = (c, d) => d === 1 ? ring(c) : c.map(k => path(k, d - 1)).join('');
    return f => path(f.geometry.coordinates, f.geometry.type === 'Polygon' ? 2 : 3);
}
/* one map drawn once per width; fills are recoloured in place */
async function drawMap(host, sc, fillOf, tipOf) {
    const geo = await loadGeo();
    const W = host.clientWidth || 800, H = Math.round(Math.min(640, W * 1.0));
    const svg = svgFor(host, W, H);
    const pid = 'hz' + Math.random().toString(36).slice(2, 7);
    const pat = E('pattern', { id: pid, width: 5, height: 5, patternUnits: 'userSpaceOnUse', patternTransform: 'rotate(45)' }, E('defs', {}, svg));
    E('rect', { width: 5, height: 5, fill: 'rgba(255,255,255,0.04)' }, pat);
    E('line', { x1: 0, y1: 0, x2: 0, y2: 5, stroke: 'rgba(255,255,255,0.42)', 'stroke-width': 1.2 }, pat);
    sc.hatch = `url(#${pid})`;
    const P = projectGeo(geo, W, H);
    sc.marks = [];
    geo.features.forEach(f => {
        const d = P(f); if (!d) return;
        const r = f.properties.k ? MN.districts[f.properties.k] : null;
        const el = E('path', { d, fill: 'rgba(255,255,255,0.035)', stroke: 'rgba(11,11,12,0.85)', 'stroke-width': .4 }, svg);
        el.addEventListener('mousemove', ev => {
            el.setAttribute('stroke', TX); el.setAttribute('stroke-width', 1.1);
            showTip(r ? tipOf(r) : `<strong>${esc(f.properties.n)}</strong><br>${esc(f.properties.s)}<br>no crop record`, ev);
        });
        el.addEventListener('mouseleave', () => { el.setAttribute('stroke', 'rgba(11,11,12,0.85)'); el.setAttribute('stroke-width', .4); hideTip(); });
        sc.marks.push({ el, r });
    });
    svg.addEventListener('mouseleave', hideTip);
    sc.marks.forEach(m => m.el.setAttribute('fill', fillOf(m.r)));
    return svg;
}

/* ---------------- 03 reading the scale ---------------- */
const SM = [
    { key: 'D1', name: 'effective number of crops', color: C.D1 },
    { key: 'D0', name: 'count of crops', color: C.D0 },
    { key: 'D2', name: 'number of main crops', color: C.D2 },
];
SC.scale = scene({
    m: 'D1', u: 1,
    controls() {
        segs($('#vs-ctl'), SM, k => {
            cancel(this); this.played = true; this.m = k; pressSeg($('#vs-ctl'), k);
            this.render(); this.u = 0; this.draw(0); tween(this, 900, t => { this.u = t; this.draw(t); });
        }, this.m);
    },
    fmt(v) { return v >= 10 ? v.toFixed(1) : v.toFixed(2); },
    render() {
        const host = $('#vs'); if (!host) return;
        const S = MN.scale.sample, m = this.m, row = id => S.rows.find(r => r.id === id), ex = S.extremes[m];
        const W = host.clientWidth || 800, nar = W < 560, fmt = v => this.fmt(v);
        const pts = DIST.filter(r => r.in606 && r[m] != null);
        const vmax = Math.max(ex.max.value, ...pts.map(r => r[m]));
        const tk0 = niceTicks(0, vmax, nar ? 4 : 7), step = tk0[1] - tk0[0], xmax = Math.ceil(vmax / step) * step;
        const ML = 14, MR = 14, X = lin(0, xmax, ML, W - MR);
        const rd = nar ? 1.7 : 2.3;
        const ys = swarmY(pts.map(r => X(r[m])), rd);
        let half = Math.max(8, ...ys.map(Math.abs)) + rd;
        const squash = Math.min(1, 70 / half); half = half * squash;

        /* marks and their label lanes */
        const marks = [
            { v: 1, lab: nar ? '1 crop' : 'one crop on all land', tip: '<strong>1</strong><br>one crop on all the land', ref: true },
            { v: 2, lab: m === 'D0' ? (nar ? '2 crops' : 'two crops') : (nar ? '2 equal' : 'two equal crops'), tip: m === 'D0' ? '<strong>2</strong><br>two crops on any split of the land' : '<strong>2</strong><br>two crops on equal halves of the land', ref: true },
            { v: ex.min.value, lab: `${ex.min.district} ${fmt(ex.min.value)}`, tip: `<strong>lowest: ${esc(ex.min.district)}</strong><br>${esc(ex.min.state)}<br>${fmt(ex.min.value)}, ${ex.min.years} years of data` },
            { v: row('median')[m], lab: `median ${fmt(row('median')[m])}`, tip: `<strong>median district</strong><br>${fmt(row('median')[m])}<br>half the districts sit below` },
            { v: row('mean')[m], lab: `${nar ? 'mean' : 'average'} ${fmt(row('mean')[m])}`, tip: `<strong>national average</strong><br>${fmt(row('mean')[m])}<br>mean over the 606 districts` },
            { v: ex.max.value, lab: `${ex.max.district} ${fmt(ex.max.value)}`, tip: `<strong>highest: ${esc(ex.max.district)}</strong><br>${esc(ex.max.state)}<br>${fmt(ex.max.value)}, ${ex.max.years} years of data` },
        ].sort((a, b) => a.v - b.v);
        const cw = nar ? 6.1 : 6.6, lanes = [];
        marks.forEach(mk => {
            const x = X(mk.v), w = mk.lab.length * cw;
            let x0 = x + 5, anchor = 'start';
            if (x0 + w > W - 2) { x0 = x - 5 - w; anchor = 'end'; }
            if (x0 < 2) { x0 = 2; anchor = 'start'; }
            /* a lane is free when its last label ends before this one starts and this
               mark's line, which runs down through every lower lane, clears their labels */
            const placed = marks.filter(o => o.lane !== undefined);
            let L = lanes.findIndex((r, li) => r < x0 - 8 &&
                placed.every(o => o.lane <= li || o.x0 + o.w < x - 4 || o.x0 > x + 4));
            if (L < 0) { lanes.push(0); L = lanes.length - 1; }
            lanes[L] = x0 + w;
            Object.assign(mk, { x, x0, w, anchor, lane: L });
        });
        /* lane 0 sits highest; a lower label never lies under an earlier mark's line */
        const laneH = 17, top = 8 + lanes.length * laneH;
        const yc = top + 14 + half, axisY = yc + half + 10;
        const H = axisY + 46;
        const svg = svgFor(host, W, H);

        /* middle 80 percent */
        const p10 = row('p10')[m], p90 = row('p90')[m];
        E('rect', { x: X(p10), y: top + 6, width: X(p90) - X(p10), height: axisY - top - 6, fill: 'rgba(255,255,255,0.045)' }, svg);
        niceTicks(0, xmax, nar ? 4 : 7).forEach(v => {
            E('line', { x1: X(v), x2: X(v), y1: axisY, y2: axisY + 4, stroke: ZERO }, svg);
            T(svg, { x: X(v), y: axisY + 16, 'text-anchor': 'middle', fill: TX3, 'font-size': 10.5, 'font-family': MONO }, v);
        });
        E('line', { x1: ML, x2: W - MR, y1: axisY, y2: axisY, stroke: ZERO }, svg);
        const zone = (a, b, s) => { if (X(b) - X(a) > s.length * 6.2 + 6) T(svg, { x: (X(a) + X(b)) / 2, y: axisY + 34, 'text-anchor': 'middle', fill: TX2, 'font-size': 10.5, 'font-family': MONO }, s); };
        zone(0, p10, 'low');
        zone(p10, p90, nar ? 'middle 80%' : 'middle 80% of districts');
        zone(p90, xmax, 'high');

        /* districts */
        this.dots = pts.map((r, i) => {
            const c = E('circle', { cx: X(r[m]), cy: yc + ys[i] * squash, r: rd, fill: C[m], opacity: .55 }, svg);
            return { c, r, x: X(r[m]), y: yc + ys[i] * squash };
        });
        this.dots.sort((a, b) => a.x - b.x);

        /* marks */
        this.marks = marks.map(mk => {
            const g = E('g', {}, svg);
            const ly = 18 + mk.lane * laneH;
            E('line', { x1: mk.x, x2: mk.x, y1: ly + 3, y2: axisY, stroke: mk.ref ? TX3 : TX, 'stroke-width': 1, 'stroke-dasharray': mk.ref ? '2 3' : null }, g);
            E('circle', { cx: mk.x, cy: axisY, r: 3, fill: mk.ref ? BG : TX, stroke: TX, 'stroke-width': 1.2 }, g);
            T(g, { x: mk.anchor === 'start' ? mk.x0 : mk.x0 + mk.w, y: ly, 'text-anchor': mk.anchor, fill: mk.ref ? TX2 : TX, 'font-size': nar ? 10 : 11, 'font-family': MONO }, mk.lab);
            const hit = E('rect', { x: Math.min(mk.x0, mk.x - 6), y: ly - 12, width: Math.max(mk.w, Math.abs(mk.x - mk.x0)) + 12, height: 16, fill: 'transparent' }, g);
            hit.addEventListener('mousemove', ev => showTip(mk.tip, ev));
            hit.addEventListener('mouseleave', hideTip);
            return g;
        });
        const ring = E('circle', { r: rd + 3, fill: 'none', stroke: TX, 'stroke-width': 1.4, opacity: 0 }, svg);
        const rdEl = $('#vs-read');
        const base = `${NAME[m]}: middle 80 percent of districts ${fmt(p10)} to ${fmt(p90)}, median ${fmt(row('median')[m])}`;
        if (rdEl) rdEl.textContent = base;
        svg.addEventListener('mousemove', ev => {
            if (ev.target.tagName === 'rect' && ev.target.getAttribute('fill') === 'transparent') return;
            const [mx, my] = svgXY(svg, ev);
            let best = null, bd = 196;
            this.dots.forEach(d => { const dd = (d.x - mx) ** 2 + (d.y - my) ** 2; if (dd < bd) { bd = dd; best = d; } });
            if (!best) { ring.setAttribute('opacity', 0); hideTip(); if (rdEl) rdEl.textContent = base; return; }
            ring.setAttribute('cx', best.x); ring.setAttribute('cy', best.y); ring.setAttribute('opacity', 1);
            const r = best.r;
            showTip(`<strong>${esc(r.d)}</strong><br>${esc(r.s)}<br>${measLine(r)}`, ev);
            if (rdEl) rdEl.textContent = `${r.d}, ${r.s}: ${NAME[m]} ${fmt(r[m])}`;
        });
        svg.addEventListener('mouseleave', () => { ring.setAttribute('opacity', 0); hideTip(); if (rdEl) rdEl.textContent = base; });
        this.draw(this.u);
    },
    draw(u) {
        if (!this.dots) return;
        const N = this.dots.length;
        this.dots.forEach((d, i) => d.c.setAttribute('opacity', (u >= i / N * .65 ? .55 : 0)));
        this.marks.forEach((g, j) => g.setAttribute('opacity', u >= .6 + j * .06 ? 1 : 0));
    },
    prep() { this.u = 0; this.draw(0); },
    play() { this.u = 0; this.draw(0); tween(this, 1600, t => { this.u = t; this.draw(t); }); },
});

/* ---------------- 05 monoculture ---------------- */
const CLS = [
    { key: 'mono', name: 'near-monoculture', col: HUE.cy },
    { key: 'pair', name: 'two crops hold the year', col: HUE.vi },
    { key: 'few', name: 'few crops, none dominant', col: HUE.ol },
    { key: 'mixed', name: 'mixed', col: 'rgba(255,255,255,0.24)' },
];
const CLSK = Object.fromEntries(CLS.map(c => [c.key, c]));
/* invented districts: plots per crop in each season, 50 plots a season */
const TOYS = [
    { t: 'One crop holds most of the land',
      mon: [['rice', 44], ['maize', 2], ['urad', 1], ['jute', 1], ['sesame', 1], ['vegetables', 1]],
      rab: [['mustard', 2], ['potato', 1], ['lentil', 1], ['khesari', 1], ['wheat', 1], ['gram', 1], ['linseed', 1], ['onion', 1]] },
    { t: 'Few crops, none holding most',
      mon: [['maize', 26], ['rice', 6], ['millet', 3], ['beans', 3]],
      rab: [['wheat', 12], ['barley', 3], ['peas', 2], ['mustard', 1]] },
    { t: 'Two crops share the year',
      mon: [['rice', 40], ['sugarcane', 3], ['maize', 2], ['tur', 2], ['urad', 1], ['sesame', 1], ['vegetables', 1]],
      rab: [['wheat', 38], ['mustard', 3], ['gram', 2], ['potato', 2], ['lentil', 1], ['barley', 1]] },
];
TOYS.forEach(ty => {
    const tot = {};
    ty.mon.concat(ty.rab).forEach(([c, n]) => { tot[c] = (tot[c] || 0) + n; });
    ty.order = Object.keys(tot).sort((a, b) => tot[b] - tot[a]);
    ty.tot = tot;
    const all = ty.order.map(c => tot[c]), sum = all.reduce((a, b) => a + b, 0);
    ty.h = hill(all); ty.sum = sum; ty.top = all[0] / sum; ty.top2 = (all[0] + (all[1] || 0)) / sum;
});
const toyFill = i => i === 0 ? '#DADAD7' : i === 1 ? '#8E8E8B' : `rgba(255,255,255,${(i % 2 ? .30 : .17)})`;
SC.mono = scene({
    view: 'dots', u: 1,
    controls() {
        segs($('#vm-ctl'), [{ key: 'dots', name: "by largest crop's share" }, { key: 'map', name: 'map' }],
            k => { cancel(this); this.played = true; this.view = k; pressSeg($('#vm-ctl'), k); this.u = 1; this.drawToys(1); this.renderChart(); }, this.view);
        const leg = $('#vm-leg');
        if (leg) {
            const n = MN.monoculture.sample;
            leg.innerHTML = CLS.map(c => `<span class="legend__item"><span class="seg__sw" style="background:${c.col}"></span>${c.name} · ${n[c.key].n}</span>`).join('');
        }
    },
    async render() { this.renderToys(); await this.renderChart(); },
    renderToys() {
        const host = $('#vm-toys'); if (!host) return;
        host.innerHTML = '';
        this.cells = [];
        TOYS.forEach((ty, ti) => {
            const card = document.createElement('div'); card.className = 'mtoy';
            card.innerHTML = `<p class="duo__t">${ty.t}</p><div class="mtoy__g"></div><div class="mtoy__v"></div>`;
            host.appendChild(card);
            const g = card.querySelector('.mtoy__g');
            const W = g.clientWidth || 300, gap = 10, cs = Math.min(14, (W - gap) / 20), H = 16 + cs * 5;
            const svg = svgFor(g, Math.max(W, cs * 20 + gap), H);
            [['monsoon', ty.mon, 0], ['rabi', ty.rab, 1]].forEach(([nm, list, si]) => {
                const x0 = si * (cs * 10 + gap);
                T(svg, { x: x0, y: 10, fill: TX3, 'font-size': 10, 'font-family': MONO }, nm);
                const seq = [];
                list.forEach(([c, n]) => { for (let k = 0; k < n; k++) seq.push(c); });
                for (let k = 0; k < 50; k++) {
                    const c = seq[k], col = k % 10, rw = Math.floor(k / 10);
                    const ci = c ? ty.order.indexOf(c) : -1;
                    const r = E('rect', { x: x0 + col * cs + .6, y: 16 + rw * cs + .6, width: cs - 1.2, height: cs - 1.2, rx: 1,
                        fill: c ? toyFill(ci) : 'none', stroke: c ? 'none' : 'rgba(255,255,255,0.13)', 'stroke-width': .8 }, svg);
                    r.addEventListener('mousemove', ev => showTip(c ? `<strong>${c}</strong><br>${nm}<br>${pct(ty.tot[c] / ty.sum)} of the year's cropped land` : `<strong>fallow</strong><br>${nm}`, ev));
                    r.addEventListener('mouseleave', hideTip);
                    this.cells.push({ r, k: k + si * 50, ti });
                }
            });
            const v = card.querySelector('.mtoy__v');
            const line = (sw, k, val) => `<span class="mtoy__k">${sw ? `<span class="seg__sw" style="background:${sw}"></span>` : '<span class="seg__sw"></span>'}${k}</span><span class="mtoy__n">${val}</span>`;
            v.innerHTML = line(C.D0, 'count of crops', ty.h.D0) + line(C.D1, 'effective number', ty.h.D1.toFixed(1)) +
                line(C.D2, 'main crops', ty.h.D2.toFixed(1)) +
                line(null, ti === 2 ? 'two largest crops' : 'largest crop', pct(ti === 2 ? ty.top2 : ty.top));
        });
        this.drawToys(this.u);
    },
    drawToys(u) {
        if (!this.cells) return;
        this.cells.forEach(c => c.r.setAttribute('opacity', u >= (c.ti * 100 + c.k) / 300 * .9 ? 1 : 0));
    },
    async renderChart() {
        const host = $('#vm'); if (!host) return;
        this.dots = null;
        const rd = $('#vm-read');
        const base = 'Hover or tap a district.';
        if (rd) rd.textContent = base;
        const tip = r => `<strong>${esc(r.d)}</strong><br>${esc(r.s)}<br>${CLSK[r.cls].name}<br>largest crop ${crop(r.top)}, ${pct(r.tsh)} of the land<br>two largest ${pct(r.t2sh)}<br>${measLine(r)}`;
        if (this.view === 'map') {
            await drawMap(host, this, r => (r && r.in606) ? CLSK[r.cls].col : 'rgba(255,255,255,0.035)',
                r => r.in606 ? tip(r) : `<strong>${esc(r.d)}</strong><br>${esc(r.s)}<br>outside the 606 districts`);
            return;
        }
        const pts = DIST.filter(r => r.in606);
        const W = host.clientWidth || 800, nar = W < 560;
        const ML = 14, MR = 14, X = lin(0, 1, ML, W - MR), rr = nar ? 1.9 : 2.6;
        const ys = swarmY(pts.map(r => X(r.tsh)), rr);
        let half = Math.max(...ys.map(Math.abs)) + rr;
        const squash = Math.min(1, (nar ? 95 : 120) / half); half *= squash;
        const top = 26, yc = top + half + 4, axisY = yc + half + 10, H = axisY + 36;
        const svg = svgFor(host, W, H);
        [0, .2, .4, .6, .8, 1].forEach(v => {
            E('line', { x1: X(v), x2: X(v), y1: axisY, y2: axisY + 4, stroke: ZERO }, svg);
            T(svg, { x: X(v), y: axisY + 16, 'text-anchor': v === 0 ? 'start' : v === 1 ? 'end' : 'middle', fill: TX3, 'font-size': 10.5, 'font-family': MONO }, pct(v));
        });
        E('line', { x1: ML, x2: W - MR, y1: axisY, y2: axisY, stroke: ZERO }, svg);
        T(svg, { x: ML, y: axisY + 32, fill: TX3, 'font-size': 10.5, 'font-family': MONO }, "largest crop's share of the land →");
        const x23 = X(2 / 3);
        E('line', { x1: x23, x2: x23, y1: top - 8, y2: axisY, stroke: TX, 'stroke-dasharray': '3 3' }, svg);
        T(svg, { x: x23 + 6, y: top - 2, fill: TX, 'font-size': 10.5, 'font-family': MONO }, nar ? '2/3' : 'two thirds');
        const nm = MN.monoculture.sample.mono.n;
        T(svg, { x: W - MR, y: top - 2, 'text-anchor': 'end', fill: TX2, 'font-size': 10.5, 'font-family': MONO }, `${nm} districts`);
        this.dots = pts.map((r, i) => ({ r, x: X(r.tsh), y: yc + ys[i] * squash,
            c: E('circle', { cx: X(r.tsh), cy: yc + ys[i] * squash, r: rr, fill: CLSK[r.cls].col }, svg) }))
            .sort((a, b) => a.x - b.x);
        const ring = E('circle', { r: rr + 3, fill: 'none', stroke: TX, 'stroke-width': 1.4, opacity: 0 }, svg);
        svg.addEventListener('mousemove', ev => {
            const [mx, my] = svgXY(svg, ev);
            let best = null, bd = 196;
            this.dots.forEach(d => { const dd = (d.x - mx) ** 2 + (d.y - my) ** 2; if (dd < bd) { bd = dd; best = d; } });
            if (!best) { ring.setAttribute('opacity', 0); hideTip(); if (rd) rd.textContent = base; return; }
            ring.setAttribute('cx', best.x); ring.setAttribute('cy', best.y); ring.setAttribute('opacity', 1);
            showTip(tip(best.r), ev);
            if (rd) rd.textContent = `${best.r.d}, ${best.r.s}: ${crop(best.r.top)} on ${pct(best.r.tsh)} of the land`;
        });
        svg.addEventListener('mouseleave', () => { ring.setAttribute('opacity', 0); hideTip(); if (rd) rd.textContent = base; });
        this.drawDots(this.u);
    },
    drawDots(u) {
        if (!this.dots) return;
        const N = this.dots.length;
        this.dots.forEach((d, i) => d.c.setAttribute('opacity', u >= i / N ? 1 : 0));
    },
    prep() { this.u = 0; this.drawToys(0); this.drawDots(0); },
    play() {
        this.u = 0; this.drawToys(0); this.drawDots(0);
        tween(this, 2000, t => { this.u = t; this.drawToys(t); this.drawDots(Math.max(0, t * 1.4 - .4)); });
    },
});

/* ---------------- 06 cropping systems ---------------- */
const PAIRS = ['maize-wheat', 'soybean-wheat', 'bajra-wheat', 'bajra-mustard', 'rice-pulse', 'rice-oilseed', 'rice-wheat-light',
    'cotton-other', 'maize-other', 'soybean-gram', 'soybean-other', 'rice-other', 'bajra-other'];
const SG = [
    { key: 'rw', name: 'rice then wheat', col: HUE.vi, has: s => s === 'rice-wheat' },
    { key: 'rr', name: 'rice twice a year', col: HUE.cy, has: s => s === 'rice-rice' },
    { key: 'pr', name: 'another two-season pair', col: HUE.ol, has: s => PAIRS.includes(s) },
    { key: 'r1', name: 'rice, little second season', col: 'rgba(255,255,255,0.46)', has: s => s === 'rice-only' },
    { key: 'ot', name: 'one main season or mixed', col: 'rgba(255,255,255,0.17)', has: s => !['rice-wheat', 'rice-rice', 'rice-only', 'whole'].includes(s) && !PAIRS.includes(s) },
    { key: 'wh', name: 'seasons not reported', col: 'hatch', has: s => s === 'whole' },
];
const sgOf = s => SG.find(g => g.has(s));
function sysName(r) {
    const d = r.sys_detail || '';
    if (r.sys === 'rice-wheat' || r.sys === 'rice-rice' || r.sys === 'whole') return MN.system_labels[r.sys];
    if (/\(dry season only\)$/.test(d)) return cap(crop(d.replace(/ \(dry season only\)$/, ''))) + ' in the dry season only';
    if (/ only$/.test(d)) return cap(crop(d.replace(/ only$/, ''))) + ', little second season';
    const i = d.indexOf('-');
    if (i > 0) return cap(crop(d.slice(0, i))) + ' then ' + crop(d.slice(i + 1));
    return MN.system_labels[r.sys] || r.sys;
}
const CAL = [
    { code: 'rice-wheat', g: 'rw', name: 'Rice then wheat', a: ['rice', 'rice_mon_median'], b: ['wheat', 'wheat_rabi_median'] },
    { code: 'rice-rice', g: 'rr', name: 'Rice twice a year', a: ['rice', 'rice_mon_median'], b: ['rice', 'rice_dry_median'] },
    { code: 'rice-only', g: 'r1', name: 'Rice, little second season', a: ['rice', 'rice_mon_median'], b: ['rice', 'rice_dry_median'] },
];
SC.sys = scene({
    sel: 'all', u: 1,
    controls() {
        const n = {};
        DIST.forEach(r => { const g = sgOf(r.sys); if (g) n[g.key] = (n[g.key] || 0) + 1; });
        const sw = c => c === 'hatch' ? 'repeating-linear-gradient(45deg, rgba(255,255,255,.5) 0 1px, transparent 1px 3px)' : c;
        segs($('#vy-ctl'), [{ key: 'all', name: 'all systems' }].concat(SG.map(g => ({ key: g.key, name: `${g.name} · ${n[g.key] || 0}`, color: sw(g.col) }))),
            k => { this.sel = k; pressSeg($('#vy-ctl'), k); this.paint(); }, this.sel);
    },
    async render() {
        const host = $('#vy'); if (!host) return;
        await drawMap(host, this, () => 'rgba(255,255,255,0.035)', r => {
            const g = sgOf(r.sys), yrs = Math.round(r.sys_agree * r.y);
            return `<strong>${esc(r.d)}</strong><br>${esc(r.s)}<br>${esc(sysName(r))}<br>in ${yrs} of ${r.y} years<br>${measLine(r)}`;
        });
        this.paint();
        this.renderCal();
    },
    paint() {
        if (!this.marks) return;
        this.marks.forEach(m => {
            const g = m.r ? sgOf(m.r.sys) : null;
            const fill = !g ? 'rgba(255,255,255,0.035)' : g.col === 'hatch' ? this.hatch : g.col;
            m.el.setAttribute('fill', fill);
            m.el.setAttribute('fill-opacity', (this.sel === 'all' || (g && g.key === this.sel)) ? 1 : .18);
        });
    },
    renderCal() {
        const host = $('#vy-cal'); if (!host) return;
        host.innerHTML = '';
        const head = document.createElement('div'); head.className = 'cal__row cal__row--h';
        head.innerHTML = `<span></span><span class="cal__sl"><span>monsoon · Jun&ndash;Oct</span><span>rabi and summer · Nov&ndash;May</span></span>
            <span class="cal__vals"><span>crops</span><span>effective</span><span>main</span><span>largest</span></span>`;
        host.appendChild(head);
        this.bars = [];
        CAL.forEach(c => {
            const s = MN.systems.sample.find(x => x.code === c.code), hh = MN.head_to_head[c.code].sample, g = SG.find(x => x.key === c.g);
            const row = document.createElement('div'); row.className = 'cal__row';
            row.innerHTML = `<span class="cal__n"><span class="seg__sw" style="background:${g.col}"></span>${c.name}<span class="cal__cnt">${hh.n} districts</span></span>
                <span class="cal__strip"></span>
                <span class="cal__vals"><span class="mtoy__n">${hh.D0.toFixed(1)}</span><span class="mtoy__n">${hh.D1.toFixed(2)}</span><span class="mtoy__n">${hh.D2.toFixed(2)}</span><span class="mtoy__n">${pct(hh.tsh)}</span></span>`;
            host.appendChild(row);
            const st = row.querySelector('.cal__strip');
            const W = st.clientWidth || 360, H = 62, gap = 6, cw = (W - gap) / 2, Y = v => v / .7 * (H - 18);
            const svg = svgFor(st, W, H);
            [c.a, c.b].forEach(([cropN, key], i) => {
                const v = s[key], x = i * (cw + gap);
                E('rect', { x, y: 0, width: cw, height: H, fill: 'rgba(255,255,255,0.03)', rx: 2 }, svg);
                const h = Math.max(1.5, Y(v));
                const bar = E('rect', { x: x + 2, y: H - h, width: cw - 4, height: h, rx: 2, fill: cropN === 'rice' ? '#DADAD7' : '#8E8E8B' }, svg);
                const lab = v < .01 ? `${cropN} under 1%` : `${cropN} ${pct(v)}`;
                const ly = Math.max(13, H - h - 5);
                T(svg, { x: x + 6, y: ly, fill: v < .01 ? TX3 : (H - h - 5 < 13 ? BG : TX), 'font-size': 10.5, 'font-family': MONO }, lab);
                const hit = E('rect', { x, y: 0, width: cw, height: H, fill: 'transparent' }, svg);
                hit.addEventListener('mousemove', ev => showTip(`<strong>${c.name}</strong><br>${i ? 'rabi and summer' : 'monsoon'}: ${cropN} on ${(v * 100).toFixed(1)}% of the year's cropped land<br>median of ${hh.n} districts`, ev));
                hit.addEventListener('mouseleave', hideTip);
                this.bars.push({ bar, h, H });
            });
        });
        this.drawCal(this.u);
    },
    drawCal(u) {
        if (!this.bars) return;
        this.bars.forEach((b, i) => {
            const t = Math.max(0, Math.min(1, u * 1.6 - i * .1)), h = b.h * easeOut(t);
            b.bar.setAttribute('height', h.toFixed(1)); b.bar.setAttribute('y', (b.H - h).toFixed(1));
        });
    },
    prep() { this.u = 0; this.drawCal(0); },
    play() { this.u = 0; this.drawCal(0); later(this, 300, () => tween(this, 1300, t => { this.u = t; this.drawCal(t); })); },
});

/* ---------------- 07 within a season ---------------- */
let REV = null;
const SEAS_ROWS = [
    { key: 'all', name: 'all districts' },
    { key: 'rw', name: 'rice then wheat', col: HUE.vi },
    { key: 'rr', name: 'rice twice a year', col: HUE.cy },
    { key: 'pr', name: 'another two-season pair' },
    { key: 'r1', name: 'rice, little second season' },
    { key: 'ot', name: 'one main season or mixed' },
];
SC.seas = scene({
    mode: 'mon', u: 1,
    controls() {
        segs($('#vss-ctl'), [{ key: 'mon', name: 'year-round crops with the monsoon' }, { key: 'own', name: 'year-round crops as a season' }],
            k => { this.mode = k; pressSeg($('#vss-ctl'), k); this.move(); }, this.mode);
    },
    rows() {
        const G = REV.seasons.groups, out = [];
        SEAS_ROWS.forEach(r => { const g = G.find(x => x.key === r.key); out.push(Object.assign({}, r, { ann: g.ann, own: g.own, mon: g.mon, n: g.n, bs_own: g.bs_own, bs_mon: g.bs_mon })); });
        REV.seasons.examples.forEach(e => {
            const r = SEAS_ROWS.find(x => x.key === e.key);
            out.push({ key: 'ex-' + e.key, name: e.d + ', ' + e.s, ex: true, col: r.col, ann: e.ann, own: e.own, mon: e.mon, sys: r.name });
        });
        return out;
    },
    render() {
        const host = $('#vss'); if (!host || !REV) return;
        const W = host.clientWidth || 760, nar = W < 560;
        const rows = this.rows();
        const rh = nar ? 44 : 34, gap = 18;
        const m = { t: 22, r: nar ? 16 : 40, b: nar ? 44 : 34, l: nar ? 12 : 220 };
        const H = m.t + rows.length * rh + gap + m.b + (nar ? 0 : 0);
        const svg = svgFor(host, W, H), iw = W - m.l - m.r;
        const X = lin(0, 7, m.l, m.l + iw);
        [0, 1, 2, 3, 4, 5, 6, 7].forEach(v => {
            E('line', { x1: X(v), x2: X(v), y1: m.t - 6, y2: H - m.b, stroke: v ? GRID : ZERO }, svg);
            T(svg, { x: X(v), y: H - m.b + 16, 'text-anchor': 'middle', fill: TX3, 'font-size': 10.5 }, String(v));
        });
        T(svg, { x: m.l + iw, y: H - 2, 'text-anchor': 'end', fill: TX3, 'font-size': 10.5 }, 'effective number of crops');
        this.marks = [];
        rows.forEach((r, i) => {
            const yc = m.t + i * rh + (r.ex ? gap : 0) + (nar ? rh - 14 : rh / 2);
            const ly = nar ? yc - 15 : yc + 4;
            const lab = T(svg, { x: nar ? m.l : m.l - 14, y: ly, 'text-anchor': nar ? 'start' : 'end', fill: r.key === 'rw' || r.key === 'rr' || r.ex ? TX : TX2, 'font-size': 11.5, 'font-style': r.ex ? 'italic' : 'normal' }, r.name);
            if (r.col && !r.ex) E('circle', { cx: nar ? m.l - 0 : m.l - 14 - lab.getComputedTextLength() - 9, cy: ly - 4, r: 3.5, fill: r.col }, svg);
            if (nar && r.col && !r.ex) lab.setAttribute('x', m.l + 10);
            const line = E('line', { y1: yc, y2: yc, stroke: 'rgba(255,255,255,0.35)', 'stroke-width': 1.6 }, svg);
            const a = E('circle', { cy: yc, r: 5, fill: BG, stroke: NEU, 'stroke-width': 1.6 }, svg);
            const b = E('circle', { cy: yc, r: 5.5, fill: r.ex ? (r.col || C.D1) : C.D1 }, svg);
            const ta = T(svg, { y: yc + 4, fill: TX2, 'font-size': 10.5, 'font-family': MONO }, '');
            const tb = T(svg, { y: yc + 4, fill: TX, 'font-size': 10.5, 'font-family': MONO, 'text-anchor': 'end' }, '');
            const hit = E('rect', { x: 0, y: yc - rh / 2 - (nar ? 8 : 0), width: W, height: rh, fill: 'transparent' }, svg);
            const tip = ev => {
                const w = r[this.mode], bs = r['bs_' + this.mode];
                showTip(`<strong>${esc(r.name)}</strong>${r.ex ? '<br>' + esc(r.sys) : ''}<br>over the year ${r.ann.toFixed(2)}<br>within a season ${w.toFixed(2)}` +
                    (bs != null ? `<br>share of annual diversity from the change of crops between seasons ${pct(bs)}<br>${r.n} districts` : ''), ev);
                this.read(r);
            };
            hit.addEventListener('mousemove', tip); hit.addEventListener('click', tip); hit.addEventListener('mouseleave', hideTip);
            this.marks.push({ r, line, a, b, ta, tb, X });
        });
        if (rows.some(r => r.ex)) {
            const y = m.t + (rows.length - 2) * rh + gap / 2 + 2;
            E('line', { x1: nar ? m.l : 8, x2: W - m.r, y1: y, y2: y, stroke: GRID, 'stroke-dasharray': '2 3' }, svg);
        }
        this.move(true);
        this.read(rows[0]);
    },
    read(r) {
        const el = $('#vss-read'); if (!el) return;
        const w = r[this.mode];
        el.textContent = `${r.name}: ${r.ann.toFixed(1)} effective crops over the year, ${w.toFixed(1)} within a season`;
    },
    move(now) {
        if (!this.marks) return;
        const from = this.marks.map(k => k.cur == null ? k.r.ann : k.cur);
        const go = t => this.marks.forEach((k, i) => {
            const target = lerp(k.r.ann, k.r[this.mode], easeOut(this.u));
            const v = lerp(from[i], target, easeOut(t)), X = k.X;
            k.cur = v;
            k.a.setAttribute('cx', X(k.r.ann)); k.b.setAttribute('cx', X(v));
            k.line.setAttribute('x1', X(v)); k.line.setAttribute('x2', X(k.r.ann));
            k.ta.setAttribute('x', X(k.r.ann) + 9); k.ta.textContent = k.r.ann.toFixed(1);
            k.tb.setAttribute('x', X(v) - 9); k.tb.textContent = this.u > .98 ? v.toFixed(1) : '';
        });
        if (now) { go(1); return; }
        tween(this, 500, go);
        const r = this.marks[0] && this.marks[0].r; if (r) this.read(r);
    },
    prep() { this.u = 0; this.marks && this.marks.forEach(k => { k.cur = k.r.ann; }); this.move(true); },
    play() {
        this.u = 0; this.marks && this.marks.forEach(k => { k.cur = k.r.ann; }); this.move(true);
        later(this, 250, () => tween(this, 1100, t => { this.u = t; this.move(true); }));
    },
});

/* ---------------- 10 count of crops with a minimum share of land ---------------- */
const TH = [
    { key: 'raw', name: 'every crop', short: 'every crop' },
    { key: 't01', name: 'at least 0.1% of land', short: 'crops on at least 0.1% of land' },
    { key: 't05', name: 'at least 0.5%', short: 'crops on at least 0.5% of land' },
    { key: 't1', name: 'at least 1%', short: 'crops on at least 1% of land' },
];
SC.small = scene({
    sel: 't01', u: 1,
    controls() {
        segs($('#vt-ctl'), TH.map(t => ({ key: t.key, name: t.name })), k => { this.sel = k; pressSeg($('#vt-ctl'), k); this.paint(); }, this.sel);
    },
    render() {
        const host = $('#vt'); if (!host || !REV) return;
        const S = REV.threshold.series;
        const W = host.clientWidth || 760, nar = W < 560;
        const m = { t: 18, r: nar ? 34 : 46, b: 30, l: nar ? 26 : 34 };
        const H = nar ? 300 : 340, iw = W - m.l - m.r, ih = H - m.t - m.b;
        const svg = this.svg = svgFor(host, W, H);
        const y0 = S[0].year, y1 = S[S.length - 1].year;
        const X = lin(y0, y1, m.l + (nar ? 32 : 40), m.l + iw), Y = lin(0, 27, m.t + ih, m.t);
        [0, 5, 10, 15, 20, 25].forEach(v => {
            E('line', { x1: m.l, x2: m.l + iw, y1: Y(v), y2: Y(v), stroke: v ? GRID : ZERO }, svg);
            T(svg, { x: m.l - 7, y: Y(v) + 4, 'text-anchor': 'end', fill: TX3, 'font-size': 10.5 }, String(v));
        });
        (nar ? [1998, 2005, 2012, 2019] : [1998, 2001, 2004, 2007, 2010, 2013, 2016, 2019]).forEach(v =>
            T(svg, { x: X(v), y: m.t + ih + 18, 'text-anchor': 'middle', fill: TX3, 'font-size': 10.5 }, String(v)));
        T(svg, { x: m.l, y: m.t - 5, fill: TX3, 'font-size': 10.5 }, 'number of crops');
        const path = k => S.map((t, i) => (i ? 'L' : 'M') + X(t.year).toFixed(1) + ',' + Y(t[k]).toFixed(1)).join('');
        this.lines = TH.concat([{ key: 'D1', short: 'effective number of crops' }]).map(t => {
            const p = E('path', { d: path(t.key), fill: 'none', 'stroke-width': 2, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' }, svg);
            const last = S[S.length - 1], first = S[0];
            const lab = T(svg, { x: X(last.year) + 6, y: Y(last[t.key]) + 4, 'font-size': 11, 'font-family': MONO }, last[t.key].toFixed(1));
            const lab0 = T(svg, { x: X(first.year) - 6, y: Y(first[t.key]) + 4, 'text-anchor': 'end', 'font-size': 11, 'font-family': MONO }, first[t.key].toFixed(1));
            return { t, p, lab, lab0, len: p.getTotalLength ? p.getTotalLength() : 1000 };
        });
        const cross = E('line', { y1: m.t, y2: m.t + ih, stroke: 'rgba(255,255,255,0.35)', visibility: 'hidden' }, svg);
        const hit = E('rect', { x: m.l, y: m.t, width: iw, height: ih, fill: 'transparent' }, svg);
        const pick = ev => {
            const r = svg.getBoundingClientRect(), k = W / r.width, x = (ev.clientX - r.left) * k;
            let best = S[0], bd = 1e9;
            S.forEach(t => { const d = Math.abs(X(t.year) - x); if (d < bd) { bd = d; best = t; } });
            cross.setAttribute('x1', X(best.year)); cross.setAttribute('x2', X(best.year)); cross.setAttribute('visibility', 'visible');
            showTip(`<strong>${best.year}&ndash;${String(best.year + 1).slice(2)}</strong>` + TH.map(t => `<br>${t.short} ${best[t.key].toFixed(1)}`).join('') +
                `<br>effective number of crops ${best.D1.toFixed(1)}`, ev);
        };
        hit.addEventListener('mousemove', pick); hit.addEventListener('click', pick);
        hit.addEventListener('mouseleave', () => { cross.setAttribute('visibility', 'hidden'); hideTip(); });
        this.paint();
    },
    paint() {
        if (!this.lines) return;
        const S = REV.threshold.series, a = S[0], b = S[S.length - 1];
        this.lines.forEach(l => {
            const on = l.t.key === this.sel, d1 = l.t.key === 'D1', raw = l.t.key === 'raw';
            const col = d1 ? C.D1 : on ? HUE.vi : raw ? 'rgba(237,237,235,0.55)' : 'rgba(255,255,255,0.18)';
            l.p.setAttribute('stroke', col);
            l.p.setAttribute('stroke-width', on ? 2.6 : 1.6);
            if (d1) l.p.setAttribute('stroke-dasharray', this.u >= 1 ? '5 4' : `${l.len} ${l.len}`);
            const showLab = on || d1 || raw;
            l.lab.setAttribute('fill', on ? TX : d1 ? C.D1 : TX2);
            l.lab0.setAttribute('fill', on ? TX : d1 ? C.D1 : TX2);
            l.lab.setAttribute('visibility', showLab ? 'visible' : 'hidden');
            l.lab0.setAttribute('visibility', showLab ? 'visible' : 'hidden');
        });
        const t = TH.find(x => x.key === this.sel);
        const el = $('#vt-read');
        if (el) el.textContent = `Counting ${t.short}: ${a[t.key].toFixed(1)} in 1998–99 and ${b[t.key].toFixed(1)} in 2019–20`;
        this.place();
    },
    place() {
        if (!this.lines) return;
        const e = easeOut(this.u);
        this.lines.forEach(l => {
            if (l.t.key === 'D1' && this.u >= 1) { l.p.setAttribute('stroke-dasharray', '5 4'); l.p.removeAttribute('stroke-dashoffset'); }
            else { l.p.setAttribute('stroke-dasharray', `${l.len} ${l.len}`); l.p.setAttribute('stroke-dashoffset', (l.len * (1 - e)).toFixed(1)); }
            const o = this.u >= 1 ? 1 : Math.max(0, (this.u - .85) / .15);
            l.lab.setAttribute('opacity', o.toFixed(2)); l.lab0.setAttribute('opacity', Math.min(1, this.u * 4).toFixed(2));
        });
    },
    prep() { this.u = 0; this.place(); },
    play() { this.u = 0; this.place(); tween(this, 1500, t => { this.u = t; this.place(); }); },
});

/* ================================================================
   wiring
   ================================================================ */
const HOSTS = { 1: '#v1', 2: '#v2', 3: '#v3', 4: '#v4', 5: '#v5a', 6: '#v6a', scale: '#vs', mono: '#vm-toys', sys: '#vy-cal', seas: '#vss', small: '#vt' };
let inited = false;

function startPlay(sc) { if (sc.played) return; sc.played = true; sc.play(); }

async function init() {
    [MD, IND, AUD, MKT, MN, REV] = await Promise.all([loadJSON('data/site_measurement_data.json'), loadJSON('data/site_indices.json'),
        loadJSON('data/site_audit.json'), loadJSON('data/site_markets_pop.json').catch(() => null), loadJSON('data/site_monoculture.json'),
        loadJSON('data/site_measurement_rev.json')]);
    DIST = Object.values(MN.districts);
    buildPresets();
    SC[3].data();
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
        }), { threshold: 0.35 });
        Object.keys(HOSTS).forEach(k => { const h = $(HOSTS[k]); if (!h) return; h.dataset.scene = k; io.observe(h); });
    } else {
        Object.values(SC).forEach(startPlay);
    }
    document.querySelectorAll('.replay').forEach(b => b.addEventListener('click', () => {
        const sc = SC[b.dataset.scene]; if (!sc) return;
        cancel(sc); sc.played = true;
        sc.play();
    }));
    inited = true;
}

let lastW = 0;
async function drawAll() {
    if (!inited) { await init(); lastW = window.innerWidth; return; }
    if (window.innerWidth === lastW) return;
    lastW = window.innerWidth;
    for (const k of Object.keys(SC)) {
        cancel(SC[k]);
        if (SC[k].u !== undefined) SC[k].u = 1;
        if (SC[k].ud !== undefined) SC[k].ud = 1;
        try { await SC[k].render(); } catch (e) { console.error('scene ' + k, e); }
        if (!SC[k].played && !RM && SC[k].prep) SC[k].prep();
    }
}
window.drawAll = drawAll;
drawAll();
})();
