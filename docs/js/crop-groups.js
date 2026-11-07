/* ================================================================
   CROP GROUPS PAGE: scene visuals.
   Reads data/site_crop_groups.json and data/districts.geojson.
   Depends on site.js for svgEl, showTip, hideTip, loadJSON, loadGeo
   and rampCyan. Everything is wrapped so no name collides with
   site.js globals.

   Every scene draws with a progress value u in [0, 1]: render() builds
   the visual, draw(u) sets the marks for that point of the entrance,
   play() tweens u from 0 to 1. Under prefers-reduced-motion u is 1
   at once. drawAll() re-renders on resize.
   ================================================================ */
(function () {
'use strict';

const RM = !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
const TX = '#EDEDEB', TX2 = '#B3B3B0', TX3 = '#7E7E7C';
const GRID = 'rgba(255,255,255,0.075)', ZERO = 'rgba(255,255,255,0.28)';
const CY = '#0E9BB5', MG = '#C63FD8', VI = '#7C5CE0';
const NODATA = 'rgba(255,255,255,0.035)';
const NAME = { rice: 'rice', wheat: 'wheat', maize: 'maize', millets: 'millets', pulses: 'pulses',
               oilseeds: 'oilseeds', cotton: 'cotton', sugarcane: 'sugarcane', other: 'other crops' };
const CAP = s => s.charAt(0).toUpperCase() + s.slice(1);
let D = null;

const $ = s => document.querySelector(s);
function E(tag, attrs, parent) { const e = svgEl(tag, attrs || {}); if (parent) parent.appendChild(e); return e; }
function T(parent, attrs, s) { const t = E('text', attrs, parent); t.textContent = s; return t; }
const sgn = (v, d) => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(d);
const fy = y => y + '–' + String((y + 1) % 100).padStart(2, '0');
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
function tipOn(el, html) {
    const f = ev => showTip(html(), ev);
    el.addEventListener('mousemove', f);
    el.addEventListener('click', f);
    el.addEventListener('mouseleave', hideTip);
}

/* ---------- motion ---------- */
function tween(sc, dur, fn) {
    if (RM || dur <= 0) { fn(1); return; }
    const token = ++sc.run, t0 = performance.now();
    (function step(now) {
        if (sc.run !== token) return;
        const t = Math.min(1, (now - t0) / dur);
        fn(t);
        if (t < 1) requestAnimationFrame(step);
    })(t0);
}
const scene = o => Object.assign({ run: 0, played: false, u: 0 }, o);

function segs(host, items, onPick, cur, label) {
    host.innerHTML = '';
    if (label) { const l = document.createElement('span'); l.className = 'ctl__lab'; l.textContent = label; host.appendChild(l); }
    items.forEach(it => {
        const b = document.createElement('button');
        b.type = 'button'; b.className = 'seg'; b.dataset.key = it.key;
        b.setAttribute('aria-pressed', it.key === cur ? 'true' : 'false');
        b.textContent = it.name;
        b.addEventListener('click', () => {
            host.querySelectorAll('.seg').forEach(x => x.setAttribute('aria-pressed', x === b ? 'true' : 'false'));
            onPick(it.key);
        });
        host.appendChild(b);
    });
}

const SC = {};

/* ================================================================
   01  map: share in 2017-19, or change since 1998-2000
   ================================================================ */
function projectGeo(geo, W, H, keyH) {
    let x0 = 180, x1 = -180, y0 = 90, y1 = -90;
    const walk = c => {
        if (typeof c[0] === 'number') { x0 = Math.min(x0, c[0]); x1 = Math.max(x1, c[0]); y0 = Math.min(y0, c[1]); y1 = Math.max(y1, c[1]); }
        else c.forEach(walk);
    };
    geo.features.forEach(f => walk(f.geometry.coordinates));
    const pad = 6, kx = Math.cos((y0 + y1) / 2 * Math.PI / 180);
    const s = Math.min((W - pad * 2) / ((x1 - x0) * kx), (H - pad * 2 - keyH) / (y1 - y0));
    const ox = pad + ((W - pad * 2) - (x1 - x0) * kx * s) / 2, oy = pad + ((H - pad * 2 - keyH) - (y1 - y0) * s) / 2;
    const PX = v => ox + (v - x0) * kx * s, PY = v => oy + (y1 - v) * s;
    const ring = c => c.map((p, i) => (i ? 'L' : 'M') + PX(p[0]).toFixed(1) + ',' + PY(p[1]).toFixed(1)).join('') + 'Z';
    const path = (c, d) => d === 1 ? ring(c) : c.map(k => path(k, d - 1)).join('');
    return { ring, fpath: f => path(f.geometry.coordinates, f.geometry.type === 'Polygon' ? 2 : 3) };
}
function hatchDef(svg, id) {
    const pat = E('pattern', { id, width: 5, height: 5, patternUnits: 'userSpaceOnUse', patternTransform: 'rotate(45)' }, E('defs', {}, svg));
    E('line', { x1: 0, y1: 0, x2: 0, y2: 5, stroke: 'rgba(255,255,255,0.20)', 'stroke-width': 1 }, pat);
    return `url(#${id})`;
}
/* diverging: magenta for a loss, cyan for a gain, the surface tone at zero */
function divRamp(t) {
    const z = [30, 31, 34], a = t < 0 ? [198, 63, 216] : [64, 214, 240];
    const k = Math.pow(Math.min(1, Math.abs(t)), 0.8);
    const c = z.map((v, i) => Math.round(v + (a[i] - v) * k));
    return `rgb(${c[0]},${c[1]},${c[2]})`;
}
const unitName = k => k.split('|')[1].toLowerCase().replace(/\b\w/g, c => c.toUpperCase());
SC[1] = scene({
    g: 'millets', mode: 'ds',
    controls() {
        segs($('#v1-group'), D.groups.map(g => ({ key: g, name: NAME[g] })), k => { this.g = k; this.recolour(); this.readout(); }, this.g);
        segs($('#v1-mode'), [{ key: 'ds', name: 'change since 1998–2000' }, { key: 's1', name: 'share in 2017–19' }],
            k => { this.mode = k; this.recolour(); this.readout(); }, this.mode);
    },
    scale() {
        const gi = D.groups.indexOf(this.g);
        const v = Object.values(D.units).map(u => u[this.mode][gi]).filter(x => x != null);
        if (this.mode === 's1') {
            const s = v.slice().sort((a, b) => a - b);
            return { gi, lo: 0, hi: Math.max(1, s[Math.floor(s.length * .98)]) };
        }
        const a = v.map(Math.abs).sort((x, y) => x - y);
        return { gi, m: Math.max(1, a[Math.floor(a.length * .98)]) };
    },
    colour(u, S) {
        const v = u[this.mode][S.gi];
        if (v == null) return NODATA;
        return this.mode === 's1' ? rampCyan(Math.max(0, Math.min(1, (v - S.lo) / (S.hi - S.lo)))) : divRamp(Math.max(-1, Math.min(1, v / S.m)));
    },
    readout() {
        const n = D.national[this.g];
        $('#v1-read').textContent = this.mode === 'ds'
            ? `${CAP(NAME[this.g])}: ${sgn(n.d, 1)} points of gross cropped area nationally, ${n.down} districts down and ${n.up} up.`
            : `${CAP(NAME[this.g])}: ${n.s1.toFixed(1)} percent of gross cropped area nationally in 2017–19, from ${n.s0.toFixed(1)} percent in 1998–2000.`;
    },
    async render() {
        const host = $('#v1'); if (!host) return;
        const geo = await loadGeo();
        const W = host.clientWidth || 900, nar = W < 560;
        const H = Math.round(Math.min(nar ? 470 : 640, W * 1.1));
        const svg = svgFor(host, W, H);
        this.svg = svg; this.W = W; this.H = H;
        this.hatch = hatchDef(svg, 'h1' + Math.random().toString(36).slice(2, 7));
        const P = projectGeo(geo, W, H, 52);
        this.paths = [];
        const g = E('g', {}, svg);
        geo.features.forEach(f => {
            const pr = f.properties, uk = D.poly[pr.k], u = uk ? D.units[uk] : null;
            const el = E('path', { d: P.fpath(f), stroke: 'rgba(11,11,12,0.8)', 'stroke-width': 0.35 }, g);
            tipOn(el, () => {
                if (!u) return `<strong>${pr.n}</strong><br>${pr.s}<br>no usable record in one of the two periods`;
                const gi = D.groups.indexOf(this.g);
                const same = unitName(uk).toLowerCase() === String(pr.n).toLowerCase();
                return `<strong>${pr.n}</strong><br>${pr.s}` + (same ? '' : `<br>counted within ${unitName(uk)} (2011 boundary)`) +
                    `<br>${NAME[this.g]}: ${u.s1[gi].toFixed(1)}% of gross cropped area in 2017–19` +
                    `<br>change since 1998–2000: ${sgn(u.ds[gi], 1)} points`;
            });
            el.addEventListener('mouseenter', () => { el.setAttribute('stroke', TX); el.setAttribute('stroke-width', 1.1); });
            el.addEventListener('mouseleave', () => { el.setAttribute('stroke', 'rgba(11,11,12,0.8)'); el.setAttribute('stroke-width', 0.35); });
            this.paths.push({ el, u });
        });
        const so = E('g', { 'pointer-events': 'none' }, svg);
        D.state_rings.forEach(r => E('path', { d: P.ring(r), fill: 'none', stroke: 'rgba(237,237,235,0.42)', 'stroke-width': 0.7, 'stroke-linejoin': 'round' }, so));
        this.key = E('g', {}, svg);
        this.recolour();
        this.readout();
        this.draw(this.played || RM ? 1 : 0);
    },
    recolour() {
        const S = this.scale();
        this.paths.forEach(p => { p.el.style.fill = p.u ? this.colour(p.u, S) : this.hatch; });
        /* key */
        const k = this.key; k.innerHTML = '';
        const W = this.W, H = this.H, kw = Math.min(260, W * .55), kx0 = W - kw - 8, ky = H - 34;
        const gid = 'g1' + Math.random().toString(36).slice(2, 7);
        const lg = E('linearGradient', { id: gid, x1: 0, x2: 1, y1: 0, y2: 0 }, E('defs', {}, k));
        for (let i = 0; i <= 10; i++) E('stop', { offset: i * 10 + '%', 'stop-color': this.mode === 's1' ? rampCyan(i / 10) : divRamp(i / 5 - 1) }, lg);
        E('rect', { x: kx0, y: ky, width: kw, height: 9, fill: `url(#${gid})`, stroke: 'rgba(255,255,255,0.18)', 'stroke-width': .6 }, k);
        const lab = this.mode === 's1'
            ? [['0%', 'start'], [S.hi.toFixed(0) + '%+', 'end']]
            : [['−' + S.m.toFixed(0) + ' pts', 'start'], ['0', 'middle'], ['+' + S.m.toFixed(0) + ' pts', 'end']];
        lab.forEach((l, i) => T(k, { x: kx0 + (lab.length === 2 ? i * kw : i * kw / 2), y: ky + 22, 'text-anchor': l[1], fill: TX3, 'font-size': 10.5, class: 'mono' }, l[0]));
        T(k, { x: kx0, y: ky - 6, fill: TX2, 'font-size': 10.5, class: 'mono' },
            this.mode === 's1' ? `${NAME[this.g]}, share of gross cropped area` : `${NAME[this.g]}, change in points of share`);
        E('rect', { x: 8, y: ky, width: 12, height: 9, fill: this.hatch, stroke: 'rgba(255,255,255,0.18)', 'stroke-width': .6 }, k);
        T(k, { x: 26, y: ky + 8, fill: TX3, 'font-size': 10.5, class: 'mono' }, 'not compared');
    },
    draw(u) { this.u = u; if (this.paths) this.paths.forEach(p => { p.el.style.opacity = Math.min(1, u * 1.4); }); },
    play() { tween(this, 900, t => this.draw(easeOut(t))); },
});

/* ================================================================
   02  nine small panels, national share by year
   ================================================================ */
SC[2] = scene({
    async render() {
        const host = $('#v2'); if (!host) return;
        const W = host.clientWidth || 900, cols = W < 560 ? 2 : 3, rows = Math.ceil(D.groups.length / cols);
        const gap = W < 560 ? 14 : 26, pw = (W - gap * (cols - 1)) / cols, ph = W < 560 ? 132 : 150;
        const H = rows * ph + (rows - 1) * 16 + 4;
        const svg = svgFor(host, W, H);
        const S = D.series, yrs = S.years, SPAN = 8;
        this.lines = [];
        D.groups.forEach((g, i) => {
            const c = i % cols, r = Math.floor(i / cols);
            const ox = c * (pw + gap), oy = r * (ph + 16);
            const v = S.aw[g], mid = (Math.min(...v) + Math.max(...v)) / 2;
            const lo = mid - SPAN / 2, hi = mid + SPAN / 2;
            const m = { l: 40, r: 40, t: 36, b: 18 };
            const X = y => ox + m.l + (y - yrs[0]) / (yrs[yrs.length - 1] - yrs[0]) * (pw - m.l - m.r);
            const Y = s => oy + m.t + (hi - s) / SPAN * (ph - m.t - m.b);
            const pg = E('g', {}, svg);
            E('rect', { x: ox + m.l - 4, y: oy + m.t - 4, width: pw - m.l - m.r + 8, height: ph - m.t - m.b + 8, fill: 'rgba(255,255,255,0.018)' }, pg);
            const n = D.national[g], col = n.d <= -1 ? MG : (n.d >= 1 ? CY : TX2);
            T(pg, { x: ox, y: oy + 13, fill: TX, 'font-size': 12.5, class: 'mono' }, NAME[g]);
            T(pg, { x: ox, y: oy + 28, fill: col, 'font-size': 11, class: 'mono' }, `${sgn(n.d, 1)} pts (${n.se.toFixed(1)})`);
            /* year ticks at the panel foot */
            [yrs[0], yrs[yrs.length - 1]].forEach((y, j) => T(pg, { x: X(y), y: oy + ph - 3, 'text-anchor': j ? 'end' : 'start', fill: TX3, 'font-size': 9.5, class: 'mono' }, fy(y)));
            const d = v.map((s, j) => (j ? 'L' : 'M') + X(yrs[j]).toFixed(1) + ',' + Y(s).toFixed(1)).join('');
            const ln = E('path', { d, fill: 'none', stroke: col, 'stroke-width': 1.8, 'stroke-linejoin': 'round' }, pg);
            const a = E('circle', { cx: X(yrs[0]), cy: Y(v[0]), r: 2.6, fill: col }, pg);
            const b = E('circle', { cx: X(yrs[yrs.length - 1]), cy: Y(v[v.length - 1]), r: 2.6, fill: col }, pg);
            T(pg, { x: X(yrs[yrs.length - 1]) + 6, y: Y(v[v.length - 1]) + 3.5, fill: TX2, 'font-size': 10, class: 'mono' }, v[v.length - 1].toFixed(1) + '%');
            T(pg, { x: X(yrs[0]) - 6, y: Y(v[0]) + 3.5, 'text-anchor': 'end', fill: TX3, 'font-size': 10, class: 'mono' }, v[0].toFixed(1) + '%');
            const hit = E('rect', { x: ox, y: oy, width: pw, height: ph, fill: 'transparent' }, pg);
            hit.addEventListener('mousemove', ev => {
                const px = ev.offsetX * (W / (host.clientWidth || W));
                let j = Math.round((px - ox - m.l) / (pw - m.l - m.r) * (yrs.length - 1));
                j = Math.max(0, Math.min(yrs.length - 1, j));
                showTip(`<strong>${CAP(NAME[g])}, ${fy(yrs[j])}</strong><br>${v[j].toFixed(1)}% of gross cropped area`, ev);
            });
            hit.addEventListener('mouseleave', hideTip);
            this.lines.push({ ln, a, b, len: null });
        });
        this.draw(this.played || RM ? 1 : 0);
    },
    draw(u) {
        this.lines.forEach(o => {
            if (o.len == null) { try { o.len = o.ln.getTotalLength(); } catch (e) { o.len = 1000; } }
            o.ln.setAttribute('stroke-dasharray', o.len);
            o.ln.setAttribute('stroke-dashoffset', o.len * (1 - u));
            o.b.style.opacity = u >= 1 ? 1 : 0;
        });
    },
    play() { tween(this, 1400, t => this.draw(easeOut(t))); },
});

/* ================================================================
   03  rice area in Andhra Pradesh and Telangana, kharif and rabi
   ================================================================ */
SC[3] = scene({
    async render() {
        const hosts = [['ap', $('#v3a')], ['ts', $('#v3b')]];
        if (!hosts[0][1]) return;
        const ymax = 3400;
        this.bars = [];
        hosts.forEach(([k, host]) => {
            const A = D.apts[k], W = host.clientWidth || 440, H = 300;
            const svg = svgFor(host, W, H);
            const m = { l: 44, r: 6, t: 10, b: 30 };
            const n = A.years.length, bw = (W - m.l - m.r) / n;
            const Y = v => m.t + (1 - v / ymax) * (H - m.t - m.b);
            niceTicks(0, ymax, 4).forEach(t => {
                E('line', { x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t), stroke: t ? GRID : ZERO }, svg);
                T(svg, { x: m.l - 6, y: Y(t) + 3.5, 'text-anchor': 'end', fill: TX3, 'font-size': 10, class: 'mono' }, t ? (t / 1000).toFixed(1) : '0');
            });
            T(svg, { x: m.l - 6, y: m.t - 0, 'text-anchor': 'end', fill: TX3, 'font-size': 9.5, class: 'mono' }, '');
            A.years.forEach((y, i) => {
                const x = m.l + i * bw + bw * .14, w = bw * .72, last = y === 2019;
                const kh = A.kharif[i], rb = A.rabi[i];
                const r1 = E('rect', { x, width: w, fill: last ? 'none' : 'rgba(237,237,235,0.38)', stroke: last ? 'rgba(237,237,235,0.55)' : 'none', 'stroke-width': .8 }, svg);
                const r2 = E('rect', { x, width: w, fill: last ? 'none' : CY, stroke: last ? CY : 'none', 'stroke-width': .8 }, svg);
                const hit = E('rect', { x: m.l + i * bw, y: m.t, width: bw, height: H - m.t - m.b, fill: 'transparent' }, svg);
                tipOn(hit, () => `<strong>${k === 'ap' ? 'Andhra Pradesh' : 'Telangana'}, ${fy(y)}</strong><br>kharif rice ${Math.round(kh)} thousand ha<br>rabi rice ${Math.round(rb)} thousand ha<br>rice share of gross cropped area ${A.share[i].toFixed(1)}%`);
                this.bars.push({ r1, r2, kh, rb, Y });
                if ((i % 5 === 0 && i < n - 3) || last) T(svg, { x: x + w / 2, y: H - m.b + 14, 'text-anchor': 'middle', fill: last ? TX2 : TX3, 'font-size': 9.5, class: 'mono' }, last ? '19–20' : String(y));
            });
            T(svg, { x: 2, y: H - 2, fill: TX3, 'font-size': 9.5, class: 'mono' }, 'million ha');
        });
        this.draw(this.played || RM ? 1 : 0);
    },
    draw(u) {
        (this.bars || []).forEach(b => {
            const kh = b.kh * u, rb = b.rb * u;
            b.r1.setAttribute('y', b.Y(kh)); b.r1.setAttribute('height', Math.max(0, b.Y(0) - b.Y(kh)));
            b.r2.setAttribute('y', b.Y(kh + rb)); b.r2.setAttribute('height', Math.max(0, b.Y(kh) - b.Y(kh + rb)));
        });
    },
    play() { tween(this, 1100, t => this.draw(easeOut(t))); },
});

/* ================================================================
   04  NFSM 2007: event-time coefficients
   ================================================================ */
SC[4] = scene({
    k: 'pulses',
    controls() {
        segs($('#v4-ctl'), [{ key: 'pulses', name: 'pulses mission: pulse share' }, { key: 'rice', name: 'rice mission: rice share' },
                            { key: 'wheat', name: 'wheat mission: wheat share' }],
            k => { this.k = k; this.render(); }, this.k);
    },
    readout() {
        const P = D.policy[this.k];
        $('#v4-read').textContent = this.k === 'pulses'
            ? `Mean pulse share before entry ${P.mean.toFixed(0)}%. Change at entry against 2002–07: ${sgn(P.w02, 2)} points (${P.w02se.toFixed(2)}).`
            : `Mean ${this.k} share before entry ${P.mean.toFixed(0)}%. Average after entry against before: ${sgn(P.did, 2)} points (${P.didse.toFixed(2)}).`;
    },
    async render() {
        const host = $('#v4'); if (!host) return;
        const P = D.policy[this.k];
        const W = host.clientWidth || 900, nar = W < 560, H = nar ? 300 : 340;
        const svg = svgFor(host, W, H);
        const m = { l: nar ? 40 : 48, r: 10, t: 26, b: 44 };
        const e0 = -9, e1 = 6;
        const X = e => m.l + (e - e0 + .5) / (e1 - e0 + 1) * (W - m.l - m.r);
        const ext = P.b.flatMap((b, i) => [b - 1.96 * P.se[i], b + 1.96 * P.se[i]]);
        const lo = Math.min(-3, Math.floor(Math.min(...ext))), hi = Math.max(3, Math.ceil(Math.max(...ext)) + 1);
        const Y = v => m.t + (hi - v) / (hi - lo) * (H - m.t - m.b);
        niceTicks(lo, hi, 5).forEach(t => {
            E('line', { x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t), stroke: t ? GRID : ZERO }, svg);
            T(svg, { x: m.l - 6, y: Y(t) + 3.5, 'text-anchor': 'end', fill: TX3, 'font-size': 10, class: 'mono' }, t > 0 ? '+' + t : (t < 0 ? '−' + (-t) : '0'));
        });
        if (this.k === 'pulses') {
            E('rect', { x: X(-5.5), y: m.t, width: X(-.5) - X(-5.5), height: H - m.t - m.b, fill: 'rgba(124,92,224,0.10)' }, svg);
            T(svg, { x: (X(-5.5) + X(-.5)) / 2, y: m.t + 12, 'text-anchor': 'middle', fill: '#A996F0', 'font-size': 10, class: 'mono' }, '2002–07 reference');
        }
        E('line', { x1: X(-.5), x2: X(-.5), y1: m.t, y2: H - m.b, stroke: 'rgba(255,255,255,0.35)', 'stroke-dasharray': '3 3' }, svg);
        T(svg, { x: X(-.5) + 5, y: m.t + 12, fill: TX2, 'font-size': 10, class: 'mono' }, 'entry 2007–08');
        for (let e = e0; e <= e1; e += 2) {
            T(svg, { x: X(e), y: H - m.b + 15, 'text-anchor': 'middle', fill: TX3, 'font-size': 10, class: 'mono' }, e > 0 ? '+' + e : (e < 0 ? '−' + (-e) : '0'));
            T(svg, { x: X(e), y: H - m.b + 28, 'text-anchor': 'middle', fill: TX3, 'font-size': 9, class: 'mono' }, String(2007 + e));
        }
        T(svg, { x: 2, y: 10, fill: TX3, 'font-size': 9.5, class: 'mono' }, 'points of share');
        this.pts = [];
        P.e.forEach((e, i) => {
            const b = P.b[i], s = P.se[i], ref = e === -1;
            const ln = E('line', { x1: X(e), x2: X(e), y1: Y(b - 1.96 * s), y2: Y(b + 1.96 * s), stroke: e >= 0 ? CY : TX2, 'stroke-width': 1.4 }, svg);
            const c = E('circle', { cx: X(e), cy: Y(b), r: ref ? 3.6 : 4, fill: ref ? 'none' : (e >= 0 ? CY : TX2), stroke: ref ? TX2 : 'none', 'stroke-width': 1.3 }, svg);
            const hit = E('rect', { x: X(e) - 12, y: m.t, width: 24, height: H - m.t - m.b, fill: 'transparent' }, svg);
            tipOn(hit, () => `<strong>${e >= 0 ? (e ? e + ' years after' : 'year of') + ' entry' : -e + (e === -1 ? ' year' : ' years') + ' before entry'}, ${fy(2007 + e)}</strong><br>` +
                (ref ? 'reference year, set to zero' : `${sgn(b, 2)} points of share against 2006–07<br>95% interval ${sgn(b - 1.96 * s, 2)} to ${sgn(b + 1.96 * s, 2)}`));
            this.pts.push({ ln, c, i });
        });
        this.readout();
        this.draw(this.played || RM ? 1 : 0);
    },
    draw(u) {
        const n = this.pts.length;
        this.pts.forEach(p => { const on = u >= (p.i + 1) / n - 1e-9 ? 1 : 0; p.ln.style.opacity = on; p.c.style.opacity = on; });
    },
    play() { tween(this, 1100, t => this.draw(t)); },
});

/* ================================================================
   05  rainfall: dose-response by June-September departure
   ================================================================ */
const RAIN = [
    { key: 'total', name: 'total rabi area', who: 'all districts', after: 'the following rabi' },
    { key: 'rice', name: 'rabi rice area', who: 'districts that grow rabi rice', after: 'the following rabi' },
    { key: 'pulses', name: 'rabi pulse area', who: 'least irrigated third of districts', after: 'the following rabi' },
    { key: 'millets', name: 'next-kharif millet area', who: 'millet-growing districts', after: 'the next kharif' },
];
const pct = b => 100 * (Math.exp(b) - 1);
SC[5] = scene({
    k: 'total',
    controls() {
        segs($('#v5-ctl'), RAIN.map(r => ({ key: r.key, name: r.name })), k => { this.k = k; this.render(); }, this.k);
    },
    readout() {
        const R = D.rain[this.k], L = RAIN.find(r => r.key === this.k), d = R.dummy;
        const what = this.k === 'millets' ? 'a monsoon 20% or more above normal' : 'a monsoon 20% or more below normal';
        $('#v5-read').textContent = `${CAP(L.who)}, ${R.units} of them. After ${what}: ${sgn(pct(d.b), 1)}%.`;
    },
    async render() {
        const host = $('#v5'); if (!host) return;
        const R = D.rain[this.k];
        const W = host.clientWidth || 900, nar = W < 560, H = nar ? 300 : 330;
        const svg = svgFor(host, W, H);
        const m = { l: nar ? 40 : 48, r: 8, t: 14, b: 46 };
        const vals = R.bins.flatMap(b => [pct(b.b - 1.96 * b.se), pct(b.b + 1.96 * b.se)]);
        let lo = Math.min(-10, ...vals), hi = Math.max(10, ...vals);
        const tk = niceTicks(lo, hi, 5); lo = Math.min(lo, tk[0]); hi = Math.max(hi, tk[tk.length - 1]);
        const nb = R.bins.length;
        const X = i => m.l + (i + .5) / nb * (W - m.l - m.r);
        const Y = v => m.t + (hi - v) / (hi - lo) * (H - m.t - m.b);
        tk.forEach(t => {
            E('line', { x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t), stroke: t ? GRID : ZERO }, svg);
            T(svg, { x: m.l - 6, y: Y(t) + 3.5, 'text-anchor': 'end', fill: TX3, 'font-size': 10, class: 'mono' }, (t > 0 ? '+' : (t < 0 ? '−' : '')) + Math.abs(t) + '%');
        });
        const LB = nar ? ['≤−40', '−40/−20', '−20/−10', '±10', '+10/+20', '+20/+40', '≥+40'] : ['≤ −40%', '−40 to −20', '−20 to −10', '−10 to +10', '+10 to +20', '+20 to +40', '≥ +40%'];
        R.bins.forEach((b, i) => T(svg, { x: X(i), y: H - m.b + 15, 'text-anchor': 'middle', fill: b.bin === 'normal' ? TX2 : TX3, 'font-size': nar ? 8.5 : 10, class: 'mono' }, LB[i]));
        T(svg, { x: (m.l + W - m.r) / 2, y: H - 6, 'text-anchor': 'middle', fill: TX3, 'font-size': 10, class: 'mono' }, 'June–September rainfall against the district’s own average');
        this.pts = [];
        const dry = this.k !== 'millets';
        R.bins.forEach((b, i) => {
            const ref = b.bin === 'normal';
            const key = dry ? (b.bin === 'le_m40' || b.bin === 'm40_m20') : (b.bin === 'p20_p40' || b.bin === 'ge_p40');
            const col = ref ? TX2 : (key ? (dry ? MG : CY) : TX2);
            const ln = ref ? null : E('line', { x1: X(i), x2: X(i), y1: Y(pct(b.b - 1.96 * b.se)), y2: Y(pct(b.b + 1.96 * b.se)), stroke: col, 'stroke-width': 1.5 }, svg);
            const c = E('circle', { cx: X(i), cy: Y(pct(b.b)), r: ref ? 3.6 : 4.4, fill: ref ? 'none' : col, stroke: ref ? TX2 : 'none', 'stroke-width': 1.3 }, svg);
            const hit = E('rect', { x: X(i) - (W - m.l - m.r) / nb / 2, y: m.t, width: (W - m.l - m.r) / nb, height: H - m.t - m.b, fill: 'transparent' }, svg);
            tipOn(hit, () => `<strong>${D.rain.labels[i]}% of average rainfall</strong><br>` + (ref
                ? `reference band, ${b.n} district-years`
                : `${sgn(pct(b.b), 1)}% against a normal year<br>95% interval ${sgn(pct(b.b - 1.96 * b.se), 1)} to ${sgn(pct(b.b + 1.96 * b.se), 1)}%<br>${b.n} district-years`));
            this.pts.push({ ln, c, i });
        });
        this.readout();
        this.draw(this.played || RM ? 1 : 0);
    },
    draw(u) {
        const n = this.pts.length;
        this.pts.forEach(p => { const on = u >= (p.i + 1) / n - 1e-9 ? 1 : 0; if (p.ln) p.ln.style.opacity = on; p.c.style.opacity = on; });
    },
    play() { tween(this, 900, t => this.draw(t)); },
});

/* ---------- crop mapping table ---------- */
function tables() {
    const tb = $('#tbl-map tbody'); if (!tb) return;
    tb.innerHTML = D.groups.map(g => `<tr><td class="grp">${CAP(NAME[g])}</td><td class="crops">${D.mapping[g].map(c => c.replace(/\s*&\s*/g, ' & ').replace(/(\S)\(/g, '$1 (')).join(', ')}</td></tr>`).join('');
}

/* ================================================================
   wiring
   ================================================================ */
const HOSTS = { 1: '#v1', 2: '#v2', 3: '#v3a', 4: '#v4', 5: '#v5' };
let inited = false;
function startPlay(sc) { if (sc.played) return; sc.played = true; sc.play(); }

async function init() {
    D = await loadJSON('data/site_crop_groups.json');
    Object.values(SC).forEach(sc => { if (sc.controls) sc.controls(); });
    tables();
    for (const k of Object.keys(SC)) {
        try { await SC[k].render(); } catch (e) { console.error('scene ' + k, e); }
    }
    if (RM) {
        Object.values(SC).forEach(sc => { sc.played = true; sc.draw(1); });
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
        sc.played = true; sc.draw(0); sc.play();
    }));
    inited = true;
}
async function drawAll() {
    if (!inited) { await init(); return; }
    for (const k of Object.keys(SC)) {
        try { await SC[k].render(); } catch (e) { console.error('scene ' + k, e); }
    }
}
window.drawAll = drawAll;
drawAll();
})();
