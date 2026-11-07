/* ================================================================
   DATA AND METHODS PAGE: scene visuals.
   Reads data/site_data_*.json and data/site_indices.json.
   Depends on site.js for svgEl, showTip, hideTip and loadJSON.
   Everything is wrapped so no name collides with site.js globals.

   Pattern for every scene (as on the markets page):
     render()  draws the visual for the scene's current state
     prep()    sets the pre-entrance state (only before first play)
     play()    runs the entrance once the visual is in view
   drawAll() re-renders every scene on resize, so render() is idempotent.
   Under prefers-reduced-motion every tween resolves at once.
   ================================================================ */
(function () {
'use strict';

const RM = !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);

const C1 = '#0E9BB5', C2 = '#7C5CE0', C3 = '#1FA35A', C4 = '#C63FD8', NEU = '#B3B3B0';
const TX = '#EDEDEB', TX2 = '#B3B3B0', TX3 = '#7E7E7C', BG = '#0B0B0C';
const GRID = 'rgba(255,255,255,0.075)', AXIS = 'rgba(255,255,255,0.22)', HAIR = 'rgba(255,255,255,0.16)';

const $ = s => document.querySelector(s);
function E(tag, attrs, parent) { const e = svgEl(tag, attrs || {}); if (parent) parent.appendChild(e); return e; }
function T(parent, attrs, s) { const t = E('text', attrs, parent); t.textContent = s; return t; }
const lerp = (a, b, t) => a + (b - a) * t;
const easeOut = t => 1 - Math.pow(1 - t, 3);
const fmt1 = v => v.toFixed(1);
const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;');
const yl = y => y + '–' + String((y + 1) % 100).padStart(2, '0');

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
/* text wrapped onto lines of at most maxW px (approximate glyph width) */
function wrapText(s, maxW, fs) {
    const cw = fs * 0.55, words = String(s).split(' '), lines = [];
    let cur = '';
    words.forEach(w => {
        const nx = cur ? cur + ' ' + w : w;
        if (nx.length * cw > maxW && cur) { lines.push(cur); cur = w; } else cur = nx;
    });
    if (cur) lines.push(cur);
    return lines;
}
function textLines(g, x, y, lines, attrs, lh) {
    const t = E('text', Object.assign({ x: x, y: y }, attrs), g);
    lines.forEach((ln, i) => { const sp = E('tspan', i ? { x: x, dy: lh } : { x: x }, t); sp.textContent = ln; });
    return t;
}
function mulberry(seed) {
    return function () {
        seed |= 0; seed = seed + 0x6D2B79F5 | 0;
        let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
        t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
        return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
}
function pearson(xs, ys) {
    const n = xs.length, mx = xs.reduce((a, b) => a + b, 0) / n, my = ys.reduce((a, b) => a + b, 0) / n;
    let sxy = 0, sxx = 0, syy = 0;
    for (let i = 0; i < n; i++) { const dx = xs[i] - mx, dy = ys[i] - my; sxy += dx * dy; sxx += dx * dx; syy += dy * dy; }
    return sxy / Math.sqrt(sxx * syy);
}
function median(a) { const s = a.slice().sort((x, y) => x - y), m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; }
const mean = a => a.reduce((x, y) => x + y, 0) / a.length;
/* hover on a mouse, tap on a touch screen */
function tipOn(el, html) {
    el.addEventListener('mousemove', ev => showTip(typeof html === 'function' ? html() : html, ev));
    el.addEventListener('mouseleave', hideTip);
    el.addEventListener('click', ev => { ev.stopPropagation(); showTip(typeof html === 'function' ? html() : html, ev); });
}

/* ---------- motion ---------- */
function tween(sc, dur, fn, done) {
    if (RM || dur <= 0) { fn(1); if (done) done(); return; }
    const token = sc.run, t0 = performance.now();
    (function step(now) {
        if (sc.run !== token) return;
        const t = Math.min(1, Math.max(0, (now - t0) / dur));
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
        b.textContent = it.name;
        b.dataset.key = it.key;
        b.addEventListener('click', () => { pressSeg(host, it.key); onPick(it.key); });
        host.appendChild(b);
    });
}
function pressSeg(host, key) {
    [...host.querySelectorAll('.seg')].forEach(b => b.setAttribute('aria-pressed', b.dataset.key === key ? 'true' : 'false'));
}

let SRC, COV, CW, FUN, CHK, MOD, IDX, K606;
const SC = {};

/* ================================================================
   01 sources to measure groups
   ================================================================ */
const SCOL = { crop: C1, ay: C2, vd: C3, pca: C3, secc: NEU, ec: NEU, nl: NEU };
const VILLAGE = { ay: 1, vd: 1, pca: 1, secc: 1 };
SC[1] = scene({
    sel: null, selG: null, u: 1,
    srcName(s) { return s.name === 'SECC' ? 'Socio-Economic and Caste Census' : s.name; },
    readout() {
        const host = $('#v1-read'); if (!host) return;
        const gName = {}; SRC.groups.forEach(g => { gName[g.id] = g.name; });
        if (this.sel) {
            const s = SRC.sources.find(x => x.id === this.sel);
            const fl = SRC.flows.filter(f => f.s === s.id);
            host.innerHTML = `<span class="readout__name">${esc(this.srcName(s))} · ${s.year} · ${s.level === 'village' ? 'village records summed to the district' : 'district figures'} · ${s.n_vars} variables</span>` +
                '<div class="ro-cols">' + fl.map(f => `<div><b>${esc(gName[f.g])}</b> <span>(${f.n})</span><br>${f.vars.map(esc).join(', ')}</div>`).join('') + '</div>';
        } else if (this.selG) {
            const g = SRC.groups.find(x => x.id === this.selG);
            const fl = SRC.flows.filter(f => f.g === g.id);
            const nm = {}; SRC.sources.forEach(s => { nm[s.id] = this.srcName(s); });
            host.innerHTML = `<span class="readout__name">${esc(g.name)} · ${fl.reduce((a, f) => a + f.n, 0)} variables</span>` +
                '<div class="ro-cols">' + fl.map(f => `<div><b>${esc(nm[f.s])}</b> <span>(${f.n})</span><br>${f.vars.map(esc).join(', ')}</div>`).join('') + '</div>';
        } else {
            const tot = SRC.sources.reduce((a, s) => a + s.n_vars, 0);
            host.innerHTML = `<span class="readout__name">Mission Antyodaya, reported by the gram panchayat</span>About half of the district measures come from what panchayats report about their own villages. Hover or tap a source or a group to list its variables.`;
        }
    },
    pick(s, g) {
        this.sel = s; this.selG = g;
        this.paint(); this.readout();
    },
    paint() {
        if (this.ribbons) this.ribbons.forEach(r => {
            const on = (!this.sel && !this.selG) || r.f.s === this.sel || r.f.g === this.selG;
            r.el.setAttribute('fill-opacity', (this.sel || this.selG) ? (on ? 0.72 : 0.06) : 0.38);
        });
        if (this.mobile) {
            [...this.mobile.querySelectorAll('.flow-m__src')].forEach(b => b.setAttribute('aria-pressed', b.dataset.id === this.sel ? 'true' : 'false'));
            const col = this.sel ? SCOL[this.sel] : TX2;
            SRC.groups.forEach(g => {
                const n = SRC.flows.filter(f => f.g === g.id && (!this.sel || f.s === this.sel)).reduce((a, f) => a + f.n, 0);
                const row = this.mobile.querySelector(`[data-g="${g.id}"]`); if (!row) return;
                row.querySelector('b').style.width = Math.min(100, n / 54 * 100) + '%';
                row.querySelector('b').style.background = col;
                row.querySelector('.n').textContent = n;
            });
        }
    },
    render() {
        const host = $('#v1'); if (!host) return;
        const W = host.clientWidth || 900;
        this.ribbons = null; this.mobile = null; this.clip = null;
        if (W < 640) { this.renderMobile(host); this.paint(); this.readout(); return; }
        const H = 500, top = 34, bot = 12, gapS = 14, gapG = 16;
        const LW = W < 900 ? 170 : 205, RW = W < 900 ? 170 : 215, bw = 10;
        const tot = SRC.flows.reduce((a, f) => a + f.n, 0);
        const gN = {}; SRC.groups.forEach(g => { gN[g.id] = SRC.flows.filter(f => f.g === g.id).reduce((a, f) => a + f.n, 0); });
        /* every node gets a slot tall enough for its two-line label */
        const MINS = 30, MING = 34;
        const solve = (ns, gap, mn) => {
            const avail = H - top - bot - gap * (ns.length - 1);
            let kk = avail / ns.reduce((a, b) => a + b, 0);
            for (let it = 0; it < 8; it++) {
                const fixed = ns.filter(n => n * kk < mn), free = ns.filter(n => n * kk >= mn);
                kk = (avail - fixed.length * mn) / Math.max(1, free.reduce((a, b) => a + b, 0));
            }
            return kk;
        };
        const k = Math.min(solve(SRC.sources.map(s => s.n_vars), gapS, MINS), solve(SRC.groups.map(g => gN[g.id]), gapG, MING));
        const slot = (n, mn) => Math.max(n * k, mn);
        const xS = LW, xG = W - RW - bw, xH = (xS + xG) / 2 - 6, hw = 12;
        const svg = svgFor(host, W, H);
        this.W = W;
        const cid = 'v1clip' + Math.random().toString(36).slice(2, 7);
        const defs = E('defs', {}, svg), cp = E('clipPath', { id: cid }, defs);
        this.clip = E('rect', { x: 0, y: 0, width: W * this.u, height: H }, cp);
        const sOrd = {}, gOrd = {};
        SRC.sources.forEach((s, i) => { sOrd[s.id] = i; });
        SRC.groups.forEach((g, i) => { gOrd[g.id] = i; });
        /* node positions */
        const sY = {}, gY = {};
        const sTot = SRC.sources.reduce((a, s) => a + slot(s.n_vars, MINS), 0) + gapS * (SRC.sources.length - 1);
        let y = top + (H - top - bot - sTot) / 2;
        SRC.sources.forEach(s => { const sl = slot(s.n_vars, MINS), h = s.n_vars * k, y0 = y + (sl - h) / 2; sY[s.id] = { y0: y0, y1: y0 + h }; y += sl + gapS; });
        const gTot = SRC.groups.reduce((a, g) => a + slot(gN[g.id], MING), 0) + gapG * (SRC.groups.length - 1);
        y = top + (H - top - bot - gTot) / 2;
        SRC.groups.forEach(g => { const sl = slot(gN[g.id], MING), h = gN[g.id] * k, y0 = y + (sl - h) / 2; gY[g.id] = { y0: y0, y1: y0 + h }; y += sl + gapG; });
        /* stacking inside nodes */
        const fl = SRC.flows.slice();
        const sCur = {}, gCur = {};
        Object.keys(sY).forEach(id => { sCur[id] = sY[id].y0; });
        Object.keys(gY).forEach(id => { gCur[id] = gY[id].y0; });
        fl.slice().sort((a, b) => sOrd[a.s] - sOrd[b.s] || gOrd[a.g] - gOrd[b.g]).forEach(f => { f._a0 = sCur[f.s]; sCur[f.s] += f.n * k; f._a1 = sCur[f.s]; });
        fl.slice().sort((a, b) => gOrd[a.g] - gOrd[b.g] || sOrd[a.s] - sOrd[b.s]).forEach(f => { f._b0 = gCur[f.g]; gCur[f.g] += f.n * k; f._b1 = gCur[f.g]; });
        const vil = fl.filter(f => VILLAGE[f.s]).sort((a, b) => sOrd[a.s] - sOrd[b.s] || gOrd[a.g] - gOrd[b.g]);
        const hubH = vil.reduce((a, f) => a + f.n, 0) * k;
        const hubTop = (sY.ay.y0 + sY.secc.y1) / 2 - hubH / 2;
        let hc = hubTop;
        vil.forEach(f => { f._h0 = hc; hc += f.n * k; f._h1 = hc; });
        const band = (x0, a0, a1, x1, b0, b1) => {
            const xm = (x0 + x1) / 2;
            return `M${x0},${a0}C${xm},${a0} ${xm},${b0} ${x1},${b0}L${x1},${b1}C${xm},${b1} ${xm},${a1} ${x0},${a1}Z`;
        };
        const gR = E('g', { 'clip-path': `url(#${cid})` }, svg);
        const gName = {}; SRC.groups.forEach(g => { gName[g.id] = g.name; });
        const sName = {}; SRC.sources.forEach(s => { sName[s.id] = this.srcName(s); });
        this.ribbons = [];
        fl.forEach(f => {
            let d;
            if (VILLAGE[f.s]) {
                d = band(xS + bw, f._a0, f._a1, xH, f._h0, f._h1) + band(xH + hw, f._h0, f._h1, xG, f._b0, f._b1);
                /* the stretch through the hub */
                d += `M${xH},${f._h0}H${xH + hw}V${f._h1}H${xH}Z`;
            } else {
                d = band(xS + bw, f._a0, f._a1, xG, f._b0, f._b1);
            }
            const el = E('path', { d: d, fill: SCOL[f.s], 'fill-opacity': 0.38, stroke: 'none' }, gR);
            tipOn(el, `<strong>${esc(sName[f.s])}</strong><br>${esc(gName[f.g])}: ${f.n} variable${f.n > 1 ? 's' : ''}`);
            this.ribbons.push({ el, f });
        });
        /* hub */
        E('rect', { x: xH, y: hubTop, width: hw, height: hubH, fill: 'rgba(255,255,255,0.10)', stroke: 'rgba(255,255,255,0.45)', 'stroke-width': 1 }, svg);
        T(svg, { x: xH + hw / 2, y: hubTop - 22, 'text-anchor': 'middle', fill: TX, 'font-size': 11.5, class: 'mono', stroke: BG, 'stroke-width': 4, 'paint-order': 'stroke' }, 'SHRUG village identifier');
        T(svg, { x: xH + hw / 2, y: hubTop - 8, 'text-anchor': 'middle', fill: TX2, 'font-size': 10.5, class: 'mono', stroke: BG, 'stroke-width': 4, 'paint-order': 'stroke' }, 'summed to district');
        /* source nodes */
        SRC.sources.forEach(s => {
            const p = sY[s.id], g = E('g', { style: 'cursor:pointer' }, svg);
            E('rect', { x: xS, y: p.y0, width: bw, height: Math.max(2, p.y1 - p.y0), fill: SCOL[s.id] }, g);
            const cy = (p.y0 + p.y1) / 2;
            T(g, { x: xS - 10, y: cy - 2, 'text-anchor': 'end', fill: TX, 'font-size': 12.5 }, this.srcName(s).replace('Socio-Economic and Caste Census', 'SECC'));
            T(g, { x: xS - 10, y: cy + 12, 'text-anchor': 'end', fill: TX3, 'font-size': 10.5, class: 'mono' }, `${s.year} · ${s.n_vars}`);
            E('rect', { x: 0, y: Math.min(p.y0, cy - 16), width: xS + bw + 4, height: Math.max(p.y1 - p.y0, 32), fill: 'transparent' }, g);
            g.addEventListener('mouseenter', () => this.pick(s.id, null));
            g.addEventListener('click', ev => { ev.stopPropagation(); this.pick(this.sel === s.id ? null : s.id, null); });
        });
        /* group nodes */
        SRC.groups.forEach(gr => {
            const p = gY[gr.id], g = E('g', { style: 'cursor:pointer' }, svg);
            E('rect', { x: xG, y: p.y0, width: bw, height: Math.max(2, p.y1 - p.y0), fill: TX2 }, g);
            const lines = wrapText(gr.name, RW - 18, 12.5), cy = (p.y0 + p.y1) / 2;
            textLines(g, xG + bw + 10, cy - (lines.length - 1) * 7.5 - 2, lines, { fill: TX, 'font-size': 12.5 }, 15);
            T(g, { x: xG + bw + 10, y: cy + lines.length * 7.5 + 7, fill: TX3, 'font-size': 10.5, class: 'mono' }, `${gN[gr.id]} variables`);
            E('rect', { x: xG - 2, y: Math.min(p.y0, cy - 20), width: RW + 12, height: Math.max(p.y1 - p.y0, 40), fill: 'transparent' }, g);
            g.addEventListener('mouseenter', () => this.pick(null, gr.id));
            g.addEventListener('click', ev => { ev.stopPropagation(); this.pick(null, this.selG === gr.id ? null : gr.id); });
        });
        T(svg, { x: xS + bw, y: 14, 'text-anchor': 'end', fill: TX3, 'font-size': 10, class: 'mono', 'letter-spacing': '.12em' }, 'SOURCES');
        T(svg, { x: xG, y: 14, fill: TX3, 'font-size': 10, class: 'mono', 'letter-spacing': '.12em' }, 'MEASURE GROUPS');
        svg.addEventListener('mouseleave', () => this.pick(null, null));
        this.paint(); this.readout();
    },
    renderMobile(host) {
        host.innerHTML = '';
        const w = document.createElement('div'); w.className = 'flow-m';
        let h = '<p class="flow-m__h">Sources</p>';
        SRC.sources.forEach(s => {
            h += `<button type="button" class="flow-m__src" data-id="${s.id}" aria-pressed="false"><i style="background:${SCOL[s.id]}"></i><span>${esc(this.srcName(s))}<em>${s.year} · ${s.level === 'village' ? 'village records' : 'district figures'}</em></span><span>${s.n_vars}</span></button>`;
        });
        h += '<div class="flow-m__tick"></div><div class="flow-m__hub">SHRUG village identifier, summed to district<br><span style="color:#7E7E7C">crop statistics, night lights and the Economic Census are already district figures</span></div><div class="flow-m__tick"></div>';
        h += '<p class="flow-m__h">Measure groups</p>';
        SRC.groups.forEach(g => { h += `<div class="flow-m__grp" data-g="${g.id}"><span>${esc(g.name)}</span><span class="flow-m__bar"><b></b></span><span class="n"></span></div>`; });
        w.innerHTML = h;
        host.appendChild(w);
        this.mobile = w;
        [...w.querySelectorAll('.flow-m__src')].forEach(b => b.addEventListener('click', () => this.pick(this.sel === b.dataset.id ? null : b.dataset.id, null)));
    },
    prep() { this.u = 0; if (this.clip) this.clip.setAttribute('width', 0); },
    play() {
        if (!this.clip) { this.u = 1; return; }
        const W = this.W;
        tween(this, 1400, t => { this.u = easeOut(t); if (this.clip) this.clip.setAttribute('width', W * this.u); });
    },
});

/* ================================================================
   02 coverage timeline
   ================================================================ */
SC[2] = scene({
    mode: 'all', u: 1, m: 0,
    controls() {
        segs($('#v2-ctl'), [{ key: 'all', name: 'all ' + CW.length }, { key: '606', name: 'the 606 analysed' }], k => {
            const from = this.m, to = k === '606' ? 1 : 0; this.mode = k;
            cancel(this);
            tween(this, 700, t => { this.m = lerp(from, to, easeOut(t)); this.draw(); });
        }, this.mode);
    },
    render() {
        const host = $('#v2'); if (!host) return;
        const W = host.clientWidth || 900, narrow = W < 640;
        const lanes = [
            { name: narrow ? 'Census 2011' : 'Census village directory and abstract, 2011', year: 2011 },
            { name: narrow ? 'SECC 2012' : 'Socio-Economic and Caste Census, 2012', year: 2012 },
            { name: narrow ? 'Economic Census 2013' : 'Economic Census, 2013', year: 2013 },
            { name: narrow ? 'Antyodaya 2019–20' : 'Mission Antyodaya, 2019–20', year: 2019 },
            { name: 'Night lights 2023', year: 2023 },
        ];
        const m = { l: narrow ? 40 : 52, r: narrow ? 14 : 96, t: 22 };
        const topH = narrow ? 180 : 210, laneH = 26, gap = 46;
        const H = m.t + topH + gap + lanes.length * laneH + 34;
        const svg = svgFor(host, W, H);
        const X = v => m.l + (v - 1997) / (2024 - 1997) * (W - m.l - m.r);
        const y0 = 450, y1 = 720;
        const Y = v => m.t + topH - (v - y0) / (y1 - y0) * topH;
        /* crop window band */
        const bandTop = m.t - 6, bandBot = m.t + topH + gap + lanes.length * laneH;
        E('rect', { x: X(1998), y: bandTop, width: X(2020) - X(1998), height: bandBot - bandTop, fill: 'rgba(255,255,255,0.028)' }, svg);
        T(svg, { x: X(2020) + 6, y: m.t + 4, fill: TX3, 'font-size': 10.5, class: 'mono' }, narrow ? '' : 'crop record ends');
        /* y grid */
        [500, 600, 700].forEach(v => {
            E('line', { x1: m.l, x2: X(2024), y1: Y(v), y2: Y(v), stroke: GRID }, svg);
            T(svg, { x: m.l - 8, y: Y(v) + 4, 'text-anchor': 'end', fill: TX3, 'font-size': 10.5, class: 'mono' }, v);
        });
        T(svg, { x: m.l, y: m.t - 8, fill: TX3, 'font-size': 10.5, class: 'mono' }, 'districts in the crop record');
        /* x axis */
        const xAxisY = bandBot + 6;
        E('line', { x1: m.l, x2: X(2024), y1: xAxisY, y2: xAxisY, stroke: AXIS }, svg);
        const tks = narrow ? [1998, 2005, 2013, 2019] : [1998, 2001, 2005, 2009, 2013, 2017, 2019, 2023];
        tks.forEach(v => {
            E('line', { x1: X(v + .5), x2: X(v + .5), y1: xAxisY, y2: xAxisY + 4, stroke: AXIS }, svg);
            T(svg, { x: X(v + .5), y: xAxisY + 17, 'text-anchor': 'middle', fill: TX3, 'font-size': 10.5, class: 'mono' }, v);
        });
        /* lanes */
        const laneY0 = m.t + topH + gap;
        this.marks = [];
        lanes.forEach((ln, i) => {
            const cy = laneY0 + i * laneH + laneH / 2;
            E('line', { x1: m.l, x2: X(2024), y1: cy, y2: cy, stroke: GRID }, svg);
            const x = X(ln.year + .5);
            const mk = E('rect', { x: x - 4, y: cy - 4, width: 8, height: 8, fill: TX }, svg);
            const right = ln.year < 2018;
            T(svg, { x: right ? x + 10 : x - 10, y: cy + 4, 'text-anchor': right ? 'start' : 'end', fill: TX2, 'font-size': narrow ? 10.5 : 11.5 }, ln.name);
            this.marks.push(mk);
        });
        /* step line */
        const cid = 'v2clip' + Math.random().toString(36).slice(2, 7);
        const cp = E('clipPath', { id: cid }, E('defs', {}, svg));
        this.clip = E('rect', { x: 0, y: 0, width: W * this.u, height: H }, cp);
        this.W = W;
        const g = E('g', { 'clip-path': `url(#${cid})` }, svg);
        this.path = E('path', { fill: 'none', stroke: TX, 'stroke-width': 2, 'stroke-linejoin': 'round' }, g);
        this.endLab = T(g, { x: X(2020) + 6, y: 0, fill: TX, 'font-size': 11.5, class: 'mono' }, '');
        this.startLab = T(g, { x: X(1998) + 4, y: 0, fill: TX2, 'font-size': 11, class: 'mono' }, '');
        this.dot = E('circle', { r: 4, fill: TX, stroke: BG, 'stroke-width': 2, opacity: 0 }, svg);
        this.X = X; this.Y = Y;
        COV.years.forEach((yr, i) => {
            const hit = E('rect', { x: X(yr), y: m.t, width: X(yr + 1) - X(yr), height: topH, fill: 'transparent' }, svg);
            const html = () => `<strong>${yl(yr)}</strong><br>${this.val(i)} districts` + (this.m > .5 ? ' among the 606' : ' in the crop record');
            hit.addEventListener('mousemove', ev => { this.dot.setAttribute('cx', X(yr + .5)); this.dot.setAttribute('cy', Y(this.val(i))); this.dot.setAttribute('opacity', 1); showTip(html(), ev); });
            hit.addEventListener('mouseleave', () => { this.dot.setAttribute('opacity', 0); hideTip(); });
            hit.addEventListener('click', ev => { ev.stopPropagation(); showTip(html(), ev); });
        });
        this.draw();
    },
    val(i) { return Math.round(lerp(COV.n_all[i], COV.n_606[i], this.m)); },
    draw() {
        if (!this.path) return;
        const X = this.X, Y = this.Y;
        let d = '';
        COV.years.forEach((yr, i) => {
            const v = lerp(COV.n_all[i], COV.n_606[i], this.m);
            d += (i ? `L${X(yr)},${Y(v)}` : `M${X(yr)},${Y(v)}`) + `L${X(yr + 1)},${Y(v)}`;
        });
        this.path.setAttribute('d', d);
        this.path.setAttribute('stroke', this.m > .5 ? C1 : TX);
        const last = COV.years.length - 1;
        this.endLab.setAttribute('y', Y(this.val(last)) + 4);
        this.endLab.textContent = this.W < 640 ? '' : this.val(last) + (this.m > .5 ? ' analysed' : ' districts');
        this.endLab.setAttribute('fill', this.m > .5 ? C1 : TX);
        this.startLab.setAttribute('y', Y(this.val(0)) + 16);
        this.startLab.textContent = this.val(0);
    },
    prep() { this.u = 0; if (this.clip) this.clip.setAttribute('width', 0); (this.marks || []).forEach(mk => mk.setAttribute('opacity', 0)); },
    play() {
        tween(this, 1300, t => {
            this.u = easeOut(t); if (this.clip) this.clip.setAttribute('width', this.W * this.u);
            (this.marks || []).forEach((mk, i) => mk.setAttribute('opacity', Math.max(0, Math.min(1, (t - .45 - i * .1) / .15))));
        });
    },
});

/* ================================================================
   03 crosswalk unit chart
   ================================================================ */
const METH = [
    { id: 'exact', name: 'exact name', col: NEU },
    { id: 'parent_map', name: 'renamed or new, traced to its 2011 district', col: C1 },
    { id: 'fuzzy_auto', name: 'similar name within the state', col: C2 },
    { id: 'fuzzy_review', name: 'similar name, checked against the census list', col: C3 },
    { id: 'unmatched', name: 'unmatched', col: C4, outline: true },
];
SC[3] = scene({
    mode: 'all', u: 1, hiPar: null,
    controls() {
        segs($('#v3-ctl'), [{ key: 'all', name: 'all ' + CW.length }, { key: '606', name: 'the 606 analysed' }, { key: 'tg', name: 'Telangana' }],
            k => { this.mode = k; this.hiPar = null; this.paint(); this.readout(); }, this.mode);
    },
    render() {
        const host = $('#v3'); if (!host) return;
        const W = host.clientWidth || 800, narrow = W < 640;
        const cell = narrow ? 10 : 12, sq = cell - 2;
        const LW = narrow ? 0 : 230, GW = W - LW, cols = Math.max(10, Math.floor(GW / cell));
        const byM = {}; METH.forEach(mt => { byM[mt.id] = []; });
        CW.forEach((r, i) => { (byM[r.m] || byM.unmatched).push(i); });
        Object.values(byM).forEach(a => a.sort((i, j) => CW[i].s.localeCompare(CW[j].s) || CW[i].d.localeCompare(CW[j].d)));
        /* layout */
        let y = 4; const blocks = [];
        METH.forEach(mt => {
            const n = byM[mt.id].length, rows = Math.ceil(n / cols);
            const labH = narrow ? 34 : 0;
            const bh = Math.max(rows * cell, narrow ? 0 : 38);
            blocks.push({ mt, y: y + labH, rows, n, labY: y });
            y += labH + bh + (narrow ? 14 : 18);
        });
        const H = y;
        const svg = svgFor(host, W, H);
        this.sqs = []; this.labs = {};
        blocks.forEach(b => {
            const lg = E('g', {}, svg);
            if (narrow) {
                T(lg, { x: 0, y: b.labY + 12, fill: TX, 'font-size': 12 }, b.mt.name);
                this.labs[b.mt.id] = T(lg, { x: 0, y: b.labY + 27, fill: TX3, 'font-size': 10.5, class: 'mono' }, '');
            } else {
                const lines = wrapText(b.mt.name, LW - 24, 12.5);
                textLines(lg, LW - 16, b.y + 10, lines, { 'text-anchor': 'end', fill: TX, 'font-size': 12.5 }, 15);
                this.labs[b.mt.id] = T(lg, { x: LW - 16, y: b.y + 10 + lines.length * 15 + 2, 'text-anchor': 'end', fill: TX3, 'font-size': 10.5, class: 'mono' }, '');
                this.labs[b.mt.id].dataset.anchor = 'end';
            }
            byM[b.mt.id].forEach((i, j) => {
                const x = LW + (j % cols) * cell, yy = b.y + Math.floor(j / cols) * cell;
                const r = E('rect', { x: x + .5, y: yy + .5, width: sq - 1, height: sq - 1, 'data-i': i }, svg);
                this.sqs.push({ el: r, r: CW[i], mt: b.mt, j: j, n: b.n });
            });
        });
        svg.addEventListener('mousemove', ev => {
            const i = ev.target.getAttribute && ev.target.getAttribute('data-i'); if (i === null || i === undefined) { hideTip(); return; }
            showTip(this.tip(CW[+i]), ev);
        });
        svg.addEventListener('mouseleave', hideTip);
        svg.addEventListener('click', ev => {
            const i = ev.target.getAttribute && ev.target.getAttribute('data-i'); if (i === null) return;
            ev.stopPropagation(); showTip(this.tip(CW[+i]), ev);
        });
        this.paint(); this.readout();
    },
    tip(r) {
        const mt = METH.find(x => x.id === r.m);
        return `<strong>${esc(r.d)}</strong><br>${esc(r.s)}<br>${esc(mt.name)}` +
            (r.p ? `<br>2011 district: ${esc(r.p)}${r.ps && r.ps !== r.s ? ', ' + esc(r.ps) : ''}` : '') +
            `<br>${r.y} years of crop data · ${r.in ? 'in the 606' : 'outside the 606'}`;
    },
    paint() {
        if (!this.sqs) return;
        const md = this.mode, cnt = {};
        METH.forEach(mt => { cnt[mt.id] = { all: 0, in: 0, tg: 0 }; });
        this.sqs.forEach(s => {
            const c = cnt[s.mt.id]; c.all++; if (s.r.in) c.in++; if (s.r.s === 'Telangana') c.tg++;
            const vis = s.j < Math.ceil(s.n * this.u);
            let fill = s.mt.outline ? 'none' : s.mt.col, stroke = s.mt.outline ? s.mt.col : 'none', op = 1;
            if (md === '606' && !s.r.in) { fill = 'none'; stroke = HAIR; }
            if (md === 'tg') {
                const isT = s.r.s === 'Telangana';
                if (!isT) op = 0.13;
                else if (this.hiPar) { fill = s.r.p === this.hiPar ? TX : s.mt.col; }
                else { stroke = s.r.in ? TX : 'none'; }
            }
            s.el.setAttribute('fill', fill);
            s.el.setAttribute('stroke', stroke);
            s.el.setAttribute('stroke-width', stroke === 'none' ? 0 : (md === 'tg' && s.r.in ? 1.6 : 1));
            s.el.setAttribute('opacity', vis ? op : 0);
        });
        METH.forEach(mt => {
            const c = cnt[mt.id], el = this.labs[mt.id]; if (!el) return;
            el.textContent = md === '606' ? `${c.in} of ${c.all} in the 606` : md === 'tg' ? (c.tg ? `${c.tg} in Telangana` : `${c.all} districts`) : `${c.all} districts · ${(c.all / CW.length * 100).toFixed(c.all < 5 ? 1 : 0)}%`;
        });
    },
    readout() {
        const host = $('#v3-read'); if (!host) return;
        if (this.mode === 'tg') {
            const tg = CW.filter(r => r.s === 'Telangana'), par = {};
            tg.forEach(r => { (par[r.p] = par[r.p] || { n: 0, in: 0, names: [] }); par[r.p].n++; if (r.in) { par[r.p].in++; par[r.p].names.push(r.d); } });
            const rows = Object.entries(par).sort((a, b) => b[1].n - a[1].n);
            const apTg = new Set(CW.filter(r => r.s === 'Andhra Pradesh' && r.in && par[r.p]).map(r => r.p)).size;
            host.innerHTML = `<span class="readout__name">Telangana, ${tg.length} districts from ${rows.length} 2011 districts</span>` +
                `${tg.filter(r => r.in).length} of them are in the 606, outlined in white. ${apTg} more of the 2011 districts enter the 606 through the years up to 2012–13 that the crop record files under Andhra Pradesh. Hover or tap a 2011 district to find its present-day districts.` +
                '<ul style="margin-top:.5rem">' + rows.map(([p, v]) => `<li class="tg-row" tabindex="0" data-p="${esc(p)}"><span>${esc(p)}</span><span class="n">${v.n}${v.in ? ' · ' + v.in + ' in 606' : ''}</span></li>`).join('') + '</ul>';
            [...host.querySelectorAll('.tg-row')].forEach(li => {
                const on = () => { this.hiPar = li.dataset.p; this.paint(); };
                li.addEventListener('mouseenter', on); li.addEventListener('focus', on); li.addEventListener('click', on);
                li.addEventListener('mouseleave', () => { this.hiPar = null; this.paint(); });
            });
        } else if (this.mode === '606') {
            host.innerHTML = '<span class="readout__name">The 606 analysed</span>Squares drawn as outlines are districts that leave the sample. Most of the renamed or new districts that leave are new districts with under ten years of crop data. 64 of the 606 share their 2011 district with another district in the crop record.';
        } else {
            const nM = CW.filter(r => r.m !== 'unmatched').length, nP = CW.filter(r => r.m === 'parent_map').length;
            host.innerHTML = `<span class="readout__name">${CW.length} districts in the crop record</span>${nM} match a 2011 census district, ${(nM / CW.length * 100).toFixed(1)} percent. The ${nP} traced to a 2011 district were renamed or created after 2011. A new district takes the village records of the district it was carved from.`;
        }
    },
    prep() { this.u = 0; this.paint(); },
    play() { tween(this, 1300, t => { this.u = easeOut(t); this.paint(); }); },
});

/* ================================================================
   04 sample funnel
   ================================================================ */
SC[4] = scene({
    u: 1, sel: 4,
    render() {
        const host = $('#v4'); if (!host) return;
        const W = host.clientWidth || 700, narrow = W < 560;
        const gut = narrow ? 44 : 64, rowH = 54, bh = 20, top = 6;
        const H = top + FUN.steps.length * rowH;
        const svg = svgFor(host, W, H);
        const bw = W - gut;
        const X = v => v / FUN.steps[0].n * bw;
        const pid = 'v4h' + Math.random().toString(36).slice(2, 7);
        const pat = E('pattern', { id: pid, width: 5, height: 5, patternUnits: 'userSpaceOnUse', patternTransform: 'rotate(45)' }, E('defs', {}, svg));
        E('line', { x1: 0, y1: 0, x2: 0, y2: 5, stroke: 'rgba(255,255,255,0.35)', 'stroke-width': 1.2 }, pat);
        this.bars = []; this.slices = [];
        FUN.steps.forEach((s, i) => {
            const y = top + i * rowH, last = i === FUN.steps.length - 1;
            T(svg, { x: 0, y: y + 13, fill: last ? C1 : TX, 'font-size': 12.5 }, `${s.n} `).appendChild(Object.assign(svgEl('tspan', { fill: TX2 }), { textContent: s.label }));
            const b = E('rect', { x: 0, y: y + 20, width: X(s.n) * this.u, height: bh, fill: last ? C1 : 'rgba(179,179,176,0.55)' }, svg);
            this.bars.push({ el: b, w: X(s.n) });
            if (i > 0) {
                const prev = FUN.steps[i - 1].n, d = prev - s.n;
                const g = E('g', { style: 'cursor:pointer', tabindex: 0, role: 'button', 'aria-label': `${d} districts removed: ${s.rule}` }, svg);
                const sl = E('rect', { x: X(s.n), y: y + 20, width: Math.max(2, X(prev) - X(s.n)), height: bh, fill: `url(#${pid})`, stroke: 'rgba(255,255,255,0.45)', 'stroke-width': 1 }, g);
                T(g, { x: W, y: y + 35, 'text-anchor': 'end', fill: this.sel === i ? TX : TX2, 'font-size': 12.5, class: 'mono' }, '−' + d);
                E('rect', { x: X(s.n) - 4, y: y + 14, width: W - X(s.n) + 4, height: bh + 12, fill: 'transparent' }, g);
                const pick = () => { this.sel = i; this.readout(); this.mark(); };
                g.addEventListener('mouseenter', pick); g.addEventListener('click', pick); g.addEventListener('focus', pick);
                this.slices.push({ g, sl, i });
            }
        });
        this.mark(); this.readout();
    },
    mark() {
        (this.slices || []).forEach(s => {
            s.sl.setAttribute('stroke', s.i === this.sel ? TX : 'rgba(255,255,255,0.45)');
            s.sl.setAttribute('stroke-width', s.i === this.sel ? 1.6 : 1);
            s.g.querySelector('text').setAttribute('fill', s.i === this.sel ? TX : TX2);
        });
    },
    readout() {
        const host = $('#v4-read'); if (!host) return;
        const s = FUN.steps[this.sel], prev = FUN.steps[this.sel - 1];
        const d = prev.n - s.n;
        const fmtD = x => esc(x.d) + (x.cov !== undefined ? ` ${Math.round(x.cov * 100)}%` : '') + (x.y !== undefined ? ` (${x.y})` : '');
        host.innerHTML = `<span class="readout__name">−${d} · ${esc(s.rule)}</span>` +
            (this.sel === 4 ? `Most of the ${d} were created after 2011. Years of crop data in brackets.` :
             this.sel === 3 ? 'Share of 2011 census villages that the survey reaches.' : '') +
            '<div class="ro-cols" style="margin-top:.35rem">' + s.dropped.map(g => `<div><b>${esc(g.state)}</b> <span>${g.districts.length}</span><br>${g.districts.map(fmtD).join(', ')}</div>`).join('') + '</div>';
    },
    prep() { this.u = 0; (this.bars || []).forEach(b => b.el.setAttribute('width', 0)); (this.slices || []).forEach(s => s.g.setAttribute('opacity', 0)); },
    play() {
        tween(this, 1200, t => {
            const e = easeOut(t);
            this.u = e;
            (this.bars || []).forEach((b, i) => b.el.setAttribute('width', b.w * Math.max(0, Math.min(1, (e * 1.4 - i * .1)))));
            (this.slices || []).forEach(s => s.g.setAttribute('opacity', Math.max(0, (t - .6) / .4)));
        });
    },
});

/* ================================================================
   05 distributions of the three measures
   ================================================================ */
SC[5] = scene({
    ev: false, u: 1,
    controls() {
        segs($('#v5-ctl'), [{ key: 'no', name: 'count and effective numbers' }, { key: 'ev', name: 'add evenness' }],
            k => { this.ev = k === 'ev'; cancel(this); this.u = 1; this.render(); }, this.ev ? 'ev' : 'no');
    },
    render() {
        const host = $('#v5'); if (!host) return;
        const rows = IDX.filter(r => K606.has(r.k) && r.D0 != null && r.D1 != null && r.D2 != null);
        const W = host.clientWidth || 900, narrow = W < 640;
        const m = { l: narrow ? 34 : 48, r: 12, t: 38, b: 40 };
        const H1 = narrow ? 260 : 320, H2 = this.ev ? 170 : 0;
        const svg = svgFor(host, W, H1 + H2);
        const bw = 0.5, hi = Math.ceil(Math.max(...rows.map(r => r.D0)) + 1), nb = Math.ceil(hi / bw);
        const S = [
            { k: 'D0', col: TX, name: 'count of crops' },
            { k: 'D1', col: C1, name: 'effective number of crops' },
            { k: 'D2', col: C2, name: 'number of main crops' },
        ];
        S.forEach(s => {
            s.c = new Array(nb).fill(0);
            rows.forEach(r => { s.c[Math.min(nb - 1, Math.floor(r[s.k] / bw))]++; });
            s.mean = mean(rows.map(r => r[s.k]));
        });
        const maxc = Math.max(...S.map(s => Math.max(...s.c)));
        const iw = W - m.l - m.r, ih = H1 - m.t - m.b;
        const X = v => m.l + v / hi * iw, Y = v => m.t + ih - v / maxc * ih;
        niceTicks(0, maxc, 4).forEach(v => {
            E('line', { x1: m.l, x2: m.l + iw, y1: Y(v), y2: Y(v), stroke: GRID }, svg);
            T(svg, { x: m.l - 7, y: Y(v) + 4, 'text-anchor': 'end', fill: TX3, 'font-size': 10.5, class: 'mono' }, v);
        });
        T(svg, { x: m.l, y: 12, fill: TX3, 'font-size': 10.5, class: 'mono' }, 'districts per half-crop bin');
        E('line', { x1: m.l, x2: m.l + iw, y1: m.t + ih, y2: m.t + ih, stroke: AXIS }, svg);
        niceTicks(0, hi, narrow ? 5 : 9).forEach(v => T(svg, { x: X(v), y: m.t + ih + 30, 'text-anchor': 'middle', fill: TX3, 'font-size': 10.5, class: 'mono' }, v));
        T(svg, { x: m.l + iw, y: H1 - 2, 'text-anchor': 'end', fill: TX3, 'font-size': 10.5, class: 'mono' }, 'crops');
        this.paths = [];
        S.forEach((s, si) => {
            let d = '';
            s.c.forEach((v, i) => { d += (i ? 'L' : 'M') + X(bw * (i + .5)).toFixed(1) + ',' + Y(v).toFixed(1); });
            const p = E('path', { d: d, fill: 'none', stroke: s.col, 'stroke-width': 2, 'stroke-linejoin': 'round' }, svg);
            this.paths.push(p);
            /* mean tick under the axis */
            const mx = X(s.mean);
            E('line', { x1: mx, x2: mx, y1: m.t + ih, y2: m.t + ih + 7, stroke: s.col, 'stroke-width': 1.5 }, svg);
            const lab = si === 0 ? fmt1(s.mean) : fmt1(s.mean);
            T(svg, { x: si === 2 ? mx - 3 : si === 1 ? mx + 3 : mx, y: m.t + ih + 17, 'text-anchor': si === 2 ? 'end' : si === 1 ? 'start' : 'middle', fill: s.col, 'font-size': 10.5, class: 'mono' }, lab);
            const pk = s.c.reduce((mx2, v, i) => v > s.c[mx2] ? i : mx2, 0), px = X(bw * (pk + .5)), py = Y(s.c[pk]);
            if (si === 0) {
                textLines(svg, px + 8, py - 18, narrow ? ['count of crops'] : ['count of crops', 'mean ' + fmt1(s.mean)], { fill: s.col, 'font-size': 12 }, 14);
            } else if (si === 1) {
                textLines(svg, px + 10, py + 2, narrow ? ['effective number', 'of crops'] : ['effective number of crops', 'mean ' + fmt1(s.mean)], { fill: s.col, 'font-size': 12 }, 14);
            } else {
                const lines = narrow ? ['number of main crops'] : ['number of main crops, mean ' + fmt1(s.mean)];
                textLines(svg, px + 9, py + 4, lines, { fill: s.col, 'font-size': 12 }, 14);
            }
        });
        for (let i = 0; i < nb; i++) {
            const h = E('rect', { x: X(bw * i), y: m.t, width: Math.max(1, X(bw) - X(0)), height: ih, fill: 'transparent' }, svg);
            const html = `<strong>between ${(bw * i).toFixed(1)} and ${(bw * (i + 1)).toFixed(1)}</strong>` + S.map(s => `<br>${s.name}: ${s.c[i]} districts`).join('');
            tipOn(h, html);
        }
        if (this.ev) {
            const y0 = H1 + 22, ih2 = H2 - 22 - 34, nb2 = 40;
            const c = new Array(nb2).fill(0);
            rows.forEach(r => { if (r.E != null) c[Math.min(nb2 - 1, Math.floor(r.E * nb2))]++; });
            const mc = Math.max(...c), em = mean(rows.filter(r => r.E != null).map(r => r.E));
            const X2 = v => m.l + v * iw, Y2 = v => y0 + ih2 - v / mc * ih2;
            T(svg, { x: m.l, y: y0 - 8, fill: TX3, 'font-size': 10.5, class: 'mono' }, 'evenness, districts per bin of 0.025');
            E('line', { x1: m.l, x2: m.l + iw, y1: y0 + ih2, y2: y0 + ih2, stroke: AXIS }, svg);
            [0, .25, .5, .75, 1].forEach(v => T(svg, { x: X2(v), y: y0 + ih2 + 26, 'text-anchor': 'middle', fill: TX3, 'font-size': 10.5, class: 'mono' }, v));
            let d = '';
            c.forEach((v, i) => { d += (i ? 'L' : 'M') + X2((i + .5) / nb2).toFixed(1) + ',' + Y2(v).toFixed(1); });
            this.paths.push(E('path', { d: d, fill: 'none', stroke: C3, 'stroke-width': 2 }, svg));
            E('line', { x1: X2(em), x2: X2(em), y1: y0 + ih2, y2: y0 + ih2 + 7, stroke: C3, 'stroke-width': 1.5 }, svg);
            T(svg, { x: X2(em) + 4, y: y0 + ih2 + 15, fill: C3, 'font-size': 10.5, class: 'mono' }, em.toFixed(3));
            const pk = c.reduce((a, v, i) => v > c[a] ? i : a, 0);
            T(svg, { x: X2((pk + .5) / nb2) + 40, y: Y2(c[pk]) + 4, fill: C3, 'font-size': 12 }, 'evenness, mean ' + em.toFixed(3));
            c.forEach((v, i) => {
                const h = E('rect', { x: X2(i / nb2), y: y0, width: iw / nb2, height: ih2, fill: 'transparent' }, svg);
                tipOn(h, `<strong>evenness between ${(i / nb2).toFixed(3)} and ${((i + 1) / nb2).toFixed(3)}</strong><br>${v} districts`);
            });
        }
        this.paths.forEach(p => { const L = p.getTotalLength(); p.dataset.len = L; p.style.strokeDasharray = L; p.style.strokeDashoffset = L * (1 - this.u); });
    },
    prep() { this.u = 0; (this.paths || []).forEach(p => { p.style.strokeDashoffset = p.dataset.len; }); },
    play() { tween(this, 1400, t => { this.u = easeOut(t); (this.paths || []).forEach(p => { p.style.strokeDashoffset = p.dataset.len * (1 - this.u); }); }); },
});

/* ================================================================
   06 pooled against year-by-year counts
   ================================================================ */
SC[6] = scene({
    m: 0, mode: 'each',
    controls() {
        segs($('#v6-ctl'), [{ key: 'each', name: 'each year, then averaged' }, { key: 'pool', name: 'all years pooled' }], k => {
            const from = this.m, to = k === 'pool' ? 1 : 0; this.mode = k;
            cancel(this);
            tween(this, 900, t => { this.m = lerp(from, to, easeOut(t)); this.draw(); });
            this.readout();
        }, this.mode);
    },
    readout() {
        const el = $('#v6-read'); if (!el) return;
        const rows = IDX.filter(r => r.D0 != null && r.D0_o != null);
        const k = this.mode === 'pool' ? 'D0_o' : 'D0';
        const r = pearson(rows.map(x => x.y), rows.map(x => x[k]));
        el.innerHTML = `${this.mode === 'pool' ? 'all years pooled' : 'each year, then averaged'} · correlation with years observed ${r.toFixed(2)} · median count ${median(rows.map(x => x[k])).toFixed(1)} · mean ${mean(rows.map(x => x[k])).toFixed(1)}`;
    },
    render() {
        const host = $('#v6'); if (!host) return;
        const rows = IDX.filter(r => r.D0 != null && r.D0_o != null);
        const W = host.clientWidth || 800, narrow = W < 640;
        const m = { l: narrow ? 34 : 46, r: 14, t: 18, b: 40 }, H = narrow ? 330 : 400;
        const svg = svgFor(host, W, H);
        const iw = W - m.l - m.r, ih = H - m.t - m.b;
        const ymax = Math.ceil(Math.max(...rows.map(r => r.D0_o)) / 10) * 10;
        const X = v => m.l + (v - .3) / 23.4 * iw, Y = v => m.t + ih - v / ymax * ih;
        niceTicks(0, ymax, 5).forEach(v => {
            E('line', { x1: m.l, x2: m.l + iw, y1: Y(v), y2: Y(v), stroke: GRID }, svg);
            T(svg, { x: m.l - 7, y: Y(v) + 4, 'text-anchor': 'end', fill: TX3, 'font-size': 10.5, class: 'mono' }, v);
        });
        T(svg, { x: m.l, y: 10, fill: TX3, 'font-size': 10.5, class: 'mono' }, 'count of crops');
        E('line', { x1: m.l, x2: m.l + iw, y1: m.t + ih, y2: m.t + ih, stroke: AXIS }, svg);
        [1, 5, 10, 15, 20, 22].forEach(v => T(svg, { x: X(v), y: m.t + ih + 18, 'text-anchor': 'middle', fill: TX3, 'font-size': 10.5, class: 'mono' }, v));
        T(svg, { x: m.l + iw, y: H - 4, 'text-anchor': 'end', fill: TX3, 'font-size': 10.5, class: 'mono' }, 'years in the crop record');
        const rnd = mulberry(7);
        const pw = narrow ? 0.34 : 0.36;
        this.dots = rows.map(r => {
            const jx = r.y + (rnd() - .5) * 2 * pw;
            const c = E('circle', { cx: X(jx), r: narrow ? 3.4 : 4, fill: C1, 'fill-opacity': .85, stroke: BG, 'stroke-width': 1.5 }, svg);
            tipOn(c, `<strong>${esc(r.d)}</strong><br>${esc(r.s)}<br>${r.y} years<br>each year, then averaged: ${fmt1(r.D0)}<br>all years pooled: ${fmt1(r.D0_o)}`);
            return { c, r };
        });
        /* median line by years observed, where at least five districts share a value */
        const byY = {};
        rows.forEach(r => { (byY[r.y] = byY[r.y] || []).push(r); });
        this.med = Object.keys(byY).map(Number).sort((a, b) => a - b).filter(y => byY[y].length >= 5)
            .map(y => ({ y, a: median(byY[y].map(r => r.D0)), b: median(byY[y].map(r => r.D0_o)) }));
        this.line = E('path', { fill: 'none', stroke: TX, 'stroke-width': 2, 'stroke-linejoin': 'round' }, svg);
        this.lineLab = T(svg, { x: 0, y: 0, fill: TX, 'font-size': 11.5, 'text-anchor': 'end', stroke: BG, 'stroke-width': 4, 'paint-order': 'stroke' }, 'median');
        this.X = X; this.Y = Y;
        this.draw(); this.readout();
    },
    draw() {
        if (!this.dots) return;
        const mm = this.m, Y = this.Y, X = this.X;
        const col = mm > .5 ? C2 : C1;
        this.dots.forEach(d => { d.c.setAttribute('cy', Y(lerp(d.r.D0, d.r.D0_o, mm))); d.c.setAttribute('fill', col); });
        let p = '';
        this.med.forEach((o, i) => { p += (i ? 'L' : 'M') + X(o.y) + ',' + Y(lerp(o.a, o.b, mm)); });
        this.line.setAttribute('d', p);
        const last = this.med[this.med.length - 1];
        if (last) { this.lineLab.setAttribute('x', X(last.y) - 12); this.lineLab.setAttribute('y', Y(lerp(last.a, last.b, mm)) - 10); }
    },
    prep() { (this.dots || []).forEach(d => d.c.setAttribute('opacity', 0)); if (this.line) this.line.setAttribute('opacity', 0); },
    play() {
        tween(this, 1000, t => {
            (this.dots || []).forEach((d, i) => d.c.setAttribute('opacity', Math.max(0, Math.min(1, t * 2 - (i / this.dots.length)))));
            if (this.line) this.line.setAttribute('opacity', Math.max(0, (t - .6) / .4));
        });
    },
});

/* ================================================================
   07 agreement with independent sources
   ================================================================ */
SC[7] = scene({
    u: 1,
    render() {
        const host = $('#v7'); if (!host) return;
        const W = host.clientWidth || 800, narrow = W < 640;
        const LW = narrow ? 0 : 300, rowH = narrow ? 64 : 58, top = 10;
        const H = top + CHK.length * rowH + 34;
        const svg = svgFor(host, W, H);
        const x0 = LW + (narrow ? 4 : 12), x1 = W - 16;
        const X = v => x0 + v * (x1 - x0);
        [0, .25, .5, .75, 1].forEach(v => {
            E('line', { x1: X(v), x2: X(v), y1: top, y2: top + CHK.length * rowH, stroke: v === 1 ? AXIS : GRID }, svg);
            T(svg, { x: X(v), y: top + CHK.length * rowH + 18, 'text-anchor': 'middle', fill: TX3, 'font-size': 10.5, class: 'mono' }, v);
        });
        this.marks = [];
        CHK.forEach((c, i) => {
            const y = top + i * rowH, cy = narrow ? y + 44 : y + rowH / 2;
            if (narrow) {
                T(svg, { x: 0, y: y + 14, fill: TX, 'font-size': 12 }, c.measure);
                T(svg, { x: 0, y: y + 28, fill: TX3, 'font-size': 10.5 }, c.against.replace('village records summed here against ', 'summed here against '));
            } else {
                T(svg, { x: LW - 8, y: cy - 3, 'text-anchor': 'end', fill: TX, 'font-size': 12.5 }, c.measure);
                const lines = wrapText(c.against, LW - 20, 10.5);
                textLines(svg, LW - 8, cy + 11, lines.slice(0, 2), { 'text-anchor': 'end', fill: TX3, 'font-size': 10.5 }, 12);
            }
            E('line', { x1: X(0), x2: X(1), y1: cy, y2: cy, stroke: GRID }, svg);
            const vals = [];
            if (c.r != null) vals.push({ v: c.r, hollow: false });
            if (c.rho != null) vals.push({ v: c.rho, hollow: true });
            vals.forEach(o => {
                const el = E('circle', { cx: X(0), cy: cy, r: 6, fill: o.hollow ? 'none' : C1, stroke: o.hollow ? C2 : BG, 'stroke-width': o.hollow ? 2 : 1.5 }, svg);
                this.marks.push({ el, x: X(o.v), x0: X(0) });
            });
            const lab = vals.map(o => (o.hollow ? 'rho ' : 'r ') + o.v.toFixed(3)).join(' · ');
            T(svg, { x: X(Math.min(...vals.map(o => o.v))) - 12, y: cy + 4, 'text-anchor': 'end', fill: TX2, 'font-size': 11, class: 'mono' }, lab);
            const hit = E('rect', { x: 0, y: y, width: W, height: rowH, fill: 'transparent' }, svg);
            tipOn(hit, `<strong>${esc(c.measure)}</strong><br>${esc(c.against)}<br>${c.n} districts` +
                (c.r != null ? `<br>r ${c.r.toFixed(3)}` : '') + (c.rho != null ? `<br>rho ${c.rho.toFixed(3)}` : '') +
                (c.mean_a != null ? `<br>mean ${c.mean_a.toFixed(3)} (${c.la}) against ${c.mean_b.toFixed(3)} (${c.lb})` : ''));
        });
        this.draw();
    },
    draw() { (this.marks || []).forEach(mk => mk.el.setAttribute('cx', lerp(mk.x0, mk.x, this.u))); },
    prep() { this.u = 0; this.draw(); },
    play() { tween(this, 1100, t => { this.u = easeOut(t); this.draw(); }); },
});

/* ================================================================
   08 control matrix
   ================================================================ */
SC[8] = scene({
    u: 1, sel: { t: 'row', i: 0 },
    render() {
        const host = $('#v8'); if (!host) return;
        const W = host.clientWidth || 800, narrow = W < 640;
        const R = MOD.rows, C = MOD.cols;
        /* on a phone the grid turns: controls run down, analyses across */
        const rowsA = narrow ? C : R, colsA = narrow ? R : C;
        const LW = narrow ? 168 : 190, HH = narrow ? 64 : 76, rh = narrow ? 34 : 42;
        const cw = (W - LW) / colsA.length;
        const H = HH + rowsA.length * rh + 6;
        const svg = svgFor(host, W, H);
        this.dots = [];
        colsA.forEach((c, j) => {
            const g = E('g', { style: 'cursor:pointer', tabindex: 0 }, svg);
            const SHORT = { 'irrigation curve': 'irrigation curve', 'irrigation source': 'irrigation source', 'single district measures': 'single measures', 'facility comparisons': 'facilities' };
            const lines = wrapText(narrow ? (SHORT[c.name] || c.name) : c.name, cw - 6, 11.5).slice(0, 4);
            textLines(g, LW + (j + .5) * cw, HH - 8 - (lines.length - 1) * 13, lines, { 'text-anchor': 'middle', fill: TX2, 'font-size': 11.5 }, 13);
            E('rect', { x: LW + j * cw, y: 0, width: cw, height: HH, fill: 'transparent' }, g);
            const pick = () => { this.sel = { t: narrow ? 'row' : 'col', i: j }; this.readout(); };
            g.addEventListener('mouseenter', pick); g.addEventListener('click', pick); g.addEventListener('focus', pick);
        });
        rowsA.forEach((r, i) => {
            const y = HH + i * rh;
            const g = E('g', { style: 'cursor:pointer', tabindex: 0 }, svg);
            E('rect', { x: 0, y: y, width: LW, height: rh, fill: 'transparent' }, g);
            const lines = wrapText(r.name, LW - 14, 12.5).slice(0, 2);
            textLines(g, 0, y + rh / 2 + 4 - (lines.length - 1) * 7, lines, { fill: TX, 'font-size': 12.5 }, 14);
            const pick = () => { this.sel = { t: narrow ? 'col' : 'row', i: i }; this.readout(); };
            g.addEventListener('mouseenter', pick); g.addEventListener('click', pick); g.addEventListener('focus', pick);
            colsA.forEach((c, j) => {
                const row = narrow ? c : r, col = narrow ? r : c;
                E('rect', { x: LW + j * cw + 2, y: y + 2, width: cw - 4, height: rh - 4, fill: 'none', stroke: GRID }, svg);
                if (row.has.indexOf(col.id) >= 0) {
                    const d = E('circle', { cx: LW + (j + .5) * cw, cy: y + rh / 2, r: 4, fill: C1 }, svg);
                    this.dots.push({ el: d, k: i * colsA.length + j });
                }
            });
        });
        this.nCells = rowsA.length * colsA.length;
        this.draw(); this.readout();
    },
    draw() { (this.dots || []).forEach(d => d.el.setAttribute('opacity', Math.max(0, Math.min(1, this.u * 1.5 - d.k / this.nCells * .5)))); },
    readout() {
        const host = $('#v8-read'); if (!host) return;
        if (this.sel.t === 'col') {
            const c = MOD.cols[this.sel.i];
            const users = MOD.rows.filter(r => r.has.indexOf(c.id) >= 0).map(r => r.name);
            host.innerHTML = `<span class="readout__name">${esc(c.name)}</span>${esc(c.def)}<br>Held constant in: ${users.map(esc).join(', ')}.`;
        } else {
            const r = MOD.rows[this.sel.i];
            let h = `<span class="readout__name">${esc(r.name)} · ${r.n} districts</span>${esc(r.note)}`;
            if (this.sel.i === 0) h += '<div class="ro-cols" style="margin-top:.35rem">' + MOD.robust.map(o => `<div>${esc(o.name)}: peak <b>${o.tp.toFixed(3)}</b> <span>(${o.n})</span></div>`).join('') + '</div>';
            host.innerHTML = h;
        }
    },
    prep() { this.u = 0; this.draw(); },
    play() { tween(this, 1000, t => { this.u = t; this.draw(); }); },
});

/* ================================================================
   method tables
   ================================================================ */
function tables() {
    const tb = $('#tbl-match tbody');
    if (tb) {
        let h = '';
        METH.forEach(mt => {
            const a = CW.filter(r => r.m === mt.id);
            h += `<tr><td>${esc(mt.name[0].toUpperCase() + mt.name.slice(1))}</td><td class="num">${a.length}</td><td class="num">${(a.length / CW.length * 100).toFixed(a.length < 5 ? 1 : 0)}%</td><td class="num">${a.filter(r => r.in).length}</td></tr>`;
        });
        h += `<tr><td><strong>Total</strong></td><td class="num"><strong>${CW.length}</strong></td><td class="num">100%</td><td class="num"><strong>${CW.filter(r => r.in).length}</strong></td></tr>`;
        tb.innerHTML = h;
    }
    const rb = $('#tbl-rob tbody');
    if (rb) {
        const ord = MOD.robust.slice();
        rb.innerHTML = ord.map(o => `<tr><td>${esc(o.name[0].toUpperCase() + o.name.slice(1))}</td><td class="num">${o.n}</td><td class="num">${o.tp.toFixed(3)}</td><td class="num">${(o.r2 * 100).toFixed(1)}%</td></tr>`).join('');
    }
    const ex = $('#excl');
    if (ex) {
        const fmtD = x => esc(x.d) + (x.cov !== undefined ? ` ${Math.round(x.cov * 100)}%` : '') + (x.y !== undefined ? ` (${x.y})` : '');
        ex.className = 'excl';
        ex.innerHTML = FUN.steps.slice(1).map((s, i) => {
            const d = FUN.steps[i].n - s.n;
            return `<div><h5>${esc(s.rule)}, ${d}</h5><div class="excl__cols">` + s.dropped.map(g => `<p><b>${esc(g.state)}</b> ${g.districts.map(fmtD).join(', ')}</p>`).join('') + '</div></div>';
        }).join('');
    }
}

/* ================================================================
   wiring
   ================================================================ */
const HOSTS = { 1: '#v1', 2: '#v2', 3: '#v3', 4: '#v4', 5: '#v5', 6: '#v6', 7: '#v7', 8: '#v8' };
let inited = false;
function startPlay(sc) { if (sc.played) return; sc.played = true; sc.play(); }

async function init() {
    let ix;
    [SRC, COV, CW, FUN, CHK, MOD, ix] = await Promise.all([
        loadJSON('data/site_data_sources.json'), loadJSON('data/site_data_coverage.json'),
        loadJSON('data/site_data_crosswalk.json'), loadJSON('data/site_data_funnel.json'),
        loadJSON('data/site_data_checks.json'), loadJSON('data/site_data_models.json'),
        loadJSON('data/site_indices.json')]);
    IDX = ix.districts;
    K606 = new Set(FUN.keys606);
    /* plain wording for the within-state comparison, and the robustness range as it now stands */
    const ws = s => String(s).replace(/without the state baseline/g, 'without the within-state comparison')
        .replace(/within the state baseline/g, 'within state').replace(/with the state baseline/g, 'within state, the main estimate')
        .replace(/state baseline/g, 'within state');
    MOD.cols.forEach(c => {
        if (c.id === 'fe') { c.name = 'within state (state fixed effects)'; c.def = 'A district is set only against other districts in the same state.'; }
        else { c.name = ws(c.name); c.def = ws(c.def); }
    });
    const tps = MOD.robust.map(o => o.tp);
    MOD.rows.forEach(r => {
        r.note = ws(r.note);
        if (r.name === 'irrigation curve') r.note = r.note.replace(/from \d\.\d+ to \d\.\d+/, `from ${Math.min(...tps).toFixed(3)} to ${Math.max(...tps).toFixed(3)}`);
    });
    MOD.robust.forEach(o => { o.name = ws(o.name); });
    Object.values(SC).forEach(sc => { if (sc.controls) sc.controls(); });
    try { tables(); } catch (e) { console.error('tables', e); }
    for (const k of Object.keys(SC)) {
        try { SC[k].render(); } catch (e) { console.error('scene ' + k, e); }
        if (!RM && SC[k].prep) SC[k].prep();
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
        if (sc.prep) sc.prep();
        sc.play();
    }));
    document.addEventListener('click', hideTip);
    inited = true;
}

async function drawAll() {
    if (!inited) { await init(); return; }
    for (const k of Object.keys(SC)) {
        try { SC[k].render(); } catch (e) { console.error('scene ' + k, e); }
        if (!SC[k].played && !RM && SC[k].prep) SC[k].prep();
    }
}
window.drawAll = drawAll;
drawAll();
})();
