/* ================================================================
   NEIGHBOURS PAGE: scene visuals.
   Reads data/site_neighbours.json and data/districts.geojson.
   Depends on site.js for svgEl, showTip, hideTip, loadJSON and loadGeo.
   Everything is wrapped so no name collides with site.js globals.

   Same pattern as the crop-groups page: each scene has render(),
   draw(u) for a progress value u in [0, 1] and play(). Under
   prefers-reduced-motion u is 1 at once. drawAll() re-renders on resize.
   ================================================================ */
(function () {
'use strict';

const RM = !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
const TX = '#EDEDEB', TX2 = '#B3B3B0', TX3 = '#7E7E7C';
const GRID = 'rgba(255,255,255,0.075)', ZERO = 'rgba(255,255,255,0.28)';
const CY = '#0E9BB5', MG = '#C63FD8', VI = '#7C5CE0', CYB = 'rgb(64,214,240)';
const NODATA = 'rgba(255,255,255,0.035)';
const NAME = { rice: 'rice', wheat: 'wheat', maize: 'maize', millets: 'millets', pulses: 'pulses',
               oilseeds: 'oilseeds', cotton: 'cotton', sugarcane: 'sugarcane', other: 'other crops' };
const CAP = s => s.charAt(0).toUpperCase() + s.slice(1);
let D = null;

const $ = s => document.querySelector(s);
function E(tag, attrs, parent) { const e = svgEl(tag, attrs || {}); if (parent) parent.appendChild(e); return e; }
function T(parent, attrs, s) { const t = E('text', attrs, parent); t.textContent = s; return t; }
const sgn = (v, d) => { const t = Math.abs(v).toFixed(d); return (+t === 0 ? '' : (v >= 0 ? '+' : '−')) + t; };
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
function segs(host, items, onPick, cur) {
    if (!host) return;
    host.innerHTML = '';
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
/* x position of a pointer event in viewBox units */
const evX = (ev, svg, W) => { const r = svg.getBoundingClientRect(); return (ev.clientX - r.left) * W / (r.width || W); };

const SC = {};

/* ================================================================
   01  likeness against distance, same state and across a border
   ================================================================ */
SC[1] = scene({
    async render() {
        const host = $('#v1'); if (!host) return;
        const W = host.clientWidth || 900, nar = W < 560, H = nar ? 330 : 400;
        const svg = svgFor(host, W, H);
        const m = { l: nar ? 34 : 44, r: nar ? 10 : 118, t: 14, b: 40 };
        const X = k => m.l + k / 600 * (W - m.l - m.r);
        const lo = 40, hi = 90;
        const Y = v => m.t + (hi - v) / (hi - lo) * (H - m.t - m.b);
        [40, 50, 60, 70, 80, 90].forEach(t => {
            E('line', { x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t), stroke: GRID }, svg);
            T(svg, { x: m.l - 6, y: Y(t) + 3.5, 'text-anchor': 'end', fill: TX3, 'font-size': 10, class: 'mono' }, String(t));
        });
        for (let k = 0; k <= 600; k += 100) T(svg, { x: X(k), y: H - m.b + 15, 'text-anchor': 'middle', fill: TX3, 'font-size': 10, class: 'mono' }, String(k));
        T(svg, { x: (m.l + W - m.r) / 2, y: H - 6, 'text-anchor': 'middle', fill: TX3, 'font-size': 10, class: 'mono' }, 'km between district centres');
        const C = (w, t) => D.sim.filter(r => r.w === w && r.t === t).sort((a, b) => a.km - b.km);
        const pth = rows => rows.map((r, i) => (i ? 'L' : 'M') + X(r.km).toFixed(1) + ',' + Y(r.m).toFixed(1)).join('');
        this.lines = [];
        [['1998-2002', 'same_state', TX2], ['1998-2002', 'cross_border', TX2]].forEach(([w, t, c]) => {
            this.lines.push({ el: E('path', { d: pth(C(w, t)), fill: 'none', stroke: 'rgba(237,237,235,0.38)', 'stroke-width': 1.1, 'stroke-dasharray': '3 3' }, svg), dash: true });
        });
        const SS = C('2015-19', 'same_state'), CB = C('2015-19', 'cross_border');
        this.lines.push({ el: E('path', { d: pth(SS), fill: 'none', stroke: CY, 'stroke-width': 2.2, 'stroke-linejoin': 'round' }, svg) });
        this.lines.push({ el: E('path', { d: pth(CB), fill: 'none', stroke: MG, 'stroke-width': 2.2, 'stroke-linejoin': 'round' }, svg) });
        /* end labels */
        this.late = E('g', {}, svg);
        const lastS = SS[SS.length - 1], lastC = CB[CB.length - 1];
        if (!nar) {
            T(this.late, { x: X(lastS.km) + 8, y: Y(lastS.m) + 3.5, fill: CY, 'font-size': 10.5, class: 'mono' }, 'same state');
            T(this.late, { x: X(lastC.km) + 8, y: Y(lastC.m) + 3.5, fill: MG, 'font-size': 10.5, class: 'mono' }, 'different states');
        }
        /* adjacent pairs and the border-as-distance gap */
        const R = D.sim_ref['2015-19'], BK = D.border_km['2015-19'];
        const aS = R['adjacent, same state'], aC = R['adjacent, across a border'];
        const g = this.late, yT = Y(BK.target);
        E('line', { x1: X(aS.km), x2: X(aS.km), y1: Y(aS.sim), y2: yT, stroke: 'rgba(237,237,235,0.35)', 'stroke-dasharray': '2 3' }, g);
        E('line', { x1: X(aS.km), x2: X(BK.km_cross), y1: yT, y2: yT, stroke: TX, 'stroke-width': 1.2 }, g);
        [X(aS.km), X(BK.km_cross)].forEach(x => E('line', { x1: x, x2: x, y1: yT - 5, y2: yT + 5, stroke: TX, 'stroke-width': 1.2 }, g));
        E('circle', { cx: X(BK.km_cross), cy: yT, r: 3.2, fill: 'none', stroke: CY, 'stroke-width': 1.4 }, g);
        const gx = (X(aS.km) + X(BK.km_cross)) / 2, lx = X(nar ? 240 : 250), ly = Y(nar ? 87 : 81);
        E('line', { x1: gx, y1: yT - 2, x2: lx - 4, y2: ly + 4, stroke: 'rgba(237,237,235,0.45)', 'stroke-width': .8 }, g);
        T(g, { x: lx, y: ly, fill: TX, 'font-size': nar ? 10.5 : 11.5, class: 'mono' }, `a state line ≈ ${BK.border_km} km`);
        T(g, { x: lx, y: ly + (nar ? 13 : 15), fill: TX3, 'font-size': nar ? 9 : 10, class: 'mono' }, `same-state pairs fall to ${BK.target.toFixed(0)} only at ${BK.km_cross} km`);
        E('circle', { cx: X(aS.km), cy: Y(aS.sim), r: 5.5, fill: CY, stroke: '#0B0B0C', 'stroke-width': 1.2 }, g);
        E('circle', { cx: X(aC.km), cy: Y(aC.sim), r: 5.5, fill: MG, stroke: '#0B0B0C', 'stroke-width': 1.2 }, g);
        T(g, { x: X(aS.km) + 9, y: Y(aS.sim) - 8, fill: CY, 'font-size': 10.5, class: 'mono' }, `adjacent, same state ${aS.sim.toFixed(0)}`);
        T(g, { x: X(2), y: Y(nar ? 56 : 57), fill: MG, 'font-size': nar ? 10 : 10.5, class: 'mono' }, 'adjacent,');
        T(g, { x: X(2), y: Y(nar ? 56 : 57) + 13, fill: MG, 'font-size': nar ? 10 : 10.5, class: 'mono' }, `different states ${aC.sim.toFixed(0)}`);
        /* hover: one band at a time */
        const guide = E('line', { y1: m.t, y2: H - m.b, stroke: 'rgba(255,255,255,0.25)', opacity: 0 }, svg);
        const hit = E('rect', { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: 'transparent' }, svg);
        const at = (rows, km) => rows.find(r => r.km === km);
        const msg = km => {
            const s = at(SS, km), c = at(CB, km);
            return `${km - 10}–${km + 10} km: same state ${s ? s.m.toFixed(1) + ' (' + s.n + ' pairs)' : 'too few pairs'}, ` +
                   `different states ${c ? c.m.toFixed(1) + ' (' + c.n + ' pairs)' : 'too few pairs'}`;
        };
        const move = ev => {
            const k = (evX(ev, svg, W) - m.l) / (W - m.l - m.r) * 600;
            const km = Math.max(30, Math.min(590, Math.round((k - 10) / 20) * 20 + 10));
            guide.setAttribute('x1', X(km)); guide.setAttribute('x2', X(km)); guide.setAttribute('opacity', 1);
            const s = at(SS, km), c = at(CB, km);
            $('#v1-read').textContent = msg(km);
            showTip(`<strong>${km - 10}–${km + 10} km apart, 2015–19</strong><br>same state: ${s ? s.m.toFixed(1) + ', ' + s.n + ' pairs' : 'too few pairs'}` +
                    `<br>different states: ${c ? c.m.toFixed(1) + ', ' + c.n + ' pairs' : 'too few pairs'}`, ev);
        };
        hit.addEventListener('mousemove', move); hit.addEventListener('click', move);
        hit.addEventListener('mouseleave', () => { hideTip(); guide.setAttribute('opacity', 0); this.readout(); });
        this.readout();
        this.draw(this.played || RM ? 1 : 0);
    },
    readout() {
        const R = D.sim_ref['2015-19'];
        $('#v1-read').textContent = `2015–19: adjacent districts in one state ${R['adjacent, same state'].sim.toFixed(1)}, adjacent districts in different states ${R['adjacent, across a border'].sim.toFixed(1)}.`;
    },
    draw(u) {
        (this.lines || []).forEach(o => {
            if (o.dash) { o.el.style.opacity = u; return; }
            if (o.len == null) { try { o.len = o.el.getTotalLength(); } catch (e) { o.len = 2000; } }
            o.el.setAttribute('stroke-dasharray', o.len);
            o.el.setAttribute('stroke-dashoffset', o.len * (1 - u));
        });
        if (this.late) this.late.style.opacity = u >= 1 ? 1 : 0;
    },
    play() { tween(this, 1300, t => this.draw(easeOut(t))); },
});

/* ================================================================
   02  explorer: a district's crop mix next to each neighbour's
   ================================================================ */
SC[2] = scene({
    k: 'HARYANA|SIRSA',
    controls() {
        const sel = $('#v2-pick'); if (!sel) return;
        const X = D.explorer.districts, by = {};
        Object.keys(X).forEach(k => { (by[X[k].s] = by[X[k].s] || []).push(k); });
        sel.innerHTML = Object.keys(by).sort().map(s => `<optgroup label="${s}">` +
            by[s].sort((a, b) => X[a].n.localeCompare(X[b].n)).map(k => `<option value="${k}"${k === this.k ? ' selected' : ''}>${X[k].n}</option>`).join('') +
            '</optgroup>').join('');
        sel.addEventListener('change', () => { this.k = sel.value; this.render(); });
    },
    readout() {
        const d = D.explorer.districts[this.k];
        const nx = d.nb.filter(n => !n.ss).length;
        $('#v2-read').textContent = `${d.n}, ${d.s}: average likeness with its ${d.nb.length} neighbours ${d.msim != null ? d.msim.toFixed(1) : 'n/a'}` +
            (d.pct != null ? `, higher than for ${Math.round(d.pct)} percent of districts` : '') +
            (nx ? `. ${nx} of the neighbours ${nx === 1 ? 'is' : 'are'} in another state.` : '. All its neighbours are in the same state.');
    },
    async render() {
        const host = $('#v2'); if (!host) return;
        const X = D.explorer.districts, d = X[this.k]; if (!d) return;
        const G = D.groups;
        const W = host.clientWidth || 900, nar = W < 620;
        const rows = [{ k: this.k, self: true, n: d.n, s: d.s, mix: d.mix }].concat(
            d.nb.slice().sort((a, b) => (b.sim == null ? -1 : b.sim) - (a.sim == null ? -1 : a.sim)).map(n => {
                const o = X[n.k] || null, nm = o || D.explorer.names[n.k] || { n: n.k, s: '' };
                return { k: n.k, n: nm.n, s: nm.s, mix: o ? o.mix : null, sim: n.sim, ss: n.ss, km: n.km };
            }));
        /* columns: every group on a wide screen; on a phone, groups that reach 3 percent somewhere */
        const cols = G.map((g, i) => i).filter(i => !nar || rows.some(r => r.mix && r.mix[i] >= 3));
        const lw = nar ? 108 : 190, sw = nar ? 64 : 120, gap = nar ? 4 : 10;
        const cw = (W - lw - sw - gap * (cols.length + 1)) / cols.length;
        const rh = nar ? 34 : 36, top = nar ? 44 : 30, H = top + rows.length * rh + 6;
        const svg = svgFor(host, W, H);
        const vmax = Math.max(10, Math.ceil(Math.max(...rows.flatMap(r => r.mix ? cols.map(i => r.mix[i]) : [0])) / 10) * 10);
        const cx = j => lw + sw + gap * (j + 1) + j * cw;
        /* headers */
        T(svg, { x: lw, y: top - 10, fill: TX3, 'font-size': nar ? 9 : 10, class: 'mono' }, 'likeness');
        cols.forEach((i, j) => {
            const lab = NAME[G[i]] === 'other crops' ? 'other' : NAME[G[i]];
            if (nar) T(svg, { x: cx(j) + 2, y: top - 8, fill: TX3, 'font-size': 9, class: 'mono', transform: `rotate(-30 ${cx(j) + 2} ${top - 8})` }, lab);
            else T(svg, { x: cx(j), y: top - 10, fill: TX3, 'font-size': 10, class: 'mono' }, lab);
        });
        this.bars = [];
        rows.forEach((r, ri) => {
            const y = top + ri * rh, by = y + rh * 0.3, bh = rh * 0.42;
            if (ri === 1) E('line', { x1: 0, x2: W, y1: y - 1, y2: y - 1, stroke: 'rgba(255,255,255,0.14)' }, svg);
            const col = r.self ? TX : (r.ss ? CY : MG);
            T(svg, { x: 0, y: y + rh * 0.45, fill: r.self ? TX : TX2, 'font-size': nar ? 10.5 : 12, class: 'mono', 'font-weight': r.self ? 500 : 400 }, r.n.length > (nar ? 13 : 24) ? r.n.slice(0, nar ? 12 : 23) + '…' : r.n);
            T(svg, { x: 0, y: y + rh * 0.45 + (nar ? 11 : 13), fill: r.self ? TX3 : (r.ss ? TX3 : MG), 'font-size': nar ? 8.5 : 9.5, class: 'mono' },
                r.self ? 'chosen district' : (nar ? (r.s.length > 15 ? r.s.slice(0, 14) + '…' : r.s) : (r.ss ? r.s : r.s + ' · other state')));
            if (!r.self) {
                E('rect', { x: lw, y: by, width: sw - 30, height: bh, fill: 'rgba(255,255,255,0.05)' }, svg);
                if (r.sim != null) {
                    const b = E('rect', { x: lw, y: by, width: 0, height: bh, fill: col, opacity: .85 }, svg);
                    this.bars.push({ el: b, w: (sw - 30) * r.sim / 100 });
                    T(svg, { x: lw + sw - 26, y: by + bh - 1, fill: TX, 'font-size': nar ? 9.5 : 10.5, class: 'mono' }, r.sim.toFixed(0));
                }
            }
            cols.forEach((i, j) => {
                const x = cx(j);
                E('rect', { x, y: by, width: cw, height: bh, fill: 'rgba(255,255,255,0.04)' }, svg);
                if (r.mix) {
                    const v = r.mix[i] || 0;
                    const b = E('rect', { x, y: by, width: 0, height: bh, fill: col, opacity: r.self ? .9 : .8 }, svg);
                    this.bars.push({ el: b, w: cw * v / vmax });
                    if (!r.self && d.mix[i] > 0) E('line', { x1: x + cw * d.mix[i] / vmax, x2: x + cw * d.mix[i] / vmax, y1: by - 3, y2: by + bh + 3, stroke: TX, 'stroke-width': 1.3 }, svg);
                }
                const hit = E('rect', { x, y, width: cw + gap, height: rh, fill: 'transparent' }, svg);
                tipOn(hit, () => `<strong>${r.n}</strong>, ${r.s}<br>${NAME[G[i]]}: ` + (r.mix ? `${(r.mix[i] || 0).toFixed(1)}% of cropped land` : 'no usable record') +
                    (r.self ? '' : `<br>${d.n}: ${(d.mix[i] || 0).toFixed(1)}%`));
            });
            if (!r.self) {
                const hit = E('rect', { x: 0, y, width: lw + sw, height: rh, fill: 'transparent' }, svg);
                tipOn(hit, () => `<strong>${r.n}</strong>, ${r.s}${r.ss ? '' : ' (other state)'}<br>` +
                    (r.sim != null ? `likeness with ${d.n}: ${r.sim.toFixed(1)}` : 'no usable record in 2015–19') + `<br>${Math.round(r.km)} km between centres`);
            }
        });
        if (!rows.some(r => r.mix)) return;
        T(svg, { x: W, y: H - 1, 'text-anchor': 'end', fill: TX3, 'font-size': 9, class: 'mono' }, `bars run 0 to ${vmax}%`);
        this.readout();
        this.draw(this.played || RM ? 1 : 0);
    },
    draw(u) { (this.bars || []).forEach(b => b.el.setAttribute('width', Math.max(0, b.w * u))); },
    play() { tween(this, 700, t => this.draw(easeOut(t))); },
});

/* ================================================================
   03  border dumbbells: rice, wheat, pulses, millets
   ================================================================ */
const CROPS3 = ['rice', 'wheat', 'pulses', 'millets'];
SC[3] = scene({
    one: 'rice',
    controls() {
        segs($('#v3-ctl'), CROPS3.map(c => ({ key: c, name: c })), k => { this.one = k; this.render(); }, this.one);
    },
    async render() {
        const host = $('#v3'); if (!host) return;
        const W = host.clientWidth || 900, nar = W < 720;
        const crops = nar ? [this.one] : CROPS3;
        const B = D.borders, order = B.order;
        const get = (b, v, w) => B.rows.find(r => r.border === b && r.v === v && r.w === w);
        const lw = nar ? 118 : 216, pg = nar ? 0 : 20;
        const pw = (W - lw - pg * (crops.length - 1)) / crops.length;
        const rh = 46, top = 26, H = top + order.length * rh + 26;
        const svg = svgFor(host, W, H);
        this.rows = [];
        /* row labels */
        order.forEach((b, i) => {
            const y = top + i * rh, r0 = B.rows.find(r => r.border === b);
            if (i) E('line', { x1: 0, x2: W, y1: y, y2: y, stroke: 'rgba(255,255,255,0.05)' }, svg);
            const t = E('text', { x: 0, y: y + rh / 2 - 3, 'font-size': nar ? 10.5 : 11.5, class: 'mono' }, svg);
            const a = E('tspan', { fill: CY }, t); a.textContent = r0.a;
            if (nar) {
                const t2 = E('text', { x: 0, y: y + rh / 2 + 10, 'font-size': 10.5, class: 'mono' }, svg);
                const b2 = E('tspan', { fill: VI }, t2); b2.textContent = r0.b_;
            } else {
                const s = E('tspan', { fill: TX3 }, t); s.textContent = ' | ';
                const b2 = E('tspan', { fill: '#A996F0' }, t); b2.textContent = r0.b_;
                const late = get(b, 'rice', '2015-19');
                T(svg, { x: 0, y: y + rh / 2 + 11, fill: TX3, 'font-size': 9.5, class: 'mono' }, `${late ? late.n : r0.n} pairs of districts`);
            }
        });
        crops.forEach((v, ci) => {
            const ox = lw + ci * (pw + pg);
            const vals = B.rows.filter(r => r.v === v).flatMap(r => [r.ma, r.mb]);
            const vmax = Math.max(10, Math.ceil(Math.max(...vals) / 10) * 10);
            const X = s => ox + 8 + s / vmax * (pw - 16);
            E('rect', { x: ox, y: top, width: pw, height: order.length * rh, fill: 'rgba(255,255,255,0.022)' }, svg);
            T(svg, { x: ox + 8, y: 12, fill: TX, 'font-size': 12, class: 'mono' }, v);
            niceTicks(0, vmax, nar ? 4 : 2).forEach(t => {
                E('line', { x1: X(t), x2: X(t), y1: top, y2: top + order.length * rh, stroke: t ? GRID : ZERO }, svg);
                T(svg, { x: X(t), y: top + order.length * rh + 14, 'text-anchor': 'middle', fill: TX3, 'font-size': 9.5, class: 'mono' }, t + '%');
            });
            order.forEach((b, i) => {
                const y = top + i * rh;
                const marks = [];
                [['1998-2002', y + rh / 2 - 7, false], ['2015-19', y + rh / 2 + 7, true]].forEach(([w, yy, fill]) => {
                    const r = get(b, v, w); if (!r) return;
                    const ln = E('line', { x1: X(Math.min(r.ma, r.mb)), x2: X(Math.max(r.ma, r.mb)), y1: yy, y2: yy, stroke: fill ? 'rgba(237,237,235,0.45)' : 'rgba(237,237,235,0.2)', 'stroke-width': fill ? 1.6 : 1 }, svg);
                    const ca = E('circle', { cx: X(r.ma), cy: yy, r: fill ? 4.6 : 3.8, fill: fill ? CY : 'none', stroke: CY, 'stroke-width': 1.4 }, svg);
                    const cb = E('circle', { cx: X(r.mb), cy: yy, r: fill ? 4.6 : 3.8, fill: fill ? VI : 'none', stroke: VI, 'stroke-width': 1.4 }, svg);
                    marks.push(ln, ca, cb);
                });
                const hit = E('rect', { x: ox, y, width: pw, height: rh, fill: 'transparent' }, svg);
                tipOn(hit, () => {
                    const e = get(b, v, '1998-2002'), l = get(b, v, '2015-19'), r0 = e || l;
                    const line = (r, lab) => r ? `<br>${lab}: ${r0.a} ${r.ma.toFixed(1)}%, ${r0.b_} ${r.mb.toFixed(1)}%<br>&nbsp;&nbsp;gap within pairs ${sgn(r.gap, 1)} points, ${r.n} pairs` : `<br>${lab}: too few usable years`;
                    return `<strong>${r0.a} | ${r0.b_}, ${v}</strong>` + line(e, '1998–2002') + line(l, '2015–19') +
                        (l && l.geo != null ? `<br>gap that measured geography predicts, 2015–19: ${sgn(l.geo, 1)}` : '');
                });
                this.rows.push({ marks, i });
            });
        });
        this.draw(this.played || RM ? 1 : 0);
    },
    draw(u) {
        const n = (this.rows || []).length ? Math.max(...this.rows.map(r => r.i)) + 1 : 1;
        (this.rows || []).forEach(r => { const on = u >= (r.i + 1) / n - 1e-9 ? 1 : 0; r.marks.forEach(m => { m.style.opacity = on; }); });
    },
    play() { tween(this, 900, t => this.draw(t)); },
});

/* ================================================================
   04  border gap over time: six small panels on one scale
   ================================================================ */
SC[4] = scene({
    async render() {
        const host = $('#v4'); if (!host) return;
        const S = D.gap_time.series;
        const W = host.clientWidth || 900, cols = W < 760 ? 2 : 3, rowsN = Math.ceil(S.length / cols);
        const gap = W < 560 ? 14 : 26, pw = (W - gap * (cols - 1)) / cols, ph = W < 560 ? 168 : 190;
        const H = rowsN * ph + (rowsN - 1) * 18 + 4;
        const svg = svgFor(host, W, H);
        const lo = -40, hi = 30, y0 = 1998, y1 = 2019;
        this.parts = [];
        S.forEach((s, i) => {
            const c = i % cols, r = Math.floor(i / cols), ox = c * (pw + gap), oy = r * (ph + 18);
            const m = { l: 30, r: 8, t: 34, b: 18 };
            const X = y => ox + m.l + (y - y0) / (y1 - y0) * (pw - m.l - m.r);
            const Y = v => oy + m.t + (hi - v) / (hi - lo) * (ph - m.t - m.b);
            const g = E('g', {}, svg);
            E('rect', { x: ox + m.l - 4, y: oy + m.t - 4, width: pw - m.l - m.r + 8, height: ph - m.t - m.b + 8, fill: 'rgba(255,255,255,0.018)' }, g);
            T(g, { x: ox, y: oy + 12, fill: TX, 'font-size': W < 560 ? 10.5 : 11.5, class: 'mono' }, `${s.a} minus ${s.b}`);
            T(g, { x: ox, y: oy + 26, fill: TX2, 'font-size': 10.5, class: 'mono' }, s.v);
            [-40, -20, 0, 20].forEach(t => {
                E('line', { x1: ox + m.l, x2: ox + pw - m.r, y1: Y(t), y2: Y(t), stroke: t ? GRID : ZERO }, g);
                T(g, { x: ox + m.l - 5, y: Y(t) + 3.5, 'text-anchor': 'end', fill: TX3, 'font-size': 9, class: 'mono' }, t > 0 ? '+' + t : (t < 0 ? '−' + (-t) : '0'));
            });
            [y0, y1].forEach((y, j) => T(g, { x: X(y), y: oy + ph - 3, 'text-anchor': j ? 'end' : 'start', fill: TX3, 'font-size': 9.5, class: 'mono' }, fy(y)));
            if (s.a === 'Telangana') E('line', { x1: X(2014), x2: X(2014), y1: oy + m.t, y2: oy + ph - m.b, stroke: 'rgba(255,255,255,0.3)', 'stroke-dasharray': '3 3' }, g);
            const half = s.nmax / 2;
            const good = s.pts.filter(p => p[1] != null);
            let d = '', prev = null;
            good.forEach(p => {
                const ok = p[2] >= half;
                if (ok) { d += (prev && prev[2] >= half && p[0] - prev[0] === 1 ? 'L' : 'M') + X(p[0]).toFixed(1) + ',' + Y(p[1]).toFixed(1); }
                prev = p;
            });
            const ln = E('path', { d, fill: 'none', stroke: CY, 'stroke-width': 1.8, 'stroke-linejoin': 'round' }, g);
            const dots = good.map(p => E('circle', { cx: X(p[0]), cy: Y(p[1]), r: 2.4, fill: p[2] >= half ? CY : 'none', stroke: CY, 'stroke-width': 1.1 }, g));
            const hit = E('rect', { x: ox, y: oy, width: pw, height: ph, fill: 'transparent' }, g);
            const f = ev => {
                const yv = Math.round(y0 + (evX(ev, svg, W) - ox - m.l) / (pw - m.l - m.r) * (y1 - y0));
                const p = s.pts.find(q => q[0] === Math.max(y0, Math.min(y1, yv)));
                showTip(`<strong>${s.a} minus ${s.b}, ${s.v}</strong><br>` + (p && p[1] != null
                    ? `${fy(p[0])}: ${sgn(p[1], 1)} points, ${p[2]} of ${s.nmax} pairs`
                    : `${fy(Math.max(y0, Math.min(y1, yv)))}: no usable record`), ev);
            };
            hit.addEventListener('mousemove', f); hit.addEventListener('click', f); hit.addEventListener('mouseleave', hideTip);
            this.parts.push({ ln, dots });
        });
        this.draw(this.played || RM ? 1 : 0);
    },
    draw(u) {
        (this.parts || []).forEach(o => {
            if (o.len == null) { try { o.len = o.ln.getTotalLength() || 1; } catch (e) { o.len = 1000; } }
            o.ln.setAttribute('stroke-dasharray', o.len);
            o.ln.setAttribute('stroke-dashoffset', o.len * (1 - u));
            o.dots.forEach(c => { c.style.opacity = u >= 1 ? 1 : 0; });
        });
    },
    play() { tween(this, 1300, t => this.draw(easeOut(t))); },
});

/* ================================================================
   05  small-multiple maps of a crop's spread
   ================================================================ */
const EP = {
    soybean_entry: { name: 'soybean take-up', early: 'grew soybean in 1998–2000',
        note: 'A district takes up soybean when the crop passes 5 percent of its cropped land and stays above it. Districts on 2011 boundaries; hatched districts lack a usable starting record. Hover or tap a district.' },
    chickpea_surge: { name: 'chickpea surges', early: null,
        note: 'A chickpea surge is the first year the crop\'s share of cropped land rises 5 points above the district\'s own 1998–2000 level and stays there. Districts on 2011 boundaries; hatched districts lack a usable starting record. Hover or tap a district.' },
};
SC[5] = scene({
    ep: 'soybean_entry',
    controls() {
        segs($('#v5-ctl'), Object.keys(EP).map(k => ({ key: k, name: EP[k].name })), k => { this.ep = k; this.render(); }, this.ep);
    },
    legend() {
        const L = $('#v5-legend'); if (!L) return;
        const sw = (fill, stroke) => `<svg width="12" height="12"><rect x="1" y="1" width="10" height="10" fill="${fill}" stroke="${stroke || 'none'}"/></svg>`;
        L.innerHTML = (EP[this.ep].early ? `<span class="legend__item">${sw('rgba(237,237,235,0.55)')}${EP[this.ep].early}</span>` : '') +
            `<span class="legend__item">${sw(CYB)}took it up since the previous map</span>` +
            `<span class="legend__item">${sw('rgba(14,155,181,0.55)')}took it up earlier</span>` +
            `<span class="legend__item">${sw('rgba(255,255,255,0.07)', 'rgba(255,255,255,0.18)')}not yet</span>`;
        const n = $('#v5-note span'); if (n) n.textContent = EP[this.ep].note;
    },
    async render() {
        const host = $('#v5'); if (!host) return;
        const geo = await loadGeo();
        const U = D.spread.episodes[this.ep].units, PY = D.spread.panel_years;
        this.legend();
        /* frame on the districts involved in this crop */
        let x0 = 180, x1 = -180, ya = 90, yb = -90;
        const walk = c => { if (typeof c[0] === 'number') { x0 = Math.min(x0, c[0]); x1 = Math.max(x1, c[0]); ya = Math.min(ya, c[1]); yb = Math.max(yb, c[1]); } else c.forEach(walk); };
        /* the whole country in every panel, scaled to fit both the width and a height cap */
        geo.features.forEach(f => walk(f.geometry.coordinates));
        const W = host.clientWidth || 900, cols = W < 760 ? 2 : 3, rowsN = Math.ceil(PY.length / cols), gap = W < 560 ? 10 : 18;
        const pw = (W - gap * (cols - 1)) / cols, kx = Math.cos((ya + yb) / 2 * Math.PI / 180), HEAD = 28;
        const sc = Math.min((pw - 8) / ((x1 - x0) * kx), (pw * 1.1) / (yb - ya));
        const mw = (x1 - x0) * kx * sc, mh = (yb - ya) * sc, ox = (pw - mw) / 2;
        const ph = HEAD + mh + 4;
        const H = rowsN * ph + (rowsN - 1) * gap + 2;
        const svg = svgFor(host, W, H);
        const PXl = v => ox + (v - x0) * kx * sc, PYl = v => HEAD + 2 + (yb - v) * sc;
        const ring = c => c.map((p, i) => (i ? 'L' : 'M') + PXl(p[0]).toFixed(1) + ',' + PYl(p[1]).toFixed(1)).join('') + 'Z';
        const path = (c, d) => d === 1 ? ring(c) : c.map(k => path(k, d - 1)).join('');
        const hid = 'h5' + Math.random().toString(36).slice(2, 7), cid = 'c5' + Math.random().toString(36).slice(2, 7);
        const defs = E('defs', {}, svg);
        const pat = E('pattern', { id: hid, width: 5, height: 5, patternUnits: 'userSpaceOnUse', patternTransform: 'rotate(45)' }, defs);
        E('line', { x1: 0, y1: 0, x2: 0, y2: 5, stroke: 'rgba(255,255,255,0.20)', 'stroke-width': 1 }, pat);
        E('rect', { x: 0, y: HEAD - 2, width: pw, height: ph - HEAD + 2 }, E('clipPath', { id: cid }, defs));
        /* features in view, path strings built once */
        const feats = [];
        geo.features.forEach(f => {
            let fx0 = 180, fx1 = -180, fy0 = 90, fy1 = -90;
            const w2 = c => { if (typeof c[0] === 'number') { fx0 = Math.min(fx0, c[0]); fx1 = Math.max(fx1, c[0]); fy0 = Math.min(fy0, c[1]); fy1 = Math.max(fy1, c[1]); } else c.forEach(w2); };
            w2(f.geometry.coordinates);
            if (fx1 < x0 || fx0 > x1 || fy1 < ya || fy0 > yb) return;
            const uk = D.poly[f.properties.k];
            feats.push({ pr: f.properties, uk, u: uk ? U[uk] : null, d: path(f.geometry.coordinates, f.geometry.type === 'Polygon' ? 2 : 3) });
        });
        const rings = D.state_rings.map(ring);
        this.panels = [];
        PY.forEach((py, i) => {
            const prev = i ? PY[i - 1] : 2000;
            const c = i % cols, r = Math.floor(i / cols);
            const g = E('g', { transform: `translate(${(c * (pw + gap)).toFixed(1)},${(r * (ph + gap)).toFixed(1)})` }, svg);
            const seen = new Set();
            const mg = E('g', { 'clip-path': `url(#${cid})` }, g);
            feats.forEach(f => {
                const u = f.u;
                let fill;
                if (!u || u.st === 'no_baseline') fill = `url(#${hid})`;
                else if (u.st === 'established') fill = 'rgba(237,237,235,0.55)';
                else if (u.y && u.y <= prev) { fill = 'rgba(14,155,181,0.55)'; seen.add(f.uk); }
                else if (u.y && u.y <= py) { fill = CYB; seen.add(f.uk); }
                else fill = 'rgba(255,255,255,0.07)';
                const el = E('path', { d: f.d, fill, stroke: 'rgba(11,11,12,0.85)', 'stroke-width': 0.3 }, mg);
                tipOn(el, () => {
                    if (!u) return `<strong>${f.pr.n}</strong><br>${f.pr.s}<br>no usable record`;
                    const sh = u.sh[i];
                    const st = u.st === 'established' ? EP[this.ep].early : u.st === 'no_baseline' ? 'no usable starting record'
                        : (u.y ? (u.y <= py ? 'took it up in ' + fy(u.y) : 'takes it up in ' + fy(u.y)) : 'had not taken it up by 2019–20');
                    return `<strong>${u.n}</strong><br>${u.s}<br>${st}` + (sh != null ? `<br>${fy(py)}: ${sh.toFixed(1)}% of cropped land` : '');
                });
            });
            const so = E('g', { 'pointer-events': 'none', 'clip-path': `url(#${cid})` }, g);
            rings.forEach(d => E('path', { d, fill: 'none', stroke: 'rgba(237,237,235,0.4)', 'stroke-width': 0.6, 'stroke-linejoin': 'round' }, so));
            T(g, { x: 2, y: 15, fill: TX, 'font-size': 12, class: 'mono' }, fy(py));
            const nAll = Object.values(U).filter(u => u.st === 'at_risk' && u.y && u.y <= py).length;
            T(g, { x: pw - 2, y: 15, 'text-anchor': 'end', fill: TX3, 'font-size': W < 560 ? 9 : 10, class: 'mono' }, `${nAll} district${nAll === 1 ? '' : 's'} so far`);
            this.panels.push(g);
        });
        this.draw(this.played || RM ? 1 : 0);
    },
    draw(u) { const n = (this.panels || []).length; (this.panels || []).forEach((g, i) => { g.style.opacity = u >= (i + 1) / n - 1e-9 ? 1 : 0.08; }); },
    play() { tween(this, 1800, t => this.draw(t)); },
});

/* ================================================================
   06  neighbours against the distant ring, per kind of take-up
   ================================================================ */
const EPN = { chickpea: 'chickpea', cotton: 'cotton', maize: 'maize', moong: 'moong', rabi_maize: 'rabi maize', rice: 'rice', soybean: 'soybean' };
const epLab = r => CAP(EPN[r.ep]) + (r.mode === 'entry' ? ' take-up' : ' surge');
SC[6] = scene({
    async render() {
        const host = $('#v6'); if (!host) return;
        const R = D.hazard.slice().sort((a, b) => b.d - a.d);
        const W = host.clientWidth || 900, nar = W < 560;
        const lw = nar ? 108 : 150, rh = 30, top = 8, H = top + R.length * rh + 34;
        const svg = svgFor(host, W, H);
        const ext = R.flatMap(r => [r.nb - 1.96 * r.nbse, r.nb + 1.96 * r.nbse, r.rg - 1.96 * r.rgse, r.rg + 1.96 * r.rgse]).map(v => v * 100);
        const lo = Math.floor(Math.min(...ext) / 5) * 5, hi = Math.ceil(Math.max(...ext) / 5) * 5;
        const X = v => lw + (v - lo) / (hi - lo) * (W - lw - 10);
        niceTicks(lo, hi, nar ? 4 : 6).forEach(t => {
            E('line', { x1: X(t), x2: X(t), y1: top, y2: top + R.length * rh, stroke: t ? GRID : ZERO }, svg);
            T(svg, { x: X(t), y: top + R.length * rh + 14, 'text-anchor': 'middle', fill: TX3, 'font-size': 10, class: 'mono' }, t > 0 ? '+' + t : (t < 0 ? '−' + (-t) : '0'));
        });
        T(svg, { x: (lw + W) / 2, y: H - 4, 'text-anchor': 'middle', fill: TX3, 'font-size': 10, class: 'mono' }, 'points of yearly chance');
        this.pts = [];
        R.forEach((r, i) => {
            const y = top + i * rh + rh / 2;
            T(svg, { x: 0, y: y + 4, fill: TX2, 'font-size': nar ? 10.5 : 11.5, class: 'mono' }, epLab(r));
            const a = E('line', { x1: X(100 * (r.nb - 1.96 * r.nbse)), x2: X(100 * (r.nb + 1.96 * r.nbse)), y1: y - 4, y2: y - 4, stroke: CY, 'stroke-width': 1.5 }, svg);
            const ac = E('circle', { cx: X(100 * r.nb), cy: y - 4, r: 4.3, fill: CY }, svg);
            const b = E('line', { x1: X(100 * (r.rg - 1.96 * r.rgse)), x2: X(100 * (r.rg + 1.96 * r.rgse)), y1: y + 4, y2: y + 4, stroke: TX2, 'stroke-width': 1.2 }, svg);
            const bc = E('circle', { cx: X(100 * r.rg), cy: y + 4, r: 3.8, fill: '#0B0B0C', stroke: TX2, 'stroke-width': 1.4 }, svg);
            const hit = E('rect', { x: 0, y: y - rh / 2, width: W, height: rh, fill: 'transparent' }, svg);
            const ci = (b, s) => `${sgn(100 * b, 1)} (95%: ${sgn(100 * (b - 1.96 * s), 1)} to ${sgn(100 * (b + 1.96 * s), 1)})`;
            tipOn(hit, () => `<strong>${epLab(r)}</strong><br>${r.ev} events; base yearly chance ${(100 * r.base).toFixed(2)}%` +
                `<br>neighbours: ${ci(r.nb, r.nbse)}<br>ring about 100 km away: ${ci(r.rg, r.rgse)}` +
                `<br>difference ${sgn(100 * r.d, 1)} (${(100 * r.dse).toFixed(1)}), q = ${r.q.toFixed(2)}` +
                (r.ss != null ? `<br>same-state neighbours ${sgn(100 * r.ss, 1)}, other-state ${sgn(100 * r.os, 1)}` : ''));
            this.pts.push({ els: [a, ac, b, bc], i });
        });
        this.draw(this.played || RM ? 1 : 0);
    },
    draw(u) { const n = (this.pts || []).length; (this.pts || []).forEach(p => { const on = u >= (p.i + 1) / n - 1e-9 ? 1 : 0; p.els.forEach(e => { e.style.opacity = on; }); }); },
    play() { tween(this, 1000, t => this.draw(t)); },
});

/* ---------- estimates list in the method drawer ---------- */
function estimates() {
    const ul = $('#est-list'); if (!ul) return;
    const xs = (o, spec, w) => {
        const u = D.xsec.find(r => r.o === o && r.spec === spec && r.w === w && r.cl === 'unit');
        const s = D.xsec.find(r => r.o === o && r.spec === spec && r.w === w && r.cl === 'state');
        return u ? `${sgn(u.b, 1)} (${u.se.toFixed(1)}; ${s.se.toFixed(1)})` : 'n/a';
    };
    const R = D.sim_ref['2015-19'], R0 = D.sim_ref['1998-2002'];
    const wd = D.widening.find(r => r.term === 'x_2013' && r.cl === 'unit' && r.spec === 'periods');
    const wds = D.widening.find(r => r.term === 'x_2013' && r.cl === 'state' && r.spec === 'periods');
    const hz = (ep, mode) => D.hazard.find(r => r.ep === ep && r.mode === mode);
    const H1 = hz('chickpea', 'surge'), H2 = hz('soybean', 'entry'), H3 = hz('cotton', 'surge');
    const f3 = v => (100 * v).toFixed(1);
    const C = D.checks;
    const bmv = (m, spec) => { const r = C.by_measure.find(x => x.w === 'w1519' && x.m === m && x.spec === spec); return r ? sgn(r.b, 1) : 'n/a'; };
    const pw = (m, spec) => { const r = C.by_measure.find(x => x.w === 'w1519' && x.m === m && x.spec === spec); return r && r.p_wild != null ? r.p_wild.toFixed(3) : 'n/a'; };
    const lov = (drop, spec) => { const r = C.leave_out.find(x => x.m === 'sim_L1' && x.drop === drop && x.spec === spec); return r ? sgn(r.b, 1) : 'n/a'; };
    const grp = os => { const v = os.reduce((a, o) => a + (C.groups.find(x => x.spec === 'M2' && x.o === o) || { share: 0 }).share, 0); return Math.round(100 * v); };
    const gci = o => { const r = C.groups.find(x => x.spec === 'M2' && x.o === o); return r ? `${sgn(r.b, 1)} (${sgn(r.lo, 1)} to ${sgn(r.hi, 1)})` : 'n/a'; };
    const wci = C.widening.find(x => x.o === 'sim_L1' && x.cl === 'state');
    const mo = (v, l) => { const r = C.moran.find(x => x.v === v && x.l === l); return r ? r.I.toFixed(2) : 'n/a'; };
    const wv = (ep, mode) => { const r = C.wave.find(x => x.ep === ep && x.mode === mode && x.sub === 'all'); return r ? `${ep} ${mode === 'entry' ? 'take-up' : 'surge'} ${r.obs}% against ${r.matched}% (p = ${r.p_matched < 0.001 ? '<0.001' : r.p_matched.toFixed(3)}, ${r.n} events)` : 'n/a'; };
    const items = [
        `Likeness, 2015–19: adjacent same-state pairs ${R['adjacent, same state'].sim} (${R['adjacent, same state'].n.toLocaleString('en-IN')} pairs); adjacent pairs in different states ${R['adjacent, across a border'].sim} (${R['adjacent, across a border'].n}); same state 200–300 km ${R['200-300 km, same state'].sim}; any pair 500–600 km ${R['500-600 km, any'].sim}. In 1998–2002 the first two were ${R0['adjacent, same state'].sim} and ${R0['adjacent, across a border'].sim}.`,
        `Cost of a state line for adjacent pairs, 2015–19, in points of likeness: distance held fixed ${xs('sim', 'M1', '2015-19')}, p = ${pw('sim_L1', 'M1')} with whole states resampled; adding geography ${xs('sim', 'M2', '2015-19')}, p = ${pw('sim_L1', 'M2')}; against each pair's own two states ${xs('sim', 'M3', '2015-19')}, p = ${pw('sim_L1', 'M3')}. With geography held fixed the cost is ${bmv('sim_L1_no_other', 'M2')} without the other-crops group, ${bmv('sim_corr', 'M2')} as a correlation of shares and ${lov('Telangana + Andhra Pradesh', 'M2')} without Telangana and Andhra Pradesh.`,
        `Distance worth a state line: ${C.border_km.interp} km (95 percent interval ${C.border_km.lo} to ${C.border_km.hi} km); ${C.border_km.reg} km from a straight-line fit. Made-up borders from ${C.placebo.n} random partitions into 25 regions: ${sgn(C.placebo.m1_mean, 1)} on average, central 95 percent ${sgn(C.placebo.m1_lo, 1)} to ${sgn(C.placebo.m1_hi, 1)}, against ${xs('sim', 'M1', '2015-19').split(' ')[0]} for real borders.`,
        `Extra gap across a state line in a crop's share, distance held fixed, then with geography: rice ${xs('g_rice', 'M1', '2015-19')} and ${xs('g_rice', 'M2', '2015-19')}; wheat ${xs('g_wheat', 'M1', '2015-19')} and ${xs('g_wheat', 'M2', '2015-19')}; pulses ${xs('g_pulses', 'M1', '2015-19')} and ${xs('g_pulses', 'M2', '2015-19')}; millets ${xs('g_millets', 'M1', '2015-19')} and ${xs('g_millets', 'M2', '2015-19')}.`,
        `Share of the likeness lost at a state line, geography held fixed, by crop group: rice and wheat ${grp(['g_rice', 'g_wheat'])} percent, oilseeds ${grp(['g_oilseeds'])}, other crops ${grp(['g_other'])}, pulses and millets ${grp(['g_pulses', 'g_millets'])}. Gap in pulse share ${gci('g_pulses')}, millets ${gci('g_millets')}, rice ${gci('g_rice')} (95 percent intervals with whole states resampled).`,
        `Change in how alike cross-border neighbours are, relative to same-state neighbours, 2013–19 against 1998–2002, holding each pair and each year fixed: ${sgn(wd.b, 2)} (${wd.se.toFixed(2)}; ${wds.se.toFixed(2)}), ${wd.n.toLocaleString('en-IN')} pair-years; 95 percent interval ${sgn(wci.lo, 1)} to ${sgn(wci.hi, 1)} with whole states resampled.`,
        `Moran's I of change, 1998–2002 to 2015–19, within states against across borders (links between Andhra Pradesh and Telangana left out): diversity ${mo('D1', 'within-state links')} and ${mo('D1', 'cross-border links, AP-TS excluded')}; rice ${mo('s_rice', 'within-state links')} and ${mo('s_rice', 'cross-border links, AP-TS excluded')}; with state means removed, diversity ${mo('D1', 'within-state links, state means removed')} and ${mo('D1', 'cross-border links, state means removed')}.`,
        `Take-ups next to an earlier grower, against the five most similar districts in the same state: ${wv('soybean', 'entry')}; ${wv('maize', 'entry')}; ${wv('cotton', 'entry')}; ${wv('chickpea', 'entry')}; ${wv('chickpea', 'surge')}.`,
        `Take-up, rise in yearly chance in percentage points, neighbours against the ring: chickpea surge ${f3(H1.nb)} (${f3(H1.nbse)}) against ${f3(H1.rg)} (${f3(H1.rgse)}), difference ${f3(H1.d)} (${f3(H1.dse)}), q = ${H1.q.toFixed(2)}; soybean take-up ${f3(H2.nb)} (${f3(H2.nbse)}) against ${f3(H2.rg)} (${f3(H2.rgse)}), difference ${f3(H2.d)} (${f3(H2.dse)}), q = ${H2.q.toFixed(2)}; cotton surge ${f3(H3.nb)} (${f3(H3.nbse)}) against ${f3(H3.rg)} (${f3(H3.rgse)}).`,
        `Year-on-year change in pulse share: neighbours in the same year ${D.yoy.pulses.same_year.toFixed(2)} (${D.yoy.pulses.same_year_se.toFixed(2)}); neighbours a year earlier ${D.yoy.pulses.hr_nbr.toFixed(3)} (${D.yoy.pulses.hr_nbr_se.toFixed(3)}) against same-state districts about 100 km away ${D.yoy.pulses.hr_ring.toFixed(3)} (${D.yoy.pulses.hr_ring_se.toFixed(3)}), difference ${D.yoy.pulses.hr_diff.toFixed(3)} (${D.yoy.pulses.hr_diff_se.toFixed(3)}). Uncertainty here lets a district's years move together.`,
    ];
    ul.innerHTML = items.map(s => `<li>${s}</li>`).join('');
}

/* ================================================================
   wiring
   ================================================================ */
const HOSTS = { 1: '#v1', 2: '#v2', 3: '#v3', 4: '#v4', 5: '#v5', 6: '#v6' };
let inited = false;
function startPlay(sc) { if (sc.played) return; sc.played = true; sc.play(); }

async function init() {
    D = await loadJSON('data/site_neighbours.json');
    Object.values(SC).forEach(sc => { if (sc.controls) sc.controls(); });
    try { estimates(); } catch (e) { console.error('estimates', e); }
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
