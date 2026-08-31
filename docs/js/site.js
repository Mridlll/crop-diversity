/* ================================================================
   Shared chart helpers.

   Conventions held throughout, deliberately:
     - one y-axis, never two
     - categorical colours assigned in fixed order from the CSS variables,
       never cycled or generated
     - a single series carries no legend; the figure title names it
     - grid and axes recede, marks do not
     - every mark has a hover target larger than the mark itself
   ================================================================ */

const PAL  = ['#0E9BB5', '#7C5CE0', '#1FA35A', '#C63FD8'];
const INK  = '#EDEDEB';                 // marks on a dark ground
const MUTED = '#7E7E7C';                // axis text
const GRIDC = 'rgba(255,255,255,0.075)';// grid hairlines
const AXISC = 'rgba(255,255,255,0.22)'; // baseline
const HOLLOW = 'rgba(255,255,255,0.30)';

/* ---------- scroll reveal ---------- */
function reveal() {
    const els = document.querySelectorAll('.fade');
    if (!('IntersectionObserver' in window)) {
        els.forEach(e => e.classList.add('in'));
        return;
    }
    const io = new IntersectionObserver((entries) => {
        entries.forEach(en => {
            if (en.isIntersecting) { en.target.classList.add('in'); io.unobserve(en.target); }
        });
    }, { rootMargin: '0px 0px -8% 0px', threshold: 0.05 });
    els.forEach(e => io.observe(e));
}

/* ---------- tooltip ---------- */
let TIP;
function tip() {
    if (!TIP) {
        TIP = document.createElement('div');
        TIP.className = 'tip';
        document.body.appendChild(TIP);
    }
    return TIP;
}
function showTip(html, ev) {
    const t = tip();
    t.innerHTML = html;
    t.classList.add('on');
    const pad = 14;
    let x = ev.clientX + pad, y = ev.clientY + pad;
    const r = t.getBoundingClientRect();
    if (x + r.width > window.innerWidth - 8) x = ev.clientX - r.width - pad;
    if (y + r.height > window.innerHeight - 8) y = ev.clientY - r.height - pad;
    t.style.left = (x + window.scrollX) + 'px';
    t.style.top = (y + window.scrollY) + 'px';
}
function hideTip() { if (TIP) TIP.classList.remove('on'); }

/* ---------- svg scaffolding ---------- */
function svgEl(tag, attrs) {
    const e = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    return e;
}
function frame(sel, opts) {
    const host = document.querySelector(sel);
    if (!host) return null;
    host.innerHTML = '';
    const W = host.clientWidth || 900;
    const H = opts.height || 380;
    const m = Object.assign({ t: 16, r: 20, b: 44, l: 56 }, opts.margin || {});
    const svg = svgEl('svg', {
        viewBox: `0 0 ${W} ${H}`, width: '100%', height: H,
        class: 'chart', role: 'img'
    });
    host.appendChild(svg);
    return { host, svg, W, H, m, iw: W - m.l - m.r, ih: H - m.t - m.b };
}
const lin = (d0, d1, r0, r1) => v => r0 + (v - d0) / (d1 - d0 || 1) * (r1 - r0);

function axes(f, xs, ys, xTicks, yTicks, xLab, yLab, fmtX, fmtY) {
    const g = svgEl('g', {});
    f.svg.appendChild(g);
    yTicks.forEach(t => {
        g.appendChild(svgEl('line', {
            x1: f.m.l, x2: f.m.l + f.iw, y1: ys(t), y2: ys(t),
            stroke: GRIDC, 'shape-rendering': 'crispEdges'
        }));
        const tx = svgEl('text', {
            x: f.m.l - 9, y: ys(t) + 4, 'text-anchor': 'end',
            fill: MUTED, 'font-size': 11, 'font-family': 'JetBrains Mono, monospace'
        });
        tx.textContent = (fmtY || (v => v))(t);
        g.appendChild(tx);
    });
    xTicks.forEach(t => {
        const tx = svgEl('text', {
            x: xs(t), y: f.m.t + f.ih + 20, 'text-anchor': 'middle',
            fill: MUTED, 'font-size': 11, 'font-family': 'JetBrains Mono, monospace'
        });
        tx.textContent = (fmtX || (v => v))(t);
        g.appendChild(tx);
    });
    g.appendChild(svgEl('line', {
        x1: f.m.l, x2: f.m.l + f.iw, y1: f.m.t + f.ih, y2: f.m.t + f.ih,
        stroke: AXISC, 'shape-rendering': 'crispEdges'
    }));
    if (xLab) {
        const t = svgEl('text', {
            x: f.m.l + f.iw / 2, y: f.H - 6, 'text-anchor': 'middle',
            fill: MUTED, 'font-size': 11.5
        });
        t.textContent = xLab; g.appendChild(t);
    }
    if (yLab) {
        const t = svgEl('text', {
            transform: `rotate(-90) translate(${-(f.m.t + f.ih / 2)},13)`,
            'text-anchor': 'middle', fill: MUTED, 'font-size': 11.5
        });
        t.textContent = yLab; g.appendChild(t);
    }
    return g;
}
function ticks(a, b, n) {
    const out = [], step = (b - a) / n;
    for (let i = 0; i <= n; i++) out.push(a + step * i);
    return out;
}

/* ---------- scatter with an optional fitted quadratic ---------- */
function scatterQuad(sel, data, o) {
    const f = frame(sel, o); if (!f) return;
    const xs0 = Math.min(...data.map(d => d.x)), xs1 = Math.max(...data.map(d => d.x));
    const ys0 = o.y0 !== undefined ? o.y0 : Math.min(...data.map(d => d.y));
    const ys1 = o.y1 !== undefined ? o.y1 : Math.max(...data.map(d => d.y));
    const X = lin(xs0, xs1, f.m.l, f.m.l + f.iw);
    const Y = lin(ys0, ys1, f.m.t + f.ih, f.m.t);
    axes(f, X, Y, ticks(xs0, xs1, 5), ticks(ys0, ys1, 5), o.xLab, o.yLab,
        o.fmtX || (v => (v * 100).toFixed(0) + '%'), o.fmtY || (v => v.toFixed(1)));
    const g = svgEl('g', {}); f.svg.appendChild(g);
    data.forEach(d => {
        const c = svgEl('circle', {
            cx: X(d.x), cy: Y(d.y), r: 2.7, fill: INK,
            'fill-opacity': .30, stroke: 'none'
        });
        const hit = svgEl('circle', { cx: X(d.x), cy: Y(d.y), r: 7, fill: 'transparent' });
        hit.addEventListener('mousemove', ev => {
            c.setAttribute('fill', PAL[0]); c.setAttribute('fill-opacity', 1);
            c.setAttribute('r', 4.2);
            showTip(o.tipFn(d), ev);
        });
        hit.addEventListener('mouseleave', () => {
            c.setAttribute('fill', INK); c.setAttribute('fill-opacity', .30);
            c.setAttribute('r', 2.7); hideTip();
        });
        g.appendChild(c); g.appendChild(hit);
    });
    if (o.quad) {
        const n = data.length;
        let sx = 0, sx2 = 0, sx3 = 0, sx4 = 0, sy = 0, sxy = 0, sx2y = 0;
        data.forEach(d => {
            const x = d.x, y = d.y;
            sx += x; sx2 += x * x; sx3 += x ** 3; sx4 += x ** 4;
            sy += y; sxy += x * y; sx2y += x * x * y;
        });
        const A = [[n, sx, sx2], [sx, sx2, sx3], [sx2, sx3, sx4]], B = [sy, sxy, sx2y];
        for (let i = 0; i < 3; i++) {
            let p = i;
            for (let r = i + 1; r < 3; r++) if (Math.abs(A[r][i]) > Math.abs(A[p][i])) p = r;
            [A[i], A[p]] = [A[p], A[i]]; [B[i], B[p]] = [B[p], B[i]];
            for (let r = 0; r < 3; r++) {
                if (r === i) continue;
                const fct = A[r][i] / A[i][i];
                for (let c = i; c < 3; c++) A[r][c] -= fct * A[i][c];
                B[r] -= fct * B[i];
            }
        }
        const b = B.map((v, i) => v / A[i][i]);
        let dpath = '';
        for (let i = 0; i <= 60; i++) {
            const x = xs0 + (xs1 - xs0) * i / 60;
            const y = b[0] + b[1] * x + b[2] * x * x;
            dpath += (i ? 'L' : 'M') + X(x) + ',' + Y(y);
        }
        f.svg.appendChild(svgEl('path', {
            d: dpath, fill: 'none', stroke: PAL[0], 'stroke-width': 2.4
        }));
        if (b[2] < 0) {
            const tp = -b[1] / (2 * b[2]);
            if (tp > xs0 && tp < xs1) {
                f.svg.appendChild(svgEl('line', {
                    x1: X(tp), x2: X(tp), y1: f.m.t, y2: f.m.t + f.ih,
                    stroke: PAL[0], 'stroke-width': 1, 'stroke-dasharray': '3 3',
                    'stroke-opacity': .65
                }));
                const t = svgEl('text', {
                    x: X(tp) + 6, y: f.m.t + 13, fill: PAL[0], 'font-size': 11,
                    'font-family': 'JetBrains Mono, monospace'
                });
                t.textContent = 'peak ' + (tp * 100).toFixed(0) + '%';
                f.svg.appendChild(t);
            }
        }
    }
}

/* ---------- profile with error bars ---------- */
function profile(sel, rows, o) {
    const f = frame(sel, o); if (!f) return;
    const xs = rows.map(r => r[o.x]), yv = rows.map(r => r[o.y]);
    const se = o.se ? rows.map(r => r[o.se] || 0) : rows.map(() => 0);
    const x0 = Math.min(...xs), x1 = Math.max(...xs);
    const y0 = Math.min(...yv.map((v, i) => v - se[i])) * .96;
    const y1 = Math.max(...yv.map((v, i) => v + se[i])) * 1.04;
    const X = lin(x0, x1, f.m.l, f.m.l + f.iw), Y = lin(y0, y1, f.m.t + f.ih, f.m.t);
    axes(f, X, Y, ticks(x0, x1, 5), ticks(y0, y1, 5), o.xLab, o.yLab,
        o.fmtX || (v => (v * 100).toFixed(0) + '%'), o.fmtY || (v => v.toFixed(1)));
    const g = svgEl('g', {}); f.svg.appendChild(g);
    rows.forEach((r, i) => {
        if (se[i]) {
            g.appendChild(svgEl('line', {
                x1: X(r[o.x]), x2: X(r[o.x]), y1: Y(r[o.y] - se[i]), y2: Y(r[o.y] + se[i]),
                stroke: 'rgba(255,255,255,0.28)', 'stroke-width': 1.2
            }));
        }
        const c = svgEl('circle', { cx: X(r[o.x]), cy: Y(r[o.y]), r: 4.6, fill: INK });
        const hit = svgEl('circle', {
            cx: X(r[o.x]), cy: Y(r[o.y]), r: 12, fill: 'transparent'
        });
        hit.addEventListener('mousemove', ev => {
            c.setAttribute('fill', PAL[0]); showTip(o.tipFn(r), ev);
        });
        hit.addEventListener('mouseleave', () => { c.setAttribute('fill', INK); hideTip(); });
        g.appendChild(c); g.appendChild(hit);
    });
}

/* ---------- coefficient / dot plot ---------- */
function coefPlot(sel, rows, o) {
    const rowH = o.rowH || 26;
    const f = frame(sel, Object.assign({}, o, {
        height: rows.length * rowH + 62,
        margin: { t: 14, r: 24, b: 44, l: o.labelW || 190 }
    }));
    if (!f) return;
    const lo = Math.min(0, ...rows.map(r => r.lo !== undefined ? r.lo : r.b));
    const hi = Math.max(0, ...rows.map(r => r.hi !== undefined ? r.hi : r.b));
    const pad = (hi - lo) * .12 || 1;
    const X = lin(lo - pad, hi + pad, f.m.l, f.m.l + f.iw);
    const tk = ticks(lo - pad, hi + pad, 4);
    tk.forEach(t => {
        f.svg.appendChild(svgEl('line', {
            x1: X(t), x2: X(t), y1: f.m.t, y2: f.m.t + f.ih,
            stroke: GRIDC, 'shape-rendering': 'crispEdges'
        }));
        const tx = svgEl('text', {
            x: X(t), y: f.m.t + f.ih + 20, 'text-anchor': 'middle', fill: MUTED,
            'font-size': 11, 'font-family': 'JetBrains Mono, monospace'
        });
        tx.textContent = t.toFixed(Math.abs(hi - lo) < 2 ? 2 : 1);
        f.svg.appendChild(tx);
    });
    f.svg.appendChild(svgEl('line', {
        x1: X(0), x2: X(0), y1: f.m.t, y2: f.m.t + f.ih, stroke: AXISC, 'stroke-width': 1.3
    }));
    rows.forEach((r, i) => {
        const y = f.m.t + rowH * (i + .5);
        const lab = svgEl('text', {
            x: f.m.l - 10, y: y + 4, 'text-anchor': 'end', fill: '#B3B3B0', 'font-size': 12.5
        });
        lab.textContent = r.label;
        f.svg.appendChild(lab);
        if (r.lo !== undefined) {
            f.svg.appendChild(svgEl('line', {
                x1: X(r.lo), x2: X(r.hi), y1: y, y2: y, stroke: 'rgba(255,255,255,0.28)', 'stroke-width': 1.4
            }));
        }
        const sig = r.p !== undefined ? r.p < 0.05 : true;
        const c = svgEl('circle', {
            cx: X(r.b), cy: y, r: 5.2,
            fill: sig ? (r.color || INK) : 'transparent',
            stroke: sig ? 'none' : HOLLOW, 'stroke-width': 1.6
        });
        const hit = svgEl('circle', { cx: X(r.b), cy: y, r: 13, fill: 'transparent' });
        hit.addEventListener('mousemove', ev => showTip(o.tipFn(r), ev));
        hit.addEventListener('mouseleave', hideTip);
        f.svg.appendChild(c); f.svg.appendChild(hit);
    });
    if (o.xLab) {
        const t = svgEl('text', {
            x: f.m.l + f.iw / 2, y: f.H - 6, 'text-anchor': 'middle',
            fill: MUTED, 'font-size': 11.5
        });
        t.textContent = o.xLab; f.svg.appendChild(t);
    }
}

/* ---------- correlation matrix ---------- */
function corrMatrix(sel, labels, mat, o) {
    const host = document.querySelector(sel); if (!host) return;
    host.innerHTML = '';
    const n = labels.length;
    const W = host.clientWidth || 900;
    const left = 168, top = 118, cell = Math.min(38, (W - left - 16) / n);
    const H = top + cell * n + 12;
    const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, width: '100%', height: H, class: 'chart' });
    host.appendChild(svg);
    labels.forEach((l, j) => {
        const t = svgEl('text', {
            transform: `translate(${left + cell * (j + .5)},${top - 8}) rotate(-42)`,
            fill: '#B3B3B0', 'font-size': 11.5
        });
        t.textContent = l; svg.appendChild(t);
    });
    labels.forEach((l, i) => {
        const t = svgEl('text', {
            x: left - 9, y: top + cell * (i + .5) + 4, 'text-anchor': 'end',
            fill: '#B3B3B0', 'font-size': 11.5
        });
        t.textContent = l; svg.appendChild(t);
        mat[i].forEach((v, j) => {
            // diverging: one hue each side of a neutral zero, never a rainbow
            const a = Math.min(1, Math.abs(v));
            const col = v >= 0 ? '14,155,181' : '198,63,216';
            const r = svgEl('rect', {
                x: left + cell * j, y: top + cell * i, width: cell - 2, height: cell - 2,
                fill: `rgba(${col},${(a * .88).toFixed(3)})`
            });
            const hit = svgEl('rect', {
                x: left + cell * j, y: top + cell * i, width: cell - 2, height: cell - 2,
                fill: 'transparent'
            });
            hit.addEventListener('mousemove', ev => showTip(
                `<strong>${labels[i]}</strong> and <strong>${labels[j]}</strong><br>r = ${v.toFixed(2)}`, ev));
            hit.addEventListener('mouseleave', hideTip);
            svg.appendChild(r);
            if (cell > 30) {
                const t2 = svgEl('text', {
                    x: left + cell * j + (cell - 2) / 2, y: top + cell * i + (cell - 2) / 2 + 4,
                    'text-anchor': 'middle', 'font-size': 10,
                    'font-family': 'JetBrains Mono, monospace',
                    fill: a > .5 ? '#EDEDEB' : '#9A9A98'
                });
                t2.textContent = v.toFixed(2).replace('0.', '.');
                svg.appendChild(t2);
            }
            svg.appendChild(hit);
        });
    });
}

/* ---------- paired dot (before / after) ---------- */
function pairedDot(sel, rows, o) {
    const rowH = o.rowH || 28;
    const f = frame(sel, Object.assign({}, o, {
        height: rows.length * rowH + 66,
        margin: { t: 14, r: 26, b: 48, l: o.labelW || 200 }
    }));
    if (!f) return;
    const all = rows.flatMap(r => [r.a, r.b]);
    const lo = Math.min(...all), hi = Math.max(...all), pad = (hi - lo) * .1 || 1;
    const X = lin(lo - pad, hi + pad, f.m.l, f.m.l + f.iw);
    ticks(lo - pad, hi + pad, 4).forEach(t => {
        f.svg.appendChild(svgEl('line', {
            x1: X(t), x2: X(t), y1: f.m.t, y2: f.m.t + f.ih, stroke: GRIDC
        }));
        const tx = svgEl('text', {
            x: X(t), y: f.m.t + f.ih + 20, 'text-anchor': 'middle', fill: MUTED,
            'font-size': 11, 'font-family': 'JetBrains Mono, monospace'
        });
        tx.textContent = t.toFixed(1); f.svg.appendChild(tx);
    });
    rows.forEach((r, i) => {
        const y = f.m.t + rowH * (i + .5);
        const lab = svgEl('text', {
            x: f.m.l - 10, y: y + 4, 'text-anchor': 'end', fill: '#B3B3B0', 'font-size': 12.5
        });
        lab.textContent = r.label; f.svg.appendChild(lab);
        f.svg.appendChild(svgEl('line', {
            x1: X(r.a), x2: X(r.b), y1: y, y2: y, stroke: 'rgba(255,255,255,0.24)', 'stroke-width': 2
        }));
        [['a', 'transparent', PAL[1]], ['b', PAL[0], PAL[0]]].forEach(([k, fill, stroke]) => {
            const c = svgEl('circle', {
                cx: X(r[k]), cy: y, r: 5, fill: fill, stroke: stroke, 'stroke-width': 1.8
            });
            const hit = svgEl('circle', { cx: X(r[k]), cy: y, r: 12, fill: 'transparent' });
            hit.addEventListener('mousemove', ev => showTip(o.tipFn(r, k), ev));
            hit.addEventListener('mouseleave', hideTip);
            f.svg.appendChild(c); f.svg.appendChild(hit);
        });
    });
    if (o.xLab) {
        const t = svgEl('text', {
            x: f.m.l + f.iw / 2, y: f.H - 6, 'text-anchor': 'middle', fill: MUTED,
            'font-size': 11.5
        });
        t.textContent = o.xLab; f.svg.appendChild(t);
    }
}

/* ---------- histogram ---------- */
function histogram(sel, values, o) {
    const f = frame(sel, o); if (!f) return;
    const lo = Math.min(...values), hi = Math.max(...values), nb = o.bins || 34;
    const w = (hi - lo) / nb, counts = new Array(nb).fill(0);
    values.forEach(v => { counts[Math.min(nb - 1, Math.floor((v - lo) / w))]++; });
    const X = lin(lo, hi, f.m.l, f.m.l + f.iw);
    const Y = lin(0, Math.max(...counts), f.m.t + f.ih, f.m.t);
    axes(f, X, Y, ticks(lo, hi, 5), ticks(0, Math.max(...counts), 4), o.xLab, o.yLab,
        o.fmtX || (v => v.toFixed(0)), v => v.toFixed(0));
    counts.forEach((c, i) => {
        if (!c) return;
        const bw = Math.max(1, (f.iw / nb) - 2);
        const r = svgEl('rect', {
            x: X(lo + i * w), y: Y(c), width: bw, height: f.m.t + f.ih - Y(c),
            fill: 'rgba(237,237,235,0.55)'
        });
        const hit = svgEl('rect', {
            x: X(lo + i * w), y: f.m.t, width: bw, height: f.ih, fill: 'transparent'
        });
        hit.addEventListener('mousemove', ev => {
            r.setAttribute('fill', PAL[0]);
            showTip(o.tipFn(lo + i * w, lo + (i + 1) * w, c), ev);
        });
        hit.addEventListener('mouseleave', () => { r.setAttribute('fill', 'rgba(237,237,235,0.55)'); hideTip(); });
        f.svg.appendChild(r); f.svg.appendChild(hit);
    });
    if (o.marker !== undefined) {
        f.svg.appendChild(svgEl('line', {
            x1: X(o.marker), x2: X(o.marker), y1: f.m.t, y2: f.m.t + f.ih,
            stroke: PAL[0], 'stroke-width': 2
        }));
    }
}

async function loadJSON(p) { const r = await fetch(p); return r.json(); }

document.addEventListener('DOMContentLoaded', reveal);
window.addEventListener('resize', () => {
    clearTimeout(window.__rz);
    window.__rz = setTimeout(() => {
        if (typeof window.drawAll === 'function') window.drawAll();
    }, 220);
});

/* ---------- choropleth ----------------------------------------------------
   Sequential encoding only: one hue, dark to bright, because the quantity is a
   magnitude. Districts with no data are drawn in the surface tone with a hair
   outline rather than being dropped, so the map stays geographically complete.
--------------------------------------------------------------------------- */
let GEO = null;
async function loadGeo(path) {
    if (!GEO) GEO = await (await fetch(path || 'data/districts.geojson')).json();
    return GEO;
}

function rampCyan(t) {
    // 0 -> near-surface slate, 1 -> full cyan. Lightness rises monotonically.
    const a = [26, 32, 38], b = [64, 214, 240];
    const c = a.map((v, i) => Math.round(v + (b[i] - v) * Math.pow(t, 0.72)));
    return `rgb(${c[0]},${c[1]},${c[2]})`;
}

async function choropleth(sel, field, o) {
    const host = document.querySelector(sel); if (!host) return;
    const geo = await loadGeo(o && o.geo);
    host.innerHTML = '';
    const opts = o || {};
    const vals = geo.features.map(f => f.properties[field])
                             .filter(v => v !== null && v !== undefined);
    if (!vals.length) return;
    const srt = vals.slice().sort((x, y) => x - y);
    const lo = opts.lo !== undefined ? opts.lo : srt[Math.floor(srt.length * 0.02)];
    const hi = opts.hi !== undefined ? opts.hi : srt[Math.floor(srt.length * 0.98)];

    let x0 = 180, x1 = -180, y0 = 90, y1 = -90;
    const walk = (c, d) => {
        if (typeof c[0] === 'number') {
            x0 = Math.min(x0, c[0]); x1 = Math.max(x1, c[0]);
            y0 = Math.min(y0, c[1]); y1 = Math.max(y1, c[1]);
        } else c.forEach(k => walk(k, d + 1));
    };
    geo.features.forEach(f => walk(f.geometry.coordinates, 0));

    const W = host.clientWidth || 900;
    const H = opts.height || Math.min(620, W * 1.02);
    const pad = 8;
    const midLat = (y0 + y1) / 2 * Math.PI / 180;
    const kx = Math.cos(midLat);                       // simple equirectangular
    const sx = (W - pad * 2) / ((x1 - x0) * kx);
    const sy = (H - pad * 2 - 34) / (y1 - y0);
    const s = Math.min(sx, sy);
    const ox = pad + ((W - pad * 2) - (x1 - x0) * kx * s) / 2;
    const oy = pad + ((H - pad * 2 - 34) - (y1 - y0) * s) / 2;
    const PX = lon => ox + (lon - x0) * kx * s;
    const PY = lat => oy + (y1 - lat) * s;

    const svg = svgEl('svg', {
        viewBox: `0 0 ${W} ${H}`, width: '100%', height: H, class: 'chart', role: 'img'
    });
    host.appendChild(svg);

    const path = (coords, depth) => {
        if (depth === 1) {
            return coords.map((p, i) => (i ? 'L' : 'M') + PX(p[0]).toFixed(1) + ',' +
                                         PY(p[1]).toFixed(1)).join('') + 'Z';
        }
        return coords.map(c => path(c, depth - 1)).join('');
    };

    geo.features.forEach(f => {
        const v = f.properties[field];
        const g = f.geometry;
        const depth = g.type === 'Polygon' ? 2 : 3;
        const d = path(g.coordinates, depth);
        if (!d) return;
        const has = v !== null && v !== undefined;
        const t = has ? Math.max(0, Math.min(1, (v - lo) / (hi - lo || 1))) : 0;
        const el = svgEl('path', {
            d: d,
            fill: has ? rampCyan(t) : 'rgba(255,255,255,0.035)',
            stroke: 'rgba(11,11,12,0.85)', 'stroke-width': 0.4
        });
        el.addEventListener('mousemove', ev => {
            el.setAttribute('stroke', '#EDEDEB');
            el.setAttribute('stroke-width', 1.1);
            showTip(opts.tipFn ? opts.tipFn(f.properties)
                : `<strong>${f.properties.n}</strong><br>${f.properties.s}<br>` +
                  (has ? (opts.label || field) + ': ' + v.toFixed(2) : 'no data'), ev);
        });
        el.addEventListener('mouseleave', () => {
            el.setAttribute('stroke', 'rgba(11,11,12,0.85)');
            el.setAttribute('stroke-width', 0.4); hideTip();
        });
        svg.appendChild(el);
    });

    // continuous key, since the encoding is continuous
    const kw = Math.min(240, W * 0.34), kx0 = W - kw - 10, ky = H - 26;
    const defs = svgEl('defs', {});
    const gid = 'g_' + Math.random().toString(36).slice(2, 8);
    const lg = svgEl('linearGradient', { id: gid, x1: '0', x2: '1', y1: '0', y2: '0' });
    for (let i = 0; i <= 10; i++) {
        lg.appendChild(svgEl('stop', {
            offset: (i * 10) + '%', 'stop-color': rampCyan(i / 10)
        }));
    }
    defs.appendChild(lg); svg.appendChild(defs);
    svg.appendChild(svgEl('rect', {
        x: kx0, y: ky, width: kw, height: 9, fill: `url(#${gid})`,
        stroke: 'rgba(255,255,255,0.18)', 'stroke-width': .6
    }));
    [[kx0, lo, 'start'], [kx0 + kw, hi, 'end']].forEach(([xx, vv, anch]) => {
        const t = svgEl('text', {
            x: xx, y: ky - 6, 'text-anchor': anch, fill: MUTED, 'font-size': 10.5,
            'font-family': 'JetBrains Mono, monospace'
        });
        t.textContent = (opts.fmt || (n => n.toFixed(1)))(vv);
        svg.appendChild(t);
    });
    if (opts.label) {
        const t = svgEl('text', {
            x: kx0 + kw / 2, y: ky + 22, 'text-anchor': 'middle', fill: MUTED,
            'font-size': 10.5
        });
        t.textContent = opts.label; svg.appendChild(t);
    }
}

/* ---------- categorical and log choropleth -------------------------------
   choropleth() handles a magnitude with one hue. Two layers need more: food
   energy per hectare spans two orders of magnitude and wants a log scale, and
   the quadrant is four named classes rather than a magnitude, so it takes the
   validated categorical palette in fixed order.
------------------------------------------------------------------------- */
const QUAD_COLOUR = {
    'diverse and energy-rich':      PAL[0],   // cyan
    'concentrated and energy-rich': PAL[1],   // violet
    'diverse and energy-poor':      PAL[2],   // green
    'concentrated and energy-poor': PAL[3],   // magenta
};

async function mapLayer(sel, cfg) {
    const host = document.querySelector(sel); if (!host) return;
    const geo = await loadGeo(cfg.geo);
    host.innerHTML = '';

    const isCat = cfg.kind === 'category';
    let lo = 0, hi = 1;
    if (!isCat) {
        const raw = geo.features.map(f => f.properties[cfg.field])
                                .filter(v => v !== null && v !== undefined && v > (cfg.log ? 0 : -Infinity));
        const vals = (cfg.log ? raw.map(Math.log10) : raw).sort((a, b) => a - b);
        lo = cfg.lo !== undefined ? cfg.lo : vals[Math.floor(vals.length * 0.02)];
        hi = cfg.hi !== undefined ? cfg.hi : vals[Math.floor(vals.length * 0.98)];
    }

    let x0 = 180, x1 = -180, y0 = 90, y1 = -90;
    const walk = c => {
        if (typeof c[0] === 'number') {
            x0 = Math.min(x0, c[0]); x1 = Math.max(x1, c[0]);
            y0 = Math.min(y0, c[1]); y1 = Math.max(y1, c[1]);
        } else c.forEach(walk);
    };
    geo.features.forEach(f => walk(f.geometry.coordinates));

    const W = host.clientWidth || 900, H = cfg.height || Math.min(640, W * 1.02), pad = 8;
    const kx = Math.cos((y0 + y1) / 2 * Math.PI / 180);
    const s = Math.min((W - pad * 2) / ((x1 - x0) * kx), (H - pad * 2 - 36) / (y1 - y0));
    const ox = pad + ((W - pad * 2) - (x1 - x0) * kx * s) / 2;
    const oy = pad + ((H - pad * 2 - 36) - (y1 - y0) * s) / 2;
    const PX = v => ox + (v - x0) * kx * s, PY = v => oy + (y1 - v) * s;

    const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, width: '100%', height: H, class: 'chart' });
    host.appendChild(svg);
    const marks = [];
    const path = (c, d) => d === 1
        ? c.map((p, i) => (i ? 'L' : 'M') + PX(p[0]).toFixed(1) + ',' + PY(p[1]).toFixed(1)).join('') + 'Z'
        : c.map(k => path(k, d - 1)).join('');

    geo.features.forEach(f => {
        const pr = f.properties, v = pr[cfg.field];
        const d = path(f.geometry.coordinates, f.geometry.type === 'Polygon' ? 2 : 3);
        if (!d) return;
        const has = v !== null && v !== undefined && v !== '';
        let fill = 'rgba(255,255,255,0.035)';
        if (has) {
            if (isCat) fill = QUAD_COLOUR[v] || 'rgba(255,255,255,0.2)';
            else {
                const t = ((cfg.log ? Math.log10(v) : v) - lo) / (hi - lo || 1);
                fill = rampCyan(Math.max(0, Math.min(1, t)));
            }
        }
        const el = svgEl('path', { d, fill, stroke: 'rgba(11,11,12,0.85)', 'stroke-width': 0.4 });
        el.addEventListener('mousemove', ev => {
            el.setAttribute('stroke', '#EDEDEB'); el.setAttribute('stroke-width', 1.1);
            showTip(cfg.tipFn(pr), ev);
        });
        el.addEventListener('mouseleave', () => {
            el.setAttribute('stroke', 'rgba(11,11,12,0.85)');
            el.setAttribute('stroke-width', 0.4); hideTip();
        });
        svg.appendChild(el);
        marks.push({ el: el, pr: pr });
    });

    if (isCat) {
        // a categorical legend is always present; colour never carries identity alone
        const keys = Object.keys(QUAD_COLOUR);
        const bw = Math.min(300, W * 0.42);
        keys.forEach((k, i) => {
            const yy = H - 36 + (i % 2) * 17, xx = W - bw * 2 - 12 + Math.floor(i / 2) * bw;
            svg.appendChild(svgEl('rect', { x: xx, y: yy - 8, width: 10, height: 10, fill: QUAD_COLOUR[k] }));
            const t = svgEl('text', { x: xx + 15, y: yy + 1, fill: MUTED, 'font-size': 10.5 });
            t.textContent = k; svg.appendChild(t);
        });
    } else {
        const kw = Math.min(240, W * 0.34), kx0 = W - kw - 10, ky = H - 26;
        const gid = 'g' + Math.random().toString(36).slice(2, 8);
        const defs = svgEl('defs', {}), lg = svgEl('linearGradient', { id: gid, x1: '0', x2: '1', y1: '0', y2: '0' });
        for (let i = 0; i <= 10; i++) lg.appendChild(svgEl('stop', { offset: i * 10 + '%', 'stop-color': rampCyan(i / 10) }));
        defs.appendChild(lg); svg.appendChild(defs);
        svg.appendChild(svgEl('rect', { x: kx0, y: ky, width: kw, height: 9, fill: `url(#${gid})`, stroke: 'rgba(255,255,255,0.18)', 'stroke-width': .6 }));
        [[kx0, lo, 'start'], [kx0 + kw, hi, 'end']].forEach(([xx, vv, a]) => {
            const t = svgEl('text', { x: xx, y: ky - 6, 'text-anchor': a, fill: MUTED, 'font-size': 10.5, 'font-family': 'JetBrains Mono, monospace' });
            t.textContent = cfg.fmt(cfg.log ? Math.pow(10, vv) : vv); svg.appendChild(t);
        });
        if (cfg.label) {
            const t = svgEl('text', { x: kx0 + kw / 2, y: ky + 22, 'text-anchor': 'middle', fill: MUTED, 'font-size': 10.5 });
            t.textContent = cfg.label; svg.appendChild(t);
        }
    }
    return {
        lo: lo, hi: hi, svg: svg,
        // Redrawing 735 polygons per frame is far too slow to animate, so a
        // caller that steps through time recolours the marks already on screen.
        recolour: function (valueOf, rlo, rhi) {
            const a = rlo === undefined ? lo : rlo, b = rhi === undefined ? hi : rhi;
            marks.forEach(function (m) {
                const v = valueOf(m.pr);
                m.el.setAttribute('fill', (v === null || v === undefined)
                    ? 'rgba(255,255,255,0.035)'
                    : rampCyan(Math.max(0, Math.min(1, (v - a) / (b - a || 1)))));
            });
        },
    };
}

/* ---------- layer switcher ---------- */
function layerTabs(sel, layers, onPick) {
    const host = document.querySelector(sel); if (!host) return;
    host.innerHTML = '';
    layers.forEach((l, i) => {
        const b = document.createElement('button');
        b.textContent = l.name;
        b.style.cssText = 'background:none;border:0;border-bottom:1px solid transparent;' +
            'color:inherit;font:inherit;font-size:.84rem;cursor:pointer;padding:0 0 .25rem 0;';
        b.addEventListener('click', () => {
            [...host.children].forEach((c, j) => {
                c.style.opacity = j === i ? '1' : '.45';
                c.style.borderBottomColor = j === i ? PAL[0] : 'transparent';
            });
            onPick(l, i);
        });
        host.appendChild(b);
    });
    host.children[0].click();
}
