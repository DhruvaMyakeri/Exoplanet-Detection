"""
Generate explorer.html - the candidate explorer - with the data inlined.

The payload from build_explorer_data.py is injected into a <script
type="application/json"> block so the page is a single self-contained file
with no network fetches (the Artifact CSP blocks them anyway).
"""

import json

TEMPLATE = r"""<title>Kepler Vetting Console</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Serif:wght@600&display=swap">
<style>
:root{
  --ground:#F2F5F7; --surface:#FFFFFF; --surface2:#E8EDF1; --raised:#FBFDFE;
  --ink:#101820; --ink2:#4A5A68; --ink3:#788A99;
  --line:#D3DDE4; --line2:#C0CDD7;
  --accent:#2E7D9A; --accent-soft:#DCEAF0;
  --planet:#1F8A6D; --planet-soft:#DCF0E8;
  --fp:#B15540; --fp-soft:#F6E3DD;
  --trace:#2A3A47; --grid:#E2E9EE;
  --shadow:0 1px 2px rgba(16,24,32,.06),0 4px 16px rgba(16,24,32,.05);
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --ground:#0A0E13; --surface:#121820; --surface2:#1A222C; --raised:#161E27;
    --ink:#E3EBF2; --ink2:#8A9BAA; --ink3:#66788A;
    --line:#26313D; --line2:#334352;
    --accent:#4FA8C7; --accent-soft:#16303C;
    --planet:#35B891; --planet-soft:#0E2E26;
    --fp:#D97A62; --fp-soft:#331912;
    --trace:#B9C9D6; --grid:#1D2731;
    --shadow:0 1px 2px rgba(0,0,0,.4),0 4px 16px rgba(0,0,0,.3);
  }
}
:root[data-theme="dark"]{
  --ground:#0A0E13; --surface:#121820; --surface2:#1A222C; --raised:#161E27;
  --ink:#E3EBF2; --ink2:#8A9BAA; --ink3:#66788A;
  --line:#26313D; --line2:#334352;
  --accent:#4FA8C7; --accent-soft:#16303C;
  --planet:#35B891; --planet-soft:#0E2E26;
  --fp:#D97A62; --fp-soft:#331912;
  --trace:#B9C9D6; --grid:#1D2731;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 4px 16px rgba(0,0,0,.3);
}
*{box-sizing:border-box}
body{
  background:var(--ground); color:var(--ink);
  font-family:"IBM Plex Sans",-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  font-size:14px; line-height:1.5; margin:0;
  -webkit-font-smoothing:antialiased;
}
.wrap{max-width:1500px;margin:0 auto;padding:22px 22px 40px}

/* ---------- masthead ---------- */
.mast{display:flex;flex-wrap:wrap;gap:20px;align-items:flex-end;
  justify-content:space-between;padding-bottom:16px;border-bottom:1px solid var(--line)}
h1{font-family:"IBM Plex Serif",Georgia,serif;font-weight:600;font-size:25px;
  margin:0 0 5px;letter-spacing:-.01em;text-wrap:balance}
.sub{color:var(--ink2);font-size:13px;max-width:66ch;margin:0}
.sub code{font-family:"IBM Plex Mono",monospace;font-size:12px;
  background:var(--surface2);padding:1px 5px;border-radius:3px}
.metrics{display:flex;gap:26px;flex-wrap:wrap}
.metric{display:flex;flex-direction:column;gap:1px}
.metric b{font-family:"IBM Plex Mono",monospace;font-size:19px;font-weight:600;
  font-variant-numeric:tabular-nums;letter-spacing:-.02em}
.metric span{font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:var(--ink3)}

/* ---------- controls ---------- */
.controls{display:flex;gap:9px;flex-wrap:wrap;align-items:center;
  margin:16px 0;padding:11px 13px;background:var(--surface);
  border:1px solid var(--line);border-radius:7px}
input[type=search],select{
  font-family:inherit;font-size:13px;color:var(--ink);
  background:var(--ground);border:1px solid var(--line2);border-radius:5px;
  padding:6px 9px}
input[type=search]{min-width:190px}
input[type=search]::placeholder{color:var(--ink3)}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.seg{display:flex;border:1px solid var(--line2);border-radius:5px;overflow:hidden}
.seg button{font-family:inherit;font-size:12px;color:var(--ink2);
  background:var(--ground);border:0;padding:6px 11px;cursor:pointer;
  border-right:1px solid var(--line2)}
.seg button:last-child{border-right:0}
.seg button[aria-pressed=true]{background:var(--accent);color:#fff;font-weight:500}
.chk{display:flex;align-items:center;gap:6px;font-size:12px;color:var(--ink2);
  cursor:pointer;user-select:none}
.count{margin-left:auto;font-family:"IBM Plex Mono",monospace;font-size:12px;
  color:var(--ink3);font-variant-numeric:tabular-nums}

/* ---------- layout ---------- */
.grid{display:grid;grid-template-columns:minmax(330px,.85fr) 1.6fr;gap:16px;align-items:start}
@media (max-width:1000px){.grid{grid-template-columns:1fr}}

.panel{background:var(--surface);border:1px solid var(--line);
  border-radius:7px;box-shadow:var(--shadow)}
.phead{display:flex;align-items:center;justify-content:space-between;gap:10px;
  padding:9px 13px;border-bottom:1px solid var(--line)}
.phead h2{font-size:11px;text-transform:uppercase;letter-spacing:.08em;
  color:var(--ink3);margin:0;font-weight:600}

/* ---------- list ---------- */
.list{max-height:74vh;overflow-y:auto;overflow-x:hidden}
.row{display:grid;grid-template-columns:auto 1fr auto;gap:9px;align-items:center;
  width:100%;text-align:left;font-family:inherit;font-size:13px;color:var(--ink);
  background:none;border:0;border-bottom:1px solid var(--line);
  padding:8px 13px;cursor:pointer}
.row:hover{background:var(--surface2)}
.row[aria-current=true]{background:var(--accent-soft);
  box-shadow:inset 3px 0 0 var(--accent)}
.koi{font-family:"IBM Plex Mono",monospace;font-size:12.5px;font-weight:500}
.rmeta{font-family:"IBM Plex Mono",monospace;font-size:10.5px;color:var(--ink3);
  font-variant-numeric:tabular-nums}
.dot{width:8px;height:8px;border-radius:50%;flex:none}
.dot.p{background:var(--planet)} .dot.f{background:var(--fp)}
.pct{font-family:"IBM Plex Mono",monospace;font-size:12px;font-weight:600;
  font-variant-numeric:tabular-nums;min-width:42px;text-align:right}
.flag{font-family:"IBM Plex Mono",monospace;font-size:9px;letter-spacing:.05em;
  padding:1px 4px;border-radius:3px;background:var(--fp-soft);color:var(--fp);
  border:1px solid var(--fp)}

/* ---------- detail ---------- */
.dbody{padding:14px}
.dtop{display:flex;flex-wrap:wrap;gap:14px;align-items:baseline;
  justify-content:space-between;margin-bottom:4px}
.dtitle{font-family:"IBM Plex Mono",monospace;font-size:20px;font-weight:600;
  letter-spacing:-.01em}
.badge{font-size:11px;font-weight:600;padding:2px 9px;border-radius:11px;
  letter-spacing:.03em}
.badge.p{background:var(--planet-soft);color:var(--planet);border:1px solid var(--planet)}
.badge.f{background:var(--fp-soft);color:var(--fp);border:1px solid var(--fp)}
.dnote{font-size:12px;color:var(--ink2);margin:0 0 13px}

.charts{display:grid;grid-template-columns:1fr 1fr;gap:13px}
@media (max-width:720px){.charts{grid-template-columns:1fr}}
.chart{background:var(--raised);border:1px solid var(--line);border-radius:6px;padding:10px}
.ctitle{font-size:10.5px;text-transform:uppercase;letter-spacing:.07em;
  color:var(--ink3);font-weight:600;margin-bottom:2px}
.chint{font-size:11px;color:var(--ink2);margin-bottom:7px;min-height:30px}
svg{display:block;width:100%;height:auto}

.models{margin-top:15px;display:flex;flex-direction:column;gap:7px}
.mrow{display:grid;grid-template-columns:74px 1fr 50px;gap:10px;align-items:center}
.mname{font-size:11.5px;color:var(--ink2);font-weight:500}
.mname em{font-style:normal;color:var(--ink3);font-size:10px;display:block}
.bar{height:16px;background:var(--surface2);border-radius:3px;overflow:hidden;
  position:relative}
.bar i{display:block;height:100%;border-radius:3px}
.bar .mid{position:absolute;left:50%;top:0;bottom:0;width:1px;background:var(--line2)}
.mval{font-family:"IBM Plex Mono",monospace;font-size:12px;font-weight:600;
  text-align:right;font-variant-numeric:tabular-nums}

.params{margin-top:16px;display:grid;
  grid-template-columns:repeat(auto-fit,minmax(112px,1fr));gap:1px;
  background:var(--line);border:1px solid var(--line);border-radius:6px;overflow:hidden}
.par{background:var(--surface);padding:8px 10px}
.par span{display:block;font-size:9.5px;text-transform:uppercase;
  letter-spacing:.07em;color:var(--ink3);margin-bottom:2px}
.par b{font-family:"IBM Plex Mono",monospace;font-size:13px;font-weight:500;
  font-variant-numeric:tabular-nums}
.foot{margin-top:22px;padding-top:14px;border-top:1px solid var(--line);
  font-size:11.5px;color:var(--ink3);max-width:88ch}
.foot b{color:var(--ink2);font-weight:600}
@media (prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
</style>

<div class="wrap">
  <header class="mast">
    <div>
      <h1>Kepler Vetting Console</h1>
      <p class="sub">Every object below is from the <b>held-out test set</b> &mdash; 1,387 signals
      the models never saw during training or selection. Scores are Platt-calibrated,
      so <code>0.90</code> means roughly 90&thinsp;%.</p>
    </div>
    <div class="metrics">
      <div class="metric"><b>0.961</b><span>PR-AUC</span></div>
      <div class="metric"><b>77.7%</b><span>recall @ 95% prec</span></div>
      <div class="metric"><b>0.019</b><span>calib. error</span></div>
      <div class="metric"><b id="mDis">&mdash;</b><span>disagreements</span></div>
    </div>
  </header>

  <div class="controls">
    <input type="search" id="q" placeholder="Search KOI or KIC&hellip;" aria-label="Search by KOI or KIC">
    <div class="seg" role="group" aria-label="Filter by truth">
      <button data-tru="all" aria-pressed="true">All</button>
      <button data-tru="1" aria-pressed="false">Planets</button>
      <button data-tru="0" aria-pressed="false">False pos.</button>
    </div>
    <div class="seg" role="group" aria-label="Filter by signal-to-noise">
      <button data-snr="all" aria-pressed="true">Any SNR</button>
      <button data-snr="0" aria-pressed="false">&lt;3</button>
      <button data-snr="1" aria-pressed="false">3&ndash;7</button>
      <button data-snr="2" aria-pressed="false">&gt;7</button>
    </div>
    <label class="chk"><input type="checkbox" id="dis"> Model disagreements only</label>
    <label class="chk"><input type="checkbox" id="wrong"> Wrong at 0.5</label>
    <select id="sort" aria-label="Sort order">
      <option value="dis">Sort: disagreement</option>
      <option value="conf">Sort: confidence</option>
      <option value="snr">Sort: SNR</option>
      <option value="per">Sort: period</option>
      <option value="koi">Sort: KOI</option>
    </select>
    <span class="count" id="count"></span>
  </div>

  <div class="grid">
    <section class="panel">
      <div class="phead"><h2>Candidates</h2><span class="rmeta" id="listNote"></span></div>
      <div class="list" id="list"></div>
    </section>
    <section class="panel"><div class="dbody" id="detail"></div></section>
  </div>

  <p class="foot"><b>How to read the curves.</b> The global view folds the whole orbit, so a
  second dip near phase &plusmn;0.5 is a secondary eclipse &mdash; light from a companion star,
  not a planet. The local view zooms to &plusmn;2.5 transit durations: a planet crossing a stellar
  disc gives a flat-bottomed <b>U</b>, while a grazing binary never fully overlaps and gives a
  <b>V</b>. Depth is normalised away (baseline&nbsp;0, depth&nbsp;&minus;1) so the network must
  read shape rather than simply flagging deep events as binaries.</p>
</div>

<script type="application/json" id="payload">__DATA__</script>
<script>
(function(){
  const D = JSON.parse(document.getElementById('payload').textContent);
  const M = D.meta, R = D.rows, N = M.n, S = M.scale;

  function decode(b64, per){
    const bin = atob(b64), u = new Uint8Array(bin.length);
    for(let i=0;i<bin.length;i++) u[i]=bin.charCodeAt(i);
    return new Int16Array(u.buffer);
  }
  const G = decode(D.g), L = decode(D.l);
  const GB = M.gbins, LB = M.lbins;

  R.forEach((r,i)=>{
    r.i=i;
    r.band = r.s<3?0:(r.s<7?1:2);
    r.gap  = Math.abs(r.c-r.r);
    r.wrong = (r.e>=0.5?1:0)!==r.y;
  });
  document.getElementById('mDis').textContent = R.filter(r=>r.gap>0.5).length;

  const st={q:'',tru:'all',snr:'all',dis:false,wrong:false,sort:'dis',sel:null};

  function filtered(){
    const q=st.q.trim().toLowerCase();
    let out=R.filter(r=>{
      if(st.tru!=='all' && r.y!==+st.tru) return false;
      if(st.snr!=='all' && r.band!==+st.snr) return false;
      if(st.dis && r.gap<=0.5) return false;
      if(st.wrong && !r.wrong) return false;
      if(q && !(r.n.toLowerCase().includes(q) || String(r.k).includes(q))) return false;
      return true;
    });
    const cmp={dis:(a,b)=>b.gap-a.gap, conf:(a,b)=>b.e-a.e,
               snr:(a,b)=>b.s-a.s, per:(a,b)=>a.p-b.p,
               koi:(a,b)=>a.n.localeCompare(b.n)}[st.sort];
    return out.sort(cmp);
  }

  const fmt=(v,d)=> v==null||!isFinite(v) ? '&mdash;' : (+v).toFixed(d);

  function renderList(){
    const rows=filtered();
    document.getElementById('count').textContent = rows.length+' of '+N;
    document.getElementById('listNote').textContent =
      st.sort==='dis' ? 'largest CNN–RF gap first' : '';
    const el=document.getElementById('list');
    if(!rows.length){ el.innerHTML='<div style="padding:22px;color:var(--ink3)">No objects match these filters.</div>'; return; }
    el.innerHTML = rows.slice(0,400).map(r=>`
      <button class="row" data-i="${r.i}" aria-current="${st.sel===r.i}">
        <span class="dot ${r.y?'p':'f'}" title="${r.y?'Confirmed planet':'False positive'}"></span>
        <span>
          <span class="koi">${r.n}</span>
          ${r.gap>0.5?'<span class="flag">SPLIT</span>':''}
          <br><span class="rmeta">SNR ${fmt(r.s,1)} &middot; P ${fmt(r.p,2)} d</span>
        </span>
        <span class="pct" style="color:${r.e>=0.5?'var(--planet)':'var(--fp)'}">${(r.e*100).toFixed(0)}%</span>
      </button>`).join('') +
      (rows.length>400?`<div style="padding:11px 13px;color:var(--ink3);font-size:12px">
        Showing first 400 of ${rows.length}. Narrow the filters to see the rest.</div>`:'');
    el.querySelectorAll('.row').forEach(b=>b.onclick=()=>{st.sel=+b.dataset.i;renderList();renderDetail();});
  }

  // ---- charts -------------------------------------------------------
  function curve(arr, off, n, opts){
    const W=520,H=190,PL=42,PR=10,PT=10,PB=26;
    let lo=Infinity,hi=-Infinity;
    const v=new Array(n);
    for(let i=0;i<n;i++){ const x=arr[off+i]/S; v[i]=x; if(x<lo)lo=x; if(x>hi)hi=x; }
    // Always show the baseline and the normalised depth, whatever the noise does.
    lo=Math.min(lo,-1.15); hi=Math.max(hi,0.3);
    const pad=(hi-lo)*0.08; lo-=pad; hi+=pad;
    const X=i=>PL+(i/(n-1))*(W-PL-PR);
    const Y=y=>PT+(1-(y-lo)/(hi-lo))*(H-PT-PB);
    let d='';
    for(let i=0;i<n;i++) d+=(i?'L':'M')+X(i).toFixed(1)+' '+Y(v[i]).toFixed(1);
    let g='';
    [0,-0.5,-1].forEach(y=>{ if(y>=lo&&y<=hi){
      g+=`<line x1="${PL}" y1="${Y(y).toFixed(1)}" x2="${W-PR}" y2="${Y(y).toFixed(1)}"
           stroke="var(--grid)" stroke-width="1" ${y===-1?'stroke-dasharray="3 3"':''}/>`;
      g+=`<text x="${PL-6}" y="${(Y(y)+3.5).toFixed(1)}" text-anchor="end" fill="var(--ink3)"
           font-family="IBM Plex Mono, monospace" font-size="9">${y.toFixed(y===0?0:1)}</text>`; }});
    let marks='';
    opts.marks.forEach(m=>{
      const px=X(m.at*(n-1));
      marks+=`<line x1="${px.toFixed(1)}" y1="${PT}" x2="${px.toFixed(1)}" y2="${H-PB}"
        stroke="${m.strong?'var(--accent)':'var(--line2)'}" stroke-width="1"
        stroke-dasharray="${m.strong?'':'2 3'}" opacity="${m.strong?.75:.9}"/>`;
      marks+=`<text x="${px.toFixed(1)}" y="${H-PB+13}" text-anchor="middle" fill="var(--ink3)"
        font-family="IBM Plex Mono, monospace" font-size="9">${m.lab}</text>`;
    });
    return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${opts.alt}">
      <rect x="${PL}" y="${PT}" width="${W-PL-PR}" height="${H-PT-PB}" fill="none" stroke="var(--line)"/>
      ${g}${marks}
      <path d="${d}" fill="none" stroke="var(--trace)" stroke-width="1.25"
            stroke-linejoin="round" vector-effect="non-scaling-stroke"/>
      <text x="${PL}" y="${H-4}" fill="var(--ink3)" font-family="IBM Plex Sans, sans-serif"
            font-size="9.5">${opts.xlab}</text></svg>`;
  }

  function renderDetail(){
    const r = R[st.sel];
    const el = document.getElementById('detail');
    if(!r){ el.innerHTML='<p class="dnote">Select a candidate to inspect it.</p>'; return; }
    const truth = r.y?'Confirmed planet':'False positive';
    const disagree = r.gap>0.5;
    const cnnRight = Math.abs(r.c-r.y) < Math.abs(r.r-r.y);

    const bars=[['Ensemble','calibrated',r.e],['CNN','shape only',r.c],['RF','catalogue features',r.r]]
      .map(([nm,sub,v])=>`<div class="mrow">
        <span class="mname">${nm}<em>${sub}</em></span>
        <span class="bar"><i style="width:${(v*100).toFixed(1)}%;background:${v>=.5?'var(--planet)':'var(--fp)'}"></i><span class="mid"></span></span>
        <span class="mval">${(v*100).toFixed(0)}%</span></div>`).join('');

    const P=[['Period',fmt(r.p,3)+' d'],['Duration',fmt(r.d,2)+' h'],
             ['Depth',r.dp==null?'&mdash;':fmt(r.dp,0)+' ppm'],['SNR',fmt(r.s,2)],
             ['Radius',r.pr==null?'&mdash;':fmt(r.pr,2)+' R⊕'],
             ['Star T<sub>eff</sub>',r.te==null?'&mdash;':r.te+' K'],
             ['Star radius',r.sr==null?'&mdash;':fmt(r.sr,2)+' R☉'],
             ['Kepler mag',fmt(r.km,2)],['Cadences',r.np_.toLocaleString()],['KIC',r.k]]
      .map(([k,v])=>`<div class="par"><span>${k}</span><b>${v}</b></div>`).join('');

    el.innerHTML=`
      <div class="dtop">
        <span class="dtitle">${r.n}</span>
        <span class="badge ${r.y?'p':'f'}">${truth}</span>
      </div>
      <p class="dnote">${
        disagree
          ? `The two models <b>disagree sharply</b> here &mdash; CNN ${(r.c*100).toFixed(0)}%,
             RF ${(r.r*100).toFixed(0)}%. The ${cnnRight?'CNN':'RF'} is closer to the truth.
             ${r.dp!=null&&r.dp>50000?'At '+Math.round(r.dp/1e4)/1e2+'% deep this is far too deep for a planet &mdash; a cue the RF sees in the catalogue depth and the CNN cannot, because depth is normalised out of the views.':''}`
          : `Both models agree. ${r.wrong?'Both are <b>wrong</b> at a 0.5 threshold.':'Both are correct at a 0.5 threshold.'}`
      }</p>
      <div class="charts">
        <div class="chart">
          <div class="ctitle">Global view &mdash; full orbit</div>
          <div class="chint">A dip near phase &plusmn;0.5 is a secondary eclipse: a companion star.</div>
          ${curve(G, r.i*GB, GB, {alt:'Folded light curve over the full orbit for '+r.n,
             xlab:'orbital phase', marks:[{at:0,lab:'−0.5'},{at:.5,lab:'0',strong:true},{at:1,lab:'+0.5'}]})}
        </div>
        <div class="chart">
          <div class="ctitle">Local view &mdash; transit</div>
          <div class="chint">Flat-bottomed <b>U</b> = planet. Sharp <b>V</b> = grazing binary.</div>
          ${curve(L, r.i*LB, LB, {alt:'Zoomed transit shape for '+r.n,
             xlab:'transit durations from centre', marks:[{at:0,lab:'−2.5'},{at:.5,lab:'0',strong:true},{at:1,lab:'+2.5'}]})}
        </div>
      </div>
      <div class="models">${bars}</div>
      <div class="params">${P}</div>`;
  }

  // ---- wiring -------------------------------------------------------
  document.getElementById('q').oninput=e=>{st.q=e.target.value;renderList();};
  document.getElementById('dis').onchange=e=>{st.dis=e.target.checked;renderList();};
  document.getElementById('wrong').onchange=e=>{st.wrong=e.target.checked;renderList();};
  document.getElementById('sort').onchange=e=>{st.sort=e.target.value;renderList();};
  document.querySelectorAll('.seg').forEach(seg=>{
    seg.querySelectorAll('button').forEach(b=>b.onclick=()=>{
      seg.querySelectorAll('button').forEach(x=>x.setAttribute('aria-pressed','false'));
      b.setAttribute('aria-pressed','true');
      if(b.dataset.tru!==undefined) st.tru=b.dataset.tru; else st.snr=b.dataset.snr;
      renderList();
    });
  });

  // Open on the most instructive object: the sharpest model disagreement.
  st.sel = R.slice().sort((a,b)=>b.gap-a.gap)[0].i;
  renderList(); renderDetail();
})();
</script>
"""


def main():
    with open("explorer_data.json") as f:
        data = f.read()
    html = TEMPLATE.replace("__DATA__", data)
    with open("explorer.html", "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[done] explorer.html  {len(html)/1e6:.2f} MB")


if __name__ == "__main__":
    main()
