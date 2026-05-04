// photon-route Cloudflare Worker
// Routes photon.ask-meridian.uk → HF Space (luuow-photon-route),
// edge-caches /rank, and serves an inline interactive UI at /.
//
// Endpoints (worker-local):
//   /         interactive HTML page
//   /health   liveness JSON
//   /api      service banner JSON
// Endpoints (proxied to HF Space):
//   /rank?q=&top_k=   edge-cached 24 h
//   /version /docs /openapi.json
//
// Cache key includes CACHE_VERSION; bump to invalidate after UI changes
// or fixture updates upstream.

const HF = 'https://luuow-photon-route.hf.space';
const PROXY = new Set(['/rank', '/version', '/docs', '/openapi.json']);
const CACHE_VERSION = 'v2';

const BANNER = {
  service: 'photon-route',
  proxy: 'cloudflare-worker',
  backend: 'huggingface-space',
  upstream: HF,
  repo: 'https://github.com/LuuOW/photon-route',
  sister: 'https://qrouter.ask-meridian.uk (DV / qubit-gate sister project)',
  endpoints: {
    ui:      '/         (interactive HTML)',
    api:     '/api      (this banner)',
    health:  '/health   (worker-local)',
    rank:    '/rank?q=<query>&top_k=N   (proxied, edge-cached 24 h)',
    version: '/version  (proxied)',
    docs:    '/docs     (proxied — FastAPI swagger)',
  },
  note: 'CV photonic retrieval. Strawberry Fields Gaussian programs, thewalrus closed-form fidelity.',
};

const CSP = [
  "default-src 'self'",
  "style-src 'self' 'unsafe-inline'",
  "script-src 'self' 'unsafe-inline'",
  "connect-src 'self'",
  "img-src 'self' data:",
  "base-uri 'none'",
  "form-action 'none'",
  "frame-ancestors 'none'",
].join('; ');

const HTML = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>photon-route — continuous-variable photonic retrieval</title>
<meta name="description" content="Semantic retrieval over Strawberry Fields Gaussian states. Query light, ranked by Gaussian-state fidelity.">
<meta name="theme-color" content="#06080f">
<meta name="color-scheme" content="dark">
<style>
  :root{
    --bg:#06080f; --panel:#0d1220; --panel2:#11182b; --line:#1c2742;
    --fg:#e7ecf5; --dim:#7a8aa8; --muted:#5a6b8c;
    --cyan:#22d3ee; --indigo:#818cf8; --magenta:#c084fc; --green:#34d399;
    --mono:'JetBrains Mono','SF Mono',ui-monospace,Menlo,Consolas,monospace;
    --radius:8px;
  }
  *{box-sizing:border-box}
  html,body{margin:0;padding:0;background:var(--bg);color:var(--fg);
    font-family:var(--mono);font-size:14px;line-height:1.55;
    -webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility}
  body{
    min-height:100vh;
    background:
      radial-gradient(ellipse at 18% -10%, rgba(129,140,248,.20), transparent 60%),
      radial-gradient(ellipse at 92% 8%,  rgba(34,211,238,.12),  transparent 55%),
      radial-gradient(ellipse at 50% 110%,rgba(192,132,252,.12), transparent 55%),
      var(--bg);
  }
  main{max-width:780px;margin:0 auto;padding:32px 20px 64px}
  header{display:flex;align-items:flex-end;justify-content:space-between;
    flex-wrap:wrap;gap:14px;margin-bottom:20px;
    border-bottom:1px solid var(--line);padding-bottom:16px}
  .brand{font-size:clamp(20px,3vw,26px);font-weight:600;letter-spacing:-.01em;
    background:linear-gradient(90deg,var(--indigo),var(--cyan) 55%,var(--magenta));
    -webkit-background-clip:text;background-clip:text;color:transparent}
  .sub{color:var(--dim);font-size:12px;margin-top:2px}
  .pill{font-size:11px;padding:4px 9px;border-radius:999px;
    border:1px solid var(--line);color:var(--dim);
    display:inline-flex;align-items:center;gap:7px;background:rgba(13,18,32,.6)}
  .pill .dot{width:6px;height:6px;border-radius:50%;background:var(--muted)}
  .pill.ok  .dot{background:var(--green);box-shadow:0 0 8px var(--green)}
  .pill.err .dot{background:#f87171}
  form{display:flex;gap:8px;margin-bottom:6px;flex-wrap:wrap}
  input[type=search]{flex:1;min-width:0;background:var(--panel);
    border:1px solid var(--line);color:var(--fg);font:inherit;
    padding:12px 14px;border-radius:var(--radius);outline:none;
    transition:border-color .12s, box-shadow .12s}
  input[type=search]:focus{border-color:var(--indigo);
    box-shadow:0 0 0 3px rgba(129,140,248,.18)}
  .topk{display:flex;align-items:center;gap:8px;background:var(--panel);
    border:1px solid var(--line);border-radius:var(--radius);
    padding:0 12px;color:var(--dim)}
  .topk input{width:54px;background:transparent;border:0;color:var(--fg);
    font:inherit;padding:10px 0;outline:none}
  .hint{color:var(--muted);font-size:11px;margin:6px 0 18px;line-height:1.6}
  .status{min-height:18px;color:var(--dim);font-size:12px;margin-bottom:10px;
    font-variant-numeric:tabular-nums}
  .status.err{color:#fca5a5}
  ol.results{list-style:none;margin:0;padding:0;display:flex;
    flex-direction:column;gap:10px}
  .card{background:var(--panel);border:1px solid var(--line);
    border-radius:var(--radius);padding:14px 16px;position:relative;overflow:hidden}
  .card .row1{display:flex;align-items:baseline;gap:10px;margin-bottom:8px}
  .rank{color:var(--muted);font-size:11px;min-width:22px}
  .score{font-variant-numeric:tabular-nums;color:var(--cyan);
    font-size:12px;letter-spacing:.02em}
  .meta{margin-left:auto;color:var(--muted);font-size:11px;
    text-align:right;line-height:1.4}
  .meta a{color:inherit;text-decoration:underline;
    text-decoration-color:var(--line);text-underline-offset:2px}
  .meta a:hover{text-decoration-color:var(--indigo);color:var(--fg)}
  .text{color:var(--fg);font-size:13px;margin:0 0 10px}
  .bar{height:3px;background:var(--panel2);border-radius:2px;overflow:hidden}
  .bar > i{display:block;height:100%;
    background:linear-gradient(90deg,var(--indigo),var(--cyan));
    border-radius:2px;transition:width .25s ease-out}
  details.about{margin-top:32px;border:1px solid var(--line);
    border-radius:var(--radius);background:var(--panel)}
  details.about summary{cursor:pointer;padding:12px 16px;color:var(--dim);
    font-size:12px;list-style:none;user-select:none}
  details.about summary::-webkit-details-marker{display:none}
  details.about summary::after{content:'▾';float:right;color:var(--muted);
    transition:transform .12s}
  details.about[open] summary::after{transform:rotate(180deg)}
  details.about .body{padding:2px 16px 16px;color:var(--fg);font-size:12px;
    line-height:1.7;border-top:1px solid var(--line)}
  details.about .body p{margin:10px 0}
  details.about a{color:var(--cyan);text-decoration:none;
    border-bottom:1px solid var(--line)}
  details.about a:hover{border-bottom-color:var(--cyan)}
  footer{margin-top:24px;display:flex;flex-wrap:wrap;gap:14px;
    justify-content:space-between;color:var(--muted);font-size:11px}
  footer a{color:var(--dim);text-decoration:none;
    border-bottom:1px dotted var(--line)}
  footer a:hover{color:var(--fg);border-bottom-color:var(--cyan)}
  .empty{color:var(--muted);text-align:center;padding:32px 12px;font-size:12px}
  @keyframes shimmer{0%{background-position:-220px 0}100%{background-position:220px 0}}
  .skeleton{height:54px;border-radius:var(--radius);
    background:linear-gradient(90deg,var(--panel) 0%,var(--panel2) 50%,var(--panel) 100%);
    background-size:440px 100%;animation:shimmer 1.2s linear infinite;
    border:1px solid var(--line)}
  @media (max-width:520px){
    main{padding:20px 14px 48px}
    .topk{order:2}
    .meta{margin-left:0;width:100%;text-align:left;margin-top:4px}
    .card .row1{flex-wrap:wrap}
    header{align-items:flex-start}
  }
  @media (prefers-reduced-motion:reduce){
    .skeleton{animation:none}
    .bar > i{transition:none}
  }
</style>
</head>
<body>
<main>
<header>
  <div>
    <div class="brand">photon-route</div>
    <div class="sub">continuous-variable photonic retrieval · Gaussian-state fidelity</div>
  </div>
  <span id="health" class="pill" aria-live="polite"><span class="dot"></span><span id="health-text">checking…</span></span>
</header>

<form id="f" role="search" aria-label="photon-route query">
  <input id="q" type="search" name="q" placeholder="quantum entanglement…" autocomplete="off" autocapitalize="off" autocorrect="off" spellcheck="false" aria-label="query text" autofocus>
  <label class="topk" title="top_k results">
    <span aria-hidden="true">k</span>
    <input id="k" type="number" min="1" max="20" value="5" inputmode="numeric" aria-label="number of results">
  </label>
</form>
<p class="hint">Each word becomes a squeezing + displacement on a bosonic mode, then a beam-splitter mixes them. Ranking is closed-form Gaussian-state fidelity (Banchi-Braunstein-Pirandola).</p>

<div id="status" class="status" role="status" aria-live="polite"></div>
<ol id="results" class="results" aria-live="polite" aria-busy="false"></ol>

<details class="about">
  <summary>what is this?</summary>
  <div class="body">
    <p><strong>photon-route</strong> is a research artifact exploring whether semantic retrieval can run in the continuous-variable (CV) photonic regime — the regime that real photonic hardware (Xanadu Borealis, fiber-loop reservoirs, coherent Ising machines) actually operates in.</p>
    <p>Each document is encoded as a <em>Gaussian state</em> over N bosonic modes via a <a href="https://strawberryfields.ai/" target="_blank" rel="noopener">Strawberry Fields</a> program: words contribute squeezing and displacement operations, then a beam-splitter network mixes the modes. Query and document fidelity is computed in closed form using the <a href="https://the-walrus.readthedocs.io/" target="_blank" rel="noopener">thewalrus</a> implementation of the Banchi-Braunstein-Pirandola formula.</p>
    <p>Day-1 parameters are SHA-256-bound (deterministic, untrained). Phase 1 will replace this with a small variational parameter set fit on an arXiv quant-ph eval set, then compare against classical bge-m3 and the DV-qubit sister project <a href="https://qrouter.ask-meridian.uk" target="_blank" rel="noopener">qrouter</a>.</p>
    <p>Source · <a href="https://github.com/LuuOW/photon-route" target="_blank" rel="noopener">github.com/LuuOW/photon-route</a></p>
  </div>
</details>

<footer>
  <span>CV photonic · gaussian backend · edge-cached at the Cloudflare boundary</span>
  <span><a href="https://qrouter.ask-meridian.uk" rel="noopener">qrouter (DV)</a> · <a href="/docs">/docs</a> · <a href="/api">json</a></span>
</footer>
</main>

<script>
(function(){
  function $(id){return document.getElementById(id)}
  var q=$('q'), k=$('k'), results=$('results'), status=$('status');
  var healthPill=$('health'), healthText=$('health-text');
  var abort=null, debounceT=0;

  fetch('/health',{cache:'no-store'}).then(function(r){return r.json()}).then(function(j){
    var ok = j && j.ok;
    healthPill.classList.add(ok?'ok':'err');
    healthText.textContent = (j && j.backend) ? j.backend : (ok?'ok':'err');
  }).catch(function(){
    healthPill.classList.add('err');
    healthText.textContent='offline';
  });

  function escapeHtml(s){return String(s).replace(/[&<>"']/g,function(c){
    return ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'})[c];
  })}

  function metaBits(meta){
    if(!meta || typeof meta!=='object') return '';
    var bits=[];
    if(meta.arxiv_id){
      bits.push('<a href="https://arxiv.org/abs/'+escapeHtml(meta.arxiv_id)+'" target="_blank" rel="noopener">arXiv:'+escapeHtml(meta.arxiv_id)+'</a>');
    } else {
      for(var kk in meta){ if(Object.prototype.hasOwnProperty.call(meta,kk)){
        bits.push(escapeHtml(kk)+':'+escapeHtml(String(meta[kk])));
      }}
    }
    return bits.join(' · ');
  }

  function render(items){
    if(!items.length){ results.innerHTML='<li class="empty">no results</li>'; return; }
    var html='';
    for(var i=0;i<items.length;i++){
      var r=items[i];
      var s=+r.score||0;
      var pct=Math.max(0,Math.min(1,s))*100;
      html += '<li class="card">'+
        '<div class="row1">'+
          '<span class="rank">#'+r.rank+'</span>'+
          '<span class="score">'+s.toFixed(4)+'</span>'+
          '<span class="meta">'+metaBits(r.meta)+'</span>'+
        '</div>'+
        '<p class="text">'+escapeHtml(r.text||'')+'</p>'+
        '<div class="bar" role="img" aria-label="fidelity '+pct.toFixed(1)+' percent">'+
          '<i style="width:'+pct.toFixed(2)+'%"></i>'+
        '</div>'+
      '</li>';
    }
    results.innerHTML=html;
  }

  function skeletons(n){
    n=Math.min(Math.max(1,n),5);
    var s='';
    for(var i=0;i<n;i++) s+='<li class="skeleton" aria-hidden="true"></li>';
    results.innerHTML=s;
  }

  async function run(){
    var text=q.value.trim();
    if(!text){
      results.innerHTML='';
      status.textContent='';
      status.classList.remove('err');
      return;
    }
    var topk=Math.max(1,Math.min(20,parseInt(k.value,10)||5));
    if(abort) abort.abort();
    abort=new AbortController();
    status.classList.remove('err');
    status.textContent='ranking…';
    results.setAttribute('aria-busy','true');
    skeletons(topk);
    var t0=performance.now();
    try{
      var url='/rank?q='+encodeURIComponent(text)+'&top_k='+topk;
      var r=await fetch(url,{signal:abort.signal});
      if(!r.ok) throw new Error('http '+r.status);
      var j=await r.json();
      var ms=(performance.now()-t0).toFixed(0);
      var cache=r.headers.get('x-photon-route-cache')||'';
      var n=(j.results||[]).length;
      status.textContent = n+' result'+(n===1?'':'s')+' · '+ms+' ms'+(cache?' · cache '+cache:'')+' · backend '+(j.backend||'?');
      render(j.results||[]);
    }catch(e){
      if(e && e.name==='AbortError') return;
      results.innerHTML='';
      status.classList.add('err');
      status.textContent='request failed: '+(e && e.message ? e.message : e);
    }finally{
      results.setAttribute('aria-busy','false');
    }
  }

  function schedule(){ clearTimeout(debounceT); debounceT=setTimeout(run,280); }
  q.addEventListener('input',schedule);
  k.addEventListener('input',schedule);
  q.addEventListener('keydown',function(e){
    if(e.key==='Enter'){ e.preventDefault(); clearTimeout(debounceT); run(); }
  });
  document.getElementById('f').addEventListener('submit',function(e){
    e.preventDefault(); clearTimeout(debounceT); run();
  });
})();
</script>
</body>
</html>`;

addEventListener('fetch', (e) => e.respondWith(handle(e.request)));

async function handle(req) {
  const url = new URL(req.url);
  const path = url.pathname;

  if (req.method === 'OPTIONS') {
    return cors(new Response(null, { status: 204 }));
  }

  if (path === '/' && (req.method === 'GET' || req.method === 'HEAD')) {
    return new Response(req.method === 'HEAD' ? null : HTML, {
      headers: {
        'content-type': 'text/html; charset=utf-8',
        'content-security-policy': CSP,
        'referrer-policy': 'strict-origin-when-cross-origin',
        'x-content-type-options': 'nosniff',
        'cache-control': 'public, max-age=300',
      },
    });
  }

  if (path === '/api' || path === '/info') {
    return jsonResp(BANNER);
  }

  if (path === '/health') {
    return jsonResp({ ok: true, proxy: 'cloudflare-worker', backend: 'gaussian' });
  }

  if (PROXY.has(path) && (req.method === 'GET' || req.method === 'HEAD')) {
    return await proxied(req, url, path);
  }

  return jsonResp({ error: 'not found' }, 404);
}

function jsonResp(obj, status = 200) {
  return new Response(JSON.stringify(obj, null, 2), {
    status,
    headers: {
      'content-type': 'application/json; charset=utf-8',
      'access-control-allow-origin': '*',
      'cache-control': 'no-store',
    },
  });
}

function cors(resp) {
  resp.headers.set('access-control-allow-origin', '*');
  resp.headers.set('access-control-allow-methods', 'GET, HEAD, OPTIONS');
  resp.headers.set('access-control-allow-headers', 'content-type');
  return resp;
}

async function proxied(req, url, path) {
  const upstream = new URL(HF);
  upstream.pathname = path;
  upstream.search = url.search;

  const isRank = path === '/rank';
  const cacheKey = new Request(upstream.toString() + '#' + CACHE_VERSION, req);
  const cache = caches.default;

  if (isRank) {
    const hit = await cache.match(cacheKey);
    if (hit) {
      const r = new Response(hit.body, hit);
      r.headers.set('x-photon-route-cache', 'hit');
      return cors(r);
    }
  }

  const fetched = await fetch(upstream.toString(), {
    method: req.method,
    headers: { accept: req.headers.get('accept') || '*/*' },
    cf: { cacheTtl: 0 },
  });

  if (!fetched.ok || fetched.status >= 500) {
    const r = new Response(fetched.body, fetched);
    r.headers.set('x-photon-route-cache', 'bypass');
    return cors(r);
  }

  const body = await fetched.arrayBuffer();
  const headers = new Headers(fetched.headers);
  headers.delete('set-cookie');
  if (isRank) headers.set('cache-control', 'public, s-maxage=86400, max-age=300');
  const ok = new Response(body, { status: fetched.status, headers });
  ok.headers.set('x-photon-route-cache', 'miss');
  if (isRank) await cache.put(cacheKey, ok.clone());
  return cors(ok);
}
