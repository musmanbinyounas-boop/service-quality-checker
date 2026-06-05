"""Stage 6/7/10 – FastAPI prediction service with MongoDB monitoring dashboard."""

from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import HTMLResponse, RedirectResponse

from app.schemas import PredictRequest, PredictResponse
from sqc import config
from sqc.drift import run_drift
from sqc.store import build_prediction_record, get_collection, save_prediction

_bundle: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    loaded = joblib.load(config.MODELS_DIR / "model.joblib")
    _bundle.update(loaded)
    yield


app = FastAPI(title="Service Quality Checker", version="0.1.0", lifespan=lifespan)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

def _load_model_metrics() -> dict | None:
    try:
        data = json.loads((config.REPORTS_DIR / "metrics.json").read_text())
        winner = data.get("winner", "RandomForest")
        m = data.get(winner, {})
        return {
            "model_type": winner,
            "precision": round(m.get("precision", 0), 4),
            "recall": round(m.get("recall", 0), 4),
            "f1": round(m.get("f1", 0), 4),
        }
    except Exception:
        return None


def _timeseries(docs: list[dict]) -> list[dict]:
    parsed = []
    for d in docs:
        try:
            ts = datetime.fromisoformat(d["ts"])
            parsed.append((ts, d.get("label", 0)))
        except (KeyError, ValueError, TypeError):
            pass
    if not parsed:
        return []
    span = (max(t for t, _ in parsed) - min(t for t, _ in parsed)).total_seconds()
    by_day = span > 2 * 86400
    buckets: dict[str, dict] = {}
    for ts, label in parsed:
        key = ts.strftime("%Y-%m-%d") if by_day else ts.strftime("%Y-%m-%d %H:00")
        if key not in buckets:
            buckets[key] = {"bucket": key, "count": 0, "pass_count": 0}
        buckets[key]["count"] += 1
        if label == 1:
            buckets[key]["pass_count"] += 1
    return sorted(buckets.values(), key=lambda x: x["bucket"])


# --------------------------------------------------------------------------- #
# Dashboard HTML                                                               #
# --------------------------------------------------------------------------- #

_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Service Quality Checker</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',Arial,sans-serif;background:#f0f2f5;color:#2c3e50;font-size:14px}
.page{max-width:960px;margin:0 auto;padding:20px 16px}
h1{font-size:1.4rem;font-weight:700;color:#1a252f;margin-bottom:3px}
.subtitle{color:#7f8c8d;font-size:0.82rem;margin-bottom:18px}
.card{background:#fff;border-radius:10px;padding:20px 22px;margin-bottom:16px;box-shadow:0 1px 5px rgba(0,0,0,0.07)}
.card h2{font-size:0.95rem;font-weight:700;color:#34495e;margin-bottom:14px;border-bottom:1px solid #ecf0f1;padding-bottom:7px;text-transform:uppercase;letter-spacing:0.04em}
.inputs-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(155px,1fr));gap:10px;margin-bottom:12px}
label{display:block;font-size:0.75rem;font-weight:600;color:#555;margin-bottom:3px}
.opt{font-weight:400;color:#95a5a6;font-size:0.7rem}
input[type=number]{width:100%;padding:6px 9px;border:1px solid #dde1e4;border-radius:6px;font-size:0.88rem;outline:none;transition:border-color 0.15s;color:#2c3e50;background:#fafbfc}
input[type=number]:focus{border-color:#3498db;background:#fff}
.btn{background:#3498db;color:#fff;border:none;padding:8px 20px;border-radius:6px;font-size:0.88rem;font-weight:600;cursor:pointer;transition:background 0.15s}
.btn:hover{background:#2980b9}
.btn:disabled{background:#bdc3c7;cursor:not-allowed}
.btn-sm{padding:5px 12px;font-size:0.78rem}
.btn-ghost{background:#ecf0f1;color:#555}
.btn-ghost:hover{background:#dde1e4}
#result-area{margin-top:12px;min-height:36px}
.rbadge{display:inline-flex;align-items:center;gap:10px;padding:9px 16px;border-radius:8px;margin-bottom:6px}
.rpass{background:#eafaf1;border:1.5px solid #2ecc71}
.rfail{background:#fdf0ef;border:1.5px solid #e74c3c}
.rverdict{font-size:1.3rem;font-weight:700}
.rpass .rverdict{color:#27ae60}
.rfail .rverdict{color:#e74c3c}
.rprob{font-size:0.88rem;color:#444}
.rdet{font-size:0.75rem;color:#999;margin-top:2px}
.hint{font-size:0.75rem;color:#aab;font-style:italic;margin-top:5px}
.errmsg{color:#e74c3c;font-size:0.83rem;margin-top:7px}
.mhdr{display:flex;align-items:center;justify-content:space-between;margin-bottom:7px;border-bottom:1px solid #ecf0f1;padding-bottom:7px}
.mhdr h2{margin-bottom:0;border-bottom:none;padding-bottom:0}
#last-upd{font-size:0.72rem;color:#aaa}
.stats-row{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:9px;margin-bottom:14px}
.sbox{background:#f8fafc;border:1px solid #e4e8ec;border-radius:8px;padding:11px 13px}
.slbl{font-size:0.7rem;font-weight:700;color:#7f8c8d;text-transform:uppercase;letter-spacing:0.04em;margin-bottom:3px}
.sval{font-size:1.25rem;font-weight:700;color:#2c3e50}
.ssub{font-size:0.7rem;color:#aaa;margin-top:1px}
.ctitle{font-size:0.75rem;font-weight:700;color:#7f8c8d;text-transform:uppercase;letter-spacing:0.04em;margin-bottom:6px}
.cwrap{margin-bottom:14px}
.drift-box{background:#f8fafc;border:1px solid #e4e8ec;border-radius:8px;padding:11px 13px;margin-bottom:14px}
.drift-insuf{background:#fff8e1;border:1px solid #f0c040;border-radius:6px;padding:7px 11px;color:#7a5f00;font-size:0.8rem}
.drift-alert{background:#fdf0ef;border:1px solid #e74c3c;border-radius:6px;padding:7px 11px;color:#922b21;font-size:0.8rem;margin-bottom:7px}
.dtbl{width:100%;border-collapse:collapse;font-size:0.8rem}
.dtbl th{text-align:left;color:#999;font-weight:600;padding:2px 6px;font-size:0.72rem}
.dtbl td{padding:2px 6px}
.slow{color:#27ae60}.smed{color:#f39c12}.shi{color:#e74c3c}
.rtbl{width:100%;border-collapse:collapse;font-size:0.78rem}
.rtbl th{text-align:left;background:#f8fafc;padding:5px 8px;color:#888;font-size:0.72rem;font-weight:700;border-bottom:2px solid #e4e8ec}
.rtbl td{padding:4px 8px;border-bottom:1px solid #f0f2f5;vertical-align:middle}
.rtbl tr:last-child td{border-bottom:none}
.lpass{background:#eafaf1;color:#27ae60;padding:1px 6px;border-radius:3px;font-size:0.72rem;font-weight:700}
.lfail{background:#fdf0ef;color:#e74c3c;padding:1px 6px;border-radius:3px;font-size:0.72rem;font-weight:700}
.empty{text-align:center;padding:28px 16px;color:#c0c6cb}
.empty-icon{font-size:2rem;margin-bottom:7px}
.empty p{font-size:0.88rem}
.triage-note{background:#f4f6f7;border-left:3px solid #aab4bb;border-radius:0 6px 6px 0;padding:8px 12px;font-size:0.78rem;color:#5d6d7e;margin-bottom:16px;line-height:1.5}
.triage-note strong{color:#4a5a65}
.form-help{font-size:0.78rem;color:#95a5a6;margin-bottom:10px;line-height:1.4}
.section-caption{font-size:0.78rem;color:#95a5a6;margin-bottom:12px;line-height:1.4}
.drift-caption{font-size:0.72rem;color:#aaa;margin-bottom:8px;line-height:1.4}
.page-footer{text-align:center;padding:18px 0 10px;font-size:0.75rem;color:#bbb;line-height:1.9;border-top:1px solid #e8ecf0;margin-top:4px}
.page-footer a{color:#95a5a6;text-decoration:none}
.page-footer a:hover{color:#7f8c8d;text-decoration:underline}
</style>
</head>
<body>
<div class="page">
  <h1>Service Quality Checker</h1>
  <p class="subtitle">Predicts whether a 5G/LTE measurement point likely delivers &ge;&nbsp;5&nbsp;Mbps downlink throughput &mdash; predict a location and monitor live stats on one screen.</p>

  <div class="triage-note">
    <strong>Screening tool, not a verdict.</strong>
    This model flags measurement points for human review.
    A &ldquo;Fail&rdquo; cannot distinguish genuine poor coverage from temporary congestion
    &mdash; the training data contains no cell-load or backhaul information.
  </div>

  <div class="card">
    <h2>Predict</h2>
    <p class="form-help">Enter radio measurements for a single location. RSRP, RSRQ, and Speed are required; SNR and CQI are optional and will be imputed from the training median if left blank.</p>
    <div class="inputs-grid">
      <div><label>RSRP&nbsp;(dBm)</label>
        <input type="number" id="f-RSRP" value="-75" min="-140" max="-40" step="1"></div>
      <div><label>RSRQ&nbsp;(dB)</label>
        <input type="number" id="f-RSRQ" value="-8" min="-30" max="10" step="1"></div>
      <div><label>SNR&nbsp;(dB) <span class="opt">(optional)</span></label>
        <input type="number" id="f-SNR" value="20" min="-30" max="40" step="1"></div>
      <div><label>CQI <span class="opt">(optional)</span></label>
        <input type="number" id="f-CQI" value="14" min="0" max="15" step="1"></div>
      <div><label>Speed&nbsp;(km/h)</label>
        <input type="number" id="f-Speed" value="0" min="0" max="130" step="1"></div>
    </div>
    <button class="btn" id="pred-btn" onclick="doPredict()">Predict &#9654;</button>
    <div id="result-area"></div>
  </div>

  <div class="card">
    <div class="mhdr">
      <h2>Monitor</h2>
      <div>
        <span id="last-upd"></span>
        <button class="btn btn-ghost btn-sm" onclick="refreshStats()" style="margin-left:8px">
          &#8635; Refresh
        </button>
      </div>
    </div>
    <p class="section-caption">Live predictions logged to the database &mdash; counts, pass rate, latency, and PSI drift (how much recent inputs differ from the training distribution).</p>
    <div id="monitor-body">
      <div class="empty"><div class="empty-icon">&#128202;</div><p>Loading&hellip;</p></div>
    </div>
  </div>

  <footer class="page-footer">
    <p>RandomForest trained on the <strong>UCC 5G dataset</strong> (single anonymised Irish operator, GPL-3.0).
    &nbsp;<strong>Screening tool &mdash; not a verdict.</strong></p>
    <p><a href="/docs">API&nbsp;docs</a> &middot; <a href="/health">Health&nbsp;check</a></p>
  </footer>
</div>

<script>
function doPredict(){
  var btn=document.getElementById('pred-btn');
  var area=document.getElementById('result-area');
  btn.disabled=true; btn.textContent='Predicting…'; area.innerHTML='';
  var body={};
  ['RSRP','RSRQ','SNR','CQI','Speed'].forEach(function(f){
    var v=document.getElementById('f-'+f).value.trim();
    if(v!=='') body[f]=parseFloat(v);
  });
  fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})
  .then(function(r){
    if(!r.ok) return r.json().then(function(e){throw new Error(e.detail||r.statusText);});
    return r.json();
  })
  .then(function(d){
    var isPass=d.label===1;
    var pct=(d.pass_probability*100).toFixed(1);
    area.innerHTML=
      '<div class="rbadge '+(isPass?'rpass':'rfail')+'">'
      +'<span class="rverdict">'+(isPass?'✓ PASS':'✗ FAIL')+'</span>'
      +'<div><div class="rprob">Pass probability: <strong>'+pct+'%</strong></div>'
      +'<div class="rdet">Threshold '+d.threshold_kbps+' kbps · '+d.model_type+' · '+d.latency_ms+' ms</div>'
      +'</div></div>'
      +'<div class="hint" id="upd-hint">Updating monitor…</div>';
    setTimeout(function(){
      refreshStats();
      var h=document.getElementById('upd-hint');
      if(h) h.textContent='';
    },1500);
  })
  .catch(function(err){
    area.innerHTML='<p class="errmsg">Error: '+esc(err.message)+'</p>';
  })
  .finally(function(){
    btn.disabled=false; btn.innerHTML='Predict &#9654;';
  });
}

function refreshStats(){
  fetch('/dashboard/data')
  .then(function(r){return r.json();})
  .then(function(d){renderMonitor(d);})
  .catch(function(err){
    document.getElementById('monitor-body').innerHTML=
      '<p class="errmsg">Failed to load monitor: '+esc(err.message)+'</p>';
  });
}

function renderMonitor(d){
  var el=document.getElementById('monitor-body');
  var upd=document.getElementById('last-upd');
  upd.textContent='Updated '+new Date().toLocaleTimeString();

  if(!d.available||d.total_predictions===0){
    el.innerHTML='<div class="empty"><div class="empty-icon">&#128202;</div>'
      +'<p>No monitoring data yet &mdash; make a prediction above.</p>'
      +(d.reason?'<p style="font-size:0.72rem;margin-top:4px;color:#ccc">'+esc(d.reason)+'</p>':'')
      +'</div>';
    return;
  }

  var passRate=d.pass_rate!=null?(d.pass_rate*100).toFixed(1)+'%':'--';
  var avgLat=d.avg_latency_ms!=null?d.avg_latency_ms+' ms':'--';
  var p95Lat=d.p95_latency_ms!=null?d.p95_latency_ms+' ms':'--';
  var f1=d.model_metrics?d.model_metrics.f1.toFixed(3):'--';
  var mtype=d.model_metrics?d.model_metrics.model_type:'';

  var html='<div class="stats-row">'
    +sbox('Total',d.total_predictions,d.pass_count+' pass / '+d.fail_count+' fail')
    +sbox('Pass Rate',passRate,'')
    +sbox('Avg Latency',avgLat,'inference')
    +sbox('P95 Latency',p95Lat,'inference')
    +sbox('Model F1',f1,mtype)
    +'</div>';

  html+='<div class="cwrap"><div class="ctitle">Predictions over time</div>'
      +'<div id="ts-chart"></div></div>';
  html+='<div class="cwrap"><div class="ctitle">Pass-probability distribution</div>'
      +'<div id="hist-chart"></div></div>';
  html+=renderDrift(d.drift);

  html+='<div class="ctitle" style="margin-bottom:7px">Recent predictions (latest 20, newest first)</div>';
  if(!d.recent||d.recent.length===0){
    html+='<p style="color:#ccc;font-size:0.83rem">None yet.</p>';
  } else {
    html+='<div style="overflow-x:auto"><table class="rtbl">'
      +'<thead><tr><th>Time (UTC)</th><th>Label</th><th>Pass&nbsp;prob</th>'
      +'<th>Latency</th><th>RSRP</th><th>Speed</th></tr></thead><tbody>';
    d.recent.forEach(function(r){
      var ts=r.ts?r.ts.replace('T',' ').substring(0,19):'--';
      var lbl=r.label===1?'<span class="lpass">PASS</span>':'<span class="lfail">FAIL</span>';
      var prob=r.pass_probability!=null?(r.pass_probability*100).toFixed(1)+'%':'--';
      var lat=r.latency_ms!=null?r.latency_ms+' ms':'--';
      var rsrp=r.features&&r.features.RSRP!=null?r.features.RSRP:'--';
      var spd=r.features&&r.features.Speed!=null?r.features.Speed:'--';
      html+='<tr><td>'+esc(ts)+'</td><td>'+lbl+'</td><td>'+prob+'</td>'
           +'<td>'+lat+'</td><td>'+rsrp+'</td><td>'+spd+'</td></tr>';
    });
    html+='</tbody></table></div>';
  }

  el.innerHTML=html;
  buildTimeseries(d.timeseries);
  buildHistogram(d.prob_histogram);
}

function sbox(label,value,sub){
  return '<div class="sbox"><div class="slbl">'+label+'</div>'
    +'<div class="sval">'+value+'</div>'
    +(sub?'<div class="ssub">'+sub+'</div>':'')+'</div>';
}

function renderDrift(drift){
  var html='<div class="drift-box"><div class="ctitle" style="margin-bottom:4px">PSI Feature Drift</div>'
    +'<p class="drift-caption">Population Stability Index &mdash; how much recent prediction inputs differ from the training distribution. High PSI (&ge;&nbsp;0.2) suggests the model may need revalidation. Requires &ge;&nbsp;100 predictions to be reliable.</p>';
  if(!drift){
    html+='<p style="color:#ccc;font-size:0.8rem">Drift data unavailable.</p>';
  } else if(drift.status==='insufficient_data'){
    html+='<div class="drift-insuf">&#9888; Drift not computed: only '+(drift.n||0)
      +' predictions logged (need &ge; '+(drift.min_sample||100)+' for a reliable signal).</div>';
  } else {
    if(drift.alert){
      html+='<div class="drift-alert">&#9888; Drift alert &mdash; max PSI = '
        +drift.max_psi.toFixed(4)+'</div>';
    }
    html+='<table class="dtbl"><thead><tr><th>Feature</th><th>PSI</th>'
        +'<th>Severity</th></tr></thead><tbody>';
    for(var feat in drift.features){
      var v=drift.features[feat];
      var sc=v.severity==='low'?'slow':v.severity==='medium'?'smed':'shi';
      html+='<tr><td>'+feat+'</td><td>'
        +(v.psi!=null?v.psi.toFixed(4):'--')+'</td>'
        +'<td class="'+sc+'">'+(v.severity||'--')+'</td></tr>';
    }
    html+='</tbody></table>';
  }
  return html+'</div>';
}

function buildTimeseries(data){
  var el=document.getElementById('ts-chart');
  if(!el) return;
  if(!data||!data.length){
    el.innerHTML='<p style="color:#ccc;font-size:0.8rem;padding:6px 0">No time-series data yet.</p>';
    return;
  }
  var W=620,H=110,PL=28,PB=20,PT=8;
  var cW=W-PL-8, cH=H-PB-PT;
  var maxC=Math.max.apply(null,data.map(function(d){return d.count;}));
  if(maxC<1) maxC=1;
  var n=data.length;
  var step=Math.floor(cW/n);
  var bw=Math.max(3,step-3);
  var svg='<svg viewBox="0 0 '+W+' '+H+'" style="width:100%;max-height:'+H+'px;display:block">';
  svg+='<line x1="'+PL+'" y1="'+PT+'" x2="'+PL+'" y2="'+(H-PB)+'" stroke="#e8ecf0" stroke-width="1"/>';
  svg+='<line x1="'+PL+'" y1="'+(H-PB)+'" x2="'+(W-8)+'" y2="'+(H-PB)+'" stroke="#e8ecf0" stroke-width="1"/>';
  data.forEach(function(d,i){
    var x=PL+i*step+Math.floor((step-bw)/2);
    var totalH=Math.round((d.count/maxC)*cH);
    var passH=d.count>0?Math.round((d.pass_count/d.count)*totalH):0;
    var failH=totalH-passH;
    var y0=H-PB;
    if(failH>0) svg+='<rect x="'+x+'" y="'+(y0-totalH)+'" width="'+bw+'" height="'+failH+'" fill="#e74c3c" fill-opacity="0.7"/>';
    if(passH>0) svg+='<rect x="'+x+'" y="'+(y0-passH)+'" width="'+bw+'" height="'+passH+'" fill="#2ecc71" fill-opacity="0.7"/>';
    if(n<=14||i%Math.ceil(n/8)===0){
      var lbl=d.bucket.slice(-5);
      svg+='<text x="'+(x+bw/2)+'" y="'+(H-4)+'" text-anchor="middle" font-size="8" fill="#aaa">'+esc(lbl)+'</text>';
    }
  });
  svg+='<rect x="'+(W-68)+'" y="'+PT+'" width="9" height="9" fill="#e74c3c" fill-opacity="0.7"/>';
  svg+='<text x="'+(W-56)+'" y="'+(PT+8)+'" font-size="9" fill="#888">Fail</text>';
  svg+='<rect x="'+(W-68)+'" y="'+(PT+13)+'" width="9" height="9" fill="#2ecc71" fill-opacity="0.7"/>';
  svg+='<text x="'+(W-56)+'" y="'+(PT+21)+'" font-size="9" fill="#888">Pass</text>';
  svg+='</svg>';
  el.innerHTML=svg;
}

function buildHistogram(data){
  var el=document.getElementById('hist-chart');
  if(!el) return;
  if(!data||!data.length){
    el.innerHTML='<p style="color:#ccc;font-size:0.8rem;padding:6px 0">No data yet.</p>';
    return;
  }
  var W=620,H=110,PL=28,PB=20,PT=8;
  var cW=W-PL-8, cH=H-PB-PT;
  var maxC=Math.max.apply(null,data.map(function(d){return d.count;}));
  if(maxC<1) maxC=1;
  var n=data.length;
  var bw=Math.floor(cW/n)-2;
  var svg='<svg viewBox="0 0 '+W+' '+H+'" style="width:100%;max-height:'+H+'px;display:block">';
  svg+='<line x1="'+PL+'" y1="'+PT+'" x2="'+PL+'" y2="'+(H-PB)+'" stroke="#e8ecf0" stroke-width="1"/>';
  svg+='<line x1="'+PL+'" y1="'+(H-PB)+'" x2="'+(W-8)+'" y2="'+(H-PB)+'" stroke="#e8ecf0" stroke-width="1"/>';
  data.forEach(function(d,i){
    var x=PL+i*(bw+2);
    var barH=Math.round((d.count/maxC)*cH);
    var t=i/(n-1||1);
    var r=Math.round(231*(1-t));
    var g=Math.round(46+174*t);
    var fill='rgb('+r+','+g+',46)';
    if(barH>0) svg+='<rect x="'+x+'" y="'+(H-PB-barH)+'" width="'+bw+'" height="'+barH+'" fill="'+fill+'" fill-opacity="0.75"/>';
    if(d.count>0) svg+='<text x="'+(x+bw/2)+'" y="'+(H-PB-barH-2)+'" text-anchor="middle" font-size="9" fill="#555">'+d.count+'</text>';
    svg+='<text x="'+(x+bw/2)+'" y="'+(H-4)+'" text-anchor="middle" font-size="8" fill="#aaa">'+(i/10).toFixed(1)+'</text>';
  });
  svg+='</svg>';
  el.innerHTML=svg;
}

function esc(s){
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

refreshStats();
</script>
</body>
</html>"""


# --------------------------------------------------------------------------- #
# Routes                                                                      #
# --------------------------------------------------------------------------- #

@app.get("/")
def root():
    return RedirectResponse(url="/dashboard")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_type": _bundle["model_type"],
        "features": _bundle["features"],
        "trained_at": _bundle["trained_at"],
        "sklearn_version": _bundle["sklearn_version"],
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, background_tasks: BackgroundTasks) -> PredictResponse:
    features: list[str] = _bundle["features"]
    data = req.model_dump()
    row = {f: (np.nan if data.get(f) is None else data[f]) for f in features}
    X = pd.DataFrame([row], columns=features)

    t0 = time.perf_counter()
    label = int(_bundle["model"].predict(X)[0])
    proba = float(_bundle["model"].predict_proba(X)[0][1])
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    print(
        f'[predict] {{"label": {label}, "pass_probability": {proba:.4f}, '
        f'"model_type": "{_bundle["model_type"]}", "latency_ms": {latency_ms}}}'
    )

    features_for_record = {
        f: (None if pd.isna(v) else float(v)) for f, v in row.items()
    }
    record = build_prediction_record(
        features=features_for_record,
        label=label,
        pass_probability=proba,
        model_type=_bundle["model_type"],
        latency_ms=latency_ms,
    )
    background_tasks.add_task(save_prediction, record)

    return PredictResponse(
        prediction="Pass" if label == 1 else "Fail",
        label=label,
        pass_probability=round(proba, 4),
        threshold_kbps=int(_bundle["threshold_kbps"]),
        model_type=_bundle["model_type"],
        latency_ms=latency_ms,
    )


@app.get("/dashboard/data")
def dashboard_data():
    try:
        col = get_collection()
    except Exception as exc:
        return {"available": False, "reason": str(exc)}

    if col is None:
        return {"available": False, "reason": "MongoDB not configured (MONGODB_URI unset)"}

    try:
        docs = list(col.find(
            {},
            {"ts": 1, "label": 1, "pass_probability": 1, "latency_ms": 1, "features": 1, "_id": 0},
        ).sort("_id", -1).limit(1000))
    except Exception as exc:
        return {"available": False, "reason": f"Query failed: {exc}"}

    empty_hist = [{"bin": f"{i/10:.1f}–{(i+1)/10:.1f}", "count": 0} for i in range(10)]
    model_metrics = _load_model_metrics()

    if not docs:
        return {
            "available": True,
            "total_predictions": 0,
            "pass_count": 0,
            "fail_count": 0,
            "pass_rate": None,
            "avg_latency_ms": None,
            "p95_latency_ms": None,
            "recent": [],
            "timeseries": [],
            "prob_histogram": empty_hist,
            "model_metrics": model_metrics,
            "drift": {"status": "insufficient_data", "n": 0, "min_sample": 100},
        }

    total = len(docs)
    pass_count = sum(1 for d in docs if d.get("label") == 1)
    fail_count = total - pass_count
    pass_rate = round(pass_count / total, 4)

    lats = sorted(d["latency_ms"] for d in docs if d.get("latency_ms") is not None)
    avg_lat = round(sum(lats) / len(lats), 2) if lats else None
    p95_idx = min(int(len(lats) * 0.95), len(lats) - 1)
    p95_lat = round(lats[p95_idx], 2) if lats else None

    recent = [
        {
            "ts": d.get("ts", ""),
            "label": d.get("label"),
            "pass_probability": d.get("pass_probability"),
            "latency_ms": d.get("latency_ms"),
            "features": d.get("features", {}),
        }
        for d in docs[:20]
    ]

    hist = [0] * 10
    for d in docs:
        p = d.get("pass_probability")
        if p is not None:
            hist[min(int(p * 10), 9)] += 1
    prob_histogram = [
        {"bin": f"{i/10:.1f}–{(i+1)/10:.1f}", "count": hist[i]}
        for i in range(10)
    ]

    feat_rows = [d.get("features", {}) for d in docs]
    feat_df = pd.DataFrame(feat_rows, columns=list(config.FEATURES))
    drift = run_drift(feat_df)

    return {
        "available": True,
        "total_predictions": total,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "pass_rate": pass_rate,
        "avg_latency_ms": avg_lat,
        "p95_latency_ms": p95_lat,
        "recent": recent,
        "timeseries": _timeseries(docs),
        "prob_histogram": prob_histogram,
        "model_metrics": model_metrics,
        "drift": drift,
    }


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    return _DASHBOARD_HTML
