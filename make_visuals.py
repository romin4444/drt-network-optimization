"""
Generate human-friendly visuals from the pipeline outputs.
==========================================================
Produces PNG charts (for the README) and a self-contained interactive map
(map.html) so non-technical readers can actually *see* the analysis instead of
reading CSVs. Everything is rendered from files already written by the pipeline.

Run (after route_design.py / route_optimizer.py):  python make_visuals.py
Outputs: assets/*.png  and  map.html
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import pandas as pd

import drt_config as cfg

ASSETS = cfg.ROOT / "assets"
ASSETS.mkdir(exist_ok=True)

# Colour per diagnostic bucket (also used by the map).
BUCKET_COLOR = {"A": "#1a9850", "B": "#4575b4", "C": "#fdae61", "D": "#d73027"}
BUCKET_LABEL = {
    "A": "A · Frequent backbone",
    "B": "B · Stable / promote",
    "C": "C · Coverage commuter",
    "D": "D · Marginal / on-demand",
}


def chart_buckets(sc: pd.DataFrame):
    counts = sc["bucket"].value_counts().reindex(["A", "B", "C", "D"]).fillna(0)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([BUCKET_LABEL[b] for b in counts.index],
           counts.values, color=[BUCKET_COLOR[b] for b in counts.index])
    ax.set_title("Route health: diagnostic buckets")
    ax.set_ylabel("number of routes")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 0.2, str(int(v)), ha="center", fontweight="bold")
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(ASSETS / "buckets.png", dpi=110)
    plt.close(fig)


def chart_fleet(opt: pd.DataFrame):
    top = opt[opt["net_new_buses_needed"] > 0].sort_values(
        "net_new_buses_needed", ascending=True).tail(10)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(top["route_id"].astype(str), top["net_new_buses_needed"], color="#4575b4")
    ax.set_title("Buses to add per corridor (frequency upgrades)")
    ax.set_xlabel("net new buses")
    for y, (v, cap) in enumerate(zip(top["net_new_buses_needed"], top["capital_cost_cad"])):
        ax.text(v + 0.05, y, f"${cap/1e6:.1f}M", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(ASSETS / "fleet.png", dpi=110)
    plt.close(fig)


def chart_speed(opt: pd.DataFrame):
    g = opt[opt["round_trip_time_saved_min"] > 0].sort_values(
        "round_trip_time_saved_min", ascending=False).head(8)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = range(len(g))
    ax.bar([i - 0.2 for i in x], g["current_speed_kmh"], width=0.4,
           label="current", color="#bbbbbb")
    ax.bar([i + 0.2 for i in x], g["optimized_speed_kmh"], width=0.4,
           label="after stop consolidation", color="#1a9850")
    ax.set_xticks(list(x))
    ax.set_xticklabels(g["route_id"].astype(str))
    ax.set_title("Speed gain from stop consolidation (no new buses)")
    ax.set_ylabel("km/h")
    ax.set_xlabel("route")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ASSETS / "speed.png", dpi=110)
    plt.close(fig)


def chart_network_map(bundle: dict):
    """Static route map coloured by bucket — quick preview for the README."""
    fig, ax = plt.subplots(figsize=(7, 7))
    for feat in bundle.get("geometries", []):
        coords = feat.get("coords", [])
        if len(coords) < 2:
            continue
        lats = [c[0] for c in coords]
        lons = [c[1] for c in coords]
        ax.plot(lons, lats, color=BUCKET_COLOR.get(feat.get("bucket"), "#888"),
                linewidth=1.1, alpha=0.8)
    handles = [plt.Line2D([0], [0], color=c, lw=2) for c in BUCKET_COLOR.values()]
    ax.legend(handles, [BUCKET_LABEL[b] for b in BUCKET_COLOR], fontsize=8, loc="upper left")
    ax.set_title("Durham Region Transit network by route health")
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(ASSETS / "network_map.png", dpi=110)
    plt.close(fig)


def interactive_map(bundle: dict):
    """Self-contained Leaflet map (geometries embedded — opens with no server)."""
    feats = [{"route_id": f["route_id"], "name": f.get("name", ""),
              "bucket": f.get("bucket"), "diagnosis": f.get("diagnosis", ""),
              "headway": f.get("headway"), "speed": f.get("speed"),
              "coords": f.get("coords", [])}
             for f in bundle.get("geometries", []) if len(f.get("coords", [])) >= 2]
    data_js = json.dumps(feats)
    colors_js = json.dumps(BUCKET_COLOR)
    labels_js = json.dumps(BUCKET_LABEL)
    html = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>DRT Network Map</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  html,body{margin:0;height:100%} #map{height:100%}
  .legend{background:#fff;padding:8px 10px;border-radius:6px;line-height:1.5;
          font:13px system-ui;box-shadow:0 1px 4px rgba(0,0,0,.3)}
  .legend i{display:inline-block;width:12px;height:12px;margin-right:6px;border-radius:2px}
  .title{position:absolute;z-index:1000;top:10px;left:50px;background:#fff;
         padding:6px 12px;border-radius:6px;font:600 15px system-ui;
         box-shadow:0 1px 4px rgba(0,0,0,.3)}
</style></head><body>
<div class="title">Durham Region Transit — routes by health bucket</div>
<div id="map"></div>
<script>
const FEATS=__DATA__, COLORS=__COLORS__, LABELS=__LABELS__;
const map=L.map('map');
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
  {maxZoom:18, attribution:'&copy; OpenStreetMap'}).addTo(map);
let all=[];
FEATS.forEach(f=>{
  const latlngs=f.coords.map(c=>[c[0],c[1]]);
  const color=COLORS[f.bucket]||'#888';
  const line=L.polyline(latlngs,{color:color,weight:3,opacity:.8}).addTo(map);
  line.bindPopup(`<b>Route ${f.route_id}</b> — ${f.name}<br>`+
    `Bucket ${f.bucket}<br>Headway: ${f.headway||'?'} min · Speed: ${f.speed||'?'} km/h<br>`+
    `<i>${f.diagnosis}</i>`);
  all=all.concat(latlngs);
});
if(all.length) map.fitBounds(all);
const legend=L.control({position:'bottomright'});
legend.onAdd=function(){const d=L.DomUtil.create('div','legend');
  d.innerHTML='<b>Route health</b><br>'+Object.keys(COLORS).map(b=>
    `<i style="background:${COLORS[b]}"></i>${LABELS[b]}`).join('<br>');
  return d;};
legend.addTo(map);
</script></body></html>"""
    html = (html.replace("__DATA__", data_js)
                .replace("__COLORS__", colors_js)
                .replace("__LABELS__", labels_js))
    (cfg.ROOT / "map.html").write_text(html, encoding="utf-8")


def main():
    sc = pd.read_csv(cfg.MAP_DATA / "route_scorecard.csv")
    opt = pd.read_csv(cfg.MAP_DATA / "route_optimization_scorecard.csv")
    bundle = json.loads((cfg.MAP_DATA / "route_bundle.json").read_text())

    chart_buckets(sc)
    chart_fleet(opt)
    chart_speed(opt)
    chart_network_map(bundle)
    interactive_map(bundle)

    print("Wrote:")
    for p in ["assets/buckets.png", "assets/fleet.png", "assets/speed.png",
              "assets/network_map.png", "map.html"]:
        print(f"  {p}")


if __name__ == "__main__":
    main()
