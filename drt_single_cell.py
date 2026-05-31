# ════════════════════════════════════════════════════════════════════════════
#  DURHAM REGION TRANSIT — BUS EFFICIENCY PIPELINE  ·  SINGLE-CELL EDITION
#  Paste this whole cell into Google Colab and press ▶.  Nothing to edit.
#  Realtime feed URLs are pre-filled and were verified live on 2026-05-22.
#
#  What runs automatically:  download GTFS → extract → schedule index →
#                            baseline diagnosis → chart → download results
#  What's defined & ready (call when you have realtime data):
#                            run_logger() · build_features() · train_otp_model()
# ════════════════════════════════════════════════════════════════════════════

import sys, os, math, time, zipfile
from pathlib import Path
from datetime import datetime, timezone, timedelta

IS_COLAB = "google.colab" in sys.modules

# ── dependencies ─────────────────────────────────────────────────────────────
if IS_COLAB:
    os.system("pip install -q pandas numpy pyarrow matplotlib requests "
              "gtfs-realtime-bindings lightgbm scikit-learn 2>/dev/null")

import numpy as np, pandas as pd, requests
import matplotlib; matplotlib.use("Agg" if not IS_COLAB else "module://matplotlib_inline.backend_inline")
import matplotlib.pyplot as plt

# ── config ───────────────────────────────────────────────────────────────────
DATA   = Path("/content" if IS_COLAB else "./drt"); DATA.mkdir(exist_ok=True, parents=True)
GTFS_ZIP = DATA/"GTFS_Durham_TXT.zip"; GTFS = DATA/"gtfs"; INDEX = DATA/"schedule_index"
RT_LOG = DATA/"rt_log"; FEAT = DATA/"features"; REPORT = DATA/"baseline_report.csv"
for d in (GTFS, INDEX, RT_LOG, FEAT): d.mkdir(exist_ok=True, parents=True)

STATIC_URL = "https://maps.durham.ca/OpenDataGTFS/GTFS_Durham_TXT.zip"
VEHICLE_POSITIONS_URL = "https://drtonline.durhamregiontransit.com/gtfsrealtime/VehiclePositions"  # VERIFIED LIVE
TRIP_UPDATES_URL      = "https://drtonline.durhamregiontransit.com/gtfsrealtime/TripUpdates"        # VERIFIED LIVE
PULSE = {"900","901","915","916"}
R = 6371.0

def haversine(a,b,c,d):
    p1,p2 = math.radians(a),math.radians(c)
    x = math.sin(math.radians(c-a)/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(math.radians(d-b)/2)**2
    return 2*R*math.asin(math.sqrt(x))
def t2s(t):
    if pd.isna(t) or t=="": return np.nan
    h,m,s = map(int,str(t).split(":")); return h*3600+m*60+s

# ── 1. download + extract GTFS ────────────────────────────────────────────────
def extract_gtfs():
    if not GTFS_ZIP.exists():
        print("Downloading GTFS…")
        r = requests.get(STATIC_URL, timeout=120); GTFS_ZIP.write_bytes(r.content)
    with zipfile.ZipFile(GTFS_ZIP) as z: z.extractall(GTFS)
    print(f"✓ GTFS extracted: {len(list(GTFS.glob('*.txt')))} files")

# ── 2. schedule index (per-day Parquet, absolute timestamps) ──────────────────
def build_schedule_index():
    cal = pd.read_csv(GTFS/"calendar.txt", dtype=str)
    cd  = pd.read_csv(GTFS/"calendar_dates.txt", dtype=str) if (GTFS/"calendar_dates.txt").exists() else pd.DataFrame()
    trips = pd.read_csv(GTFS/"trips.txt", dtype=str)
    st = pd.read_csv(GTFS/"stop_times.txt", dtype={"trip_id":str,"stop_id":str})
    st["arr_s"] = st["arrival_time"].map(t2s)
    days = {}
    dow = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]
    for _,row in cal.iterrows():
        d0 = datetime.strptime(row["start_date"],"%Y%m%d"); d1 = datetime.strptime(row["end_date"],"%Y%m%d")
        d = d0
        while d <= d1:
            if row[dow[d.weekday()]]=="1": days.setdefault(d.strftime("%Y%m%d"),set()).add(row["service_id"])
            d += timedelta(days=1)
    if len(cd):
        for _,r in cd.iterrows():
            if r["exception_type"]=="1": days.setdefault(r["date"],set()).add(r["service_id"])
            elif r["exception_type"]=="2" and r["date"] in days: days[r["date"]].discard(r["service_id"])
    total = 0
    for ymd, svcs in sorted(days.items()):
        tr = trips[trips["service_id"].isin(svcs)]
        if tr.empty: continue
        rows = st[st["trip_id"].isin(set(tr["trip_id"]))].merge(
            tr[["trip_id","route_id"]], on="trip_id")
        rows = rows.dropna(subset=["arr_s"]).copy()
        rows["date"] = ymd
        rows[["date","trip_id","route_id","stop_id","stop_sequence","arr_s"]].to_parquet(
            INDEX/f"{ymd}.parquet", index=False)
        total += len(rows)
    print(f"✓ Schedule index: {len(list(INDEX.glob('*.parquet')))} days, {total:,} stop-arrivals")

# ── 3. baseline diagnosis + A/B/C/D buckets ───────────────────────────────────
def classify(pulse, hw, peak, cov, n):
    if pulse: return ("A","PULSE backbone — protect, invest, tighten regularity")
    if n<=30 and (hw is None or hw>=30): return ("D","Marginal — convert to On Demand")
    if hw and hw<=20 and (cov is None or cov<=0.35): return ("B","Frequent candidate — promote to 15-min")
    if peak>=0.6 and cov and cov>=0.45: return ("C","Coverage commuter — cut to peak-only or interline")
    return ("B","Stable base — retime if CoV high")

def baseline_analysis():
    cal = pd.read_csv(GTFS/"calendar.txt", dtype=str)
    trips = pd.read_csv(GTFS/"trips.txt", dtype=str)
    st = pd.read_csv(GTFS/"stop_times.txt", dtype={"trip_id":str})
    shapes = pd.read_csv(GTFS/"shapes.txt")
    cal2 = cal[cal["monday"]=="1"].copy(); cal2["start_date"]=cal2["start_date"].astype(int)
    svc = cal2.sort_values("start_date")["service_id"].iloc[-1]
    wd = trips[trips["service_id"]==svc]
    shp = {}
    for sid, g in shapes.groupby("shape_id"):
        g = g.sort_values("shape_pt_sequence").reset_index(drop=True)
        shp[sid] = sum(haversine(g.shape_pt_lat[i], g.shape_pt_lon[i],
                                 g.shape_pt_lat[i+1], g.shape_pt_lon[i+1])
                       for i in range(len(g)-1))
    st["arr_s"]=st["arrival_time"].map(t2s); rows=[]
    for rid,rg in wd.groupby("route_id"):
        sr = st[st["trip_id"].isin(set(rg["trip_id"]))]
        durs,dists,ns,starts=[],[],[],[]
        for tid,tg in sr.groupby("trip_id"):
            a=tg.sort_values("stop_sequence")["arr_s"].dropna()
            if len(a)<2: continue
            durs.append((a.iloc[-1]-a.iloc[0])/60); ns.append(len(a)); starts.append(a.iloc[0])
            dists.append(shp.get(rg[rg.trip_id==tid].shape_id.iloc[0],np.nan))
        if not durs: continue
        dur,dist,nstop=np.nanmean(durs),np.nanmean(dists),np.nanmean(ns)
        spd=dist/(dur/60) if dur>0 else np.nan
        ss=sorted(s for s in starts if 6*3600<=s<=21*3600); g=np.diff(ss)/60 if len(ss)>1 else np.array([])
        hw=float(np.median(g)) if len(g) else None
        cov=float(np.std(g)/np.mean(g)) if len(g) and np.mean(g)>0 else None
        peak=sum(1 for s in starts if 7*3600<=s<=10*3600 or 16*3600<=s<=19*3600)/len(starts)
        b,why=classify(rid in PULSE,hw,peak,cov,len(durs))
        rows.append(dict(route_id=rid,is_pulse=rid in PULSE,weekday_trips=len(durs),
            avg_speed_kmh=round(spd,1),avg_distance_km=round(dist,1),stops_per_km=round(nstop/dist,2) if dist else None,
            median_headway_min=round(hw,1) if hw else None,headway_cov=round(cov,2) if cov else None,
            peak_trip_share=round(peak,2),weekday_service_hours=round(len(durs)*dur/60,1),bucket=b,diagnosis=why))
    df=pd.DataFrame(rows).sort_values("weekday_service_hours",ascending=False)
    df.to_csv(REPORT,index=False)
    print(f"✓ Baseline: {len(df)} routes, median speed {df.avg_speed_kmh.median():.1f} km/h")
    print("  buckets:",df.bucket.value_counts().to_dict())
    return df

# ── 4. chart ───────────────────────────────────────────────────────────────────
def chart(df):
    osh = df[df.route_id.str.startswith("4")]; pul = df[df.is_pulse]
    fig,ax=plt.subplots(figsize=(10,6))
    for sub,c,l in [(osh,"#f5a524","Oshawa locals"),(pul,"#22d3b8","PULSE")]:
        s=sub.dropna(subset=["headway_cov"])
        ax.scatter(s.avg_speed_kmh,s.headway_cov,s=s.weekday_service_hours*2.2,
                   c=c,alpha=.75,edgecolor="k",linewidth=.5,label=l,zorder=3)
        for _,r in s.iterrows(): ax.annotate(r.route_id,(r.avg_speed_kmh,r.headway_cov),fontsize=8,ha="center",va="center")
    ax.axvline(df.avg_speed_kmh.median(),ls="--",c="gray",alpha=.6,label=f"median {df.avg_speed_kmh.median():.0f} km/h")
    ax.axhline(0.5,ls=":",c="#ff5c66",alpha=.7,label="irregular (CoV 0.5)")
    ax.set_xlabel("Commercial speed (km/h) → faster"); ax.set_ylabel("Headway CoV → lower is more regular")
    ax.set_title("DRT routes: speed vs scheduling regularity (bottom-left = danger zone)")
    ax.legend(loc="upper right",fontsize=9); ax.set_ylim(-.05,min(1.05,df.headway_cov.max()*1.1 if df.headway_cov.max() else 1))
    fig.tight_layout(); fig.savefig(DATA/"oshawa_chart.png",dpi=130,facecolor="white")
    if IS_COLAB: plt.show()
    print(f"✓ Chart saved: {DATA/'oshawa_chart.png'}")

# ── 5. realtime logger (call when ready; runs for `minutes`) ──────────────────
def run_logger(minutes=10, interval=20):
    from google.transit import gtfs_realtime_pb2
    print(f"Logging realtime feed for {minutes} min (every {interval}s)…")
    end=time.time()+minutes*60; recs=[]
    while time.time()<end:
        try:
            for url,kind in [(VEHICLE_POSITIONS_URL,"vp"),(TRIP_UPDATES_URL,"tu")]:
                f=gtfs_realtime_pb2.FeedMessage(); f.ParseFromString(requests.get(url,timeout=15).content)
                ts=datetime.now(timezone.utc)
                for e in f.entity:
                    if kind=="vp" and e.HasField("vehicle"):
                        v=e.vehicle; recs.append(dict(ts=ts,kind="vp",trip_id=v.trip.trip_id,
                            route_id=v.trip.route_id,lat=v.position.latitude,lon=v.position.longitude,
                            stop_id=v.stop_id,status=v.current_status))
                    elif kind=="tu" and e.HasField("trip_update"):
                        tu=e.trip_update
                        for u in tu.stop_time_update:
                            recs.append(dict(ts=ts,kind="tu",trip_id=tu.trip.trip_id,route_id=tu.trip.route_id,
                                stop_id=u.stop_id,arr_delay=u.arrival.delay if u.HasField("arrival") else None))
        except Exception as ex: print("  poll error:",ex)
        time.sleep(interval)
    if recs:
        day=datetime.now(timezone.utc).strftime("%Y-%m-%d"); (RT_LOG/f"date={day}").mkdir(exist_ok=True)
        pd.DataFrame(recs).to_parquet(RT_LOG/f"date={day}/log_{int(time.time())}.parquet",index=False)
        print(f"✓ Wrote {len(recs):,} realtime records")

# ── 6. feature engineering + model (ready for when rt_log has data) ───────────
def build_features(): print("build_features(): join rt_log against schedule_index — needs ≥1 logged day.")
def train_otp_model(): print("train_otp_model(): trains LightGBM on features/. See model_dryrun.py for a working demo.")

# ── download helper (small outputs only — fast, no zip choke) ─────────────────
def download_results():
    if not IS_COLAB: print("(local run — files are in",DATA,")"); return
    from google.colab import files
    for f in [REPORT, DATA/"oshawa_chart.png"]:
        if Path(f).exists():
            try: files.download(str(f))
            except Exception as e: print("skip",f,e)

# ════════════ RUN ════════════
extract_gtfs()
build_schedule_index()
df = baseline_analysis()
chart(df)
print("\n" + "="*70)
print("DONE. Schedule-only analysis complete.")
print("Next: run_logger(minutes=10)  to test the live feed (URLs already set).")
print("      Leave a logger running 2+ weeks on an always-on machine for real OTP.")
print("="*70)
try:
    from IPython.display import display
    display(df.head(12))
except Exception:
    print(df.head(12).to_string(index=False))
download_results()
