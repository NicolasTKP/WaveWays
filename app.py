# app.py
"""
Streamlit app — Weather-aware routing (A* & K-shortest) + fuel/CO2 + ROI + disk cache + interactive map.

Save as: app.py
Run: streamlit run app.py
"""

import os
import time
import json
import math
import shutil
from pathlib import Path
from itertools import islice

import requests
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from bs4 import BeautifulSoup
from shapely.geometry import Polygon, Point, LineString, mapping
from geopy.distance import geodesic
import networkx as nx
from folium.plugins import HeatMap

# Optional geodata (GEBCO) support
try:
    import xarray as xr
except Exception:
    xr = None

# Optional folium streaming
try:
    import folium
    from streamlit_folium import st_folium
    FOLIUM_OK = True
except Exception:
    FOLIUM_OK = False

st.set_page_config(page_title="Waves Ways — Weather-aware Routing", layout="wide")

# -----------------------
# CONFIG (edit paths if needed)
# -----------------------
DEFAULT_PORTS_XLSX = "data/Ports/ports.xlsx"   # -> change if your file is elsewhere
GEBCO_PATH = None  # e.g. "data/Bathymetry/GEBCO_2024_sub_ice_topo.nc" or None

# global bounds (clamp ROI)
GLOBAL_LAT_MIN, GLOBAL_LAT_MAX = 0.0, 8.0
GLOBAL_LON_MIN, GLOBAL_LON_MAX = 99.0, 120.0

# defaults
LAT_STEP_DEFAULT = 0.20
LON_STEP_DEFAULT = 0.20
RISK_WEIGHT_DEFAULT = 40.0
AVOID_THRESHOLD_DEFAULT = 2.5

# MET cache
CACHE_DIR = Path(".met_cache")
CACHE_DIR.mkdir(exist_ok=True)
CACHE_TTL_SECONDS_DEFAULT = 3600  # 1 hour

# region rectangles & MET codes (from your earlier data)
REGION_RECTS = {
    "Northern_Straits_of_Malacca": [
        {"lat": 5.95, "lon": 98.025},
        {"lat": 5.95, "lon": 100.9},
        {"lat": 3.0, "lon": 101.4},
        {"lat": 3.0, "lon": 99.0}
    ],
    "Southern_Straits_of_Malacca": [
        {"lat": 3.0, "lon": 100.8},
        {"lat": 3.0, "lon": 103.3},
        {"lat": 1.2667, "lon": 103.5167},
        {"lat": 1.1667, "lon": 103.3917},
        {"lat": 1.1, "lon": 102.9667}
    ],
    "Tioman_Island": [
        {"lat": 2.9, "lon": 104.05},
        {"lat": 2.9, "lon": 104.3},
        {"lat": 2.7, "lon": 104.3},
        {"lat": 2.7, "lon": 104.05}
    ],
    "Bunguran_Island": [
        {"lat": 4.2, "lon": 107.8},
        {"lat": 4.2, "lon": 108.6},
        {"lat": 3.6, "lon": 108.6},
        {"lat": 3.6, "lon": 107.8}
    ],
    "Layang_Layang_Island": [
        {"lat": 7.39, "lon": 113.82},
        {"lat": 7.39, "lon": 113.87},
        {"lat": 7.35, "lon": 113.87},
        {"lat": 7.35, "lon": 113.82}
    ],
    "Labuan": [
        {"lat": 5.36, "lon": 115.16},
        {"lat": 5.36, "lon": 115.36},
        {"lat": 5.22, "lon": 115.36},
        {"lat": 5.22, "lon": 115.16}
    ]
}

REGION_TO_CODE = {
    "Northern_Straits_of_Malacca": "Sh002",
    "Southern_Straits_of_Malacca": "Sh003",
    "Tioman_Island": "Sh005",
    "Bunguran_Island": "Sh007",
    "Layang_Layang_Island": "Sh010",
    "Labuan": "Sh012"
}

# -----------------------
# Utilities: parsers & risk
# -----------------------
SEA_STATE_MAP = {"slight": 1.0, "moderate": 2.0, "rough": 3.0, "very rough": 4.5, "high": 6.0}
WIND_DESC_TO_KT = {"light": 8, "moderate": 16, "fresh": 20, "strong": 26, "gale": 34}

def extract_wave_height(text: str):
    if not text: return None
    t = text.lower()
    m = __import__("re").search(r'(\d+(?:\.\d+)?)\s*(?:-|–|to)\s*(\d+(?:\.\d+)?)\s*m', t)
    if m:
        a, b = float(m.group(1)), float(m.group(2)); return (a + b) / 2.0
    m = __import__("re").search(r'(\d+(?:\.\d+)?)\s*m', t)
    if m: return float(m.group(1))
    for k,v in SEA_STATE_MAP.items():
        if k in t: return v
    return None

def extract_wind_speed(text: str):
    if not text: return None
    t = text.lower()
    m = __import__("re").search(r'(\d+)\s*(?:-|–|to)\s*(\d+)\s*(?:knot|kt)', t)
    if m:
        a, b = int(m.group(1)), int(m.group(2)); return (a + b)/2.0
    m = __import__("re").search(r'(\d+)\s*(?:knot|kt)', t)
    if m: return float(m.group(1))
    for k,v in WIND_DESC_TO_KT.items():
        if k in t: return float(v)
    return None

def risk_score(wave_m, wind_kt):
    wv = 0.0 if wave_m is None else wave_m
    wd = 0.0 if wind_kt is None else wind_kt
    wave_pen = 0.0
    if wv >= 4.0: wave_pen = 4.0
    elif wv >= 3.0: wave_pen = 3.0
    elif wv >= 2.0: wave_pen = 2.0
    elif wv >= 1.0: wave_pen = 1.0
    wind_pen = 0.0
    if wd >= 30: wind_pen = 3.0
    elif wd >= 20: wind_pen = 2.0
    elif wd >= 12: wind_pen = 1.0
    return wave_pen + wind_pen

# -----------------------
# Disk cache for MET pages (per region code)
# -----------------------
def ensure_cache_dir():
    CACHE_DIR.mkdir(exist_ok=True)

def cache_path(code: str) -> Path:
    ensure_cache_dir()
    return CACHE_DIR / f"{code}.json"

def fetch_met_cached(code: str, ttl_seconds: int):
    p = cache_path(code)
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf8"))
            ts = data.get("_fetched_ts", 0)
            if time.time() - ts < ttl_seconds:
                return data
        except Exception:
            pass
    # fetch
    url = f"https://www.met.gov.my/en/forecast/marine/shipping/{code}/"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        html = r.text
    except Exception as e:
        return {"wave_m": None, "wind_kt": None, "raw": f"ERROR: {e}", "_fetched_ts": time.time()}
    soup = BeautifulSoup(html, "html.parser")
    all_text = soup.get_text(separator=" ", strip=True)
    wave = extract_wave_height(all_text)
    wind = extract_wind_speed(all_text)
    out = {"wave_m": wave, "wind_kt": wind, "raw": all_text[:1000], "_fetched_ts": time.time()}
    try:
        p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf8")
    except Exception:
        pass
    return out

def fetch_all_region_forecasts_cached(ttl_seconds: int):
    out = {}
    for region, code in REGION_TO_CODE.items():
        out[region] = fetch_met_cached(code, ttl_seconds)
    return out

# -----------------------
# Regions polygons
# -----------------------
REGION_POLYS = {name: Polygon([(p["lon"], p["lat"]) for p in pts]) for name, pts in REGION_RECTS.items()}

def region_for_point(lat, lon):
    pt = Point(lon, lat)
    for name, poly in REGION_POLYS.items():
        if poly.contains(pt) or poly.touches(pt):
            return name
    return None

# -----------------------
# Load ports file
# -----------------------
@st.cache_data
def load_ports(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, engine="openpyxl")
    if "Decimal" in df.columns:
        def parse_decimal(val):
            try:
                s = str(val).strip()
                if "," in s:
                    lat_s, lon_s = s.split(",", 1)
                else:
                    parts = s.split()
                    if len(parts) >= 2:
                        lat_s, lon_s = parts[0], parts[1]
                    else:
                        return (np.nan, np.nan)
                return (float(lat_s), float(lon_s))
            except Exception:
                return (np.nan, np.nan)
        df[["lat","lon"]] = df["Decimal"].apply(lambda v: pd.Series(parse_decimal(v)))
    elif "Latitude" in df.columns and "Longitude" in df.columns:
        df = df.rename(columns={"Latitude":"lat","Longitude":"lon"})
    if "Ports" not in df.columns:
        df = df.rename(columns={df.columns[0]:"Ports"})
    return df[["Ports","lat","lon"]].dropna().reset_index(drop=True)

# -----------------------
# Optional GEBCO land mask
# -----------------------
@st.cache_data
def maybe_land_mask(path: str):
    if (xr is None) or (path is None) or (not os.path.exists(path)):
        return None
    try:
        ds = xr.open_dataset(path)
        ds_sub = ds.sel(lon=slice(GLOBAL_LON_MIN, GLOBAL_LON_MAX), lat=slice(GLOBAL_LAT_MIN, GLOBAL_LAT_MAX))
        elev = ds_sub["elevation"].values
        lats = ds_sub["lat"].values
        lons = ds_sub["lon"].values
        mask = (elev >= 0).astype(np.uint8)
        return {"mask": mask, "lats": lats, "lons": lons}
    except Exception:
        return None

LAND = maybe_land_mask(GEBCO_PATH)

def is_land(lat, lon):
    if LAND is None:
        return False
    lats, lons, mask = LAND["lats"], LAND["lons"], LAND["mask"]
    i = int(np.clip(np.searchsorted(lats, lat), 1, len(lats)-1))
    j = int(np.clip(np.searchsorted(lons, lon), 1, len(lons)-1))
    cand = [(i-1,j-1),(i-1,j),(i,j-1),(i,j)]
    vals = []
    for ii,jj in cand:
        ii = np.clip(ii,0,len(lats)-1)
        jj = np.clip(jj,0,len(lons)-1)
        vals.append(mask[ii,jj])
    return int(np.mean(vals) > 0.5)

# -----------------------
# ROI & grid & risk
# -----------------------
def build_roi_bounds(start_lat, start_lon, goal_lat, goal_lon, pad_deg=1.0):
    lat_min = max(GLOBAL_LAT_MIN, min(start_lat, goal_lat) - pad_deg)
    lat_max = min(GLOBAL_LAT_MAX, max(start_lat, goal_lat) + pad_deg)
    lon_min = max(GLOBAL_LON_MIN, min(start_lon, goal_lon) - pad_deg)
    lon_max = min(GLOBAL_LON_MAX, max(start_lon, goal_lon) + pad_deg)
    return lat_min, lat_max, lon_min, lon_max

def build_grid(lat_min, lat_max, lat_step, lon_min, lon_max, lon_step):
    lats = np.arange(lat_min, lat_max + 1e-12, lat_step)
    lons = np.arange(lon_min, lon_max + 1e-12, lon_step)
    return lats, lons

def compute_risk_field(lats, lons, region_forecasts, wave_block_thresh, wind_block_thresh):
    risk = np.zeros((len(lats), len(lons)), dtype=float)
    for i, la in enumerate(lats):
        for j, lo in enumerate(lons):
            if is_land(la, lo):
                risk[i,j] = np.inf
                continue
            reg = region_for_point(la, lo)
            if reg is None:
                risk[i,j] = 0.5
            else:
                info = region_forecasts.get(reg, {})
                wave_m = info.get("wave_m")
                wind_kt = info.get("wind_kt")
                # block if above thresholds
                if (wave_m is not None and wave_m >= wave_block_thresh) or (wind_kt is not None and wind_kt >= wind_block_thresh):
                    risk[i,j] = np.inf
                else:
                    rv = risk_score(wave_m, wind_kt)
                    risk[i,j] = rv
    return risk

# -----------------------
# Graph conversion (grid -> DiGraph)
# -----------------------
def build_graph_from_grid(lats, lons, risk, risk_weight=RISK_WEIGHT_DEFAULT, avoid_threshold=AVOID_THRESHOLD_DEFAULT):
    G = nx.DiGraph()
    ni, nj = risk.shape
    # add nodes for water cells only
    for i in range(ni):
        for j in range(nj):
            if np.isinf(risk[i,j]): continue
            G.add_node((i,j), lat=float(lats[i]), lon=float(lons[j]), risk=float(risk[i,j]))
    # edges (8-connected)
    deltas = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    for i in range(ni):
        for j in range(nj):
            if np.isinf(risk[i,j]): continue
            for di,dj in deltas:
                ii, jj = i+di, j+dj
                if 0 <= ii < ni and 0 <= jj < nj:
                    if np.isinf(risk[ii,jj]): continue
                    lat1, lon1 = float(lats[i]), float(lons[j])
                    lat2, lon2 = float(lats[ii]), float(lons[jj])
                    dist_km = geodesic((lat1, lon1), (lat2, lon2)).km
                    r_edge = 0.5 * (risk[i,j] + risk[ii,jj])
                    penalty = 0.0
                    if r_edge >= avoid_threshold:
                        penalty = 1e6
                    weight = dist_km + risk_weight * r_edge + penalty
                    G.add_edge((i,j), (ii,jj), weight=weight, distance=dist_km, risk=r_edge)
    return G

# -----------------------
# K-shortest & A*
# -----------------------
def k_shortest_paths_graph(G, source_node, target_node, k=2):
    try:
        paths_gen = nx.shortest_simple_paths(G, source_node, target_node, weight='weight')
    except Exception as e:
        st.error(f"Error generating shortest_simple_paths: {e}")
        return []
    results = []
    for p in islice(paths_gen, k):
        results.append(p)
    return results

def astar_single_path(G, source_node, target_node, lats, lons):
    # heuristic: geodesic (km) from node to target
    def heuristic(u, v):
        lat_u, lon_u = float(lats[u[0]]), float(lons[u[1]])
        lat_v, lon_v = float(lats[v[0]]), float(lons[v[1]])
        return geodesic((lat_u, lon_u), (lat_v, lon_v)).km
    try:
        path = nx.astar_path(G, source_node, target_node, heuristic=lambda n: heuristic(n, target_node), weight='weight')
        return path
    except Exception as e:
        st.error(f"A* failed: {e}")
        return None

# -----------------------
# Fuel & CO2 estimator (cubic scaling)
# -----------------------
def estimate_fuel_co2_for_coords(coords, speed_kn, ref_speed_kn=12.0, ref_fuel_tpd=20.0, co2_per_tonne=3.114):
    total_km = 0.0
    leg_info = []
    for a,b in zip(coords[:-1], coords[1:]):
        d = geodesic(a, b).km
        total_km += d
        leg_info.append({"distance_km": d})
    speed_kmh = max(1e-6, speed_kn * 1.852)
    total_time_h = total_km / speed_kmh
    ref_rate_tph = ref_fuel_tpd / 24.0
    scaled_rate_tph = ref_rate_tph * (speed_kn / ref_speed_kn) ** 3
    fuel_tons = scaled_rate_tph * total_time_h
    co2_tons = fuel_tons * co2_per_tonne
    # per-leg breakdown
    per_leg = []
    for leg in leg_info:
        d = leg["distance_km"]
        t_h = d / speed_kmh
        fuel = scaled_rate_tph * t_h
        co2 = fuel * co2_per_tonne
        per_leg.append({"distance_km": d, "time_h": t_h, "fuel_tons": fuel, "co2_tons": co2})
    return total_km, total_time_h, fuel_tons, co2_tons, per_leg

# -----------------------
# Helpers
# -----------------------
def path_nodes_to_coords(path, lats, lons):
    return [(float(lats[i]), float(lons[j])) for (i,j) in path]

def nearest_node_to_latlon(lat, lon, lats, lons, risk):
    i = int(np.abs(lats - lat).argmin())
    j = int(np.abs(lons - lon).argmin())
    if np.isinf(risk[i,j]):
        ni, nj = risk.shape
        for r in range(1,8):
            for di in range(-r, r+1):
                for dj in range(-r, r+1):
                    ii, jj = i+di, j+dj
                    if 0 <= ii < ni and 0 <= jj < nj and not np.isinf(risk[ii,jj]):
                        return (ii, jj)
        raise RuntimeError("Unable to find nearby water node for start/goal.")
    return (i,j)

# -----------------------
# Streamlit UI
# -----------------------
st.title("Waves Ways — Weather-aware Routing (A* & K-shortest)")
st.markdown("Compute weather-aware routes that avoid high waves/winds, compare best vs shadow routes, estimate fuel & CO₂, and export results.")

# layout
left, right = st.columns([1,2])

with left:
    st.header("Inputs")
    ports_path = st.text_input("Ports XLSX path", value=DEFAULT_PORTS_XLSX)
    try:
        ports_df = load_ports(ports_path)
    except Exception as e:
        st.error(f"Cannot load ports.xlsx: {e}")
        st.stop()

    port_names = sorted(ports_df["Ports"].tolist())
    origin = st.selectbox("Origin port", port_names, index=0)
    dest = st.selectbox("Destination port", port_names, index=min(1, len(port_names)-1))

    pad_deg = st.slider("ROI padding (degrees)", min_value=0.2, max_value=3.0, value=1.0, step=0.1)
    lat_step = st.number_input("Lat step (deg)", value=float(LAT_STEP_DEFAULT), step=0.05, format="%.2f")
    lon_step = st.number_input("Lon step (deg)", value=float(LON_STEP_DEFAULT), step=0.05, format="%.2f")

    st.subheader("Weather safety thresholds (block cell if exceeded)")
    wave_block_thresh = st.number_input("Wave height threshold (m) — block if ≥", value=4.0, step=0.5, format="%.1f")
    wind_block_thresh = st.number_input("Wind speed threshold (knots) — block if ≥", value=25.0, step=1.0, format="%.1f")

    st.subheader("Routing options")
    routing_mode = st.radio("Routing algorithm", ["A*", "K-shortest (k=2)"])
    risk_weight = st.number_input("Risk weight (multiplier)", value=float(RISK_WEIGHT_DEFAULT), step=1.0)
    avoid_threshold = st.number_input("Avoid threshold (risk score)", value=float(AVOID_THRESHOLD_DEFAULT), step=0.5)

    st.subheader("Vessel & emissions")
    ref_speed_kn = st.number_input("Reference speed (knots) at ref_fuel", value=12.0, step=0.5)
    ref_fuel_tpd = st.number_input("Reference fuel (tonnes/day) at ref speed", value=20.0, step=1.0)
    eval_speed_kn = st.number_input("Evaluation speed (knots) for estimates", value=float(ref_speed_kn), step=0.5)

    st.subheader("Map & cache")
    use_mapbox = st.checkbox("Use Mapbox tiles (requires token)")
    mapbox_token = st.text_input("Mapbox token (only if using Mapbox)", value="")
    cache_ttl = st.number_input("MET cache TTL (seconds)", value=int(CACHE_TTL_SECONDS_DEFAULT), step=60)
    if st.button("Clear MET cache"):
        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR)
            CACHE_DIR.mkdir()
            st.success("MET cache cleared.")

    compute_btn = st.button("Compute routes")

with right:
    st.header("Status")
    st.write("After pressing **Compute routes**, the app fetches regional forecasts (cached), builds an ROI grid, computes a risk field (blocking cells where wave/wind exceed thresholds), builds a graph, and finds routes.")

# Main compute
if compute_btn:
    with st.spinner("Fetching forecasts..."):
        forecasts = fetch_all_region_forecasts_cached(int(cache_ttl))
    st.write("Region forecasts (wave_m, wind_kt):")
    st.json({k: {"wave_m": v.get("wave_m"), "wind_kt": v.get("wind_kt"), "fetched": v.get("_fetched_ts")} for k,v in forecasts.items()})

    # map ports -> coords
    srow = ports_df[ports_df["Ports"]==origin].iloc[0]
    grow = ports_df[ports_df["Ports"]==dest].iloc[0]
    start_la, start_lo = float(srow["lat"]), float(srow["lon"])
    goal_la, goal_lo = float(grow["lat"]), float(grow["lon"])

    # build ROI
    lat_min, lat_max, lon_min, lon_max = build_roi_bounds(start_la, start_lo, goal_la, goal_lo, pad_deg=pad_deg)
    st.write(f"ROI bounds: lat {lat_min:.3f}–{lat_max:.3f}, lon {lon_min:.3f}–{lon_max:.3f}")

    # build grid & risk
    lats, lons = build_grid(lat_min, lat_max, lat_step, lon_min, lon_max, lon_step)
    st.write(f"Grid: {len(lats)} x {len(lons)} cells ({len(lats)*len(lons)} total)")

    st.info("Computing risk field (this will mark blocked cells where wave/wind exceed thresholds)...")
    risk = compute_risk_field(lats, lons, forecasts, wave_block_thresh, wind_block_thresh)

    # quick diagnostics counts
    total_cells = risk.size
    blocked_cells = np.isinf(risk).sum()
    st.write(f"Total cells: {total_cells} — blocked cells (land or weather): {blocked_cells}")

    # build graph
    st.info("Building graph from grid...")
    G = build_graph_from_grid(lats, lons, risk, risk_weight=risk_weight, avoid_threshold=avoid_threshold)
    st.write(f"Graph nodes: {G.number_of_nodes()} edges: {G.number_of_edges()}")

    # map origin/dest to nearest node
    try:
        src = nearest_node_to_latlon(start_la, start_lo, lats, lons, risk)
        tgt = nearest_node_to_latlon(goal_la, goal_lo, lats, lons, risk)
    except Exception as e:
        st.error(f"Failed to map ports to water grid nodes: {e}")
        st.stop()

    st.write("Mapped start ->", src, "goal ->", tgt)

    # routing
    routes_nodes = []
    if routing_mode == "A*":
        with st.spinner("Running A*..."):
            path = astar_single_path(G, src, tgt, lats, lons)
        if path is None:
            st.error("A* did not find a path. Try increasing ROI or lowering thresholds.")
            st.stop()
        routes_nodes = [path]
    else:
        with st.spinner("Computing K-shortest (k=2)..."):
            routes_nodes = k_shortest_paths_graph(G, src, tgt, k=2)
        if not routes_nodes:
            st.error("No K-shortest paths found. Try A* or adjust grid/padding.")
            st.stop()

    # compute metrics, fuel/co2
    results = []
    for idx, p in enumerate(routes_nodes):
        coords = path_nodes_to_coords(p, lats, lons)
        total_km, total_h, fuel_tons, co2_tons, per_leg = estimate_fuel_co2_for_coords(coords, eval_speed_kn, ref_speed_kn, ref_fuel_tpd)
        results.append({
            "idx": idx,
            "nodes": len(p),
            "distance_km": total_km,
            "time_h": total_h,
            "fuel_tons": fuel_tons,
            "co2_tons": co2_tons,
            "coords": coords,
            "per_leg": per_leg
        })

    # summary table
    summary = pd.DataFrame([{
        "route": "Best" if r["idx"]==0 else f"Shadow {r['idx']}",
        "nodes": r["nodes"],
        "distance_km": round(r["distance_km"],1),
        "time_h": round(r["time_h"],1),
        "fuel_tons": round(r["fuel_tons"],2),
        "co2_tons": round(r["co2_tons"],2)
    } for r in results])
    st.subheader("Routes summary")
    st.table(summary)

    # Plot risk heatmap (matplotlib) + overlay
    fig, ax = plt.subplots(figsize=(10,6))
    rp = np.copy(risk); rp[np.isinf(rp)] = np.nan
    im = ax.imshow(rp.T, origin="lower", extent=[lon_min, lon_max, lat_min, lat_max], aspect="auto", cmap="YlOrRd", alpha=0.85)
    plt.colorbar(im, ax=ax, label="Risk score (higher = worse)")

    for name, poly in REGION_POLYS.items():
        xs, ys = poly.exterior.xy
        ax.plot(xs, ys, "--", lw=1)

    # ports
    ax.scatter(ports_df["lon"], ports_df["lat"], s=20, c="dodgerblue", zorder=6)
    for _, r in ports_df.iterrows():
        ax.text(r["lon"], r["lat"], r["Ports"], fontsize=7)

    colors = ["black", "cyan", "magenta", "lime"]
    for idx, res in enumerate(results):
        xs = [lo for (_, lo) in res["coords"]]
        ys = [la for (la, _) in res["coords"]]
        ax.plot(xs, ys, color=colors[idx%len(colors)], lw=2.5, label=("Best" if idx==0 else f"Shadow {idx}"))
    ax.scatter([start_lo, goal_lo], [start_la, goal_la], c=["green","red"], s=80, zorder=9)
    ax.set_xlim(lon_min, lon_max); ax.set_ylim(lat_min, lat_max)
    ax.set_title(f"{origin} → {dest} (routes overlay)")
    ax.legend(fontsize=8)
    st.pyplot(fig)

    # prepare heat points for folium
    heat_points = []
    for i, la in enumerate(lats):
        for j, lo in enumerate(lons):
            rv = risk[i,j]
            if np.isfinite(rv) and rv > 0:
                heat_points.append([la, lo, float(rv)])

    # downsample heat points if too many
    heat_limit = 2000
    if len(heat_points) > heat_limit:
        factor = max(1, len(heat_points) // heat_limit)
        heat_points = heat_points[::factor]

    # Folium interactive map (if available)
    if FOLIUM_OK:
        center_lat = (start_la + goal_la) / 2.0
        center_lon = (start_lo + goal_lo) / 2.0
        tiles = "OpenStreetMap"
        if use_mapbox and mapbox_token.strip():
            mb_url = f"https://api.mapbox.com/styles/v1/mapbox/streets-v11/tiles/{{z}}/{{x}}/{{y}}?access_token={mapbox_token}"
            tiles = mb_url
        m = folium.Map(location=[center_lat, center_lon], zoom_start=6, tiles=tiles, attr="Mapbox" if use_mapbox else "OSM")
        if heat_points:
            HeatMap(heat_points, radius=10, blur=12, max_zoom=10).add_to(m)
        # add routes
        fol_colors = ["black","blue","purple","green"]
        for idx, r in enumerate(results):
            folium.PolyLine(locations=[(la,lo) for (la,lo) in r["coords"]], color=fol_colors[idx%len(fol_colors)], weight=4, opacity=0.8, popup=f"Route {idx}").add_to(m)
        folium.Marker([start_la, start_lo], popup="START", icon=folium.Icon(color="green")).add_to(m)
        folium.Marker([goal_la, goal_lo], popup="GOAL", icon=folium.Icon(color="red")).add_to(m)
        st.subheader("Interactive map")
        st.write("Pan / zoom. Heat layer shows risk; polylines show routes.")
        st_folium(m, width=900, height=500)
    else:
        st.info("Folium or streamlit-folium not installed; interactive map disabled.")

    # per-leg breakdown & charts for best route
    st.subheader("Best route per-leg breakdown")
    best = results[0]
    leg_df = pd.DataFrame(best["per_leg"])
    leg_df.index = range(1, len(leg_df)+1)
    leg_df.index.name = "leg"
    st.dataframe(leg_df.round(4))

    # charts
    fig2, ax2 = plt.subplots(1,2, figsize=(12,4))
    ax2[0].bar(leg_df.index, leg_df["fuel_tons"])
    ax2[0].set_xlabel("Leg"); ax2[0].set_ylabel("Fuel (t)"); ax2[0].set_title("Fuel per leg")
    ax2[1].plot(leg_df["fuel_tons"].cumsum(), marker="o")
    ax2[1].set_xlabel("Leg"); ax2[1].set_ylabel("Cumulative fuel (t)"); ax2[1].set_title("Cumulative fuel")
    st.pyplot(fig2)

    # export options
    st.subheader("Export routes")
    for idx, r in enumerate(results):
        df_coords = pd.DataFrame(r["coords"], columns=["lat","lon"])
        csv_buf = df_coords.to_csv(index=False)
        st.download_button(f"Download route {idx} CSV", data=csv_buf, file_name=f"route_{idx}_{origin}_to_{dest}.csv", mime="text/csv")
        # GeoJSON
        line = LineString([(lon, lat) for (lat, lon) in r["coords"]])
        gj = {"type":"FeatureCollection", "features":[{"type":"Feature","geometry":mapping(line),
                                                       "properties":{"route": ("best" if idx==0 else f"shadow_{idx}"),
                                                                    "distance_km": r["distance_km"],
                                                                    "fuel_tons": r["fuel_tons"],
                                                                    "co2_tons": r["co2_tons"]}}]}
        geojson_str = json.dumps(gj)
        st.download_button(f"Download route {idx} GeoJSON", data=geojson_str, file_name=f"route_{idx}_{origin}_to_{dest}.geojson", mime="application/geo+json")

    st.success("Done — routes computed and exported as needed.")
