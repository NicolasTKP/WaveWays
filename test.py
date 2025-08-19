import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import LineString
import networkx as nx
from networkx.algorithms import approximation as approx
from geopy.distance import geodesic

def safe_float(val):
    """Convert to float safely, return np.nan if invalid."""
    try:
        if val is None:
            return 0.0
        fval = float(val)
        if np.isnan(fval) or np.isinf(fval):
            return 0.0
        return round(fval,2)
    except Exception:
        return 0.0

def get_wave_wind(lat, lon, time="latest"):
    """
    Fetch wave and wind data for given lat, lon, and time.
    """
    url = "https://pae-paha.pacioos.hawaii.edu/erddap/griddap/ww3_global"
    ds = xr.open_dataset(url, decode_times=True, decode_timedelta=True)

    point = ds.sel(longitude=lon, latitude=lat, method="nearest")

    if time == "latest":
        idx = -1
    else:
        idx = dict(time=np.datetime64(time))

    return {
        "wave_height": safe_float(point["Thgt"].isel(time=idx).values),
        "wave_period": safe_float(point["Tper"].isel(time=idx).values),
        "wave_direction": safe_float(point["Tdir"].isel(time=idx).values),
        "wind_wave_height": safe_float(point["whgt"].isel(time=idx).values),
        "wind_wave_period": safe_float(point["wper"].isel(time=idx).values),
        "wind_wave_direction": safe_float(point["wdir"].isel(time=idx).values),
    }
    
def interpolate_points(lat1, lon1, lat2, lon2, n=5):
    """Generate n points between two coordinates and round to 3 decimals"""
    line = LineString([(lon1, lat1), (lon2, lat2)])
    return [
        (round(lat, 3), round(lon, 3))
        for lon, lat in [line.interpolate(d, normalized=True).coords[0] 
                         for d in np.linspace(0, 1, n)]
    ]

def reorder_route(route, start):
    """Rotate the route so it always starts at the chosen departure port"""
    if start not in route:
        raise ValueError(f"{start} not found in route!")
    idx = route.index(start)
    return route[idx:] + route[:idx]

def compute_edge_weight(edge_data, alpha=1.0, weights=None, normalizers=None):
    """
    Compute edge weight from distance + normalized environmental penalties.
    """
    if weights is None:
        weights = {
            "wave_height": 10.0,
            "wave_period": 5.0,
            "wind_wave_height": 8.0,
            "wind_wave_period": 5.0,
        }
    
    if normalizers is None:
        # Reasonable scaling denominators so values map into ~0–1
        normalizers = {
            "wave_height": 5.0,       # typical max ~5 m
            "wave_period": 20.0,      # seconds
            "wind_wave_height": 5.0,  # meters
            "wind_wave_period": 20.0, # seconds
        }
    
    distance = edge_data["distance_km"]
    penalty = 0.0

    for var, w in weights.items():
        values = [c.get(var, 0.0) for c in edge_data["conditions"]]
        avg_val = np.nanmean(values) if values else 0.0

        norm = normalizers.get(var, 1.0)
        penalty += w * (avg_val / norm)

    return alpha * distance + penalty

def enforce_tsp_constraints(route, start, cycle=True):
    """
    Remove duplicates in route while keeping TSP constraints:
    - Each node visited once
    - Start/end match if cycle=True
    """
    seen = set()
    cleaned = []
    for node in route:
        if node not in seen:
            cleaned.append(node)
            seen.add(node)
    # Close the cycle if needed
    if cycle:
        if cleaned[0] != start:
            cleaned = reorder_route(cleaned, start)
        if cleaned[-1] != cleaned[0]:
            cleaned.append(cleaned[0])
    else:
        if cleaned[0] != start:
            cleaned = reorder_route(cleaned, start)
    return cleaned


# ds = xr.open_dataset("data\\Bathymetry\\GEBCO_2025_sub_ice.nc")
# print(ds)


# # Subset to Malaysia region: Longitude 99–120, Latitude 0 to 8
# ds_subset = ds.sel(
#     lon=slice(99, 120),
#     lat=slice(0, 8)
# )

# # Load the ocean mask based on elevation
# elevation = ds_subset['elevation']
# ocean_mask = xr.where(elevation < 0, 1, np.nan)

# # Load and query EEZ shapefile
# eez = gpd.read_file("data\\Fishing_Zones\\EEZ_land_union_v4_202410.shp")
# malaysia_eez = eez[eez['TERRITORY1'] == 'Malaysia']

# load ports data
ports_df = pd.read_excel("data\\Ports\\ports.xlsx")
ports_df[["lat", "lon"]] = ports_df["Decimal"].str.split(",", expand=True).astype(float)
ports_gdf = gpd.GeoDataFrame(
    ports_df,
    geometry=gpd.points_from_xy(ports_df["lon"], ports_df["lat"]),
    crs="EPSG:4326"
)


# Select ports (example)
selected_ports = ports_gdf.sample(4, random_state=50)

# Build graph
G = nx.Graph()

# Add nodes
for idx, row in selected_ports.iterrows():
    G.add_node(row["Ports"], pos=(row["lon"], row["lat"]))


edges_dict = {}

for i, row_i in selected_ports.iterrows():
    for j, row_j in selected_ports.iterrows():
        if i < j:
            points = interpolate_points(
                row_i["lat"], row_i["lon"], 
                row_j["lat"], row_j["lon"], 
                n=5
            )
            
            # Add wave/wind data for each interpolated point
            point_conditions = []
            for (lat, lon) in points:
                try:
                    cond = get_wave_wind(lat, lon, time="latest")
                except Exception as e:
                    cond = {"error": str(e)}
                point_conditions.append({
                    "lat": lat, "lon": lon, **cond
                })

            edges_dict[(row_i["Ports"], row_j["Ports"])] = {
                "points": points,
                "distance_km": round(geodesic(
                    (row_i["lat"], row_i["lon"]),
                    (row_j["lat"], row_j["lon"])
                ).km, 1),
                "conditions": point_conditions
            }

import pprint
pprint.pprint(edges_dict)
    

# Add edges with computed weights (distance + sea state)
for (u, v), data in edges_dict.items():
    weight = compute_edge_weight(data)  # <-- use your function
    G.add_edge(u, v, weight=weight, **data)
            
for u, v, data in G.edges(data=True):
    print(f"{u} -> {v}: distance={data['distance_km']} km, weight={data['weight']:.2f}")
    
pos = nx.get_node_attributes(G, 'pos')

# Choose a departure port
departure = selected_ports.iloc[0]["Ports"]

# Case A: return to departure (cycle)
cycle = approx.traveling_salesman_problem(G, weight="weight", cycle=True)
cycle = enforce_tsp_constraints(cycle, departure, cycle=True)
print("Optimal cycle route:", cycle)

# Case B: do not return (path)
path = approx.traveling_salesman_problem(G, weight="weight", cycle=False)
path = enforce_tsp_constraints(path, departure, cycle=False)
print("Optimal path route:", path)

# Visualize the routes
plt.figure(figsize=(10, 8))
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10, font_color='black', edge_color='gray')
plt.title("Port Network Graph with Routes")
for route in [cycle, path]:
    route_edges = [(route[i], route[i + 1]) for i in range(len(route) - 1)]
    nx.draw_networkx_edges(G, pos, edgelist=route_edges, edge_color='red', width=2)
    
plt.show()