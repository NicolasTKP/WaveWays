import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import LineString
import networkx as nx
from networkx.algorithms import approximation as approx
from geopy.distance import geodesic
import requests
from bs4 import BeautifulSoup
import re
from shapely.geometry import Point, Polygon

vessel = {
    "speed_knots": 20.0,  # cruising speed in knots
    "fuel_consumption_t_per_day": 20.0,  # fuel consumption in tons per day at cruising speed
    "fuel_tank_capacity_t": 1000.0,  # total fuel tank capacity in tons
    "safety_fuel_margin_t": 10.0,  # safety margin in tons
    "fuel_tank_remaining_t": 1000.0  # current fuel remaining in tons
}

def safe_float(val, var_name=None):
    """Convert to float safely, return np.nan if invalid."""
    try:
        if val is None:
            return 0.0
        # Ensure val is a scalar before converting to float
        if isinstance(val, np.ndarray) and val.size == 1:
            fval = float(val.item()) # Use .item() to get the scalar from any-dim array with one element
        elif isinstance(val, (int, float, np.number)): # Handle direct scalars and NumPy scalar types
            fval = float(val)
        elif val is None:
            return 0.0
        else:
            # If it's a multi-element array or other unexpected type, return 0.0
            # or raise an error if this case is truly an issue.
            return 0.0
            
        if np.isnan(fval) or np.isinf(fval):
            return 0.0
        # Cap wind_wave_period to a reasonable maximum if it's excessively large
        if var_name == "wind_wave_period" and fval > 100.0: # Assuming 100 seconds is a very generous upper bound
            print(f"Warning: Capping unusually large wind_wave_period value: {fval} to 100.0")
            return 100.0
        return round(fval,2)
    except Exception:
        return 0.0

def get_wave_wind(lat, lon, time="latest"):
    """
    Fetch wave and wind data for given lat, lon, and time.
    """
    url = "https://pae-paha.pacioos.hawaii.edu/erddap/griddap/ww3_global"
    try:
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
            "wind_wave_period": safe_float(point["wper"].isel(time=idx).values, var_name="wind_wave_period"), # Pass var_name
            "wind_wave_direction": safe_float(point["wdir"].isel(time=idx).values),
        }
    except Exception as e:
        print(f"Error fetching wave/wind data for ({lat}, {lon}): {e}")
        return {
            "wave_height": 0.0,
            "wave_period": 0.0,
            "wave_direction": 0.0,
            "wind_wave_height": 0.0,
            "wind_wave_period": 0.0,
            "wind_wave_direction": 0.0,
            "error": str(e)
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

def compute_edge_weight(edge_data, vessel_data, alpha=1.0, weights=None, normalizers=None):
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
    environmental_penalty = 0.0
    weather_penalty = 0.0

    for var, w in weights.items():
        values = [c.get(var, 0.0) for c in edge_data["conditions"]]
        avg_val = np.nanmean(values) if values else 0.0

        norm = normalizers.get(var, 1.0)
        environmental_penalty += w * (avg_val / norm)
    
    # Add worst-case weather penalty if available
    if "worst_weather_penalty" in edge_data:
        weather_penalty = edge_data["worst_weather_penalty"]

    # Calculate fuel range and apply penalty if distance exceeds it
    max_range_km = calculate_vessel_range_km(vessel_data)
    fuel_penalty = 0.0
    if distance > max_range_km:
        fuel_penalty = 1000000.0 # Very high penalty for unreachable routes
        print(f"Route segment of {distance:.1f} km exceeds vessel range of {max_range_km:.1f} km. Applying fuel penalty.")

    return alpha * distance + environmental_penalty + weather_penalty + fuel_penalty

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

locations_data = {
    # Practical, coverage-first split of the Malacca Strait at 3.00°N.
    # Northern section spans ~3.00°N–5.95°N and across the full width of the strait.
    "Northern_Straits_of_Malacca": [
        {"lat": 5.9500, "lon": 98.0250},  # near I-M-T tripoint sector (NW entrance line vicinity)
        {"lat": 5.9500, "lon": 100.9000}, # off Kedah/Penang side (NE)
        {"lat": 3.0000, "lon": 101.4000}, # split line at 3°N (SE)
        {"lat": 3.0000, "lon": 99.0000}   # split line at 3°N (SW)
    ],

    # Southern section uses the IHO eastern exit line points and reaches up to the 3.00°N split.
    "Southern_Straits_of_Malacca": [
        {"lat": 3.0000, "lon": 100.8000},  # split line near Sumatra side (NW)
        {"lat": 3.0000, "lon": 103.3000},  # split line near Malay Peninsula side (NE)
        {"lat": 1.2667, "lon": 103.5167},  # Tanjong Piai (IHO eastern limit point)
        {"lat": 1.1667, "lon": 103.3917},  # Klein Karimoen (IHO eastern limit point)
        {"lat": 1.1000, "lon": 102.9667}   # Tanjong Kedabu on Sumatra coast (IHO southern edge)
    ],

    # Islands below use coverage rectangles that fully contain the island;
    # replace with detailed coastlines if you later load shore polygons.
    "Tioman_Island": [
        {"lat": 2.9000, "lon": 104.0500},  # NW
        {"lat": 2.9000, "lon": 104.3000},  # NE
        {"lat": 2.7000, "lon": 104.3000},  # SE
        {"lat": 2.7000, "lon": 104.0500}   # SW
    ],
    "Bunguran_Island": [  # aka Natuna Besar
        {"lat": 4.2000, "lon": 107.8000},  # NW
        {"lat": 4.2000, "lon": 108.6000},  # NE
        {"lat": 3.6000, "lon": 108.6000},  # SE
        {"lat": 3.6000, "lon": 107.8000}   # SW
    ],
    "Layang_Layang_Island": [
        {"lat": 7.3900, "lon": 113.8200},  # NW
        {"lat": 7.3900, "lon": 113.8700},  # NE
        {"lat": 7.3500, "lon": 113.8700},  # SE
        {"lat": 7.3500, "lon": 113.8200}   # SW
    ],
    "Labuan": [
        {"lat": 5.3600, "lon": 115.1600},  # NW
        {"lat": 5.3600, "lon": 115.3600},  # NE
        {"lat": 5.2200, "lon": 115.3600},  # SE
        {"lat": 5.2200, "lon": 115.1600}   # SW
    ]
}

location_codes = {
    "Northern_Straits_of_Malacca": "Sh002",
    "Southern_Straits_of_Malacca": "Sh003",
    "Tioman_Island": "Sh005",
    "Bunguran_Island": "Sh007",
    "Layang_Layang_Island": "Sh010",
    "Labuan": "Sh012"
}

def is_point_in_polygon(lat, lon, polygon_points):
    """
    Check if a (lat, lon) point is inside a polygon defined by a list of points.
    polygon_points: list of {"lat": float, "lon": float}
    """
    if not polygon_points:
        return False
    
    # Create a shapely Point object
    point = Point(lon, lat)
    
    # Create a shapely Polygon object
    # The coordinates for Polygon should be (lon, lat)
    polygon_coords = [(p["lon"], p["lat"]) for p in polygon_points]
    polygon = Polygon(polygon_coords)
    
    return polygon.contains(point)

def get_weather_for_point(lat, lon):
    """
    Checks if a point is within a predefined location and fetches weather data if it is.
    Returns weather data and the location name if found, otherwise None, None.
    """
    for location_name, points in locations_data.items():
        if is_point_in_polygon(lat, lon, points):
            location_code = location_codes.get(location_name)
            if location_code:
                url = f"https://www.met.gov.my/en/forecast/marine/shipping/{location_code}/"
                print(f"Fetching weather for {location_name} at ({lat}, {lon})...")
                weather = scrape_weather_data(url)
                return weather, location_name
    return None, None

def get_weather_penalty(weather_data):
    """
    Assigns a penalty based on weather conditions.
    Higher penalty for bad weather.
    """
    penalty = 0.0
    if not weather_data:
        return penalty

    # Iterate through all tables and forecasts
    for table_key in weather_data:
        for entry in weather_data[table_key]:
            if 'Forecast' in entry and isinstance(entry['Forecast'], dict):
                for period, details in entry['Forecast'].items():
                    if 'Weather' in details and details['Weather']:
                        weather_desc = details['Weather'].lower()
                        if "thunderstorm" in weather_desc or "isolated thunderstorms" in weather_desc:
                            penalty += 50.0 # High penalty
                        elif "rain" in weather_desc or "showers" in weather_desc:
                            penalty += 20.0 # Medium penalty
                        elif "cloudy" in weather_desc:
                            penalty += 5.0 # Low penalty
                        elif "fair" in weather_desc or "no rain" in weather_desc or "sunny" in weather_desc:
                            penalty += 0.0 # No penalty
                        else:
                            penalty += 10.0 # Default penalty for unknown conditions
    return penalty

def calculate_vessel_range_km(vessel_data):
    """
    Calculate the maximum distance the vessel can travel in km with current fuel.
    """
    speed_knots = vessel_data["speed_knots"]
    fuel_consumption_t_per_day = vessel_data["fuel_consumption_t_per_day"]
    fuel_remaining_t = vessel_data["fuel_tank_remaining_t"] - vessel_data["safety_fuel_margin_t"]

    if fuel_remaining_t <= 0:
        return 0.0 # No effective fuel remaining

    # Convert speed from knots to km/h (1 knot = 1.852 km/h)
    speed_km_per_hour = speed_knots * 1.852
    
    # Convert fuel consumption from tons/day to tons/hour
    fuel_consumption_t_per_hour = fuel_consumption_t_per_day / 24.0

    if fuel_consumption_t_per_hour <= 0:
        return float('inf') # Avoid division by zero, implies infinite range if no consumption

    # Calculate total hours of travel possible
    total_hours_possible = fuel_remaining_t / fuel_consumption_t_per_hour

    # Calculate maximum distance in km
    max_range_km = total_hours_possible * speed_km_per_hour
    return max_range_km


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
            
            point_conditions = []
            all_weather_data_for_edge = [] # Collect all weather data found along the edge
            for (lat, lon) in points:
                try:
                    cond = get_wave_wind(lat, lon, time="latest")
                except Exception as e:
                    cond = {"error": str(e)}
                point_conditions.append({
                    "lat": lat, "lon": lon, **cond
                })
                
                # Check if this point is within a special weather location
                weather_for_point, location_name = get_weather_for_point(lat, lon)
                if weather_for_point:
                    all_weather_data_for_edge.append(weather_for_point)
                    print(f"Weather data found for {location_name} at ({lat}, {lon}) along edge between {row_i['Ports']} and {row_j['Ports']}.")
            
            # Determine the worst-case weather penalty for the edge
            worst_weather_penalty = 0.0
            if all_weather_data_for_edge:
                # Calculate penalty for each collected weather data and take the maximum
                penalties = [get_weather_penalty(wd) for wd in all_weather_data_for_edge]
                worst_weather_penalty = max(penalties)
                print(f"Worst-case weather penalty for edge between {row_i['Ports']} and {row_j['Ports']}: {worst_weather_penalty:.2f}")

            edge_data = {
                "points": points,
                "distance_km": round(geodesic(
                    (row_i["lat"], row_i["lon"]),
                    (row_j["lat"], row_j["lon"])
                ).km, 1),
                "conditions": point_conditions,
                "worst_weather_penalty": worst_weather_penalty # Store the worst-case penalty
            }
            
            edges_dict[(row_i["Ports"], row_j["Ports"])] = edge_data

import pprint
pprint.pprint(edges_dict)
    

# Add edges with computed weights (distance + sea state)
for (u, v), data in edges_dict.items():
    weight = compute_edge_weight(data, vessel)  # Pass vessel data
    G.add_edge(u, v, weight=weight, **data)
            
for u, v, data in G.edges(data=True):
    print(f"{u} -> {v}: distance={data['distance_km']} km, weight={data['weight']:.2f}")
    
pos = nx.get_node_attributes(G, 'pos')

# Choose a departure port
departure = selected_ports.iloc[0]["Ports"]

# Check for reachable nodes from departure
reachable_edges = [edge for edge in G.edges(data=True) if edge[2]['weight'] < 1000000.0] # Filter out highly penalized (unreachable) edges

if not reachable_edges:
    print(f"Alert: No reachable nodes from {departure} with current fuel and weather conditions.")
else:
    # Create a subgraph with only reachable edges for TSP calculation
    G_reachable = nx.Graph()
    for u, v, data in reachable_edges:
        G_reachable.add_edge(u, v, weight=data['weight'])
    
    if not G_reachable.nodes:
        print(f"Alert: No reachable nodes from {departure} with current fuel and weather conditions.")
    else:
        # Identify and warn about individual unreachable ports
        reachable_nodes_set = set(G_reachable.nodes)
        all_selected_nodes_set = set(selected_ports["Ports"].tolist())
        unreachable_ports = all_selected_nodes_set - reachable_nodes_set

        if unreachable_ports:
            print(f"Warning: The following ports are unreachable from {departure} with current fuel and weather conditions: {', '.join(unreachable_ports)}")

        # Case A: return to departure (cycle)
        try:
            cycle = approx.traveling_salesman_problem(G_reachable, weight="weight", cycle=True)
            cycle = enforce_tsp_constraints(cycle, departure, cycle=True)
            print("Optimal cycle route:", cycle)
        except nx.NetworkXPointlessConcept:
            print(f"Alert: No feasible cycle route from {departure} with current fuel and weather conditions.")
        except Exception as e:
            print(f"Error generating cycle route: {e}")

        # Case B: do not return (path)
        try:
            path = approx.traveling_salesman_problem(G_reachable, weight="weight", cycle=False)
            path = enforce_tsp_constraints(path, departure, cycle=False)
            print("Optimal path route:", path)
        except nx.NetworkXPointlessConcept:
            print(f"Alert: No feasible path route from {departure} with current fuel and weather conditions.")
        except Exception as e:
            print(f"Error generating path route: {e}")

# Visualize the routes
plt.figure(figsize=(10, 8))
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10, font_color='black', edge_color='gray')
plt.title("Port Network Graph with Routes")
for route in [cycle, path]:
    route_edges = [(route[i], route[i + 1]) for i in range(len(route) - 1)]
    nx.draw_networkx_edges(G, pos, edgelist=route_edges, edge_color='red', width=2)
    
plt.show()

