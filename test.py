import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point, Polygon
import networkx as nx
from networkx.algorithms import approximation as approx
from geopy.distance import geodesic
import requests
from bs4 import BeautifulSoup
import re
import heapq
from math import radians, sin, cos, sqrt, atan2
import matplotlib.pyplot as pyplot
from d_star_lite import find_d_star_lite_route, create_bathymetry_grid, lat_lon_to_grid_coords, grid_coords_to_lat_lon

vessel = {
    "speed_knots": 20.0,
    "fuel_consumption_t_per_day": 20.0,  
    "fuel_tank_capacity_t": 1000.0,
    "safety_fuel_margin_t": 10.0,
    "fuel_tank_remaining_t": 1000.0, 
    "size": (1000, 500, 60),  
    "percent_of_height_underwater": 0.3,
}

# Calculate vessel height underwater for bathymetry checks
vessel_height_underwater = vessel["size"][2] * vessel["percent_of_height_underwater"]

def safe_float(val, var_name=None):
    """Convert to float safely, return np.nan if invalid."""
    try:
        if val is None:
            return 0.0
        if isinstance(val, np.ndarray) and val.size == 1:
            fval = float(val.item())
        elif isinstance(val, (int, float, np.number)):
            fval = float(val)
        else:
            return 0.0
            
        if np.isnan(fval) or np.isinf(fval):
            return 0.0
        if var_name == "wind_wave_period" and fval > 100.0:
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
            "wind_wave_period": safe_float(point["wper"].isel(time=idx).values, var_name="wind_wave_period"),
            "wind_wave_direction": safe_float(point["wdir"].isel(time=idx).values),
        }
    except Exception as e:
        print(f"Error fetching wave/wind data for ({lat}, {lon}): {e}")
        return {
            "wave_height": 0.0, "wave_period": 0.0, "wave_direction": 0.0,
            "wind_wave_height": 0.0, "wind_wave_period": 0.0, "wind_wave_direction": 0.0,
            "error": str(e)
        }
    
def interpolate_points(lat1, lon1, lat2, lon2, n=5):
    """Generate n points between two coordinates"""
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
            "wave_height": 10.0, "wave_period": 5.0,
            "wind_wave_height": 8.0, "wind_wave_period": 5.0,
        }
    
    if normalizers is None:
        normalizers = {
            "wave_height": 5.0, "wave_period": 20.0,
            "wind_wave_height": 5.0, "wind_wave_period": 20.0,
        }
    
    distance = edge_data["distance_km"]
    environmental_penalty = 0.0
    weather_penalty = 0.0

    for var, w in weights.items():
        values = [c.get(var, 0.0) for c in edge_data["conditions"]]
        avg_val = np.nanmean(values) if values else 0.0

        norm = normalizers.get(var, 1.0)
        environmental_penalty += w * (avg_val / norm)
    
    if "worst_weather_penalty" in edge_data:
        weather_penalty = edge_data["worst_weather_penalty"]

    max_range_km = calculate_vessel_range_km(vessel_data)
    fuel_penalty = 0.0
    if distance > max_range_km:
        fuel_penalty = 1000000.0
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
    "Northern_Straits_of_Malacca": [
        {"lat": 5.9500, "lon": 98.0250}, {"lat": 5.9500, "lon": 100.9000},
        {"lat": 3.0000, "lon": 101.4000}, {"lat": 3.0000, "lon": 99.0000}
    ],
    "Southern_Straits_of_Malacca": [
        {"lat": 3.0000, "lon": 100.8000}, {"lat": 3.0000, "lon": 103.3000},
        {"lat": 1.2667, "lon": 103.5167}, {"lat": 1.1667, "lon": 103.3917},
        {"lat": 1.1000, "lon": 102.9667}
    ],
    "Tioman_Island": [
        {"lat": 2.9000, "lon": 104.0500}, {"lat": 2.9000, "lon": 104.3000},
        {"lat": 2.7000, "lon": 104.3000}, {"lat": 2.7000, "lon": 104.0500}
    ],
    "Bunguran_Island": [
        {"lat": 4.2000, "lon": 107.8000}, {"lat": 4.2000, "lon": 108.6000},
        {"lat": 3.6000, "lon": 108.6000}, {"lat": 3.6000, "lon": 107.8000}
    ],
    "Layang_Layang_Island": [
        {"lat": 7.3900, "lon": 113.8200}, {"lat": 7.3900, "lon": 113.8700},
        {"lat": 7.3500, "lon": 113.8700}, {"lat": 7.3500, "lon": 113.8200}
    ],
    "Labuan": [
        {"lat": 5.3600, "lon": 115.1600}, {"lat": 5.3600, "lon": 115.3600},
        {"lat": 5.2200, "lon": 115.3600}, {"lat": 5.2200, "lon": 115.1600}
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

location_weather_cache = {}

def is_point_in_polygon(lat, lon, polygon_points):
    """
    Check if a (lat, lon) point is inside a polygon defined by a list of points.
    """
    if not polygon_points:
        return False
    point = Point(lon, lat)
    polygon_coords = [(p["lon"], p["lat"]) for p in polygon_points]
    polygon = Polygon(polygon_coords)
    return polygon.contains(point)

def scrape_weather_data(url):
    """
    Scrapes weather data
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        weather_data = {}
        
        # find all tables with class 'table table-hover'
        tables = soup.find_all('table', class_='table table-hover')
        
        for i, table in enumerate(tables):
            table_name = f"Table_{i+1}"
            current_table_data = []
            
            headers = [th.get_text(strip=True) for th in table.find('thead').find_all('th')]
            
            # get table rows
            for row in table.find('tbody').find_all('tr'):
                row_data = [td.get_text(strip=True) for td in row.find_all('td')]
                
                row_dict = {}
                for j, header in enumerate(headers):
                    if j < len(row_data):
                        row_dict[header] = row_data[j]
                    else:
                        row_dict[header] = None 
                
                if 'Forecast' in row_dict and row_dict['Forecast']:
                    forecast_text = row_dict['Forecast']
                    # Example: "08:00 AM - 08:00 PM: Fair. 08:00 PM - 08:00 AM: Fair."
                    forecast_periods = re.findall(r'(\d{2}:\d{2}\s[APM]{2}\s-\s\d{2}:\d{2}\s[APM]{2}):\s(.*?)(?=\s\d{2}:\d{2}\s[APM]{2}\s-\s\d{2}:\d{2}\s[APM]{2}:|\s*$)', forecast_text)
                    
                    parsed_forecast = {}
                    for period, weather in forecast_periods:
                        parsed_forecast[period.strip()] = {"Weather": weather.strip()}
                    row_dict['Forecast'] = parsed_forecast
                
                current_table_data.append(row_dict)
            weather_data[table_name] = current_table_data
            
        return weather_data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL {url}: {e}")
        return None
    except Exception as e:
        print(f"Error scraping weather data from {url}: {e}")
        return None

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
                if weather:
                    location_weather_cache[location_name] = weather 
                return weather, location_name
    return None, None

def get_weather_penalty(weather_data):
    """
    Assigns a penalty based on weather conditions.
    """
    penalty = 0.0
    if not weather_data:
        return penalty

    for table_key in weather_data:
        for entry in weather_data[table_key]:
            if 'Forecast' in entry and isinstance(entry['Forecast'], dict):
                for period, details in entry['Forecast'].items():
                    if 'Weather' in details and details['Weather']:
                        weather_desc = details['Weather'].lower()
                        if "thunderstorm" in weather_desc or "isolated thunderstorms" in weather_desc:
                            penalty += 50.0
                        elif "rain" in weather_desc or "showers" in weather_desc:
                            penalty += 20.0
                        elif "cloudy" in weather_desc:
                            penalty += 5.0
                        elif "fair" in weather_desc or "no rain" in weather_desc or "sunny" in weather_desc:
                            penalty += 0.0
                        else:
                            penalty += 10.0
    return penalty

def calculate_vessel_range_km(vessel_data):
    """
    Calculate the maximum distance the vessel can travel in km with current fuel.
    """
    speed_knots = vessel_data["speed_knots"]
    fuel_consumption_t_per_day = vessel_data["fuel_consumption_t_per_day"]
    fuel_remaining_t = vessel_data["fuel_tank_remaining_t"] - vessel_data["safety_fuel_margin_t"]

    if fuel_remaining_t <= 0:
        return 0.0

    speed_km_per_hour = speed_knots * 1.852
    fuel_consumption_t_per_hour = fuel_consumption_t_per_day / 24.0

    if fuel_consumption_t_per_hour <= 0:
        return float('inf')

    total_hours_possible = fuel_remaining_t / fuel_consumption_t_per_hour
    max_range_km = total_hours_possible * speed_km_per_hour
    return max_range_km

def simulate_vessel_route(vessel_data, G, route):
    """
    Simulate the vessel's journey along a given route, including fuel consumption and refueling.
    """
    current_vessel_state = vessel_data.copy()
    journey_log = []
    
    print("\n--- Simulating Vessel Route ---")
    print(f"Starting fuel: {current_vessel_state['fuel_tank_remaining_t']:.2f} tons")

    for i in range(len(route) - 1):
        start_port = route[i]
        end_port = route[i+1]
        
        if G.has_edge(start_port, end_port):
            edge_data = G[start_port][end_port]
        elif G.has_edge(end_port, start_port):
            edge_data = G[end_port][start_port]
        else:
            print(f"Error: No edge found between {start_port} and {end_port}. Skipping segment.")
            continue

        distance_km = edge_data["distance_km"]
        
        speed_knots = current_vessel_state["speed_knots"]
        fuel_consumption_t_per_day = current_vessel_state["fuel_consumption_t_per_day"]
        
        speed_km_per_hour = speed_knots * 1.852
        travel_time_hours = distance_km / speed_km_per_hour
        fuel_consumed_t = (fuel_consumption_t_per_day / 24.0) * travel_time_hours
        
        current_vessel_state["fuel_tank_remaining_t"] -= fuel_consumed_t
        
        log_entry = {
            "segment": f"{start_port} to {end_port}",
            "distance_km": distance_km,
            "fuel_consumed_t": fuel_consumed_t,
            "fuel_remaining_before_refuel_t": current_vessel_state["fuel_tank_remaining_t"]
        }
        
        if i > 0:
            current_vessel_state["fuel_tank_remaining_t"] = current_vessel_state["fuel_tank_capacity_t"]
            log_entry["refueled"] = True
            log_entry["fuel_remaining_after_refuel_t"] = current_vessel_state["fuel_tank_remaining_t"]
        else:
            log_entry["refueled"] = False
            log_entry["fuel_remaining_after_refuel_t"] = current_vessel_state["fuel_tank_remaining_t"]

        journey_log.append(log_entry)
        print(f"  Segment {start_port} -> {end_port}: Distance={distance_km:.1f} km, Fuel Consumed={fuel_consumed_t:.2f} t, Fuel Remaining={current_vessel_state['fuel_tank_remaining_t']:.2f} t {'(Refueled)' if i > 0 else ''}")

    print("--- Simulation Complete ---")
    return current_vessel_state, journey_log

def get_optimal_path_route(vessel_data, selected_ports):
    """
    Generates an optimal path route (list of port names) using TSP approximation.
    """
    
    G = nx.Graph()

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
                all_weather_data_for_edge = []
                for (lat, lon) in points:
                    try:
                        cond = get_wave_wind(lat, lon, time="latest")
                    except Exception as e:
                        cond = {"error": str(e)}
                    point_conditions.append({
                        "lat": lat, "lon": lon, **cond
                    })
                    
                    weather_for_point, location_name = get_weather_for_point(lat, lon)
                    if weather_for_point:
                        all_weather_data_for_edge.append(weather_for_point)
                
                worst_weather_penalty = 0.0
                if all_weather_data_for_edge:
                    penalties = [get_weather_penalty(wd) for wd in all_weather_data_for_edge]
                    worst_weather_penalty = max(penalties)

                edge_data = {
                    "points": points,
                    "distance_km": round(geodesic(
                        (row_i["lat"], row_i["lon"]),
                        (row_j["lat"], row_j["lon"])
                    ).km, 1),
                    "conditions": point_conditions,
                    "worst_weather_penalty": worst_weather_penalty
                }
                
                edges_dict[(row_i["Ports"], row_j["Ports"])] = edge_data
        
    for (u, v), data in edges_dict.items():
        weight = compute_edge_weight(data, vessel_data)
        G.add_edge(u, v, weight=weight, **data)
                
    departure = selected_ports.iloc[0]["Ports"]
    reachable_edges = [edge for edge in G.edges(data=True) if edge[2]['weight'] < 1000000.0]

    if not reachable_edges:
        print(f"Alert: No reachable nodes from {departure} with current fuel and weather conditions.")
        return None, None
    else:
        G_reachable = nx.Graph()
        for u, v, data in reachable_edges:
            G_reachable.add_edge(u, v, weight=data['weight'])
        
        if not G_reachable.nodes:
            print(f"Alert: No reachable nodes from {departure} with current fuel and weather conditions.")
            return None, None
        else:
            all_selected_nodes_set = set(selected_ports["Ports"].tolist())
            reachable_nodes_set = set(G_reachable.nodes)
            unreachable_ports = all_selected_nodes_set - reachable_nodes_set
            if unreachable_ports:
                print(f"Warning: The following ports are unreachable from {departure} with current fuel and weather conditions: {', '.join(unreachable_ports)}")

            path = None
            try:
                path = approx.traveling_salesman_problem(G_reachable, weight="weight", cycle=False)
                path = enforce_tsp_constraints(path, departure, cycle=False)
            except nx.NetworkXPointlessConcept:
                print(f"Alert: No feasible path route from {departure} with current fuel and weather conditions.")
            except Exception as e:
                print(f"Error generating path route: {e}")
            
            return path, G, selected_ports 

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance

def generate_landmark_points(path_latlon, interval_km=100):
    """
    Generates landmark points along a given path at specified intervals.
    """
    if not path_latlon:
        return []

    landmark_points = [path_latlon[0]] # Start with the first point
    cumulative_distance = 0.0

    for i in range(len(path_latlon) - 1):
        p1 = path_latlon[i]
        p2 = path_latlon[i+1]
        segment_distance = geodesic(p1, p2).km
        
        # Check if adding this segment crosses a landmark interval
        while cumulative_distance + segment_distance >= interval_km:
            remaining_distance_to_landmark = interval_km - cumulative_distance
            
            # Calculate the fraction of the current segment needed to reach the landmark
            fraction = remaining_distance_to_landmark / segment_distance
            
            # Interpolate the point
            # Note: geodesic expects (lat, lon) tuples
            lat_interp = p1[0] + fraction * (p2[0] - p1[0])
            lon_interp = p1[1] + fraction * (p2[1] - p1[1])
            
            landmark_points.append((lat_interp, lon_interp))
            cumulative_distance += remaining_distance_to_landmark
            segment_distance -= remaining_distance_to_landmark
            
            # Reset cumulative_distance for the next interval
            if cumulative_distance >= interval_km:
                cumulative_distance = 0.0 # Start new interval from this landmark
                
        cumulative_distance += segment_distance
        
    return landmark_points

def create_weather_penalty_grid(min_lon, max_lon, min_lat, max_lat, lat_step, lon_step, num_lat_cells, num_lon_cells):
    """
    Creates a grid with weather penalties for each cell.
    """
    weather_penalty_grid = np.zeros((num_lat_cells, num_lon_cells), dtype=float)

    for location_name, weather_data in location_weather_cache.items():
        penalty = get_weather_penalty(weather_data)
        if penalty > 0:
            polygon_points = locations_data.get(location_name)
            if polygon_points:
                polygon_coords = [(p["lon"], p["lat"]) for p in polygon_points]
                location_polygon = Polygon(polygon_coords)

                for r in range(num_lat_cells):
                    for c in range(num_lon_cells):
                        cell_lat = min_lat + r * lat_step + lat_step / 2
                        cell_lon = min_lon + c * lon_step + lon_step / 2
                        
                        if location_polygon.contains(Point(cell_lon, cell_lat)):
                            weather_penalty_grid[r, c] = max(weather_penalty_grid[r, c], penalty) # Take max penalty if areas overlap
    return weather_penalty_grid.tolist()

class Node:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

    def __lt__(self, other):
        return self.f < other.f

    def __hash__(self):
        return hash(self.position)

def astar(maze, start, end, weather_penalty_grid=None):
    start_node = Node(start)
    end_node = Node(end)

    open_list = []
    open_list_positions = {start_node.position: start_node}
    closed_list_positions = set()

    heapq.heappush(open_list, (start_node.f, start_node))

    while open_list:
        f_cost, current_node = heapq.heappop(open_list)

        if current_node.position in closed_list_positions:
            continue

        closed_list_positions.add(current_node.position)

        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1]

        (x, y) = current_node.position
        neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]

        for next_pos in neighbors:
            (nx, ny) = next_pos

            if not (0 <= nx < len(maze) and 0 <= ny < len(maze[0])):
                continue

            # Check for bathymetry obstacles
            if maze[nx][ny] != 0:
                continue

            # Calculate movement cost, including weather penalty
            move_cost = 1 
            if weather_penalty_grid and weather_penalty_grid[nx][ny] > 0:
                move_cost += weather_penalty_grid[nx][ny] # Add weather penalty

            new_g = current_node.g + move_cost
            new_h = abs(nx - end_node.position[0]) + abs(ny - end_node.position[1])
            new_f = new_g + new_h

            if next_pos in open_list_positions and new_g >= open_list_positions[next_pos].g:
                continue

            new_node = Node(next_pos, current_node)
            new_node.g = new_g
            new_node.h = new_h
            new_node.f = new_f

            heapq.heappush(open_list, (new_node.f, new_node))
            open_list_positions[next_pos] = new_node

    return None

def create_bathymetry_grid(ds_subset, min_lon, max_lon, min_lat, max_lat, cell_size_m=500, vessel_height=0):
    """
    Creates a grid (maze) from bathymetry data.
    """
    center_lat = (min_lat + max_lat) / 2
    lat_deg_per_m = 1 / haversine(center_lat, 0, center_lat + 0.001, 0) * 0.001
    lon_deg_per_m = 1 / haversine(center_lat, 0, center_lat, 0.001) * 0.001

    lat_step = cell_size_m * lat_deg_per_m
    lon_step = cell_size_m * lon_deg_per_m

    num_lat_cells = int(np.ceil((max_lat - min_lat) / lat_step))
    num_lon_cells = int(np.ceil((max_lon - min_lon) / lon_step))

    grid = np.zeros((num_lat_cells, num_lon_cells), dtype=int)
    elevation_data = np.zeros((num_lat_cells, num_lon_cells), dtype=float)
    grid_lats = np.linspace(min_lat, max_lat, num_lat_cells)
    grid_lons = np.linspace(min_lon, max_lon, num_lon_cells)

    for r in range(num_lat_cells):
        for c in range(num_lon_cells):
            cell_lat = grid_lats[r] + lat_step / 2
            cell_lon = grid_lons[c] + lon_step / 2

            try:
                elevation = ds_subset['elevation'].sel(lat=cell_lat, lon=cell_lon, method='nearest').item()
            except KeyError:
                elevation = 0

            elevation_data[r, c] = elevation

            if elevation >= -vessel_height:
                grid[r, c] = 1 
            else:
                grid[r, c] = 0 

    elevation_da = xr.DataArray(
        elevation_data,
        coords={'lat': grid_lats, 'lon': grid_lons},
        dims=['lat', 'lon']
    )
    return grid.tolist(), grid_lats, grid_lons, lat_step, lon_step, elevation_da

def lat_lon_to_grid_coords(lat, lon, min_lat, min_lon, lat_step, lon_step, num_lat_cells, num_lon_cells):
    """Converts latitude and longitude to grid (row, col) coordinates."""
    row = int((lat - min_lat) / lat_step)
    col = int((lon - min_lon) / lon_step)
    row = max(0, min(row, num_lat_cells - 1))
    col = max(0, min(col, num_lon_cells - 1))
    return row, col

def grid_coords_to_lat_lon(row, col, min_lat, min_lon, lat_step, lon_step):
    """Converts grid (row, col) coordinates to approximate latitude and longitude (center of cell)."""
    lat = min_lat + row * lat_step + lat_step / 2
    lon = min_lon + col * lon_step + lon_step / 2
    return lat, lon

if __name__ == "__main__":
    # Load ports data
    ports_df = pd.read_excel("data\\Ports\\ports.xlsx")
    ports_df[["lat", "lon"]] = ports_df["Decimal"].str.split(",", expand=True).astype(float)
    ports_gdf = gpd.GeoDataFrame(
        ports_df,
        geometry=gpd.points_from_xy(ports_df["lon"], ports_df["lat"]),
        crs="EPSG:4326"
    )

    print("Generating optimal path route...")
    selected_ports = ports_gdf.sample(4, random_state=50) # Randomly select 4 ports
    optimal_path_route, G_main, selected_ports_main = get_optimal_path_route(vessel, selected_ports)

    if optimal_path_route:
        print("\nOptimal path route:", optimal_path_route)
        
        # Load bathymetry data
        ds_bathymetry = xr.open_dataset("data\\Bathymetry\\GEBCO_2025_sub_ice.nc")

        # Define the region of interest (Malaysia)
        min_lon_astar, max_lon_astar = 99, 120
        min_lat_astar, max_lat_astar = 0, 8

        ds_subset_astar = ds_bathymetry.sel(
            lon=slice(min_lon_astar, max_lon_astar),
            lat=slice(min_lat_astar, max_lat_astar)
        )

        cell_size_m_astar = 5000 # 5km x 5km grid cells

        all_astar_paths_latlon = []
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        color_index = 0

        bathymetry_maze, grid_lats, grid_lons, lat_step, lon_step, elevation_data = create_bathymetry_grid(
            ds_subset_astar, min_lon_astar, max_lon_astar, min_lat_astar, max_lat_astar, cell_size_m_astar, vessel_height_underwater
        )
        num_lat_cells = len(grid_lats)
        num_lon_cells = len(grid_lons)

        weather_penalty_grid = create_weather_penalty_grid(
            min_lon_astar, max_lon_astar, min_lat_astar, max_lat_astar, lat_step, lon_step, num_lat_cells, num_lon_cells
        )

        cmap_maze = pyplot.get_cmap('Greys_r', 2)
        ax.imshow(bathymetry_maze, extent=[min_lon_astar, max_lon_astar, min_lat_astar, max_lat_astar],
                       origin='lower', cmap=cmap_maze, interpolation='nearest', alpha=0.7)
        
        weather_penalty_np = np.array(weather_penalty_grid)
        masked_weather_penalty = np.ma.masked_where(weather_penalty_np == 0, weather_penalty_np)
        
        cmap_weather = pyplot.get_cmap('YlOrRd', 5) # Yellow-Orange-Red for penalties
        ax.imshow(masked_weather_penalty, extent=[min_lon_astar, max_lon_astar, min_lat_astar, max_lat_astar],
                       origin='lower', cmap=cmap_weather, interpolation='nearest', alpha=0.5)

        ax.scatter(selected_ports_main["lon"], selected_ports_main["lat"], color='purple', s=150, marker='^', label="Selected Ports (TSP)")
        for idx, row in selected_ports_main.iterrows():
            ax.text(row["lon"] + 0.1, row["lat"] + 0.1, row["Ports"], fontsize=9, color='purple')

        for i in range(len(optimal_path_route) - 1):
            start_port_name = optimal_path_route[i]
            end_port_name = optimal_path_route[i+1]

            start_port_coords = ports_gdf[ports_gdf["Ports"] == start_port_name].iloc[0]
            end_port_coords = ports_gdf[ports_gdf["Ports"] == end_port_name].iloc[0]

            start_latlon = (start_port_coords["lat"], start_port_coords["lon"])
            end_latlon = (end_port_coords["lat"], end_port_coords["lon"])

            print(f"\nGenerating A* path for segment: {start_port_name} (({start_latlon[0]:.7f}, {start_latlon[1]:.7f})) to {end_port_name} (({end_latlon[0]:.7f}, {end_latlon[1]:.7f}))")

            start_grid = lat_lon_to_grid_coords(start_latlon[0], start_latlon[1], min_lat_astar, min_lon_astar, lat_step, lon_step, num_lat_cells, num_lon_cells)
            end_grid = lat_lon_to_grid_coords(end_latlon[0], end_latlon[1], min_lat_astar, min_lon_astar, lat_step, lon_step, num_lat_cells, num_lon_cells)

            search_radius = 10
            
            if bathymetry_maze[start_grid[0]][start_grid[1]] == 1:
                print(f"Start point {start_grid} is an obstacle. Adjusting.")
                found_new_start = False
                for dr in range(-search_radius, search_radius + 1):
                    for dc in range(-search_radius, search_radius + 1):
                        if dr == 0 and dc == 0: continue
                        new_r, new_c = start_grid[0] + dr, start_grid[1] + dc
                        if 0 <= new_r < num_lat_cells and 0 <= new_c < num_lon_cells and bathymetry_maze[new_r][new_c] == 0:
                            start_grid = (new_r, new_c)
                            print(f"New start grid for segment: {start_grid}")
                            found_new_start = True
                            break
                    if found_new_start: break
                if not found_new_start:
                    print(f"Could not find walkable start point for {start_port_name}. Skipping segment.")
                    continue

            if bathymetry_maze[end_grid[0]][end_grid[1]] == 1:
                print(f"End point {end_grid} is an obstacle. Adjusting.")
                found_new_end = False
                for dr in range(-search_radius, search_radius + 1):
                    for dc in range(-search_radius, search_radius + 1):
                        if dr == 0 and dc == 0: continue
                        new_r, new_c = end_grid[0] + dr, end_grid[1] + dc
                        if 0 <= new_r < num_lat_cells and 0 <= new_c < num_lon_cells and bathymetry_maze[new_r][new_c] == 0:
                            end_grid = (new_r, new_c)
                            print(f"New end grid for segment: {end_grid}")
                            found_new_end = True
                            break
                    if found_new_end: break
                if not found_new_end:
                    print(f"Could not find walkable end point for {end_port_name}. Skipping segment.")
                    continue

            path_grid = astar(bathymetry_maze, start_grid, end_grid, weather_penalty_grid)

            if not path_grid:
                print(f"No A* path found for segment: {start_port_name} to {end_port_name} with weather penalties. Trying without weather penalties as fallback.")
                path_grid = astar(bathymetry_maze, start_grid, end_grid, weather_penalty_grid=None)
                if path_grid:
                    print(f"Fallback path found for segment: {start_port_name} to {end_port_name} (ignoring weather penalties).")
                else:
                    print(f"No A* path found even with fallback for segment: {start_port_name} to {end_port_name}. Skipping segment.")
                    continue

            path_latlon = [grid_coords_to_lat_lon(r, c, min_lat_astar, min_lon_astar, lat_step, lon_step) for r, c in path_grid]
            all_astar_paths_latlon.extend(path_latlon)

            path_lons = [p[1] for p in path_latlon]
            path_lats = [p[0] for p in path_latlon]
            current_color = colors[color_index % len(colors)]
            ax.plot(path_lons, path_lats, color=current_color, linewidth=2, alpha=0.8, label=f"A* Path: {start_port_name}-{end_port_name}")
            ax.scatter(start_latlon[1], start_latlon[0], color='green', s=80, marker='o')
            ax.scatter(end_latlon[1], end_latlon[0], color='red', s=80, marker='X')
            color_index += 1
        
        # Generate and plot landmark points
        landmark_points = generate_landmark_points(all_astar_paths_latlon, interval_km=50)
        if landmark_points:
            landmark_lons = [p[1] for p in landmark_points]
            landmark_lats = [p[0] for p in landmark_points]
            ax.scatter(landmark_lons, landmark_lats, color='cyan', s=100, marker='*', edgecolor='black', label="Landmark Points (100km)")
            print(f"\nGenerated {len(landmark_points)} landmark points:")
            for lp in landmark_points:
                print(f"  Lat: {lp[0]:.4f}, Lon: {lp[1]:.4f}")

        ax.set_xlim(min_lon_astar, max_lon_astar)
        ax.set_ylim(min_lat_astar, max_lat_astar)
        plt.title("Combined Optimal TSP Route with A* Paths and Landmark Points on Bathymetry Grid")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = list(dict.fromkeys(labels)) # Remove duplicate labels
        unique_handles = [handles[labels.index(l)] for l in unique_labels]
        ax.legend(unique_handles, unique_labels)
        plt.grid(True, linestyle=':', alpha=0.6)
        
        # --- Simulate Obstacle and Perform D* Lite Rerouting ---
        if len(landmark_points) >= 2:
            # Choose two consecutive landmark points for rerouting
            reroute_start_latlon = landmark_points[0]
            reroute_end_latlon = landmark_points[1] # Or any other two consecutive points

            print(f"\nSimulating obstacle and performing D* Lite rerouting between landmark points:")
            print(f"  Start: Lat={reroute_start_latlon[0]:.4f}, Lon={reroute_start_latlon[1]:.4f}")
            print(f"  End: Lat={reroute_end_latlon[0]:.4f}, Lon={reroute_end_latlon[1]:.4f}")

            # Define an obstacle in grid coordinates
            # For demonstration, let's place an obstacle near the middle of the segment
            mid_lat = (reroute_start_latlon[0] + reroute_end_latlon[0]) / 2
            mid_lon = (reroute_start_latlon[1] + reroute_end_latlon[1]) / 2
            
            obstacle_grid_coords = lat_lon_to_grid_coords(mid_lat, mid_lon, min_lat_astar, min_lon_astar, lat_step, lon_step, num_lat_cells, num_lon_cells)
            
            # Create a copy of the maze to modify for D* Lite
            dstar_maze = [row[:] for row in bathymetry_maze]
            
            # Define obstacle size in grid cells (20km / 5km_per_cell = 4 cells)
            obstacle_grid_size = 4 
            
            # Calculate the top-left corner of the obstacle block
            ox, oy = obstacle_grid_coords
            obstacle_start_row = max(0, ox - obstacle_grid_size // 2)
            obstacle_end_row = min(num_lat_cells, ox + obstacle_grid_size // 2)
            obstacle_start_col = max(0, oy - obstacle_grid_size // 2)
            obstacle_end_col = min(num_lon_cells, oy + obstacle_grid_size // 2)

            # Mark the obstacle in the D* Lite maze
            actual_obstacle_cells = []
            for r in range(obstacle_start_row, obstacle_end_row):
                for c in range(obstacle_start_col, obstacle_end_col):
                    if 0 <= r < num_lat_cells and 0 <= c < num_lon_cells:
                        dstar_maze[r][c] = 1 # Mark as obstacle
                        actual_obstacle_cells.append((r, c))
                        
            if actual_obstacle_cells:
                print(f"  Obstacle simulated as a {obstacle_grid_size}x{obstacle_grid_size} block around grid coords: {obstacle_grid_coords}")
                # Visualize the obstacle block
                obstacle_lats = [grid_coords_to_lat_lon(r, c, min_lat_astar, min_lon_astar, lat_step, lon_step)[0] for r, c in actual_obstacle_cells]
                obstacle_lons = [grid_coords_to_lat_lon(r, c, min_lat_astar, min_lon_astar, lat_step, lon_step)[1] for r, c in actual_obstacle_cells]
                ax.scatter(obstacle_lons, obstacle_lats, color='black', s=50, marker='s', label="Simulated Obstacle (20km x 20km)")
            else:
                print(f"  Warning: Obstacle coordinates {obstacle_grid_coords} are out of grid bounds or no cells could be marked. Not placing obstacle.")
                obstacle_grid_coords = None # Don't pass invalid obstacle

            rerouted_path_latlon = find_d_star_lite_route(
                dstar_maze, 
                reroute_start_latlon, 
                reroute_end_latlon, 
                min_lat_astar, min_lon_astar, 
                lat_step, lon_step, 
                num_lat_cells, num_lon_cells, 
                weather_penalty_grid,
                obstacle_coords=obstacle_grid_coords # Pass the obstacle to D* Lite
            )

            if rerouted_path_latlon:
                rerouted_lons = [p[1] for p in rerouted_path_latlon]
                rerouted_lats = [p[0] for p in rerouted_path_latlon]
                ax.plot(rerouted_lons, rerouted_lats, color='magenta', linewidth=3, linestyle='--', label="D* Lite Rerouted Path")
                print(f"D* Lite rerouted path found with {len(rerouted_path_latlon)} points.")
            else:
                print("No D* Lite rerouted path found.")
        else:
            print("Not enough landmark points to simulate D* Lite rerouting.")

        plt.show()

    else:
        print("Could not generate an optimal path route.")