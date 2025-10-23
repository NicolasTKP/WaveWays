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
from collections import deque # Added import for deque
from math import radians, sin, cos, sqrt, atan2
import matplotlib.pyplot as pyplot

vessel = {
    "speed_knots": 20.0,
    "fuel_consumption_t_per_day": 20.0,  
    "fuel_tank_capacity_t": 1500.0,
    "safety_fuel_margin_t": 10.0,
    "fuel_tank_remaining_t": 1500.0, 
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

_wave_wind_cache = {} # Global cache for get_wave_wind

def get_wave_wind(lat, lon, time="latest"):
    """
    Fetch wave and wind data for given lat, lon, and time.
    Uses an in-memory cache to avoid redundant API calls.
    """
    cache_key = (round(lat, 3), round(lon, 3), time)
    if cache_key in _wave_wind_cache:
        return _wave_wind_cache[cache_key]

    url = "https://pae-paha.pacioos.hawaii.edu/erddap/griddap/ww3_global"
    try:
        ds = xr.open_dataset(url, decode_times=True, decode_timedelta=True)

        point = ds.sel(longitude=lon, latitude=lat, method="nearest")

        if time == "latest":
            idx = -1
        else:
            idx = dict(time=np.datetime64(time))

        result = {
            "wave_height": safe_float(point["Thgt"].isel(time=idx).values),
            "wave_period": safe_float(point["Tper"].isel(time=idx).values),
            "wave_direction": safe_float(point["Tdir"].isel(time=idx).values),
            "wind_wave_height": safe_float(point["whgt"].isel(time=idx).values),
            "wind_wave_period": safe_float(point["wper"].isel(time=idx).values, var_name="wind_wave_period"),
            "wind_wave_direction": safe_float(point["wdir"].isel(time=idx).values),
        }
        _wave_wind_cache[cache_key] = result
        return result
    except Exception as e:
        print(f"Error fetching wave/wind data for ({lat}, {lon}): {e}")
        error_result = {
            "wave_height": 0.0, "wave_period": 0.0, "wave_direction": 0.0,
            "wind_wave_height": 0.0, "wind_wave_period": 0.0, "wind_wave_direction": 0.0,
            "error": str(e)
        }
        _wave_wind_cache[cache_key] = error_result # Cache error results too
        return error_result
    
_weather_for_point_cache = {} # Global cache for get_weather_for_point

def get_weather_for_point(lat, lon):
    """
    Checks if a point is within a predefined location and fetches weather data if it is.
    Uses an in-memory cache to avoid redundant API calls.
    Returns weather data and the location name if found, otherwise None, None.
    """
    cache_key = (round(lat, 3), round(lon, 3))
    if cache_key in _weather_for_point_cache:
        return _weather_for_point_cache[cache_key]

    for location_name, points in locations_data.items():
        if is_point_in_polygon(lat, lon, points):
            location_code = location_codes.get(location_name)
            if location_code:
                url = f"https://www.met.gov.my/en/forecast/marine/shipping/{location_code}/"
                # Removed verbose print statement
                weather = scrape_weather_data(url)
                
                result = (weather, location_name)
                _weather_for_point_cache[cache_key] = result
                return result
    
    _weather_for_point_cache[cache_key] = (None, None) # Cache negative result
    return None, None

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

def get_optimal_path_route(vessel_data, selected_ports, bathymetry_maze, grid_params, weather_penalty_grid):
    """
    Generates an optimal path route (list of port names) using TSP approximation for sequencing,
    then uses A* for each segment to find navigable paths avoiding obstacles and bad weather.
    """
    
    # Step 1: TSP for Sequencing Only (using straight-line distances)
    G_tsp_sequencing = nx.Graph()
    for idx, row in selected_ports.iterrows():
        G_tsp_sequencing.add_node(row["Ports"], pos=(row["lon"], row["lat"]), lat=row["lat"], lon=row["lon"])

    for i, row_i in selected_ports.iterrows():
        for j, row_j in selected_ports.iterrows():
            if i < j:
                distance_km = round(geodesic(
                    (row_i["lat"], row_i["lon"]),
                    (row_j["lat"], row_j["lon"])
                ).km, 1)
                # Add edge with straight-line distance as weight for TSP sequencing
                G_tsp_sequencing.add_edge(row_i["Ports"], row_j["Ports"], weight=distance_km)
    
    departure = selected_ports.iloc[0]["Ports"]
    optimal_path_route_names = None
    try:
        # Run TSP approximation to get the sequence of ports
        optimal_path_route_names = approx.traveling_salesman_problem(G_tsp_sequencing, weight="weight", cycle=False)
        optimal_path_route_names = enforce_tsp_constraints(optimal_path_route_names, departure, cycle=False)
        print(f"TSP generated sequence: {optimal_path_route_names}")
    except nx.NetworkXPointlessConcept:
        print(f"Alert: No feasible TSP sequence from {departure}.")
        return None, None, None
    except Exception as e:
        print(f"Error generating TSP sequence: {e}")
        return None, None, None

    if not optimal_path_route_names:
        print("Could not generate an optimal path route sequence. Exiting.")
        return None, None, None

    # Step 2: A* Pathfinding for Each Segment in the TSP Sequence
    full_astar_path_latlon = []
    G_final_route = nx.DiGraph() # Use DiGraph to store actual A* paths and their properties

    for i in range(len(optimal_path_route_names) - 1):
        start_port_name = optimal_path_route_names[i]
        end_port_name = optimal_path_route_names[i+1]

        start_port_data = selected_ports[selected_ports["Ports"] == start_port_name].iloc[0]
        end_port_data = selected_ports[selected_ports["Ports"] == end_port_name].iloc[0]

        start_lat, start_lon = start_port_data["lat"], start_port_data["lon"]
        end_lat, end_lon = end_port_data["lat"], end_port_data["lon"]

        start_grid = lat_lon_to_grid_coords(start_lat, start_lon, *grid_params)
        end_grid = lat_lon_to_grid_coords(end_lat, end_lon, *grid_params)

        # Adjust start/end grid if they are on land
        if bathymetry_maze[start_grid[0]][start_grid[1]] != 0:
            start_grid = find_closest_sea_node(start_grid, bathymetry_maze)
            if bathymetry_maze[start_grid[0]][start_grid[1]] != 0:
                print(f"  Error: Could not find a valid sea node for start port {start_port_name}. A* will likely fail.")

        if bathymetry_maze[end_grid[0]][end_grid[1]] != 0:
            end_grid = find_closest_sea_node(end_grid, bathymetry_maze)
            if bathymetry_maze[end_grid[0]][end_grid[1]] != 0:
                print(f"  Error: Could not find a valid sea node for end port {end_port_name}. A* will likely fail.")

        path_grid = astar(bathymetry_maze, start_grid, end_grid, weather_penalty_grid)
        
        path_latlon_segment = []
        path_distance_km = 1000000.0 # Default to unreachable if no path found

        if path_grid:
            path_latlon_segment = [grid_coords_to_lat_lon(r, c, *grid_params[:4]) for r, c in path_grid]
            # Calculate actual distance of the A* path
            path_distance_km = 0.0
            for k in range(len(path_latlon_segment) - 1):
                path_distance_km += geodesic(path_latlon_segment[k], path_latlon_segment[k+1]).km
            full_astar_path_latlon.extend(path_latlon_segment)
        else:
            print(f"  No A* path found between {start_port_name} and {end_port_name} (land/weather). Assigning high penalty.")
            
        edge_data = {
            "distance_km": path_distance_km,
            "conditions": [], # Placeholder, actual conditions would be fetched here
            "worst_weather_penalty": 0.0, # Placeholder
            "path_latlon": path_latlon_segment
        }
        
        weight = compute_edge_weight(edge_data, vessel_data)
        G_final_route.add_edge(start_port_name, end_port_name, weight=weight, **edge_data)
                
    # Check if the overall route is feasible based on fuel
    total_route_distance = 0.0
    for i in range(len(optimal_path_route_names) - 1):
        start_port_name = optimal_path_route_names[i]
        end_port_name = optimal_path_route_names[i+1]
        if G_final_route.has_edge(start_port_name, end_port_name):
            total_route_distance += G_final_route[start_port_name][end_port_name]["distance_km"]
        else:
            print(f"Warning: Missing edge in G_final_route for {start_port_name} to {end_port_name}")
            total_route_distance += 1000000.0

    max_vessel_range = calculate_vessel_range_km(vessel_data)
    if total_route_distance > max_vessel_range:
        print(f"Warning: The total A* route distance ({total_route_distance:.1f} km) exceeds vessel range ({max_vessel_range:.1f} km). Fuel penalty will apply.")

    return optimal_path_route_names, G_final_route, selected_ports # selected_ports is not used in main.py/DRL_train.py after this call

def get_optimal_path_route_for_api(vessel_data, start_point_latlon, destination_points_latlon, 
                           bathymetry_maze, grid_params, weather_penalty_grid, cycle_route=False):
    """
    Generates an optimal path route (list of (lat, lon) tuples) using TSP approximation for sequencing,
    then uses A* for each segment to find navigable paths avoiding obstacles and bad weather.
    Also identifies unreachable destinations based on initial fuel range.
    This version is specifically for the API, accepting (lat, lon) tuples.
    """
    
    # Create a list of all points including start and destinations
    all_points = [start_point_latlon] + destination_points_latlon
    point_names = [f"Start_{start_point_latlon[0]:.2f},{start_point_latlon[1]:.2f}"] + \
                  [f"Dest_{p[0]:.2f},{p[1]:.2f}" for p in destination_points_latlon]
    
    # Map names to (lat, lon) for easy lookup
    name_to_coords = {name: point for name, point in zip(point_names, all_points)}

    # Step 1: TSP for Sequencing Only (using straight-line distances and reachability)
    G_tsp_sequencing = nx.Graph()
    unreachable_destinations = []
    max_vessel_range = calculate_vessel_range_km(vessel_data)

    for i in range(len(all_points)):
        G_tsp_sequencing.add_node(point_names[i], pos=(all_points[i][1], all_points[i][0]), 
                                  lat=all_points[i][0], lon=all_points[i][1])

    for i in range(len(all_points)):
        for j in range(i + 1, len(all_points)):
            p1_name = point_names[i]
            p2_name = point_names[j]
            p1_coords = all_points[i]
            p2_coords = all_points[j]

            distance_km = round(geodesic(p1_coords, p2_coords).km, 1)
            
            # Check reachability for straight-line distance
            if distance_km > max_vessel_range:
                # Mark as unreachable, but still add edge with high weight for TSP to avoid if possible
                G_tsp_sequencing.add_edge(p1_name, p2_name, weight=distance_km + 1000000.0) # High penalty
                if p2_name not in [d['name'] for d in unreachable_destinations]:
                    unreachable_destinations.append({"name": p2_name, "lat": p2_coords[0], "lon": p2_coords[1], "reason": "Exceeds vessel straight-line range"})
            else:
                G_tsp_sequencing.add_edge(p1_name, p2_name, weight=distance_km)
    
    departure_name = point_names[0]
    optimal_path_route_names = None
    try:
        # Run TSP approximation to get the sequence of points
        optimal_path_route_names = approx.traveling_salesman_problem(G_tsp_sequencing, weight="weight", cycle=cycle_route)
        optimal_path_route_names = enforce_tsp_constraints(optimal_path_route_names, departure_name, cycle=cycle_route)
        print(f"TSP generated sequence: {optimal_path_route_names}")
    except nx.NetworkXPointlessConcept:
        print(f"Alert: No feasible TSP sequence from {departure_name}.")
        return None, None, unreachable_destinations
    except Exception as e:
        print(f"Error generating TSP sequence: {e}")
        return None, None, unreachable_destinations

    if not optimal_path_route_names:
        print("Could not generate an optimal path route sequence. Exiting.")
        return None, None, unreachable_destinations
    
    # Convert optimal_path_route_names back to (lat, lon) tuples
    optimal_path_route_latlon = [(name_to_coords[name][0], name_to_coords[name][1]) for name in optimal_path_route_names]

    return optimal_path_route_latlon, G_tsp_sequencing, unreachable_destinations # Return G_tsp_sequencing for now, as G_final_route is not built here

def generate_multi_leg_astar_path_and_landmarks(vessel_data, sequenced_destination_points_latlon, 
                                                min_lon_region=99, max_lon_region=120, 
                                                min_lat_region=0, max_lat_region=8, 
                                                cell_size_m=10000, landmark_interval_km=20):
    """
    Generates a combined A* path and landmark points for a given sequence of (lat, lon) destinations.
    This function assumes the sequence is already determined (e.g., by TSP or user input).
    """
    if not sequenced_destination_points_latlon or len(sequenced_destination_points_latlon) < 2:
        print("Error: At least two points (start and end) are required for multi-leg A* path generation.")
        return None, None, None, None, None

    ds_bathymetry = xr.open_dataset("..\\data\\Bathymetry\\GEBCO_2025_sub_ice.nc")
    ds_subset_astar = ds_bathymetry.sel(
        lon=slice(min_lon_region, max_lon_region),
        lat=slice(min_lat_region, max_lat_region)
    )
    
    vessel_height_underwater_calc = vessel_data["size"][2] * vessel_data["percent_of_height_underwater"]
    bathymetry_maze, grid_lats, grid_lons, lat_step, lon_step, elevation_data = create_bathymetry_grid(
        ds_subset_astar, min_lon_region, max_lon_region, min_lat_region, max_lat_region, 
        cell_size_m, vessel_height_underwater_calc
    )
    num_lat_cells = len(grid_lats)
    num_lon_cells = len(grid_lons)
    grid_params = (min_lat_region, min_lon_region, lat_step, lon_step, num_lat_cells, num_lon_cells)

    weather_penalty_grid = create_weather_penalty_grid(
        min_lon_region, max_lon_region, min_lat_region, max_lat_region, 
        lat_step, lon_step, num_lat_cells, num_lon_cells
    )

    full_astar_path_latlon = []
    
    for i in range(len(sequenced_destination_points_latlon) - 1):
        start_point = sequenced_destination_points_latlon[i]
        end_point = sequenced_destination_points_latlon[i+1]

        start_lat, start_lon = start_point[0], start_point[1]
        end_lat, end_lon = end_point[0], end_point[1]

        start_grid = lat_lon_to_grid_coords(start_lat, start_lon, *grid_params)
        end_grid = lat_lon_to_grid_coords(end_lat, end_lon, *grid_params)

        # Adjust start/end grid if they are on land
        if bathymetry_maze[start_grid[0]][start_grid[1]] != 0:
            start_grid = find_closest_sea_node(start_grid, bathymetry_maze)
            if bathymetry_maze[start_grid[0]][start_grid[1]] != 0:
                print(f"  Error: Could not find a valid sea node for start point {start_point}. A* will likely fail.")

        if bathymetry_maze[end_grid[0]][end_grid[1]] != 0:
            end_grid = find_closest_sea_node(end_grid, bathymetry_maze)
            if bathymetry_maze[end_grid[0]][end_grid[1]] != 0:
                print(f"  Error: Could not find a valid sea node for end point {end_point}. A* will likely fail.")

        path_grid = astar(bathymetry_maze, start_grid, end_grid, weather_penalty_grid)
        
        if path_grid:
            path_latlon_segment = [grid_coords_to_lat_lon(r, c, *grid_params[:4]) for r, c in path_grid]
            full_astar_path_latlon.extend(path_latlon_segment)
        else:
            print(f"  No A* path found between {start_point} and {end_point} (land/weather). This segment will be skipped.")
            # Depending on desired behavior, you might want to raise an error or return None here
            # For now, we'll continue, but the path will be incomplete.

    if not full_astar_path_latlon:
        print("Error: No A* path could be generated for the entire sequence.")
        return None, None, None, None, None

    landmark_points = generate_landmark_points(full_astar_path_latlon, interval_km=landmark_interval_km)
    if not landmark_points:
        print("No landmark points generated. Exiting.")
        return None, None, None, None, None
    
    print(f"Generated {len(landmark_points)} landmarks at {landmark_interval_km}km intervals for multi-leg route")

    return full_astar_path_latlon, landmark_points, bathymetry_maze, grid_params, weather_penalty_grid


def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance


def generate_landmark_points(path_latlon, interval_km=20): # Reduced interval to 20km
    """
    Generates landmark points along a given path at specified intervals.
    Ensures at least the start and end points are included if the path is valid.
    """
    if not path_latlon:
        return []

    # Always include the first point
    landmark_points = [path_latlon[0]]
    distance_since_last_landmark = 0.0

    for i in range(len(path_latlon) - 1):
        p1 = path_latlon[i]
        p2 = path_latlon[i+1]
        segment_length = geodesic(p1, p2).km
        
        current_segment_covered = 0.0
        while distance_since_last_landmark + segment_length - current_segment_covered >= interval_km:
            remaining_to_next_interval = interval_km - distance_since_last_landmark
            
            # Calculate fraction along the current segment to place the landmark
            # Ensure denominator is not zero for very short segments
            if (segment_length - current_segment_covered) <= 0:
                break # Avoid division by zero, move to next segment
            
            fraction_of_segment = remaining_to_next_interval / (segment_length - current_segment_covered)
            
            # Interpolate the point
            lat_interp = p1[0] + (p2[0] - p1[0]) * (current_segment_covered + remaining_to_next_interval) / segment_length
            lon_interp = p1[1] + (p2[1] - p1[1]) * (current_segment_covered + remaining_to_next_interval) / segment_length
            
            landmark_points.append((lat_interp, lon_interp))
            
            current_segment_covered += remaining_to_next_interval
            distance_since_last_landmark = 0.0 # Reset for the new landmark
            
        distance_since_last_landmark += (segment_length - current_segment_covered)
    
    # Always include the last point if it's not already the first or last landmark
    if path_latlon[-1] not in landmark_points:
        landmark_points.append(path_latlon[-1])

    # Ensure at least two landmarks if the path has more than one point
    if len(path_latlon) > 1 and len(landmark_points) < 2:
        # If only one landmark was generated (the start point), add the end point
        if path_latlon[-1] != path_latlon[0]: # Only add if different
            landmark_points.append(path_latlon[-1])
        elif len(path_latlon) > 1: # If start and end are the same, but path has intermediate points
            # Add an intermediate point if available and different from start/end
            if len(path_latlon) > 2 and path_latlon[1] != path_latlon[0]:
                landmark_points.append(path_latlon[1])
            else: # Fallback: if path is very short, just duplicate the start point to avoid error
                landmark_points.append(path_latlon[0])


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

def find_closest_sea_node(land_grid_coords, bathymetry_maze):
    """
    Finds the closest non-land grid cell (sea node) to a given land grid cell.
    Uses a Breadth-First Search (BFS) to explore neighbors.
    """
    (start_row, start_col) = land_grid_coords
    rows, cols = len(bathymetry_maze), len(bathymetry_maze[0])
    
    # If the start node is already sea, return it
    if bathymetry_maze[start_row][start_col] == 0:
        return land_grid_coords

    queue = deque([(start_row, start_col)])
    visited = {(start_row, start_col)}

    while queue:
        r, c = queue.popleft()

        # Check neighbors
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]: # 4-directional movement
            nr, nc = r + dr, c + dc

            # Check bounds
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue

            if (nr, nc) not in visited:
                visited.add((nr, nc))
                # If neighbor is sea, return it
                if bathymetry_maze[nr][nc] == 0:
                    return (nr, nc)
                # If neighbor is land, add to queue to explore its neighbors
                queue.append((nr, nc))
    
    # Should ideally not happen if there's any sea around, but as a fallback
    print(f"Warning: No sea node found around {land_grid_coords}. This might indicate an isolated landmass or error.")
    return land_grid_coords # Return original if no sea found (will likely lead to A* failure)

def create_astar_grid(full_astar_path_latlon, grid_params):
    """
    Creates a grid representing the A* path.
    Cells on the A* path are marked with 1, others with 0.
    """
    min_lat, min_lon, lat_step, lon_step, num_lat_cells, num_lon_cells = grid_params
    astar_grid = np.zeros((num_lat_cells, num_lon_cells), dtype=int)

    for lat, lon in full_astar_path_latlon:
        row, col = lat_lon_to_grid_coords(lat, lon, min_lat, min_lon, lat_step, lon_step, num_lat_cells, num_lon_cells)
        astar_grid[row, col] = 1
    
    return astar_grid.tolist()

def generate_optimized_route_and_landmarks(vessel_data, num_sample_ports=3, random_state=50, 
                                           min_lon_region=99, max_lon_region=120, 
                                           min_lat_region=0, max_lat_region=8, 
                                           cell_size_m=10000, landmark_interval_km=20):
    """
    High-level function to generate an optimized route using TSP and A*,
    and then extract high-value landmark points.
    """
    ports_df = pd.read_excel("..\\data\\Ports\\ports.xlsx")
    ports_df[["lat", "lon"]] = ports_df["Decimal"].str.split(",", expand=True).astype(float)
    ports_gdf = gpd.GeoDataFrame(
        ports_df,
        geometry=gpd.points_from_xy(ports_df["lon"], ports_df["lat"]),
        crs="EPSG:4326"
    )

    print("Generating optimal path route...")
    selected_ports = ports_gdf.sample(num_sample_ports, random_state=random_state)
    
    ds_bathymetry = xr.open_dataset("..\\data\\Bathymetry\\GEBCO_2025_sub_ice.nc")
    ds_subset_astar = ds_bathymetry.sel(
        lon=slice(min_lon_region, max_lon_region),
        lat=slice(min_lat_region, max_lat_region)
    )
    
    vessel_height_underwater_calc = vessel_data["size"][2] * vessel_data["percent_of_height_underwater"]
    bathymetry_maze, grid_lats, grid_lons, lat_step, lon_step, elevation_data = create_bathymetry_grid(
        ds_subset_astar, min_lon_region, max_lon_region, min_lat_region, max_lat_region, 
        cell_size_m, vessel_height_underwater_calc
    )
    num_lat_cells = len(grid_lats)
    num_lon_cells = len(grid_lons)
    grid_params = (min_lat_region, min_lon_region, lat_step, lon_step, num_lat_cells, num_lon_cells)

    weather_penalty_grid = create_weather_penalty_grid(
        min_lon_region, max_lon_region, min_lat_region, max_lat_region, 
        lat_step, lon_step, num_lat_cells, num_lon_cells
    )

    # Call the original get_optimal_path_route which expects selected_ports
    optimal_path_route_names, G_main, selected_ports_main = get_optimal_path_route(
        vessel_data, selected_ports, bathymetry_maze, grid_params, weather_penalty_grid
    )

    if optimal_path_route_names is None: # Check for None explicitly
        print("Could not generate an optimal path route. Exiting.")
        return None, None, None, None, None

    full_astar_path_latlon = []
    if optimal_path_route_names and G_main:
        for i in range(len(optimal_path_route_names) - 1):
            start_port = optimal_path_route_names[i]
            end_port = optimal_path_route_names[i+1]
            if G_main.has_edge(start_port, end_port):
                path_segment = G_main[start_port][end_port].get("path_latlon", [])
                full_astar_path_latlon.extend(path_segment)
            elif G_main.has_edge(end_port, start_port):
                path_segment = G_main[end_port][start_port].get("path_latlon", [])
                full_astar_path_latlon.extend(path_segment[::-1])
    
    landmark_points = generate_landmark_points(full_astar_path_latlon, interval_km=landmark_interval_km)
    if not landmark_points:
        print("No landmark points generated. Exiting.")
        return None, None, None, None, None
    
    print(f"Generated {len(landmark_points)} landmarks at {landmark_interval_km}km intervals")

    return full_astar_path_latlon, landmark_points, bathymetry_maze, grid_params, weather_penalty_grid
