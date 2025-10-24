from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware # Import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Tuple
import uuid
import torch
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from shapely.geometry import Point, Polygon, LineString

# Import DRL components and utilities
from training_codes.DRL_train import Actor, DDPG, MarineEnv
from utils import (
    vessel, get_optimal_path_route_for_api, generate_multi_leg_astar_path_and_landmarks,
    create_bathymetry_grid, create_weather_penalty_grid, lat_lon_to_grid_coords,
    grid_coords_to_lat_lon, find_closest_sea_node, create_astar_grid, haversine,
    generate_landmark_points # Ensure this is imported if needed separately
)
from d_star_lite import find_d_star_lite_route # Import the D* Lite function

app = FastAPI()

# Configure CORS
origins = ["*"] # Temporarily allow all origins for debugging

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global dictionary to store simulation states for single-user demonstration
# In a production environment, this would be replaced by a persistent store like Redis
simulations: Dict[str, Dict[str, Any]] = {}

# Load ports data once at startup
ports_df = pd.read_excel("data\\Ports\\ports.xlsx")
ports_df[["lat", "lon"]] = ports_df["Decimal"].str.split(",", expand=True).astype(float)
ports_gdf = gpd.GeoDataFrame(
    ports_df,
    geometry=gpd.points_from_xy(ports_df["lon"], ports_df["lat"]),
    crs="EPSG:4326"
)

# --- Pydantic Models for Request/Response Bodies ---

class PointModel(BaseModel):
    lat: float
    lon: float
    name: str = "" # Add a name field, default to empty string

class VesselConfig(BaseModel):
    speed_knots: float = vessel["speed_knots"]
    fuel_consumption_t_per_day: float = vessel["fuel_consumption_t_per_day"]
    fuel_tank_capacity_t: float = vessel["fuel_tank_capacity_t"]
    safety_fuel_margin_t: float = vessel["safety_fuel_margin_t"]

class SuggestRouteSequenceRequest(BaseModel):
    start_point: PointModel # This PointModel should now include the name
    destinations: List[PointModel] # These PointModels should now include names
    is_cycle_route: bool = False
    vessel_config: VesselConfig

class SuggestedRouteSequenceResponse(BaseModel):
    session_id: str
    suggested_sequence: List[PointModel]
    unreachable_destinations: List[Dict[str, Any]] # Includes lat, lon, reason

class InitializeMultiLegSimulationRequest(BaseModel):
    session_id: str # From the suggest_route_sequence response
    start_point: PointModel
    sequenced_destinations: List[PointModel]
    initial_speed: float
    initial_heading: float
    vessel_config: VesselConfig

class InitializeMultiLegSimulationResponse(BaseModel):
    session_id: str
    message: str
    initial_vessel_state: Dict[str, float]
    full_astar_path: List[List[float]]
    landmark_points: List[List[float]]

class GetNextActionRequest(BaseModel):
    session_id: str
    current_lat: float
    current_lon: float
    current_speed: float
    current_heading: float

class GetNextActionResponse(BaseModel):
    new_speed: float
    new_heading: float
    current_landmark_idx: int
    current_leg_idx: int
    total_fuel_consumed: float
    total_emissions: float
    total_eta_hours: float
    done: bool
    message: str = "OK"

class ObstaclePolygonModel(BaseModel):
    points: List[PointModel]

class DStarLiteRouteRequest(BaseModel):
    session_id: str
    obstacle_polygon: ObstaclePolygonModel

class DStarLiteRouteResponse(BaseModel):
    session_id: str
    d_star_lite_path: List[PointModel]
    message: str = "OK"

# --- API Endpoints ---

@app.post("/api/suggest_route_sequence", response_model=SuggestedRouteSequenceResponse)
async def suggest_route_sequence(request: SuggestRouteSequenceRequest):
    """
    Suggests an optimal sequence of destinations using TSP and identifies unreachable ones.
    """
    try:
        # Create a vessel dictionary from the provided config
        current_vessel = {
            "speed_knots": request.vessel_config.speed_knots,
            "fuel_consumption_t_per_day": request.vessel_config.fuel_consumption_t_per_day,
            "fuel_tank_capacity_t": request.vessel_config.fuel_tank_capacity_t,
            "safety_fuel_margin_t": request.vessel_config.safety_fuel_margin_t,
            "fuel_tank_remaining_t": request.vessel_config.fuel_tank_capacity_t # Assume full tank for initial check
        }

        # Store original PointModel objects with names for later lookup
        original_points_with_names = [request.start_point] + request.destinations
        start_point_latlon = (request.start_point.lat, request.start_point.lon)
        destination_points_latlon = [(d.lat, d.lon) for d in request.destinations]

        # For TSP, we need bathymetry and weather grids, but these are static for the region
        # We'll use default region parameters for now, or load them from a config if available
        min_lon_region, max_lon_region = 99, 120
        min_lat_region, max_lat_region = 0, 8
        cell_size_m = 10000 # 10km resolution

        ds_bathymetry = xr.open_dataset("data\\Bathymetry\\GEBCO_2025_sub_ice.nc")
        ds_subset_astar = ds_bathymetry.sel(
            lon=slice(min_lon_region, max_lon_region),
            lat=slice(min_lat_region, max_lat_region)
        )
        
        
        # Use the global 'vessel' for static properties like size and percent_of_height_underwater
        vessel_height_underwater_calc = vessel["size"][2] * vessel["percent_of_height_underwater"]


        bathymetry_maze, grid_lats, grid_lons, lat_step, lon_step, _ = create_bathymetry_grid(
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

        # Call the modified get_optimal_path_route
        suggested_sequence_latlon, _, unreachable_destinations_raw = get_optimal_path_route_for_api(
            current_vessel, start_point_latlon, destination_points_latlon,
            bathymetry_maze, grid_params, weather_penalty_grid, request.is_cycle_route
        )

        if suggested_sequence_latlon is None:
            raise HTTPException(status_code=500, detail="Failed to generate route sequence.")

        # Convert unreachable_destinations_raw to a more user-friendly format
        unreachable_formatted = []
        for dest in unreachable_destinations_raw:
            # dest['name'] is like "Dest_X.XX,Y.YY" or "Start_X.XX,Y.YY"
            # We only care about actual destinations, not the start point if it was marked unreachable
            if dest['name'].startswith("Dest_"):
                unreachable_formatted.append({
                    "lat": dest['lat'],
                    "lon": dest['lon'],
                    "reason": dest['reason']
                })
        
        # Store initial data for the session, even if only sequence is generated
        session_id = str(uuid.uuid4())
        simulations[session_id] = {
            "vessel_config": request.vessel_config.dict(),
            "start_point": request.start_point, # Store the full PointModel with name
            "destinations": request.destinations, # Store full PointModels with names
            "is_cycle_route": request.is_cycle_route,
            "bathymetry_maze": bathymetry_maze,
            "grid_params": grid_params,
            "weather_penalty_grid": weather_penalty_grid,
            "sequenced_destinations_latlon": suggested_sequence_latlon, # Store the suggested sequence as lat/lon tuples
            "full_astar_path_latlon": None, # Will be filled by next API
            "landmark_points": None, # Will be filled by next API
            "marine_env": None,
            "ddpg_agent": None,
            "current_leg_idx": 0,
            "current_landmark_idx": 0,
            "original_points_with_names": original_points_with_names # Store for lookup
        }

        # Reconstruct suggested_sequence with names
        suggested_sequence_with_names: List[PointModel] = []
        for lat, lon in suggested_sequence_latlon:
            # Find the matching original PointModel by lat/lon
            matched_point = next(
                (p for p in original_points_with_names if p.lat == lat and p.lon == lon),
                None
            )
            if matched_point:
                suggested_sequence_with_names.append(matched_point)
            else:
                # Fallback for points not directly matching (e.g., intermediate points if any)
                suggested_sequence_with_names.append(PointModel(lat=lat, lon=lon, name=f"Lat {lat:.4f}, Lon {lon:.4f}"))


        return SuggestedRouteSequenceResponse(
            session_id=session_id,
            suggested_sequence=suggested_sequence_with_names,
            unreachable_destinations=unreachable_formatted
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/api/initialize_multi_leg_simulation", response_model=InitializeMultiLegSimulationResponse)
async def initialize_multi_leg_simulation(request: InitializeMultiLegSimulationRequest):
    """
    Initializes a multi-leg simulation based on a user-defined sequence of destinations,
    generating the full A* path and landmarks for the entire route.
    """
    try:
        session_data = simulations.get(request.session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session ID not found. Please call /api/suggest_route_sequence first.")

        # Use vessel config from request, or from session if not provided (for consistency)
        current_vessel_config = request.vessel_config.dict()
        
        # Create a mutable vessel dictionary for the environment
        initial_vessel_state_for_env = {
            **vessel, # Start with global vessel defaults for static properties
            **current_vessel_config, # Override with user-provided config
            "fuel_tank_remaining_t": current_vessel_config["fuel_tank_capacity_t"] # Start with full fuel
        }
        # Set initial position, speed, heading
        initial_vessel_state_for_env["current_position_latlon"] = (request.start_point.lat, request.start_point.lon)
        initial_vessel_state_for_env["current_speed_knots"] = request.initial_speed
        initial_vessel_state_for_env["current_heading_deg"] = request.initial_heading


        # Retrieve environment parameters from session
        bathymetry_maze = session_data["bathymetry_maze"]
        grid_params = session_data["grid_params"]
        weather_penalty_grid = session_data["weather_penalty_grid"]

        # Combine start point with sequenced destinations for A* path generation
        full_sequenced_points = [(request.start_point.lat, request.start_point.lon)] + \
                                [(d.lat, d.lon) for d in request.sequenced_destinations]

        # Generate full A* path and landmarks for the entire multi-leg route
        full_astar_path_latlon, landmark_points, _, _, _ = generate_multi_leg_astar_path_and_landmarks(
            initial_vessel_state_for_env, # Pass the vessel data for height calculation etc.
            full_sequenced_points,
            min_lon_region=grid_params[1], max_lon_region=grid_params[1] + grid_params[4] * grid_params[3],
            min_lat_region=grid_params[0], max_lat_region=grid_params[0] + grid_params[5] * grid_params[2],
            cell_size_m=10000, # Consistent with initial grid generation
            landmark_interval_km=20 # Consistent with DRL_train.py
        )

        if full_astar_path_latlon is None or not landmark_points:
            raise HTTPException(status_code=500, detail="Failed to generate A* path or landmarks for the multi-leg route.")

        # Initialize DRL environment
        sequence_length = 3 # Consistent with DRL_train.py
        marine_env = MarineEnv(
            initial_vessel_state_for_env, 
            full_astar_path_latlon, 
            landmark_points, 
            bathymetry_maze, 
            grid_params, 
            weather_penalty_grid, 
            sequence_length=sequence_length
        )
        
        # Load trained DDPG agent
        state_dim = marine_env.state_dim
        action_dim = marine_env.action_dim
        max_action = 1.0

        ddpg_agent = DDPG(state_dim, action_dim, max_action, sequence_length=sequence_length)
        try:
            ddpg_agent.actor.load_state_dict(torch.load("models/ddpg_actor.pth", map_location=ddpg_agent.device))
            print("✓ Trained DDPG actor model loaded successfully for session", request.session_id)
        except FileNotFoundError:
            raise HTTPException(status_code=500, detail="DRL model (ddpg_actor.pth) not found. Please ensure it's trained and available.")

        # Update session data with initialized environment and agent
        session_data["full_astar_path_latlon"] = full_astar_path_latlon
        session_data["landmark_points"] = landmark_points
        session_data["marine_env"] = marine_env
        session_data["ddpg_agent"] = ddpg_agent
        session_data["current_leg_idx"] = 0 # Reset for new simulation
        session_data["current_landmark_idx"] = 0 # Reset for new simulation

        return InitializeMultiLegSimulationResponse(
            session_id=request.session_id,
            message="Multi-leg simulation initialized successfully",
            initial_vessel_state={
                "lat": request.start_point.lat,
                "lon": request.start_point.lon,
                "speed": request.initial_speed,
                "heading": request.initial_heading
            },
            full_astar_path=full_astar_path_latlon,
            landmark_points=landmark_points
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during simulation initialization: {str(e)}")

@app.post("/api/get_next_action", response_model=GetNextActionResponse)
async def get_next_action(request: GetNextActionRequest):
    """
    Receives the current vessel state from the frontend and returns the DRL agent's
    recommended next speed and heading.
    """
    try:
        session_data = simulations.get(request.session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session ID not found. Please initialize a simulation first.")

        marine_env: MarineEnv = session_data["marine_env"]
        ddpg_agent: DDPG = session_data["ddpg_agent"]

        if marine_env is None or ddpg_agent is None:
            raise HTTPException(status_code=500, detail="DRL environment or agent not initialized for this session.")

        # Update the MarineEnv with the current state from the frontend
        marine_env.update_vessel_state(
            request.current_lat,
            request.current_lon,
            request.current_speed,
            request.current_heading
        )

        # Get the current state sequence for the DRL agent
        current_state_sequence = marine_env._get_state()

        # Get action from the trained DRL agent
        action = ddpg_agent.select_action(current_state_sequence)

        # Simulate one step in the environment with the agent's action
        # For API demonstration, we might not introduce random obstacles
        next_state, reward, done, info = marine_env.step(action, obstacle_present=False, episode=0)

        # Extract new speed and heading from the environment's vessel state
        new_speed = marine_env.current_speed_knots
        new_heading = marine_env.current_heading_deg

        # Update session's current landmark and leg index
        session_data["current_landmark_idx"] = marine_env.current_landmark_idx
        # The MarineEnv itself manages current_landmark_idx.
        # For multi-leg, we need to determine current_leg_idx based on landmarks.
        # This logic might need to be more sophisticated if landmarks are not strictly sequential per leg.
        # For now, we'll assume current_leg_idx advances when the last landmark of a leg is reached.
        
        # Simple logic for current_leg_idx (can be refined based on how landmarks are structured per leg)
        # Assuming landmarks are sequential across all legs
        total_landmarks = len(marine_env.landmark_points)
        current_landmark_idx = marine_env.current_landmark_idx
        
        # If all landmarks are reached, the simulation is done
        if current_landmark_idx >= total_landmarks:
            current_leg_idx = len(session_data["sequenced_destinations"]) - 1 # Last leg
            done = True
        else:
            # Determine which leg the current landmark belongs to
            # This requires knowing which landmarks correspond to which leg.
            # For simplicity, let's assume each leg has roughly equal number of landmarks,
            # or we need to store leg-specific landmark ranges in session_data.
            # For now, we'll just use the overall landmark index.
            current_leg_idx = 0 # Placeholder, needs more robust logic if actual leg tracking is required
            # A more robust approach would involve storing the landmark ranges for each leg
            # e.g., session_data["leg_landmark_ranges"] = [(0, 5), (5, 12), ...]
            # Then:
            # for i, (start_lm, end_lm) in enumerate(session_data["leg_landmark_ranges"]):
            #     if start_lm <= current_landmark_idx < end_lm:
            #         current_leg_idx = i
            #         break
            # if current_landmark_idx >= session_data["leg_landmark_ranges"][-1][1]:
            #     current_leg_idx = len(session_data["leg_landmark_ranges"]) - 1

        return GetNextActionResponse(
            new_speed=float(new_speed),
            new_heading=float(new_heading),
            current_landmark_idx=int(current_landmark_idx),
            current_leg_idx=int(current_leg_idx),
            total_fuel_consumed=float(marine_env.total_fuel_consumed),
            total_emissions=float(marine_env.total_emissions),
            total_eta_hours=float(marine_env.total_eta_hours),
            done=bool(done),
            message="OK" if not done else "Simulation complete."
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during getting next action: {str(e)}")

@app.post("/api/d_star_lite_route", response_model=DStarLiteRouteResponse)
async def get_d_star_lite_route(request: DStarLiteRouteRequest):
    """
    Calculates a D* Lite route avoiding a dynamic obstacle (polygon).
    If the obstacle intersects the A* path, it finds two nearest landmark points
    and uses them as start and end for D* Lite.
    """
    try:
        session_data = simulations.get(request.session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session ID not found. Please initialize a simulation first.")

        full_astar_path_latlon = session_data.get("full_astar_path_latlon")
        landmark_points = session_data.get("landmark_points")
        bathymetry_maze = session_data["bathymetry_maze"]
        grid_params = session_data["grid_params"]
        weather_penalty_grid = session_data["weather_penalty_grid"]

        if not full_astar_path_latlon or not landmark_points:
            raise HTTPException(status_code=400, detail="A* path or landmark points not found in session. Please initialize simulation.")

        # Create Shapely Polygon from obstacle points
        obstacle_coords = [(p.lon, p.lat) for p in request.obstacle_polygon.points]
        obstacle_polygon = Polygon(obstacle_coords)

        # Check for intersection with A* path
        astar_line = LineString([(p[1], p[0]) for p in full_astar_path_latlon]) # (lon, lat) for shapely
        
        intersection_found = False
        intersection_point = None

        if astar_line.intersects(obstacle_polygon):
            intersection_found = True
            # Find the actual intersection geometry
            intersection_geometry = astar_line.intersection(obstacle_polygon)
            
            # For simplicity, let's take the centroid of the intersection as a reference point
            # If the intersection is a LineString, its centroid is a good representative.
            # If it's a MultiPoint or MultiLineString, the overall centroid works.
            if not intersection_geometry.is_empty:
                intersection_point = intersection_geometry.centroid
            else:
                # Fallback if centroid is empty (e.g., point intersection)
                # Find the closest point on the A* path to the obstacle
                min_dist = float('inf')
                closest_astar_point = None
                for astar_lat, astar_lon in full_astar_path_latlon:
                    p_astar = Point(astar_lon, astar_lat)
                    dist = p_astar.distance(obstacle_polygon)
                    if dist < min_dist:
                        min_dist = dist
                        closest_astar_point = (astar_lat, astar_lon)
                if closest_astar_point:
                    intersection_point = Point(closest_astar_point[1], closest_astar_point[0])


        d_star_lite_path_latlon = []
        if intersection_found and intersection_point:
            # Find two nearest landmark points to the intersection_point
            # Convert intersection_point to (lat, lon)
            ref_lat, ref_lon = intersection_point.y, intersection_point.x

            distances = []
            for i, (lm_lat, lm_lon) in enumerate(landmark_points):
                dist = haversine(ref_lat, ref_lon, lm_lat, lm_lon)
                distances.append((dist, i, (lm_lat, lm_lon)))
            
            distances.sort()

            if len(distances) < 2:
                raise HTTPException(status_code=400, detail="Not enough landmark points to reroute.")

            # Select the two nearest landmark points
            # The first one is the start for D* Lite, the second is the end.
            # We need to ensure they are "around" the obstacle.
            # A more robust approach would be to find the closest landmark *before* the obstacle
            # and the closest landmark *after* the obstacle along the A* path.
            # For simplicity, let's pick the two closest for now.
            
            # Let's refine this: find the landmark closest to the start of the intersection
            # and the landmark closest to the end of the intersection along the A* path.
            # This requires finding the segment of the A* path that intersects.

            # For now, a simpler approach: find the two closest landmarks to the intersection point.
            # This might not always yield a path around the obstacle if the landmarks are on the same side.
            # A better approach would be to find the closest landmark *before* the intersection
            # and the closest landmark *after* the intersection along the original A* path.

            # Let's try to find the closest landmark *before* the intersection and *after* the intersection.
            # This requires knowing the index of the intersection point on the A* path.
            
            # Find the index of the A* path point closest to the intersection point
            min_dist_to_astar_path = float('inf')
            closest_astar_path_idx = -1
            for i, (path_lat, path_lon) in enumerate(full_astar_path_latlon):
                dist = haversine(ref_lat, ref_lon, path_lat, path_lon)
                if dist < min_dist_to_astar_path:
                    min_dist_to_astar_path = dist
                    closest_astar_path_idx = i
            
            if closest_astar_path_idx == -1:
                raise HTTPException(status_code=500, detail="Could not find closest A* path point to intersection.")

            # Find the closest landmark before and after the closest_astar_path_idx
            start_lm_for_dstar = None
            end_lm_for_dstar = None
            
            # Search backwards for the first landmark not in the obstacle
            for i in range(closest_astar_path_idx, -1, -1):
                path_point = full_astar_path_latlon[i]
                p_point = Point(path_point[1], path_point[0])
                if not p_point.intersects(obstacle_polygon):
                    # Find the closest landmark to this point
                    min_lm_dist = float('inf')
                    for lm_lat, lm_lon in landmark_points:
                        dist = haversine(path_point[0], path_point[1], lm_lat, lm_lon)
                        if dist < min_lm_dist:
                            min_lm_dist = dist
                            start_lm_for_dstar = (lm_lat, lm_lon)
                    break
            
            # Search forwards for the first landmark not in the obstacle
            for i in range(closest_astar_path_idx, len(full_astar_path_latlon)):
                path_point = full_astar_path_latlon[i]
                p_point = Point(path_point[1], path_point[0])
                if not p_point.intersects(obstacle_polygon):
                    # Find the closest landmark to this point
                    min_lm_dist = float('inf')
                    for lm_lat, lm_lon in landmark_points:
                        dist = haversine(path_point[0], path_point[1], lm_lat, lm_lon)
                        if dist < min_lm_dist:
                            min_lm_dist = dist
                            end_lm_for_dstar = (lm_lat, lm_lon)
                    break

            if not start_lm_for_dstar or not end_lm_for_dstar:
                raise HTTPException(status_code=400, detail="Could not find suitable landmark points for D* Lite rerouting.")

            # Create a temporary grid with the obstacle
            temp_bathymetry_maze = np.copy(bathymetry_maze)
            min_lat, min_lon, lat_step, lon_step, num_lat_cells, num_lon_cells = grid_params

            # Mark obstacle cells in the temporary grid
            # Iterate over a bounding box around the obstacle for efficiency
            min_o_lon, min_o_lat, max_o_lon, max_o_lat = obstacle_polygon.bounds
            
            # Convert bounds to grid coordinates
            min_r, min_c = lat_lon_to_grid_coords(max_o_lat, min_o_lon, min_lat, min_lon, lat_step, lon_step, num_lat_cells, num_lon_cells)
            max_r, max_c = lat_lon_to_grid_coords(min_o_lat, max_o_lon, min_lat, min_lon, lat_step, lon_step, num_lat_cells, num_lon_cells)

            # Ensure coordinates are within grid bounds
            min_r, min_c = max(0, min_r), max(0, min_c)
            max_r, max_c = min(num_lat_cells - 1, max_r), min(num_lon_cells - 1, max_c)

            for r in range(min_r, max_r + 1):
                for c in range(min_c, max_c + 1):
                    cell_lat, cell_lon = grid_coords_to_lat_lon(r, c, min_lat, min_lon, lat_step, lon_step)
                    cell_point = Point(cell_lon, cell_lat)
                    if obstacle_polygon.contains(cell_point) or obstacle_polygon.touches(cell_point):
                        temp_bathymetry_maze[r][c] = 1 # Mark as obstacle

            # Run D* Lite
            d_star_lite_path_latlon = find_d_star_lite_route(
                temp_bathymetry_maze,
                start_lm_for_dstar,
                end_lm_for_dstar,
                min_lat, min_lon, lat_step, lon_step, num_lat_cells, num_lon_cells,
                weather_penalty_grid=weather_penalty_grid
            )
            if not d_star_lite_path_latlon:
                raise HTTPException(status_code=500, detail="D* Lite failed to find a reroute path.")

        return DStarLiteRouteResponse(
            session_id=request.session_id,
            d_star_lite_path=[PointModel(lat=p[0], lon=p[1]) for p in d_star_lite_path_latlon],
            message="D* Lite reroute calculated successfully." if intersection_found else "No intersection found, no reroute needed."
        )

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during D* Lite route calculation: {str(e)}")
