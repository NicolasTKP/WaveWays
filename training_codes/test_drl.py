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
import torch
import random

# Import common utilities from utils.py
from utils import (
    vessel, vessel_height_underwater, safe_float, get_wave_wind,
    interpolate_points, reorder_route, compute_edge_weight, enforce_tsp_constraints,
    locations_data, location_codes, location_weather_cache,
    is_point_in_polygon, scrape_weather_data, get_weather_for_point,
    get_weather_penalty, calculate_vessel_range_km, simulate_vessel_route,
    get_optimal_path_route, haversine, generate_landmark_points,
    create_weather_penalty_grid, Node, astar, create_bathymetry_grid,
    lat_lon_to_grid_coords, grid_coords_to_lat_lon,
    generate_optimized_route_and_landmarks # Added this import
)

# Import D* Lite specific function
from d_star_lite import find_d_star_lite_route

# Import DRL components - IMPORTANT: Import after other modules to avoid circular imports
try:
    from DRL_train import Actor, DDPG, MarineEnv, plot_simulation_episode # Added plot_simulation_episode
except ImportError as e:
    print(f"Warning: Could not import DRL components: {e}")
    print("If you haven't trained the model yet, please run DRL_train.py first.")
    DDPG = None
    MarineEnv = None
    plot_simulation_episode = None # Set to None if not imported

if __name__ == "__main__":
    # Load ports data
    ports_df = pd.read_excel("..\\data\\Ports\\ports.xlsx")
    ports_df[["lat", "lon"]] = ports_df["Decimal"].str.split(",", expand=True).astype(float)
    ports_gdf = gpd.GeoDataFrame(
        ports_df,
        geometry=gpd.points_from_xy(ports_df["lon"], ports_df["lat"]),
        crs="EPSG:4326"
    )

    # Use the same random state for consistency
    random_state_value = 50

    print("Generating optimized route and landmarks (consistent with DRL_train.py)...")
    full_astar_path_latlon, landmark_points, bathymetry_maze, grid_params, weather_penalty_grid = \
        generate_optimized_route_and_landmarks(vessel, num_sample_ports=3, random_state=random_state_value, 
                                               cell_size_m=10000, landmark_interval_km=100)

    if full_astar_path_latlon is None:
        print("Failed to generate optimized route and landmarks. Exiting.")
        plt.show()
        exit()

    print("Generating optimal path route...")
    # ALIGNED: Use 3 ports for consistency with training
    selected_ports = ports_gdf.sample(3, random_state=random_state_value)
    optimal_path_route, G_main, selected_ports_main = get_optimal_path_route(
        vessel, selected_ports, bathymetry_maze, grid_params, weather_penalty_grid
    )

    if optimal_path_route:
        print("\nOptimal path route:", optimal_path_route)
        
        
        # Extract grid dimensions from grid_params for plotting
        min_lat_astar, min_lon_astar, lat_step, lon_step, num_lat_cells, num_lon_cells = grid_params
        max_lon_astar = min_lon_astar + num_lon_cells * lon_step
        max_lat_astar = min_lat_astar + num_lat_cells * lat_step

        # Check if DRL components are available
        if DDPG is None or MarineEnv is None or plot_simulation_episode is None:
            print("✗ DRL components not available")
            print("  Skipping DRL simulation. Run DRL_train.py first to train the agent.")
            plt.show()
            exit()
        
        # Define sequence length for LSTM (consistent with DRL_train.py)
        sequence_length = 4

        # Initialize DRL environment
        env = MarineEnv(vessel.copy(), full_astar_path_latlon, landmark_points, 
                       bathymetry_maze, grid_params, weather_penalty_grid, 
                       sequence_length=sequence_length)
        
        # Load trained agent
        state_dim = env.state_dim
        action_dim = env.action_dim
        max_action = 1.0

        agent = DDPG(state_dim, action_dim, max_action, sequence_length=sequence_length)
        
        try:
            # Corrected path to load the model from the parent directory
            agent.actor.load_state_dict(torch.load("../models/ddpg_actor.pth", map_location=agent.device))
            print("✓ Trained DDPG actor model loaded successfully")
        except FileNotFoundError:
            print("✗ Error: ddpg_actor.pth not found at expected path '../models/ddpg_actor.pth'")
            print("  Please run DRL_train.py first to train the agent")
            plt.show()
            exit()

        # Run simulation
        print("\nRunning DRL simulation...")
        # The env.full_episode_drl_path will be populated by the env.step calls
        
        current_position = env.reset()
        done = False
        step_count = 0
        max_steps = 500  # Increased for full route completion

        while not done and step_count < max_steps:
            # Get action from trained agent
            action = agent.select_action(current_position)
            
            # Simulate obstacle (1% chance, aligned with training)
            obstacle_present = random.random() < 0.01
            
            next_state, reward, done, info = env.step(action, obstacle_present, episode=0) # Pass episode=0 for simulation
            
            current_position = next_state
            step_count += 1
            
            # Progress update every 50 steps
            if step_count % 50 == 0:
                print(f"  Step {step_count}: Landmark {env.current_landmark_idx}/{len(landmark_points)}, "
                      f"Fuel: {env.vessel['fuel_tank_remaining_t']:.1f}t")

        # Simulation results
        print("\n" + "="*60)
        print("SIMULATION RESULTS")
        print("="*60)
        print(f"Total steps: {step_count}")
        print(f"Landmarks reached: {env.current_landmark_idx}/{len(landmark_points)}")
        print(f"Final fuel: {env.vessel['fuel_tank_remaining_t']:.2f} t")
        print(f"Total fuel consumed: {env.total_fuel_consumed:.2f} t")
        print(f"Total emissions: {env.total_emissions:.2f} tCO2")
        print(f"Total ETA: {env.total_eta_hours:.2f} hours ({env.total_eta_hours/24:.2f} days)")
        print(f"Obstacles encountered: {len(env.active_obstacles)}") # Use env.active_obstacles
        print(f"D* Lite reroutes: {len(env.rerouted_paths_history)}") # Use env.rerouted_paths_history
        print("="*60 + "\n")

        # Use the plot_simulation_episode function from DRL_train.py for consistent visualization
        plot_simulation_episode(
            episode=0, # Use 0 for a single simulation run
            env=env,
            full_astar_path_latlon=full_astar_path_latlon,
            landmark_points=landmark_points,
            bathymetry_maze=bathymetry_maze,
            grid_params=grid_params,
            weather_penalty_grid=weather_penalty_grid
        )

    else:
        print("Could not generate an optimal path route.")
