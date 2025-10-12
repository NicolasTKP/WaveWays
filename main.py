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
    lat_lon_to_grid_coords, grid_coords_to_lat_lon
)

# Import D* Lite specific function
from d_star_lite import find_d_star_lite_route

# Import DRL components - IMPORTANT: Import after other modules to avoid circular imports
try:
    from DRL_train import Actor, DDPG, MarineEnv
except ImportError as e:
    print(f"Warning: Could not import DRL components: {e}")
    print("If you haven't trained the model yet, please run DRL_train.py first.")
    DDPG = None
    MarineEnv = None

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
    # ALIGNED: Use 3 ports for consistency with training
    selected_ports = ports_gdf.sample(3, random_state=50)
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

        # ALIGNED: Use 10km grid cells (consistent with training)
        cell_size_m_astar = 10000  # 10km x 10km grid cells

        all_astar_paths_latlon = []
        
        fig, ax = plt.subplots(figsize=(14, 12))
        
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        color_index = 0

        print("Creating bathymetry grid...")
        bathymetry_maze, grid_lats, grid_lons, lat_step, lon_step, elevation_data = create_bathymetry_grid(
            ds_subset_astar, min_lon_astar, max_lon_astar, min_lat_astar, max_lat_astar, 
            cell_size_m_astar, vessel_height_underwater
        )
        num_lat_cells = len(grid_lats)
        num_lon_cells = len(grid_lons)

        print("Creating weather penalty grid...")
        weather_penalty_grid = create_weather_penalty_grid(
            min_lon_astar, max_lon_astar, min_lat_astar, max_lat_astar, 
            lat_step, lon_step, num_lat_cells, num_lon_cells
        )

        # Plot bathymetry
        cmap_maze = pyplot.get_cmap('Greys_r', 2)
        ax.imshow(bathymetry_maze, extent=[min_lon_astar, max_lon_astar, min_lat_astar, max_lat_astar],
                       origin='lower', cmap=cmap_maze, interpolation='nearest', alpha=0.6)
        
        # Plot weather penalties
        weather_penalty_np = np.array(weather_penalty_grid)
        masked_weather_penalty = np.ma.masked_where(weather_penalty_np == 0, weather_penalty_np)
        
        cmap_weather = pyplot.get_cmap('YlOrRd', 5)
        ax.imshow(masked_weather_penalty, extent=[min_lon_astar, max_lon_astar, min_lat_astar, max_lat_astar],
                       origin='lower', cmap=cmap_weather, interpolation='nearest', alpha=0.4)

        # Plot selected ports
        ax.scatter(selected_ports_main["lon"], selected_ports_main["lat"], 
                  color='purple', s=200, marker='^', label="Selected Ports", zorder=5)
        for idx, row in selected_ports_main.iterrows():
            ax.text(row["lon"] + 0.15, row["lat"] + 0.15, row["Ports"], 
                   fontsize=10, color='purple', fontweight='bold')

        # Generate A* paths for each segment
        print("\nGenerating A* paths between ports...")
        for i in range(len(optimal_path_route) - 1):
            start_port_name = optimal_path_route[i]
            end_port_name = optimal_path_route[i+1]

            start_port_coords = ports_gdf[ports_gdf["Ports"] == start_port_name].iloc[0]
            end_port_coords = ports_gdf[ports_gdf["Ports"] == end_port_name].iloc[0]

            start_latlon = (start_port_coords["lat"], start_port_coords["lon"])
            end_latlon = (end_port_coords["lat"], end_port_coords["lon"])

            print(f"  Segment {i+1}: {start_port_name} → {end_port_name}")

            start_grid = lat_lon_to_grid_coords(start_latlon[0], start_latlon[1], 
                                               min_lat_astar, min_lon_astar, lat_step, 
                                               lon_step, num_lat_cells, num_lon_cells)
            end_grid = lat_lon_to_grid_coords(end_latlon[0], end_latlon[1], 
                                             min_lat_astar, min_lon_astar, lat_step, 
                                             lon_step, num_lat_cells, num_lon_cells)

            search_radius = 5  # Reduced search radius
            
            # Adjust start point if on obstacle
            if bathymetry_maze[start_grid[0]][start_grid[1]] == 1:
                print(f"    Adjusting start point (on obstacle)")
                found_new_start = False
                for dr in range(-search_radius, search_radius + 1):
                    for dc in range(-search_radius, search_radius + 1):
                        if dr == 0 and dc == 0: continue
                        new_r, new_c = start_grid[0] + dr, start_grid[1] + dc
                        if 0 <= new_r < num_lat_cells and 0 <= new_c < num_lon_cells and bathymetry_maze[new_r][new_c] == 0:
                            start_grid = (new_r, new_c)
                            found_new_start = True
                            break
                    if found_new_start: break
                if not found_new_start:
                    print(f"    Could not find valid start. Skipping segment.")
                    continue

            # Adjust end point if on obstacle
            if bathymetry_maze[end_grid[0]][end_grid[1]] == 1:
                print(f"    Adjusting end point (on obstacle)")
                found_new_end = False
                for dr in range(-search_radius, search_radius + 1):
                    for dc in range(-search_radius, search_radius + 1):
                        if dr == 0 and dc == 0: continue
                        new_r, new_c = end_grid[0] + dr, end_grid[1] + dc
                        if 0 <= new_r < num_lat_cells and 0 <= new_c < num_lon_cells and bathymetry_maze[new_r][new_c] == 0:
                            end_grid = (new_r, new_c)
                            found_new_end = True
                            break
                    if found_new_end: break
                if not found_new_end:
                    print(f"    Could not find valid end. Skipping segment.")
                    continue

            # Run A* pathfinding
            path_grid = astar(bathymetry_maze, start_grid, end_grid, weather_penalty_grid)

            if not path_grid:
                print(f"    No path found with weather penalties. Trying without...")
                path_grid = astar(bathymetry_maze, start_grid, end_grid, weather_penalty_grid=None)
                if not path_grid:
                    print(f"    No path found. Skipping segment.")
                    continue

            # Convert grid path to lat/lon
            path_latlon = [grid_coords_to_lat_lon(r, c, min_lat_astar, min_lon_astar, lat_step, lon_step) 
                          for r, c in path_grid]
            all_astar_paths_latlon.extend(path_latlon)

            # Plot path
            path_lons = [p[1] for p in path_latlon]
            path_lats = [p[0] for p in path_latlon]
            current_color = colors[color_index % len(colors)]
            ax.plot(path_lons, path_lats, color=current_color, linewidth=2.5, alpha=0.8, 
                   label=f"A* Path: {start_port_name}→{end_port_name}")
            ax.scatter(start_latlon[1], start_latlon[0], color='green', s=100, marker='o', 
                      edgecolor='black', linewidth=1.5, zorder=4)
            ax.scatter(end_latlon[1], end_latlon[0], color='red', s=100, marker='X', 
                      edgecolor='black', linewidth=1.5, zorder=4)
            color_index += 1
        
        # ALIGNED: Generate landmark points at 100km intervals
        print(f"\nGenerating landmark points at 100km intervals...")
        landmark_points = generate_landmark_points(all_astar_paths_latlon, interval_km=100)
        
        if landmark_points:
            landmark_lons = [p[1] for p in landmark_points]
            landmark_lats = [p[0] for p in landmark_points]
            ax.scatter(landmark_lons, landmark_lats, color='cyan', s=120, marker='o', 
                      edgecolor='black', linewidth=2, label="Landmarks (100km)", zorder=6)
            print(f"  Generated {len(landmark_points)} landmarks")

        # Grid parameters for DRL
        grid_params = (min_lat_astar, min_lon_astar, lat_step, lon_step, num_lat_cells, num_lon_cells)

        # --- DRL Agent Simulation ---
        print("\n" + "="*60)
        print("DRL AGENT SIMULATION")
        print("="*60)
        
        # Check if DRL components are available
        if DDPG is None or MarineEnv is None:
            print("✗ DRL components not available")
            print("  Skipping DRL simulation. Run DRL_train.py first to train the agent.")
            plt.show()
            exit()
        
        # Initialize DRL environment
        env = MarineEnv(vessel.copy(), all_astar_paths_latlon, landmark_points, 
                       bathymetry_maze, grid_params, weather_penalty_grid)
        
        # Load trained agent
        state_dim = env.state_dim
        action_dim = env.action_dim
        max_action = 1.0

        agent = DDPG(state_dim, action_dim, max_action)
        
        try:
            agent.actor.load_state_dict(torch.load("ddpg_actor.pth", map_location=agent.device))
            print("✓ Trained DDPG actor model loaded successfully")
        except FileNotFoundError:
            print("✗ Error: ddpg_actor.pth not found")
            print("  Please run DRL_train.py first to train the agent")
            plt.show()
            exit()

        # Run simulation
        print("\nRunning DRL simulation...")
        drl_path_history = []
        rerouted_paths_history = []
        obstacle_history = []
        
        current_position = env.reset()
        drl_path_history.append(env.current_position_latlon)
        done = False
        step_count = 0
        max_steps = 500  # Increased for full route completion

        while not done and step_count < max_steps:
            # Get action from trained agent
            action = agent.select_action(current_position)
            
            # Simulate obstacle (1% chance, aligned with training)
            obstacle_present = random.random() < 0.01
            
            if obstacle_present:
                obstacle_info = env._generate_obstacle()
                obstacle_history.append(obstacle_info)

            next_state, reward, done, info = env.step(action, obstacle_present)
            
            drl_path_history.append(env.current_position_latlon)
            
            if "rerouted_path" in info and info["rerouted_path"]:
                rerouted_paths_history.append(info["rerouted_path"])

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
        print(f"Obstacles encountered: {len(obstacle_history)}")
        print(f"D* Lite reroutes: {len(rerouted_paths_history)}")
        print("="*60 + "\n")

        # Plot DRL path
        drl_lons = [p[1] for p in drl_path_history]
        drl_lats = [p[0] for p in drl_path_history]
        ax.plot(drl_lons, drl_lats, color='darkgreen', linewidth=3, linestyle='-', 
               label="DRL Agent Path", zorder=3)
        ax.scatter(drl_lons[0], drl_lats[0], color='lime', s=150, marker='o', 
                  edgecolor='black', linewidth=2, label="DRL Start", zorder=7)
        ax.scatter(drl_lons[-1], drl_lats[-1], color='darkgreen', s=180, marker='P', 
                  edgecolor='black', linewidth=2, label="DRL Final Position", zorder=7)

        # Plot D* Lite rerouted paths
        for idx, r_path in enumerate(rerouted_paths_history):
            r_lons = [p[1] for p in r_path]
            r_lats = [p[0] for p in r_path]
            label = "D* Lite Reroute" if idx == 0 else ""
            ax.plot(r_lons, r_lats, color='magenta', linewidth=2.5, linestyle=':', 
                   label=label, alpha=0.8, zorder=4)

        # Plot obstacles encountered
        for idx, obs in enumerate(obstacle_history):
            # Draw obstacle as a rectangle
            obs_lat, obs_lon = obs['center']
            size_lat, size_lon = obs['size_lat'], obs['size_lon']
            
            from matplotlib.patches import Rectangle
            rect = Rectangle(
                (obs_lon - size_lon/2, obs_lat - size_lat/2),
                size_lon, size_lat,
                linewidth=1.5, edgecolor='red', facecolor='red', alpha=0.3,
                label='Obstacles' if idx == 0 else ""
            )
            ax.add_patch(rect)

        # Finalize plot
        ax.set_xlim(min_lon_astar, max_lon_astar)
        ax.set_ylim(min_lat_astar, max_lat_astar)
        plt.title("Maritime Route Optimization: A* Planning + DRL Agent Simulation\n" + 
                 f"(100km landmarks, 10km grid, {len(obstacle_history)} obstacles)", fontsize=14)
        plt.xlabel("Longitude", fontsize=12)
        plt.ylabel("Latitude", fontsize=12)
        
        # Create legend
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = list(dict.fromkeys(labels))
        unique_handles = [handles[labels.index(l)] for l in unique_labels]
        ax.legend(unique_handles, unique_labels, loc='upper right', fontsize=9, 
                 framealpha=0.9)
        
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.tight_layout()
        
        # Save figure
        plt.savefig("maritime_route_simulation.png", dpi=200, bbox_inches='tight')
        print("Visualization saved to: maritime_route_simulation.png")
        
        plt.show()

    else:
        print("Could not generate an optimal path route.")