import heapq
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2
import matplotlib.pyplot as pyplot


vessel = {
    "speed_knots": 20.0,  # cruising speed in knots
    "fuel_consumption_t_per_day": 20.0,  # fuel consumption in tons per day at cruising speed
    "fuel_tank_capacity_t": 1000.0,  # total fuel tank capacity in tons
    "safety_fuel_margin_t": 10.0,  # safety margin in tons
    "fuel_tank_remaining_t": 1000.0,  # current fuel remaining in tons
    "size": (1000, 500, 60),  # size of the vessel in meters (length, width, height)
    "percent_of_height_underwater": 0.3,
}

# Haversine formula to calculate distance between two lat/lon points in meters
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # Radius of Earth in meters

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance

class Node:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.g = 0  # Cost from start to this node
        self.h = 0  # Heuristic cost from this node to end
        self.f = 0  # Total cost (g + h)

    def __eq__(self, other):
        return self.position == other.position

    def __lt__(self, other):
        return self.f < other.f

    def __hash__(self): # Added for set/dict usage
        return hash(self.position)

def astar(maze, start, end):
    """
    Finds the shortest path from start to end in a maze using the A* algorithm.

    Args:
        maze (list of list of int): A 2D grid where 0 represents a walkable path
                                    and 1 represents an obstacle.
        start (tuple): The starting coordinates (row, col).
        end (tuple): The ending coordinates (row, col).

    Returns:
        list of tuple: The path from start to end, or None if no path is found.
    """
    start_node = Node(start)
    end_node = Node(end)

    open_list = [] # Min-heap for (f_cost, node)
    open_list_positions = {start_node.position: start_node} # Dictionary to quickly access nodes in open_list by position
    closed_list_positions = set() # Set to store positions of nodes already evaluated

    heapq.heappush(open_list, (start_node.f, start_node))

    while open_list:
        f_cost, current_node = heapq.heappop(open_list)

        # If we already processed a better path to this node, skip
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

            if maze[nx][ny] != 0:
                continue

            if next_pos in closed_list_positions:
                continue

            new_g = current_node.g + 1
            new_h = abs(nx - end_node.position[0]) + abs(ny - end_node.position[1])
            new_f = new_g + new_h

            # If the neighbor is already in open_list_positions and we found a worse path, skip
            if next_pos in open_list_positions and new_g >= open_list_positions[next_pos].g:
                continue

            # This is either a new node or a better path to an existing node in open_list
            new_node = Node(next_pos, current_node)
            new_node.g = new_g
            new_node.h = new_h
            new_node.f = new_f

            heapq.heappush(open_list, (new_node.f, new_node))
            open_list_positions[next_pos] = new_node # Update or add the node with the better path

    return None  # No path found

def create_bathymetry_grid(ds_subset, min_lon, max_lon, min_lat, max_lat, cell_size_m=500, vessel_height=0):
    """
    Creates a grid (maze) from bathymetry data.

    Args:
        ds_subset (xarray.Dataset): Subset of GEBCO bathymetry data.
        min_lon, max_lon, min_lat, max_lat (float): Bounding box for the grid.
        cell_size_m (int): Desired cell size in meters (e.g., 500m x 500m).
        vessel_height (float): The height of the vessel in meters. Areas shallower than this are obstacles.

    Returns:
        tuple: (grid (list of list of int), grid_lats, grid_lons, lat_step, lon_step, elevation_data)
    """
    # Calculate approximate degrees per meter at the center latitude of the region
    # This is a simplification; for high accuracy, a more complex projection would be needed.
    center_lat = (min_lat + max_lat) / 2
    lat_deg_per_m = 1 / haversine(center_lat, 0, center_lat + 0.001, 0) * 0.001
    lon_deg_per_m = 1 / haversine(center_lat, 0, center_lat, 0.001) * 0.001

    lat_step = cell_size_m * lat_deg_per_m
    lon_step = cell_size_m * lon_deg_per_m

    num_lat_cells = int(np.ceil((max_lat - min_lat) / lat_step))
    num_lon_cells = int(np.ceil((max_lon - min_lon) / lon_step))

    grid = np.zeros((num_lat_cells, num_lon_cells), dtype=int)
    elevation_data = np.zeros((num_lat_cells, num_lon_cells), dtype=float) # Store actual elevation for visualization
    grid_lats = np.linspace(min_lat, max_lat, num_lat_cells)
    grid_lons = np.linspace(min_lon, max_lon, num_lon_cells)

    for r in range(num_lat_cells):
        for c in range(num_lon_cells):
            # Get the center of the grid cell
            cell_lat = grid_lats[r] + lat_step / 2
            cell_lon = grid_lons[c] + lon_step / 2

            # Query bathymetry data for elevation at this point
            # Use .sel(method='nearest') to find the closest data point
            try:
                elevation = ds_subset['elevation'].sel(lat=cell_lat, lon=cell_lon, method='nearest').item()
            except KeyError: # Handle cases where the point might be outside the ds_subset's exact bounds
                elevation = 0 # Assume land/obstacle if no data

            elevation_data[r, c] = elevation # Store elevation

            # Mark as obstacle (1) if shallower than vessel_height or land (elevation >= 0)
            if elevation >= -vessel_height:
                grid[r, c] = 1
            else:
                grid[r, c] = 0 # Walkable

    # Convert elevation_data to xarray.DataArray for better plotting with imshow
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
    # Ensure coordinates are within bounds
    row = max(0, min(row, num_lat_cells - 1))
    col = max(0, min(col, num_lon_cells - 1))
    return row, col

def grid_coords_to_lat_lon(row, col, min_lat, min_lon, lat_step, lon_step):
    """Converts grid (row, col) coordinates to approximate latitude and longitude (center of cell)."""
    lat = min_lat + row * lat_step + lat_step / 2
    lon = min_lon + col * lon_step + lon_step / 2
    return lat, lon

if __name__ == "__main__":
    # Load bathymetry data
    ds = xr.open_dataset("data\\Bathymetry\\GEBCO_2024_sub_ice_topo.nc")

    # Define the region of interest (e.g., around Malaysia)
    min_lon, max_lon = 99, 120 # Original bathymetry.py range
    min_lat, max_lat = 0, 8   # Original bathymetry.py range

    ds_subset = ds.sel(
        lon=slice(min_lon, max_lon),
        lat=slice(min_lat, max_lat)
    )

    # Get vessel height from the vessel dictionary
    vessel_height = vessel["size"][2] * vessel["percent_of_height_underwater"] # Assuming vessel height is the third element in the 'size' tuple

    # Create the bathymetry grid (maze)
    cell_size_m = 5000 # 10km x 10km grid cells for faster computation
    bathymetry_maze, grid_lats, grid_lons, lat_step, lon_step, elevation_data = create_bathymetry_grid(
        ds_subset, min_lon, max_lon, min_lat, max_lat, cell_size_m, vessel_height
    )

    num_lat_cells = len(grid_lats)
    num_lon_cells = len(grid_lons)

    print(f"Generated a grid of size: {num_lat_cells} rows x {num_lon_cells} columns")

    # Debugging: Print a small section of the grid and walkable percentage
    print("\nSample of the generated bathymetry grid (0=walkable, 1=obstacle):")
    for r_idx in range(min(5, num_lat_cells)):
        print(bathymetry_maze[r_idx][:min(10, num_lon_cells)])
    walkable_cells = np.sum(np.array(bathymetry_maze) == 0)
    total_cells = num_lat_cells * num_lon_cells
    print(f"Walkable cells: {walkable_cells} ({walkable_cells / total_cells * 100:.2f}%)")

    # Define start and end points in lat/lon
    start_latlon = (2.5, 101.0) # Adjusted start for better chance of being in walkable area
    end_latlon = (4.0, 104.5)   # Adjusted end for better chance of being in walkable area

    # Convert lat/lon to grid coordinates
    start_grid = lat_lon_to_grid_coords(start_latlon[0], start_latlon[1], min_lat, min_lon, lat_step, lon_step, num_lat_cells, num_lon_cells)
    end_grid = lat_lon_to_grid_coords(end_latlon[0], end_latlon[1], min_lat, min_lon, lat_step, lon_step, num_lat_cells, num_lon_cells)

    print(f"Start Lat/Lon: {start_latlon} -> Grid: {start_grid}")
    print(f"End Lat/Lon: {end_latlon} -> Grid: {end_grid}")

    # Check if start or end points are obstacles
    if bathymetry_maze[start_grid[0]][start_grid[1]] == 1:
        print(f"Start point {start_grid} is an obstacle (depth >= {vessel_height}m or land). Adjusting start point.")
        # Try to find a nearby walkable point for start
        for dr in range(-10, 11): # Increased search radius further
            for dc in range(-10, 11): # Increased search radius further
                if dr == 0 and dc == 0:
                    continue
                new_start_r, new_start_c = start_grid[0] + dr, start_grid[1] + dc
                if 0 <= new_start_r < num_lat_cells and 0 <= new_start_c < num_lon_cells and bathymetry_maze[new_start_r][new_start_c] == 0:
                    start_grid = (new_start_r, new_start_c)
                    print(f"New start grid: {start_grid}")
                    break
            else:
                continue
            break
        else:
            print("Could not find a walkable start point nearby. Exiting.")
            exit()

    if bathymetry_maze[end_grid[0]][end_grid[1]] == 1:
        print(f"End point {end_grid} is an obstacle (depth >= {vessel_height}m or land). Adjusting end point.")
        # Try to find a nearby walkable point for end
        for dr in range(-10, 11): # Increased search radius further
            for dc in range(-10, 11): # Increased search radius further
                if dr == 0 and dc == 0:
                    continue
                new_end_r, new_end_c = end_grid[0] + dr, end_grid[1] + dc
                if 0 <= new_end_r < num_lat_cells and 0 <= new_end_c < num_lon_cells and bathymetry_maze[new_end_r][new_end_c] == 0:
                    end_grid = (new_end_r, new_end_c)
                    print(f"New end grid: {new_end_c}")
                    break
            else:
                continue
            break
        else:
            print("Could not find a walkable end point nearby. Exiting.")
            exit()

    # Run A* on the bathymetry maze
    print(f"\nFinding path from {start_grid} to {end_grid} in the bathymetry maze:")
    path_grid = astar(bathymetry_maze, start_grid, end_grid)

    if path_grid:
        print("Path found in grid coordinates:")
        for r, c in path_grid:
            print(f"({r}, {c})", end=" -> ")
        print("End")

        # Convert grid path back to lat/lon for visualization
        path_latlon = [grid_coords_to_lat_lon(r, c, min_lat, min_lon, lat_step, lon_step) for r, c in path_grid]

        # Visualize the path on the bathymetry map
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot the bathymetry maze (0=walkable, 1=obstacle)
        # Using a custom colormap to clearly distinguish walkable vs. obstacle
        cmap_maze = pyplot.get_cmap('Greys_r', 2) # 2 colors: 0 and 1
        im = ax.imshow(bathymetry_maze, extent=[min_lon, max_lon, min_lat, max_lat],
                       origin='lower', cmap=cmap_maze, interpolation='nearest')
        plt.colorbar(im, ax=ax, ticks=[0, 1], label="Maze: 0=Walkable, 1=Obstacle")

        # Plot grid lines
        for lat in grid_lats:
            ax.axhline(lat, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        for lon in grid_lons:
            ax.axvline(lon, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

        # Plot the path
        path_lons = [p[1] for p in path_latlon]
        path_lats = [p[0] for p in path_latlon]
        ax.plot(path_lons, path_lats, color='cyan', linewidth=3, marker='o', markersize=5, label="A* Path")

        # Plot start and end points
        ax.scatter(start_latlon[1], start_latlon[0], color='green', s=100, marker='o', label="Start")
        ax.scatter(end_latlon[1], end_latlon[0], color='red', s=100, marker='X', label="End")

        ax.set_xlim(min_lon, max_lon)
        ax.set_ylim(min_lat, max_lat)
        plt.title("A* Path on Bathymetry Grid")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.show()

    else:
        print("No path found in the bathymetry maze!")
