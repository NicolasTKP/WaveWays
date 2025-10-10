import heapq
from math import radians, sin, cos, sqrt, atan2
import numpy as np
import xarray as xr
from shapely.geometry import Point, Polygon

# Helper functions copied from main.py for now, will refactor later if needed
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance

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

# D* Lite Implementation
class DStarLite:
    def __init__(self, grid, start, goal, weather_penalty_grid=None):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.weather_penalty_grid = weather_penalty_grid
        self.rows = len(grid)
        self.cols = len(grid[0])

        self.km = 0 # Key modifier
        self.g = {} # Cost from start to node
        self.rhs = {} # Heuristic cost from start to node
        self.U = [] # Priority queue

        self.pred = {} # Predecessors
        self.succ = {} # Successors

        self.initialize()

    def initialize(self):
        for r in range(self.rows):
            for c in range(self.cols):
                pos = (r, c)
                self.g[pos] = float('inf')
                self.rhs[pos] = float('inf')
                self.pred[pos] = set()
                self.succ[pos] = set()

        self.rhs[self.goal] = 0
        heapq.heappush(self.U, (self.calculate_key(self.goal), self.goal))

    def calculate_key(self, s):
        return (min(self.g[s], self.rhs[s]) + self.h(s, self.start) + self.km, min(self.g[s], self.rhs[s]))

    def h(self, s1, s2):
        # Manhattan distance heuristic
        return abs(s1[0] - s2[0]) + abs(s1[1] - s2[1])

    def get_neighbors(self, s):
        neighbors = []
        x, y = s
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]: # 4-directional movement
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.rows and 0 <= ny < self.cols:
                neighbors.append((nx, ny))
        return neighbors

    def cost(self, s1, s2):
        # Cost of moving from s1 to s2
        if self.grid[s2[0]][s2[1]] == 1: # Obstacle
            return float('inf')
        
        move_cost = 1
        if self.weather_penalty_grid and self.weather_penalty_grid[s2[0]][s2[1]] > 0:
            move_cost += self.weather_penalty_grid[s2[0]][s2[1]]
        return move_cost

    def update_vertex(self, u):
        if u != self.goal:
            self.rhs[u] = min(self.cost(u, s_prime) + self.g[s_prime] for s_prime in self.get_neighbors(u))
        
        # Remove u from U if it's there
        self.U = [(key, node) for key, node in self.U if node != u]
        heapq.heapify(self.U)

        if self.g[u] != self.rhs[u]:
            heapq.heappush(self.U, (self.calculate_key(u), u))

    def compute_shortest_path(self):
        while self.U and (self.U[0][0] < self.calculate_key(self.start) or self.rhs[self.start] != self.g[self.start]):
            k_old, u = heapq.heappop(self.U)

            if k_old < self.calculate_key(u):
                heapq.heappush(self.U, (self.calculate_key(u), u))
            elif self.g[u] > self.rhs[u]:
                self.g[u] = self.rhs[u]
                for s_prime in self.get_neighbors(u):
                    self.update_vertex(s_prime)
            else:
                self.g[u] = float('inf')
                for s_prime in self.get_neighbors(u) + [u]:
                    self.update_vertex(s_prime)

    def find_path(self):
        path = []
        current = self.start
        while current != self.goal:
            path.append(current)
            neighbors = self.get_neighbors(current)
            if not neighbors:
                return None # No path found
            
            min_cost = float('inf')
            next_node = None
            for neighbor in neighbors:
                c = self.cost(current, neighbor)
                if c == float('inf'): # Obstacle
                    continue
                
                if self.g[neighbor] + c < min_cost:
                    min_cost = self.g[neighbor] + c
                    next_node = neighbor
            
            if next_node is None:
                return None # Stuck
            current = next_node
        path.append(self.goal)
        return path

    def rescan(self, changed_cells):
        self.km += self.h(self.start, self.goal) # Update key modifier
        for u in changed_cells:
            self.update_vertex(u)
        self.compute_shortest_path()

def find_d_star_lite_route(grid, start_latlon, end_latlon, min_lat, min_lon, lat_step, lon_step, num_lat_cells, num_lon_cells, weather_penalty_grid=None, obstacle_coords=None):
    """
    Finds a route using D* Lite between two lat/lon points on a grid.
    """
    start_grid = lat_lon_to_grid_coords(start_latlon[0], start_latlon[1], min_lat, min_lon, lat_step, lon_step, num_lat_cells, num_lon_cells)
    end_grid = lat_lon_to_grid_coords(end_latlon[0], end_latlon[1], min_lat, min_lon, lat_step, lon_step, num_lat_cells, num_lon_cells)

    # Ensure start and end are not obstacles initially
    if grid[start_grid[0]][start_grid[1]] == 1:
        print(f"D* Lite: Start point {start_grid} is an obstacle. Cannot start.")
        return None
    if grid[end_grid[0]][end_grid[1]] == 1:
        print(f"D* Lite: End point {end_grid} is an obstacle. Cannot reach.")
        return None

    d_star = DStarLite(grid, start_grid, end_grid, weather_penalty_grid)
    d_star.compute_shortest_path()
    path_grid = d_star.find_path()

    if obstacle_coords:
        print(f"D* Lite: Simulating obstacle at {obstacle_coords}")
        ox, oy = obstacle_coords
        if 0 <= ox < d_star.rows and 0 <= oy < d_star.cols:
            original_cost = d_star.grid[ox][oy]
            d_star.grid[ox][oy] = 1 # Mark as obstacle
            changed_cells = [(ox, oy)]
            d_star.rescan(changed_cells)
            path_grid = d_star.find_path()
            d_star.grid[ox][oy] = original_cost # Revert for future calls if needed
        else:
            print(f"D* Lite: Obstacle coordinates {obstacle_coords} are out of grid bounds.")

    if path_grid:
        path_latlon = [grid_coords_to_lat_lon(r, c, min_lat, min_lon, lat_step, lon_step) for r, c in path_grid]
        return path_latlon
    return None
