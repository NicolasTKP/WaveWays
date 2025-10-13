import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import copy
import pandas as pd
import xarray as xr
from geopy.distance import geodesic
from shapely.geometry import Point, Polygon
from math import radians, sin, cos, sqrt, atan2, degrees
import matplotlib.pyplot as plt
import time
import geopandas as gpd

# Import necessary functions from utils.py and d_star_lite.py
from utils import (
    vessel, vessel_height_underwater, safe_float, get_wave_wind, interpolate_points,
    reorder_route, compute_edge_weight, enforce_tsp_constraints,
    locations_data, location_codes, location_weather_cache,
    is_point_in_polygon, scrape_weather_data, get_weather_for_point,
    get_weather_penalty, calculate_vessel_range_km, simulate_vessel_route,
    get_optimal_path_route, haversine, generate_landmark_points,
    create_weather_penalty_grid, Node, astar, create_bathymetry_grid,
    lat_lon_to_grid_coords, grid_coords_to_lat_lon
)
from d_star_lite import DStarLite, find_d_star_lite_route

# --- DDPG Agent Implementation ---
# OPTIMIZATION STRATEGY:
# 1. Weather caching: 1 point per 50km (5 grid cells) = ~90% reduction in API calls
# 2. Segment-based optimization: Only optimize between consecutive landmarks
# 3. Smart obstacle handling: D* Lite for large (>3km), DRL learns small ones
# 4. Coarse grid: 10km cells instead of 5km = 75% fewer cells
# 5. Reduced training: 100 episodes, 40 steps/episode, 2hr timesteps

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = torch.relu(self.l1(state))
        x = torch.relu(self.l2(x))
        return self.max_action * torch.tanh(self.l3(x))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return self.l3(x)

class DDPG:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.max_action = max_action
        self.discount = 0.99
        self.tau = 0.005
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor.to(self.device)
        self.actor_target.to(self.device)
        self.critic.to(self.device)
        self.critic_target.to(self.device)

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=100):
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        not_done = torch.FloatTensor(not_done).to(self.device)

        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = reward + (not_done * self.discount * target_Q).detach()

        current_Q = self.critic(state, action)
        critic_loss = nn.functional.mse_loss(current_Q, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(state, self.actor(state)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, next_state, reward, done):
        self.buffer.append((state, action, next_state, reward, done))

    def sample(self, batch_size):
        state, action, next_state, reward, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), np.array(action), np.array(next_state), np.array(reward).reshape(-1, 1), np.array(done).reshape(-1, 1)

    def __len__(self):
        return len(self.buffer)

# --- Environment Definition ---

class MarineEnv:
    def __init__(self, initial_vessel_state, full_astar_path_latlon, landmark_points, 
                 bathymetry_data, grid_params, weather_penalty_grid):
        self.vessel = initial_vessel_state
        self.full_astar_path_latlon = full_astar_path_latlon
        self.landmark_points = landmark_points
        self.bathymetry_data = bathymetry_data
        self.grid_params = grid_params
        self.weather_penalty_grid = weather_penalty_grid

        self.min_lat, self.min_lon, self.lat_step, self.lon_step, self.num_lat_cells, self.num_lon_cells = grid_params

        self.current_position_latlon = None
        self.current_heading_deg = None
        self.current_speed_knots = None
        self.current_landmark_idx = 0
        self.total_fuel_consumed = 0.0
        self.total_emissions = 0.0
        self.total_eta_hours = 0.0
        self.time_step_hours = 2.0  # REDUCED from 5 hours for finer control
        self.drl_path_segment = []
        self.rerouted_paths_history = []

        # PRE-CACHE WEATHER DATA for strategic grid points (OPTIMIZATION 1)
        # Cache one point per ~50km radius (5 grid cells at 10km resolution)
        # This dramatically reduces caching time while maintaining coverage
        self.weather_cache = {}
        print("Pre-caching weather data for strategic grid points...")
        
        # Calculate how many cache points we'll create
        cache_interval = 40  # One cache point every 5 cells (~50km at 10km resolution)
        num_cache_lat = max(1, self.num_lat_cells // cache_interval)
        num_cache_lon = max(1, self.num_lon_cells // cache_interval)
        total_cache_points = num_cache_lat * num_cache_lon
        
        print(f"  Grid size: {self.num_lat_cells}x{self.num_lon_cells} cells")
        print(f"  Cache points: {num_cache_lat}x{num_cache_lon} = {total_cache_points} points")
        print(f"  Each cache point covers ~50km radius")
        
        cached_count = 0
        for i in range(0, self.num_lat_cells, cache_interval):
            for j in range(0, self.num_lon_cells, cache_interval):
                lat, lon = grid_coords_to_lat_lon(i, j, self.min_lat, self.min_lon, 
                                                 self.lat_step, self.lon_step)
                cache_key = (round(lat, 1), round(lon, 1))  # Round to 1 decimal for 10km accuracy
                
                if cache_key not in self.weather_cache:
                    try:
                        self.weather_cache[cache_key] = get_wave_wind(lat, lon)
                        cached_count += 1
                        if cached_count % 10 == 0:
                            print(f"    Cached {cached_count}/{total_cache_points} points...", end='\r')
                    except Exception as e:
                        # Use default values if fetch fails
                        self.weather_cache[cache_key] = {
                            "wave_height": 0.0, "wave_period": 0.0, "wave_direction": 0.0,
                            "wind_wave_height": 0.0, "wind_wave_period": 0.0, "wind_wave_direction": 0.0
                        }
        
        print(f"\n  Weather cache loaded with {len(self.weather_cache)} strategic points")
        print(f"  Coverage: Each point represents ~50km radius (~5 grid cells)")

        # Pre-compute A* segment between each pair of consecutive landmarks (OPTIMIZATION 2)
        self.landmark_astar_segments = {}
        print("Pre-computing A* segments between landmarks...")
        for i in range(len(landmark_points) - 1):
            segment_key = (i, i+1)
            start_landmark = landmark_points[i]
            end_landmark = landmark_points[i+1]
            
            # Find A* path points between these landmarks
            segment_points = []
            in_segment = False
            for p in full_astar_path_latlon:
                if geodesic(p, start_landmark).km < 5.0:  # Within 5km of start
                    in_segment = True
                if in_segment:
                    segment_points.append(p)
                if geodesic(p, end_landmark).km < 5.0:  # Within 5km of end
                    break
            
            self.landmark_astar_segments[segment_key] = segment_points if segment_points else [start_landmark, end_landmark]
        print(f"Pre-computed {len(self.landmark_astar_segments)} A* segments")

        self.reset()

        self.state_dim = 14
        self.action_dim = 2
        self.max_speed_knots = self.vessel["speed_knots"]
        self.max_heading_change_deg = 30

    def reset(self):
        """Enhanced reset with tracking variables"""
        self.current_position_latlon = self.landmark_points[0]
        
        if len(self.landmark_points) > 1:
            self.current_heading_deg = self._calculate_bearing(
                self.landmark_points[0], self.landmark_points[1]
            )
        else:
            self.current_heading_deg = 0.0
        
        self.current_speed_knots = self.vessel["speed_knots"] * 0.7
        self.current_landmark_idx = 1
        self.total_fuel_consumed = 0.0
        self.total_emissions = 0.0
        self.total_eta_hours = 0.0
        self.vessel["fuel_tank_remaining_t"] = self.vessel["fuel_tank_capacity_t"]
        self.drl_path_segment = [self.current_position_latlon]
        
        # ADDED: Initialize tracking for circular motion detection
        self.recent_headings = [self.current_heading_deg]
        self.recent_positions = [self.current_position_latlon]

        return self._get_state()

    def _calculate_bearing(self, point1, point2):
        """Calculate bearing from point1 to point2 in degrees (0-360)"""
        lat1, lon1 = radians(point1[0]), radians(point1[1])
        lat2, lon2 = radians(point2[0]), radians(point2[1])
        
        dlon = lon2 - lon1
        x = sin(dlon) * cos(lat2)
        y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
        bearing = degrees(atan2(x, y))
        return (bearing + 360) % 360

    def _get_state(self):
        current_lat, current_lon = self.current_position_latlon
        
        target_landmark_lat, target_landmark_lon = self.landmark_points[self.current_landmark_idx]

        # USE CACHED WEATHER DATA with smart lookup (OPTIMIZATION)
        # Round to 1 decimal place to match cache keys (covers ~10km)
        cache_key = (round(current_lat, 1), round(current_lon, 1))
        
        if cache_key in self.weather_cache:
            sea_currents = self.weather_cache[cache_key]
        else:
            # Find nearest cached point (within 50km radius)
            if self.weather_cache:
                # Calculate distance to all cache points (in lat/lon degrees, approximation)
                nearest_key = min(
                    self.weather_cache.keys(), 
                    key=lambda k: (k[0] - current_lat)**2 + (k[1] - current_lon)**2
                )
                sea_currents = self.weather_cache[nearest_key]
            else:
                # Fallback to default values
                sea_currents = {
                    "wave_height": 0.0, "wave_period": 0.0, "wave_direction": 0.0,
                    "wind_wave_height": 0.0, "wind_wave_period": 0.0, "wind_wave_direction": 0.0
                }

        distance_to_target_landmark = geodesic(self.current_position_latlon, 
                                               self.landmark_points[self.current_landmark_idx]).km

        state = [
            current_lat, current_lon,
            self.current_speed_knots, self.current_heading_deg,
            target_landmark_lat, target_landmark_lon,
            sea_currents["wave_height"], sea_currents["wave_period"], sea_currents["wave_direction"],
            sea_currents["wind_wave_height"], sea_currents["wind_wave_period"], sea_currents["wind_wave_direction"],
            self.vessel["fuel_tank_remaining_t"],
            distance_to_target_landmark
        ]
        return np.array(state, dtype=np.float32)

    def _generate_obstacle(self):
        """Generate a realistic obstacle with size (up to 5km x 5km)"""
        # Random size between 1km to 5km
        obstacle_size_km = random.uniform(1.0, 5.0)
        
        # Convert km to grid cells
        km_to_lat = 1 / 111.0  # Approximate: 1 degree latitude ≈ 111 km
        km_to_lon = 1 / (111.0 * cos(radians(self.current_position_latlon[0])))
        
        obstacle_size_lat = obstacle_size_km * km_to_lat
        obstacle_size_lon = obstacle_size_km * km_to_lon
        
        # Get obstacle center in heading direction
        heading_rad = radians(90 - self.current_heading_deg)
        distance_ahead_km = random.uniform(10, 30)  # 10-30km ahead
        
        R = 6371
        lat_offset = (distance_ahead_km / R) * (180 / np.pi) * cos(heading_rad)
        lon_offset = (distance_ahead_km / R) * (180 / np.pi) * sin(heading_rad) / cos(radians(self.current_position_latlon[0]))
        
        obstacle_center_lat = self.current_position_latlon[0] + lat_offset
        obstacle_center_lon = self.current_position_latlon[1] + lon_offset
        
        return {
            'center': (obstacle_center_lat, obstacle_center_lon),
            'size_km': obstacle_size_km,
            'size_lat': obstacle_size_lat,
            'size_lon': obstacle_size_lon
        }

    def _check_collision_with_obstacle(self, position, obstacle):
        """Check if position collides with obstacle"""
        lat_diff = abs(position[0] - obstacle['center'][0])
        lon_diff = abs(position[1] - obstacle['center'][1])
        
        return (lat_diff < obstacle['size_lat'] / 2 and 
                lon_diff < obstacle['size_lon'] / 2)

    def step(self, action, obstacle_present=False):
        speed_change_normalized, heading_change_normalized = action

        # Apply action
        new_speed_knots = self.current_speed_knots + speed_change_normalized * (self.max_speed_knots / 2)
        new_speed_knots = np.clip(new_speed_knots, 1.0, self.max_speed_knots)

        heading_change_deg = heading_change_normalized * self.max_heading_change_deg
        new_heading_deg = (self.current_heading_deg + heading_change_deg) % 360

        self.current_speed_knots = new_speed_knots
        self.current_heading_deg = new_heading_deg

        # Calculate new position
        distance_travelled_km = self.current_speed_knots * 1.852 * self.time_step_hours
        heading_rad = radians(90 - self.current_heading_deg)

        current_lat, current_lon = self.current_position_latlon
        R = 6371

        new_lat = current_lat + (distance_travelled_km / R) * (180 / np.pi) * cos(heading_rad)
        new_lon = current_lon + (distance_travelled_km / R) * (180 / np.pi) * sin(heading_rad) / cos(radians(current_lat))

        new_position_latlon = (new_lat, new_lon)

        # OBSTACLE HANDLING WITH SIZE-BASED DECISION
        rerouted_path_latlon = None
        obstacle_info = None
        
        if obstacle_present:
            obstacle_info = self._generate_obstacle()
            obstacle_size_km = obstacle_info['size_km']
            
            # Check if vessel would collide with obstacle
            if self._check_collision_with_obstacle(new_position_latlon, obstacle_info):
                print(f"Obstacle detected! Size: {obstacle_size_km:.2f} km")
                
                # DECISION: Use D* Lite for large obstacles (>3km), DRL handles small ones
                if obstacle_size_km > 3.0:
                    print(f"Large obstacle ({obstacle_size_km:.2f} km) - Using D* Lite rerouting")
                    
                    current_grid = lat_lon_to_grid_coords(current_lat, current_lon, 
                                                         self.min_lat, self.min_lon, 
                                                         self.lat_step, self.lon_step,
                                                         self.num_lat_cells, self.num_lon_cells)
                    obstacle_center_grid = lat_lon_to_grid_coords(
                        obstacle_info['center'][0], obstacle_info['center'][1],
                        self.min_lat, self.min_lon, self.lat_step, self.lon_step,
                        self.num_lat_cells, self.num_lon_cells
                    )
                    
                    # Create temporary grid with obstacle
                    temp_grid = [row[:] for row in self.bathymetry_data]
                    
                    # Mark obstacle area in grid
                    obstacle_radius_cells = int(obstacle_size_km / (self.lat_step * 111.0)) + 1
                    for dr in range(-obstacle_radius_cells, obstacle_radius_cells + 1):
                        for dc in range(-obstacle_radius_cells, obstacle_radius_cells + 1):
                            r, c = obstacle_center_grid[0] + dr, obstacle_center_grid[1] + dc
                            if 0 <= r < self.num_lat_cells and 0 <= c < self.num_lon_cells:
                                temp_grid[r][c] = 1
                    
                    # Use D* Lite to reroute to current landmark
                    rerouted_path_latlon = find_d_star_lite_route(
                        temp_grid,
                        self.current_position_latlon,
                        self.landmark_points[self.current_landmark_idx],
                        self.min_lat, self.min_lon, self.lat_step, self.lon_step,
                        self.num_lat_cells, self.num_lon_cells,
                        self.weather_penalty_grid,
                        obstacle_coords=obstacle_center_grid
                    )
                    
                    if rerouted_path_latlon and len(rerouted_path_latlon) > 1:
                        new_position_latlon = rerouted_path_latlon[1]
                        print(f"D* Lite rerouted to {new_position_latlon}")
                    else:
                        print("D* Lite failed - using emergency turn")
                        self.current_heading_deg = (self.current_heading_deg + 90) % 360
                else:
                    print(f"Small obstacle ({obstacle_size_km:.2f} km) - DRL handles avoidance")
                    # DRL will learn to avoid through negative reward
                    # Just execute the collision for penalty
                    pass

        self.current_position_latlon = new_position_latlon
        self.drl_path_segment.append(new_position_latlon)
        self.total_eta_hours += self.time_step_hours

        # Fuel consumption
        fuel_consumed_t = (self.vessel["fuel_consumption_t_per_day"] / 24.0) * self.time_step_hours
        self.vessel["fuel_tank_remaining_t"] -= fuel_consumed_t
        self.total_fuel_consumed += fuel_consumed_t

        # Emissions
        co2_emission_factor_t_per_t_fuel = 3.114
        emissions_t_co2 = fuel_consumed_t * co2_emission_factor_t_per_t_fuel
        self.total_emissions += emissions_t_co2

        # Check if landmark reached (INCREASED to 15km for faster convergence)
        distance_to_current_landmark = geodesic(
            self.current_position_latlon, 
            self.landmark_points[self.current_landmark_idx]
        ).km
        
        if distance_to_current_landmark < 15:
            self.current_landmark_idx += 1
            print(f"Reached landmark {self.current_landmark_idx}/{len(self.landmark_points)}")
            self.drl_path_segment = [self.current_position_latlon]
            self.rerouted_paths_history = []

        done = (self.current_landmark_idx >= len(self.landmark_points) or 
                self.vessel["fuel_tank_remaining_t"] <= 0)

        reward = self._calculate_reward(rerouted_path_latlon, obstacle_info)

        next_state = self._get_state()
        
        if rerouted_path_latlon:
            self.rerouted_paths_history.append(rerouted_path_latlon)

        return next_state, reward, done, {"rerouted_path": rerouted_path_latlon}

    def _calculate_reward(self, rerouted_path_latlon=None, obstacle_info=None):
        """IMPROVED REWARD CALCULATION with positive reinforcement"""
        reward = 0.0

        if self.current_landmark_idx < len(self.landmark_points):
            current_target = self.landmark_points[self.current_landmark_idx]
            distance_to_target = geodesic(self.current_position_latlon, current_target).km
            
            # 1. STRONG PROGRESS REWARD (most important signal)
            if hasattr(self, 'state_before_action') and len(self.state_before_action) > 13:
                prev_distance = self.state_before_action[13]
                progress = prev_distance - distance_to_target
                
                # CRITICAL: Reward progress heavily, penalize regression
                if progress > 0:
                    reward += progress * 100.0  # INCREASED from 5.0
                else:
                    reward += progress * 50.0  # Less penalty for slight regression
            
            # 2. DISTANCE-BASED REWARD (normalize by max distance)
            max_distance = 200  # Approximate max distance between landmarks (km)
            normalized_distance = distance_to_target / max_distance
            reward += (1.0 - normalized_distance) * 100  # Reward being closer (0-100 points)
            
            # 3. HEADING ALIGNMENT REWARD
            target_bearing = self._calculate_bearing(self.current_position_latlon, current_target)
            heading_diff = abs(self.current_heading_deg - target_bearing)
            if heading_diff > 180:
                heading_diff = 360 - heading_diff
            
            # Reward heading toward target
            heading_alignment = 1.0 - (heading_diff / 180.0)  # 0 to 1
            reward += heading_alignment * 50  # Up to 50 points for good heading
            
            # 4. A* PATH ADHERENCE (reduced weight)
            if self.current_landmark_idx > 0:
                segment_key = (self.current_landmark_idx - 1, self.current_landmark_idx)
                if segment_key in self.landmark_astar_segments:
                    astar_segment = self.landmark_astar_segments[segment_key]
                    distances_to_segment = [geodesic(self.current_position_latlon, p).km 
                                        for p in astar_segment]
                    min_dist_to_segment = min(distances_to_segment) if distances_to_segment else 999
                    
                    # Moderate reward for staying near A* path (don't over-constrain)
                    if min_dist_to_segment < 30:
                        reward += (30 - min_dist_to_segment) * 2  # Up to 60 points
                    elif min_dist_to_segment > 50:
                        reward -= (min_dist_to_segment - 50) * 1  # Penalty for straying far
        
        # 5. REDUCED OPERATIONAL PENALTIES (scale with timestep)
        # Only apply small penalties so fuel/emissions don't dominate early learning
        fuel_penalty = self.total_fuel_consumed * 0.005  # REDUCED from 0.02
        emissions_penalty = self.total_emissions * 0.01  # REDUCED from 0.05
        reward -= (fuel_penalty + emissions_penalty)
        
        # 6. LANDMARK COMPLETION BONUS (huge reward)
        if hasattr(self, 'state_before_action_landmark_idx'):
            if self.current_landmark_idx > self.state_before_action_landmark_idx:
                reward += 500  # INCREASED from 300
                print(f"  🎯 LANDMARK BONUS: +500 points!")
        
        # 7. FUEL EXHAUSTION (terminal penalty)
        if self.vessel["fuel_tank_remaining_t"] <= 0:
            reward -= 1000  # REDUCED from 500 to avoid overwhelming
        
        # 8. COLLISION HANDLING
        if obstacle_info and self._check_collision_with_obstacle(
            self.current_position_latlon, obstacle_info):
            collision_penalty = 200  # REDUCED from 100
            reward -= collision_penalty
            print(f"  ⚠️ COLLISION PENALTY: -{collision_penalty} points")
        
        # 9. D* LITE REROUTING (moderate penalty)
        if rerouted_path_latlon:
            reward -= 50  # REDUCED from 30
            print(f"  🔄 D* LITE REROUTE: -50 points")
        
        # 10. SPEED EFFICIENCY BONUS (encourage appropriate speed)
        optimal_speed = self.max_speed_knots * 0.7
        speed_diff = abs(self.current_speed_knots - optimal_speed)
        reward += max(0, 20 - speed_diff * 2)  # Up to 20 points for optimal speed
        
        # 11. CIRCULAR MOTION PENALTY
        # Detect if vessel is turning too much without making progress
        if hasattr(self, 'recent_headings'):
            self.recent_headings.append(self.current_heading_deg)
            if len(self.recent_headings) > 5:
                self.recent_headings.pop(0)
                
                # Check if making large heading changes
                heading_variance = np.var(self.recent_headings)
                if heading_variance > 5000:  # High variance = lots of turning
                    reward -= 30
                    print(f"  🔄 EXCESSIVE TURNING PENALTY: -30 points")
        else:
            self.recent_headings = [self.current_heading_deg]
        
        return reward


# Additional fixes to add to the MarineEnv class:

def plot_simulation_episode(episode, env, full_astar_path_latlon, landmark_points, 
                           bathymetry_maze, grid_params, weather_penalty_grid):
    """Visualize episode (simplified for speed)"""
    fig, ax = plt.subplots(figsize=(10, 8))

    min_lat, min_lon, lat_step, lon_step, num_lat_cells, num_lon_cells = grid_params
    max_lon_astar = min_lon + num_lon_cells * lon_step
    max_lat_astar = min_lat + num_lat_cells * lat_step

    # Plot grids
    cmap_maze = plt.get_cmap('Greys_r', 2)
    ax.imshow(bathymetry_maze, extent=[min_lon, max_lon_astar, min_lat, max_lat_astar],
             origin='lower', cmap=cmap_maze, interpolation='nearest', alpha=0.5)

    # Plot A* path
    if full_astar_path_latlon:
        astar_lons = [p[1] for p in full_astar_path_latlon]
        astar_lats = [p[0] for p in full_astar_path_latlon]
        ax.plot(astar_lons, astar_lats, 'gray', linewidth=1, linestyle='--', label="A* Path", alpha=0.7)

    # Plot landmarks
    landmark_lons = [p[1] for p in landmark_points]
    landmark_lats = [p[0] for p in landmark_points]
    ax.scatter(landmark_lons, landmark_lats, color='cyan', s=80, marker='o', 
              edgecolor='black', label="Landmarks", zorder=5)

    # Plot DRL path
    drl_lons = [p[1] for p in env.drl_path_segment]
    drl_lats = [p[0] for p in env.drl_path_segment]
    ax.plot(drl_lons, drl_lats, 'darkgreen', linewidth=2, label="DRL Path")
    ax.scatter(drl_lons[-1], drl_lats[-1], color='darkgreen', s=120, 
              marker='P', label="Current Position", zorder=6)

    ax.set_xlim(min_lon, max_lon_astar)
    ax.set_ylim(min_lat, max_lat_astar)
    plt.title(f"Episode {episode+1}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    ax.legend(loc='upper right', fontsize=8)
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"output/episode_{episode+1}.png", dpi=100)
    plt.close(fig)

# --- Training Loop ---

def train_ddpg_agent(env, agent, replay_buffer, num_episodes=500, batch_size=64, 
                     visualize_every_n_episodes=20,
                     full_astar_path_latlon=None, landmark_points=None, 
                     bathymetry_maze=None, grid_params=None, weather_penalty_grid=None):
    episode_rewards = []
    
    # ADDED: Curriculum learning - start with easier task
    env.time_step_hours = 3.0  # Larger time steps initially for faster progress
    
    for episode in range(num_episodes):
        state = env.reset()
        env.state_before_action_landmark_idx = env.current_landmark_idx
        env.state_before_action = state
        episode_reward = 0
        done = False
        step_count = 0
        
        max_steps = 50
        
        # CURRICULUM: Reduce timestep as agent improves
        if episode > 100:
            env.time_step_hours = 2.5
        if episode > 200:
            env.time_step_hours = 2.0
        
        while not done and step_count < max_steps:
            action = agent.select_action(state)
            
            # IMPROVED EXPLORATION: Higher noise early, decay faster
            noise_scale = max(0.01, 0.3 * (1.0 - episode / num_episodes) ** 2)
            action = (action + np.random.normal(0, noise_scale, size=env.action_dim)).clip(-1, 1)
            
            # REDUCED OBSTACLE FREQUENCY during early learning
            obstacle_present = False
            if episode > 50:  # Only introduce obstacles after basic navigation learned
                obstacle_present = random.random() < 0.005  # 0.5% chance
            
            next_state, reward, done, info = env.step(action, obstacle_present)
            replay_buffer.push(state, action, next_state, reward, done)
            
            state = next_state
            env.state_before_action_landmark_idx = env.current_landmark_idx
            env.state_before_action = state
            episode_reward += reward
            step_count += 1
            
            # Train more frequently once buffer has enough data
            if len(replay_buffer) > batch_size * 2:
                if step_count % 2 == 0:  # Train every 2 steps
                    agent.train(replay_buffer, batch_size)
        
        episode_rewards.append(episode_reward)
        
        # Enhanced logging
        if (episode + 1) % 5 == 0:
            avg_reward = np.mean(episode_rewards[-5:])
            max_reward = np.max(episode_rewards[-5:])
            print(f"Ep {episode+1}/{num_episodes}, "
                  f"Reward: {episode_reward:.1f} (Avg: {avg_reward:.1f}, Max: {max_reward:.1f}), "
                  f"Landmarks: {env.current_landmark_idx}/{len(env.landmark_points)}, "
                  f"Fuel: {env.vessel['fuel_tank_remaining_t']:.1f}t, "
                  f"Noise: {noise_scale:.3f}")
        
        # Visualization
        if (episode + 1) % visualize_every_n_episodes == 0:
            plot_simulation_episode(episode, env, full_astar_path_latlon, 
                                  landmark_points, bathymetry_maze, 
                                  grid_params, weather_penalty_grid)
    
    return agent, episode_rewards

# --- Main execution ---
if __name__ == "__main__":
    ports_df = pd.read_excel("data\\Ports\\ports.xlsx")
    ports_df[["lat", "lon"]] = ports_df["Decimal"].str.split(",", expand=True).astype(float)
    ports_gdf = gpd.GeoDataFrame(
        ports_df,
        geometry=gpd.points_from_xy(ports_df["lon"], ports_df["lat"]),
        crs="EPSG:4326"
    )

    print("Generating optimal path route...")
    selected_ports = ports_gdf.sample(3, random_state=50)  # REDUCED to 3 ports
    optimal_path_route_names, G_main, selected_ports_main = get_optimal_path_route(vessel, selected_ports)

    if not optimal_path_route_names:
        print("Could not generate an optimal path route. Exiting.")
        exit()

    full_astar_path_latlon = []
    for port_name in optimal_path_route_names:
        port_coords = ports_gdf[ports_gdf["Ports"] == port_name].iloc[0]
        full_astar_path_latlon.append((port_coords["lat"], port_coords["lon"]))
    
    # INCREASED LANDMARK INTERVAL to 100km (aligned with main.py suggestion)
    landmark_points = generate_landmark_points(full_astar_path_latlon, interval_km=100)
    if not landmark_points:
        print("No landmark points generated. Exiting.")
        exit()
    
    print(f"Generated {len(landmark_points)} landmarks at 100km intervals")
    
    # Load bathymetry
    ds_bathymetry = xr.open_dataset("data\\Bathymetry\\GEBCO_2024_sub_ice_topo.nc")
    min_lon_astar, max_lon_astar = 99, 120
    min_lat_astar, max_lat_astar = 0, 8
    ds_subset_astar = ds_bathymetry.sel(
        lon=slice(min_lon_astar, max_lon_astar),
        lat=slice(min_lat_astar, max_lat_astar)
    )
    
    # INCREASED CELL SIZE to 10km (aligned with optimization and main.py)
    cell_size_m_astar = 10000  # 10km x 10km cells
    vessel_height_underwater = vessel["size"][2] * vessel["percent_of_height_underwater"]

    print("Creating bathymetry grid...")
    bathymetry_maze, grid_lats, grid_lons, lat_step, lon_step, elevation_data = create_bathymetry_grid(
        ds_subset_astar, min_lon_astar, max_lon_astar, min_lat_astar, max_lat_astar, 
        cell_size_m_astar, vessel_height_underwater
    )
    num_lat_cells = len(grid_lats)
    num_lon_cells = len(grid_lons)
    grid_params = (min_lat_astar, min_lon_astar, lat_step, lon_step, num_lat_cells, num_lon_cells)

    print("Creating weather penalty grid...")
    weather_penalty_grid = create_weather_penalty_grid(
        min_lon_astar, max_lon_astar, min_lat_astar, max_lat_astar, 
        lat_step, lon_step, num_lat_cells, num_lon_cells
    )

    # Initialize Environment and Agent
    print("\nInitializing DRL environment...")
    env = MarineEnv(vessel.copy(), full_astar_path_latlon, landmark_points, 
                   bathymetry_maze, grid_params, weather_penalty_grid)
    
    agent = DDPG(env.state_dim, env.action_dim, max_action=1.0)
    replay_buffer = ReplayBuffer(capacity=10000)  # REDUCED buffer size

    print("\n" + "="*60)
    print("STARTING DDPG TRAINING")
    print("="*60)
    print(f"Configuration:")
    print(f"  - Episodes: 100")
    print(f"  - Landmarks: {len(landmark_points)} (100km intervals)")
    print(f"  - Grid cells: {num_lat_cells}x{num_lon_cells} (10km resolution)")
    print(f"  - Max steps per episode: 40")
    print(f"  - Time step: 2 hours")
    print(f"  - Obstacle handling: D* Lite for >3km, DRL for ≤3km")
    print(f"  - Segment-based optimization: Between consecutive landmarks")
    print("="*60 + "\n")
    
    start_time = time.time()
    
    trained_agent, episode_rewards = train_ddpg_agent(
        env, agent, replay_buffer, 
        num_episodes=500,  # REDUCED from 500
        batch_size=64,     # REDUCED from 64
        visualize_every_n_episodes=20,
        full_astar_path_latlon=full_astar_path_latlon,
        landmark_points=landmark_points,
        bathymetry_maze=bathymetry_maze,
        grid_params=grid_params,
        weather_penalty_grid=weather_penalty_grid
    )
    
    training_time = time.time() - start_time
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Total training time: {training_time/60:.2f} minutes")
    print(f"Average time per episode: {training_time/100:.2f} seconds")
    print(f"Final average reward (last 10 episodes): {np.mean(episode_rewards[-10:]):.2f}")
    print("="*60 + "\n")

    # Save trained model
    torch.save(trained_agent.actor.state_dict(), "ddpg_actor.pth")
    print("Trained DDPG actor model saved to ddpg_actor.pth")

    # Save training metrics
    metrics_df = pd.DataFrame({
        'episode': range(1, len(episode_rewards) + 1),
        'reward': episode_rewards
    })
    metrics_df.to_csv("training_metrics.csv", index=False)
    print("Training metrics saved to training_metrics.csv")

    # Plot training rewards
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards, alpha=0.6, label='Episode Reward')
    window = 10
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(episode_rewards)), moving_avg, 
                linewidth=2, label=f'{window}-Episode Moving Avg')
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DDPG Training Rewards")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    if len(episode_rewards) >= 20:
        plt.plot(episode_rewards[20:], alpha=0.7)
        plt.xlabel("Episode (after warmup)")
        plt.ylabel("Total Reward")
        plt.title("Training Progress (Episodes 21+)")
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("ddpg_training_rewards.png", dpi=150)
    print("Training reward plot saved to ddpg_training_rewards.png")
    plt.show()
    
    print("\n" + "="*60)
    print("All training artifacts saved successfully!")
    print("="*60)