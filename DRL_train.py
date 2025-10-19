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
from scipy.spatial import KDTree # Added for faster nearest neighbor search

# Import necessary functions from utils.py and d_star_lite.py
from utils import (
    vessel, vessel_height_underwater, safe_float, get_wave_wind, interpolate_points,
    reorder_route, compute_edge_weight, enforce_tsp_constraints,
    locations_data, location_codes, location_weather_cache,
    is_point_in_polygon, scrape_weather_data, get_weather_for_point,
    get_weather_penalty, calculate_vessel_range_km, simulate_vessel_route,
    get_optimal_path_route, haversine, generate_landmark_points,
    create_weather_penalty_grid, Node, astar, create_bathymetry_grid,
    lat_lon_to_grid_coords, grid_coords_to_lat_lon,
    generate_optimized_route_and_landmarks, create_astar_grid # Added this import
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
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)  # REDUCED from 1e-4

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=5e-4)  # REDUCED from 1e-3

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
    def _normalize_state(self, state):
        """Normalize state values to reasonable ranges for neural network"""
        # State: [lat, lon, speed, heading, target_lat, target_lon, 
        #         wave_h, wave_p, wave_d, wind_wave_h, wind_wave_p, wind_wave_d, fuel, distance]
        
        normalized = state.copy()
        
        # Normalize lat/lon to [-1, 1] based on grid bounds
        normalized[0] = (state[0] - self.min_lat) / (8.0) * 2 - 1  # lat
        normalized[1] = (state[1] - self.min_lon) / (21.0) * 2 - 1  # lon
        normalized[4] = (state[4] - self.min_lat) / (8.0) * 2 - 1  # target_lat
        normalized[5] = (state[5] - self.min_lon) / (21.0) * 2 - 1  # target_lon
        
        # Normalize speed [0, max_speed] to [0, 1]
        normalized[2] = state[2] / self.max_speed_knots
        
        # Normalize heading [0, 360] to [-1, 1]
        normalized[3] = (state[3] / 180.0) - 1.0
        
        # Normalize wave/wind data (typical ranges)
        normalized[6] = state[6] / 5.0  # wave_height (0-5m typical)
        normalized[7] = state[7] / 15.0  # wave_period (0-15s typical)
        normalized[8] = (state[8] / 180.0) - 1.0  # wave_direction
        normalized[9] = state[9] / 5.0  # wind_wave_height
        normalized[10] = state[10] / 15.0  # wind_wave_period
        normalized[11] = (state[11] / 180.0) - 1.0  # wind_wave_direction
        
        # Normalize fuel [0, max_capacity] to [0, 1]
        normalized[12] = state[12] / self.vessel["fuel_tank_capacity_t"]
        
        # Normalize distance (typical max distance between landmarks)
        normalized[13] = state[13] / 200.0  # Clamp to [0, 1]

        # Normalize distance to A* path grid (typical max distance)
        normalized[14] = state[14] / 200.0  # Clamp to [0, 1]

        # Normalize bearing to A* path [0, 360] to [-1, 1]
        normalized[15] = (state[15] / 180.0) - 1.0
        
        return normalized

    def _get_state(self):
        """Modified to return normalized state"""
        current_lat, current_lon = self.current_position_latlon
        
        # Handle case where all landmarks are visited
        if self.current_landmark_idx >= len(self.landmark_points):
            target_landmark_lat, target_landmark_lon = self.landmark_points[-1] # Target the last landmark
        else:
            target_landmark_lat, target_landmark_lon = self.landmark_points[self.current_landmark_idx]

        # Get cached weather data (keep existing code)
        cache_key = (round(current_lat, 1), round(current_lon, 1))
        if cache_key in self.weather_cache:
            sea_currents = self.weather_cache[cache_key]
        else:
            if self.weather_cache:
                nearest_key = min(
                    self.weather_cache.keys(), 
                    key=lambda k: (k[0] - current_lat)**2 + (k[1] - current_lon)**2
                )
                sea_currents = self.weather_cache[nearest_key]
            else:
                sea_currents = {
                    "wave_height": 0.0, "wave_period": 0.0, "wave_direction": 0.0,
                    "wind_wave_height": 0.0, "wind_wave_period": 0.0, "wind_wave_direction": 0.0
                }

        distance_to_target_landmark = geodesic(
            self.current_position_latlon, 
            self.landmark_points[self.current_landmark_idx]
        ).km

        # Calculate minimum distance and closest point to A* path grid
        min_dist_to_astar_path_grid, closest_astar_point = self._get_min_distance_to_astar_path_grid(current_lat, current_lon)
        
        # Calculate bearing to the closest A* path point
        bearing_to_astar_path = self._calculate_bearing(self.current_position_latlon, closest_astar_point)

        state = np.array([
            current_lat, current_lon,
            self.current_speed_knots, self.current_heading_deg,
            target_landmark_lat, target_landmark_lon,
            sea_currents["wave_height"], sea_currents["wave_period"], sea_currents["wave_direction"],
            sea_currents["wind_wave_height"], sea_currents["wind_wave_period"], sea_currents["wind_wave_direction"],
            self.vessel["fuel_tank_remaining_t"],
            distance_to_target_landmark,
            min_dist_to_astar_path_grid, # New state variable
            bearing_to_astar_path # New state variable: bearing to closest A* path point
        ], dtype=np.float32)
        
        # NORMALIZE before returning
        return self._normalize_state(state)
    
    def _calculate_reward(self, reward_info, episode): # Added episode parameter
        """
        REDESIGNED REWARD SYSTEM v2 - Balanced and scaled properly
        
        Key changes:
        1. Reduced penalty magnitudes (were causing -150k rewards)
        2. Stronger emphasis on distance reduction
        3. Earlier detection of stuck behavior
        4. Reward shaping to guide agent even when far from target
        """
        reward_breakdown = {
            'progress_reward': 0.0,
            'regression_penalty': 0.0,
            'proximity_bonus': 0.0,
            'distance_penalty': 0.0,
            'heading_reward': 0.0,
            'landmark_bonus': 0.0,
            'stuck_penalty': 0.0,
            'fuel_cost_penalty': 0.0,
            'emissions_penalty': 0.0,
            'fuel_exhaustion_penalty': 0.0,
            'collision_penalty': 0.0,
            'rerouting_penalty': 0.0,
            'revisit_landmark_penalty': 0.0,
            'speed_efficiency_penalty': 0.0,
            'too_slow_penalty': 0.0,
            'circular_motion_penalty': 0.0,
            'excessive_turning_penalty': 0.0,
            'land_collision_penalty': 0.0, # Added for land collision
            'out_of_bounds_penalty': 0.0, # Added for out of bounds
            'astar_path_following_reward': 0.0, # New reward for A* path following
            'total_reward': 0.0
        }
        
        # Extract info
        prev_distance = reward_info['prev_distance']
        new_distance = reward_info['new_distance']
        prev_landmark_idx = reward_info['prev_landmark_idx']
        rerouted_path = reward_info['rerouted_path']
        obstacle_info = reward_info['obstacle_info']
        distance_travelled = reward_info['distance_travelled']
        revisited_landmark = reward_info['revisited_landmark'] # Extract the new flag
        land_collision_occurred = reward_info.get('land_collision_occurred', False)
        out_of_bounds_occurred = reward_info.get('out_of_bounds_occurred', False)

        # Extract A* path info from the current state (last element)
        current_state = self._get_state() # Get the current (normalized) state
        min_dist_to_astar_path_grid = current_state[14] * 200.0 # Denormalize

        # ============================================================================
        # 1. PROGRESS REWARD (PRIMARY SIGNAL)
        # ============================================================================
        progress = prev_distance - new_distance  # Positive = moving closer
        
        if progress > 2:
            base_reward = progress / 2 * 4.0
            efficiency = min(1.0, progress / max(distance_travelled, 0.1))
            if efficiency > 0.8:
                base_reward *= 1.3
            reward_breakdown['progress_reward'] = base_reward
        else:
            regression_penalty = abs(progress) * 7.5
            reward_breakdown['regression_penalty'] = -regression_penalty
        
        # ============================================================================
        # 2. DISTANCE SHAPING (SECONDARY - PROVIDES GRADIENT)
        # ============================================================================
        if new_distance < 10:
            proximity_bonus = (10 - new_distance) * 25.0
            reward_breakdown['proximity_bonus'] = proximity_bonus
        
        distance_penalty = new_distance * 0.10
        reward_breakdown['distance_penalty'] = -distance_penalty
        
        # ============================================================================
        # 3. HEADING ALIGNMENT (MODERATE WEIGHT)
        # ============================================================================
        if self.current_landmark_idx < len(self.landmark_points):
            current_target = self.landmark_points[self.current_landmark_idx]
            target_bearing = self._calculate_bearing(self.current_position_latlon, current_target)
            heading_diff = abs(self.current_heading_deg - target_bearing)
            if heading_diff > 180:
                heading_diff = 360 - heading_diff
            
            heading_reward = (180 - heading_diff) / 180.0 * 13 # heading reward multiplier
            reward_breakdown['heading_reward'] = heading_reward
        
        # ============================================================================
        # 4. LANDMARK COMPLETION (HUGE BONUS)
        # ============================================================================
        if self.current_landmark_idx > prev_landmark_idx:
            if self.current_landmark_idx not in self.bonus_received_landmarks:
                landmark_bonus = 2300.0
                reward_breakdown['landmark_bonus'] = landmark_bonus
                self.bonus_received_landmarks.add(self.current_landmark_idx)
                print(f"  🎯 LANDMARK REACHED! +{landmark_bonus} points (first time)")
            else:
                print(f"  🎯 LANDMARK REACHED! (already received bonus)")
        
        # ============================================================================
        # 5. MOVEMENT REQUIREMENT (PREVENT STAYING STILL)
        # ============================================================================
        if distance_travelled < 3.0:
            stuck_penalty = (3.0 - distance_travelled) * 80.0
            reward_breakdown['stuck_penalty'] = -stuck_penalty
            if distance_travelled < 1.0:
                print(f"  🐌 BARELY MOVING: -{stuck_penalty:.1f} (only {distance_travelled:.1f}km)")
        
        # ============================================================================
        # 6. OPERATIONAL COSTS (MINIMAL)
        # ============================================================================
        fuel_cost_penalty = self.total_fuel_consumed * 0.002
        emissions_penalty = self.total_emissions * 0.005
        reward_breakdown['fuel_cost_penalty'] = -fuel_cost_penalty
        reward_breakdown['emissions_penalty'] = -emissions_penalty
        
        # ============================================================================
        # 7. FAILURE PENALTIES (MODERATE - NOT OVERWHELMING)
        # ============================================================================
        
        if self.vessel["fuel_tank_remaining_t"] <= 0:
            reward_breakdown['fuel_exhaustion_penalty'] = -500.0
            print(f"  ⛽ FUEL EXHAUSTED: -500")
        
        if obstacle_info and self._check_collision_with_obstacle(
            self.current_position_latlon, obstacle_info):
            reward_breakdown['collision_penalty'] = -2000.0
            print(f"  💥 COLLISION: -200")
        
        if rerouted_path:
            reward_breakdown['rerouting_penalty'] = -50.0
        
        if revisited_landmark:
            revisit_penalty = 500.0
            reward_breakdown['revisit_landmark_penalty'] = -revisit_penalty
            print(f"  ↩️ REVISITED LANDMARK: -{revisit_penalty}")

        # Land/Out of bounds penalties (handled in step, but added to breakdown here for completeness)
        if land_collision_occurred:
            # Scale land collision penalty with episode number
            base_penalty = 600.0
            scaled_penalty = base_penalty * max(1, episode / 300) # Increase penalty by 1% per episode
            reward_breakdown['land_collision_penalty'] = -scaled_penalty
            print(f"  ⚠️ LAND COLLISION: -{scaled_penalty:.2f} (scaled by episode {episode})")
        if out_of_bounds_occurred:
            reward_breakdown['out_of_bounds_penalty'] = -1000.0 # Consistent with step()

        # ============================================================================
        # 8. SPEED EFFICIENCY (SMALL PENALTY)
        # ============================================================================
        optimal_speed = self.max_speed_knots * 0.75
        speed_diff = abs(self.current_speed_knots - optimal_speed)
        speed_penalty = speed_diff * 0.5
        reward_breakdown['speed_efficiency_penalty'] = -speed_penalty
        
        if self.current_speed_knots < self.max_speed_knots * 0.4:
            reward_breakdown['too_slow_penalty'] = -7.0
        
        # ============================================================================
        # 9. ANTI-CIRCULAR MOTION (EARLY DETECTION)
        # ============================================================================
        
        if not hasattr(self, 'position_history'):
            self.position_history = []
        
        self.position_history.append(self.current_position_latlon)
        if len(self.position_history) > 5:
            self.position_history.pop(0)
            
            if len(self.position_history) == 5:
                lats = [p[0] for p in self.position_history]
                lons = [p[1] for p in self.position_history]
                
                lat_range = max(lats) - min(lats)
                lon_range = max(lons) - min(lons)
                
                lat_range_km = lat_range * 111.0
                lon_range_km = lon_range * 111.0 * cos(radians(lats[0]))
                
                if lat_range_km < 15 and lon_range_km < 15:
                    circular_penalty = 1000.0
                    reward_breakdown['circular_motion_penalty'] = -circular_penalty
                    print(f"  🔄 CIRCULAR MOTION: -{circular_penalty} (range: {lat_range_km:.1f}x{lon_range_km:.1f}km)")
                    
                    self.position_history = [self.current_position_latlon]
        
        if not hasattr(self, 'heading_history'):
            self.heading_history = []
        
        self.heading_history.append(self.current_heading_deg)
        if len(self.heading_history) > 4:
            self.heading_history.pop(0)
            
            if len(self.heading_history) == 4:
                total_change = 0
                for i in range(len(self.heading_history) - 1):
                    diff = abs(self.heading_history[i+1] - self.heading_history[i])
                    if diff > 180:
                        diff = 360 - diff
                    total_change += diff
                
                if total_change > 90:
                    zigzag_penalty = 200.0
                    reward_breakdown['excessive_turning_penalty'] = -zigzag_penalty
                    print(f"  ↩️ EXCESSIVE TURNING: -{zigzag_penalty} ({total_change:.0f}° in 4 steps)")
        
        # ============================================================================
        # 10. A* PATH FOLLOWING REWARD (GUIDANCE)
        # ============================================================================
        astar_following_reward = 0.0
        if min_dist_to_astar_path_grid < 60: # Only reward/penalize if relatively close to A* path
            # Reward for being close to the A* path
            astar_proximity_reward = (60 - min_dist_to_astar_path_grid) * 0.5 # Max 25 reward if on path
            astar_following_reward += astar_proximity_reward
        else:
            # Penalty for being too far from the A* path
            astar_following_reward -= (min_dist_to_astar_path_grid - 50) * 0.25 # Penalty increases with distance

        reward_breakdown['astar_path_following_reward'] = astar_following_reward
        
        # ============================================================================
        # 11. REWARD CLIPPING (PREVENT EXTREME VALUES)
        # ============================================================================
        total_reward = sum(reward_breakdown.values())
        reward_breakdown['total_reward'] = np.clip(total_reward, -2500, 2500)
        
        return reward_breakdown

    def __init__(self, initial_vessel_state, full_astar_path_latlon, landmark_points, 
             bathymetry_maze, grid_params, weather_penalty_grid):
        self.vessel = initial_vessel_state
        self.full_astar_path_latlon = full_astar_path_latlon
        self.landmark_points = landmark_points
        self.bathymetry_maze = bathymetry_maze
        self.grid_params = grid_params
        self.weather_penalty_grid = weather_penalty_grid

        self.min_lat, self.min_lon, self.lat_step, self.lon_step, self.num_lat_cells, self.num_lon_cells = grid_params

        # CRITICAL: Define these BEFORE reset() is called
        # State: [lat, lon, speed, heading, target_lat, target_lon, 
        #         wave_h, wave_p, wave_d, wind_wave_h, wind_wave_p, wind_wave_d, fuel, distance,
        #         min_dist_to_astar_path_grid, bearing_to_astar_path]
        self.state_dim = 16 # Increased state dimension by 1
        self.action_dim = 2
        self.max_speed_knots = self.vessel["speed_knots"]
        self.max_heading_change_deg = 30
        self.time_step_hours = 2.0
        self.astar_proximity_threshold_km = 10.0 # New constant for A* path proximity check

        # Initialize tracking variables
        self.current_position_latlon = None
        self.current_heading_deg = None
        self.current_speed_knots = None
        self.current_landmark_idx = 0
        self.total_fuel_consumed = 0.0
        self.total_emissions = 0.0
        self.total_eta_hours = 0.0
        self.drl_path_segment = []
        self.full_episode_drl_path = [] # Added for full episode path tracking
        self.rerouted_paths_history = []
        self.visited_landmarks = set() # Initialize set to store visited landmark indices

        # Create A* path grid
        self.astar_path_grid = create_astar_grid(full_astar_path_latlon, grid_params)
        print(f"A* path grid created with {np.sum(self.astar_path_grid)} path cells.")

        # Pre-compute (lat, lon) coordinates of all A* path cells for faster distance calculation
        self.astar_path_latlon_points = []
        for r in range(self.num_lat_cells):
            for c in range(self.num_lon_cells):
                if self.astar_path_grid[r][c] == 1:
                    lat, lon = grid_coords_to_lat_lon(r, c, *self.grid_params[:4])
                    self.astar_path_latlon_points.append((lat, lon))
        print(f"Pre-computed {len(self.astar_path_latlon_points)} A* path (lat,lon) points.")

        # Create a KDTree for efficient nearest neighbor queries on A* path points
        if self.astar_path_latlon_points:
            self.astar_path_kdtree = KDTree(self.astar_path_latlon_points)
            print("KDTree created for A* path points.")
        else:
            self.astar_path_kdtree = None
            print("Warning: No A* path points to build KDTree.")

        # PRE-CACHE WEATHER DATA (keep your existing code)
        self.weather_cache = {}
        print("Pre-caching weather data for strategic grid points...")
        
        cache_interval = 40
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
                cache_key = (round(lat, 1), round(lon, 1))
                
                if cache_key not in self.weather_cache:
                    try:
                        self.weather_cache[cache_key] = get_wave_wind(lat, lon)
                        cached_count += 1
                        if cached_count % 10 == 0:
                            print(f"    Cached {cached_count}/{total_cache_points} points...", end='\r')
                    except Exception as e:
                        self.weather_cache[cache_key] = {
                            "wave_height": 0.0, "wave_period": 0.0, "wave_direction": 0.0,
                            "wind_wave_height": 0.0, "wind_wave_period": 0.0, "wind_wave_direction": 0.0
                        }
        
        print(f"\n  Weather cache loaded with {len(self.weather_cache)} strategic points")
        print(f"  Coverage: Each point represents ~50km radius (~5 grid cells)")

        # Pre-compute A* segments (keep your existing code)
        self.landmark_astar_segments = {}
        print("Pre-computing A* segments between landmarks...")
        for i in range(len(landmark_points) - 1):
            segment_key = (i, i+1)
            start_landmark = landmark_points[i]
            end_landmark = landmark_points[i+1]
            
            segment_points = []
            in_segment = False
            for p in full_astar_path_latlon:
                if geodesic(p, start_landmark).km < 5.0:
                    in_segment = True
                if in_segment:
                    segment_points.append(p)
                if geodesic(p, end_landmark).km < 5.0:
                    break
            
            self.landmark_astar_segments[segment_key] = segment_points if segment_points else [start_landmark, end_landmark]
        print(f"Pre-computed {len(self.landmark_astar_segments)} A* segments")

        # NOW call reset() - all attributes are defined
        self.reset()

    def _get_min_distance_to_astar_path_grid(self, current_lat, current_lon):
        """
        Calculates the minimum geodesic distance from the current position
        to any 'on-path' cell in the astar_path_grid.
        """
        if self.astar_path_kdtree is None or not self.astar_path_latlon_points:
            return 200.0, (current_lat, current_lon) # Return max distance and current position as dummy

        avg_lat_rad = radians((self.min_lat + self.num_lat_cells * self.lat_step) / 2)
        deg_per_km_lat = 1 / 111.0
        radius_deg = self.astar_proximity_threshold_km * deg_per_km_lat

        indices_in_range = self.astar_path_kdtree.query_ball_point(
            (current_lat, current_lon), 
            r=radius_deg
        )
        
        min_geodesic_dist = 200.0
        closest_astar_point = (current_lat, current_lon) # Default to current position

        if indices_in_range:
            geodesic_distances = []
            for idx in indices_in_range:
                path_lat, path_lon = self.astar_path_latlon_points[idx]
                dist = geodesic((current_lat, current_lon), (path_lat, path_lon)).km
                geodesic_distances.append((dist, (path_lat, path_lon)))
            
            if geodesic_distances:
                min_geodesic_dist, closest_astar_point = min(geodesic_distances, key=lambda x: x[0])
        
        return min_geodesic_dist, closest_astar_point
    def reset(self):
        """Enhanced reset with all tracking variables properly initialized and robust landmark handling"""
        if not self.landmark_points:
            print("Error: No landmark points available for reset. Cannot initialize environment.")
            # Depending on desired behavior, you might raise an exception or exit here.
            # For now, we'll set a dummy state and mark as done.
            self.current_position_latlon = (0.0, 0.0)
            self.current_heading_deg = 0.0
            self.current_speed_knots = 0.0
            self.current_landmark_idx = 0 # No landmarks to target
            self.total_fuel_consumed = self.vessel["fuel_tank_capacity_t"] # Mark as out of fuel
            self.total_emissions = 0.0
            self.total_eta_hours = 0.0
            self.vessel["fuel_tank_remaining_t"] = 0.0
            self.drl_path_segment = []
            self.rerouted_paths_history = []
            self.position_history = []
            self.heading_history = []
            self.visited_landmarks.clear()
            return self._normalize_state(np.zeros(self.state_dim, dtype=np.float32)) # Return a dummy normalized state

        self.current_position_latlon = self.landmark_points[0]
        
        if len(self.landmark_points) > 1:
            self.current_heading_deg = self._calculate_bearing(
                self.landmark_points[0], self.landmark_points[1]
            )
            self.current_landmark_idx = 1 # Target the second landmark
        else:
            # If only one landmark (start point), it's effectively the destination
            self.current_heading_deg = 0.0 # No clear next heading
            self.current_landmark_idx = 0 # Target the first (and only) landmark
        
        self.current_speed_knots = self.vessel["speed_knots"] * 0.7
        self.total_fuel_consumed = 0.0
        self.total_emissions = 0.0
        self.total_eta_hours = 0.0
        self.vessel["fuel_tank_remaining_t"] = self.vessel["fuel_tank_capacity_t"]
        self.drl_path_segment = [self.current_position_latlon]
        self.full_episode_drl_path = [self.current_position_latlon] # Initialize full path
        
        # Initialize ALL tracking lists
        self.recent_headings = [self.current_heading_deg]
        self.recent_positions = [self.current_position_latlon]
        self.rerouted_paths_history = []
        self.position_history = [self.current_position_latlon]
        self.heading_history = [self.current_heading_deg]
        self.visited_landmarks.clear() # Clear visited landmarks on reset
        self.visited_landmarks.add(0) # Add the starting landmark (index 0)
        self.bonus_received_landmarks = set() # Initialize set to track landmarks for which bonus has been received
        self.bonus_received_landmarks.add(0) # Bonus for starting landmark is implicitly received
        
        # A* path tracking is no longer needed as we use the grid
        # self.current_astar_path_idx = 0

        print(f"DEBUG: Resetting episode. Starting position: {self.current_position_latlon}, Target landmark index: {self.current_landmark_idx}")

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

    def step(self, action, obstacle_present=False, episode=0): # Added episode parameter
        """Enhanced step function with proper state tracking"""
        # CRITICAL: Store previous state BEFORE taking action
        prev_position = self.current_position_latlon
        prev_landmark_idx = self.current_landmark_idx
        
        # Store distance to target before action
        if self.current_landmark_idx < len(self.landmark_points):
            current_target = self.landmark_points[self.current_landmark_idx]
            prev_distance_to_target = geodesic(prev_position, current_target).km
        else:
            prev_distance_to_target = 0.0
        
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

        # Check for land collision (bathymetry depth > 0)
        is_on_land = False
        out_of_bounds_occurred = False # New flag for clarity
        grid_lat, grid_lon = lat_lon_to_grid_coords(new_lat, new_lon, *self.grid_params)
        
        # Ensure grid coordinates are within bounds before accessing bathymetry_maze
        if 0 <= int(grid_lat) < self.num_lat_cells and 0 <= int(grid_lon) < self.num_lon_cells:
            # Correct indexing for a list of lists (2D array)
            if self.bathymetry_maze[int(grid_lat)][int(grid_lon)] > 0: # Assuming >0 means land
                is_on_land = True
                new_position_latlon = prev_position # Revert position
                # Removed: done = True # End episode on land collision
        else:
            # If outside grid bounds, treat as land or invalid area
            out_of_bounds_occurred = True
            print(f"  ⚠️ OUT OF BOUNDS at {new_position_latlon}! Treating as land.")
            new_position_latlon = prev_position # Revert position
            # Removed: done = True # End episode on out of bounds

        # OBSTACLE HANDLING (keep your existing code)
        rerouted_path_latlon = None
        obstacle_info = None
        
        if obstacle_present:
            obstacle_info = self._generate_obstacle()
            obstacle_size_km = obstacle_info['size_km']
            
            if self._check_collision_with_obstacle(new_position_latlon, obstacle_info):
                print(f"Obstacle detected! Size: {obstacle_size_km:.2f} km")
                
                if obstacle_size_km > 3.0:
                    print(f"Large obstacle ({obstacle_size_km:.2f} km) - Using D* Lite rerouting")
        # ... your D* Lite code ...
                    pass

        # If a land collision or out of bounds occurred, the position has already been reverted.
        # The episode will not terminate immediately, but a penalty will be applied.
        # The agent will get a chance to learn from this penalty in the next step.

        self.current_position_latlon = new_position_latlon
        self.drl_path_segment.append(new_position_latlon)
        self.full_episode_drl_path.append(new_position_latlon) # Append to full episode path
        self.total_eta_hours += self.time_step_hours

        # A* path tracking is no longer needed as we use the grid
        # if self.current_astar_path_idx < len(self.full_astar_path_latlon):
        #     current_astar_target = self.full_astar_path_latlon[self.current_astar_path_idx]
        #     distance_to_current_astar_target = geodesic(self.current_position_latlon, current_astar_target).km
            
        #     # If the agent is within a certain threshold of the current A* path point, advance the index
        #     if distance_to_current_astar_target < 5.0: # 5km threshold
        #         self.current_astar_path_idx += 1
        #         # print(f"  ➡️ Advanced A* path index to {self.current_astar_path_idx}")

        # Fuel consumption

        # Fuel consumption
        fuel_consumed_t = (self.vessel["fuel_consumption_t_per_day"] / 24.0) * self.time_step_hours
        self.vessel["fuel_tank_remaining_t"] -= fuel_consumed_t
        self.total_fuel_consumed += fuel_consumed_t

        # Emissions
        co2_emission_factor_t_per_t_fuel = 3.114
        emissions_t_co2 = fuel_consumed_t * co2_emission_factor_t_per_t_fuel
        self.total_emissions += emissions_t_co2

        # Calculate NEW distance to target
        if self.current_landmark_idx < len(self.landmark_points):
            current_target = self.landmark_points[self.current_landmark_idx]
            new_distance_to_target = geodesic(self.current_position_latlon, current_target).km
        else:
            new_distance_to_target = 0.0

        # Check if landmark reached
        revisited_landmark = False
        if new_distance_to_target < 8:
            # Before incrementing, add the *current* landmark (which is now reached) to visited
            if self.current_landmark_idx < len(self.landmark_points):
                self.visited_landmarks.add(self.current_landmark_idx)

            self.current_landmark_idx += 1
            print(f"Reached landmark {self.current_landmark_idx}/{len(self.landmark_points)}")
            self.drl_path_segment = [self.current_position_latlon]
            self.rerouted_paths_history = []

            # After incrementing, check if the *new* target landmark has been visited before
            if self.current_landmark_idx < len(self.landmark_points) and \
               self.current_landmark_idx in self.visited_landmarks:
                revisited_landmark = True
                print(f"  ⚠️ Revisiting landmark {self.current_landmark_idx}!")


        done = (self.current_landmark_idx >= len(self.landmark_points) or 
                self.vessel["fuel_tank_remaining_t"] <= 0)

        # PASS ALL NECESSARY INFO TO REWARD CALCULATION
        reward_info = {
            'prev_position': prev_position,
            'prev_distance': prev_distance_to_target,
            'new_distance': new_distance_to_target,
            'prev_landmark_idx': prev_landmark_idx,
            'rerouted_path': rerouted_path_latlon,
            'obstacle_info': obstacle_info,
            'distance_travelled': distance_travelled_km,
            'revisited_landmark': revisited_landmark, # Add this flag
            'land_collision_occurred': is_on_land, # Pass land collision status
            'out_of_bounds_occurred': out_of_bounds_occurred # Pass out of bounds status
        }
        
        reward_breakdown = self._calculate_reward(reward_info, episode) # Pass current episode
        total_reward = reward_breakdown['total_reward']

        next_state = self._get_state()
        
        if rerouted_path_latlon:
            self.rerouted_paths_history.append(rerouted_path_latlon)

        # Pass the full breakdown in the info dictionary
        info = {"rerouted_path": rerouted_path_latlon, "reward_breakdown": reward_breakdown}

        return next_state, total_reward, done, info
    
 
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
        ax.plot(astar_lons, astar_lats, 'red', linewidth=1, linestyle='--', label="A* Path", alpha=0.7)

    # Plot landmarks
    landmark_lons = [p[1] for p in landmark_points]
    landmark_lats = [p[0] for p in landmark_points]
    ax.scatter(landmark_lons, landmark_lats, color='cyan', s=80, marker='o', 
              edgecolor='black', label="Landmarks", zorder=5)

    # Highlight the first landmark
    if landmark_points:
        first_landmark_lon = landmark_points[0][1]
        first_landmark_lat = landmark_points[0][0]
        ax.scatter(first_landmark_lon, first_landmark_lat, color='yellow', s=150, 
                   marker='*', edgecolor='red', label="Start Landmark", zorder=7)
    
    # Plot DRL path
    drl_lons = [p[1] for p in env.full_episode_drl_path] # Use full_episode_drl_path for plotting
    drl_lats = [p[0] for p in env.full_episode_drl_path] # Use full_episode_drl_path for plotting
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
    
def normalize_reward_linear(reward):
    old_min, old_max = -2500, 2500
    new_min, new_max = -1, 1
    normalized = ((reward - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
    return np.clip(normalized, new_min, new_max)  # optional safety clip

# --- Training Loop ---

def train_ddpg_agent(env, agent, replay_buffer, num_episodes=500, batch_size=64, 
                     visualize_every_n_episodes=20,
                     full_astar_path_latlon=None, landmark_points=None, 
                     bathymetry_maze=None, grid_params=None, weather_penalty_grid=None):
    episode_rewards = []
    episode_landmarks = []  # Track progress
    
    # Curriculum learning
    env.time_step_hours = 3
    
    # IMPROVED: Track best performance for early stopping
    best_avg_landmarks = 0
    episodes_without_improvement = 0
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        step_count = 0
        
        # Initialize episode reward breakdown
        episode_reward_breakdown = {
            'progress_reward': 0.0, 'regression_penalty': 0.0, 'proximity_bonus': 0.0,
            'distance_penalty': 0.0, 'heading_reward': 0.0, 'landmark_bonus': 0.0,
            'stuck_penalty': 0.0, 'fuel_cost_penalty': 0.0, 'emissions_penalty': 0.0,
            'fuel_exhaustion_penalty': 0.0, 'collision_penalty': 0.0, 'rerouting_penalty': 0.0,
            'revisit_landmark_penalty': 0.0, 'speed_efficiency_penalty': 0.0,
            'too_slow_penalty': 0.0, 'circular_motion_penalty': 0.0,
            'excessive_turning_penalty': 0.0, 'land_collision_penalty': 0.0,
            'out_of_bounds_penalty': 0.0, 'astar_path_following_reward': 0.0, 'total_reward': 0.0
        }

        max_steps = 80  # INCREASED from 50
        
        # Curriculum: Adjust difficulty
        if episode > 100:
            env.time_step_hours = 2.5
        if episode > 200:
            env.time_step_hours = 2
        
        while not done and step_count < max_steps:
            action = agent.select_action(state)
            
            # CRITICAL: Keep exploration higher for longer
            if episode < 100:
                noise_scale = 0.4  # High exploration
            elif episode < 250:
                noise_scale = 0.2  # Medium exploration
            else:
                noise_scale = max(0.05, 0.15 * (1.0 - (episode - 250) / 250))  # Gradual decay
            
            action = (action + np.random.normal(0, noise_scale, size=env.action_dim)).clip(-1, 1)
            
            # Introduce obstacles gradually
            obstacle_present = False
            if episode > 150:  # Wait longer before adding obstacles
                obstacle_present = random.random() < 0.003  # 0.3% chance
            
            next_state, reward, done, info = env.step(action, obstacle_present, episode) # Pass current episode
            
            # Accumulate reward breakdown
            for key, value in info['reward_breakdown'].items():
                episode_reward_breakdown[key] += value
            
            # CLIP reward before storing (prevent extreme values in buffer)
            # The 'reward' returned by env.step is already clipped total_reward
            reward = normalize_reward_linear(reward)
            replay_buffer.push(state, action, next_state, reward, done)
            
            state = next_state
            episode_reward += reward # This is the total reward for the step
            step_count += 1
            
            # Train every step once buffer is ready
            if len(replay_buffer) > batch_size * 4:
                agent.train(replay_buffer, batch_size)
        
        episode_rewards.append(episode_reward)
        episode_landmarks.append(env.current_landmark_idx)
        
        # Print detailed reward breakdown at the end of the episode
        if (episode % 10 == 0):
            print(f"\n--- Episode {episode+1} Reward Breakdown ---")
            for key, value in episode_reward_breakdown.items():
                if key != 'total_reward':
                    print(f"  {key.replace('_', ' ').title()}: {value:.2f}")
            print(f"  Total Episode Reward: {episode_reward_breakdown['total_reward']:.2f}")
            print("------------------------------------")

        # Enhanced logging with progress tracking
        if (episode + 1) % 5 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_landmarks = np.mean(episode_landmarks[-10:])
            max_landmarks = max(episode_landmarks[-10:])
            
            print(f"Ep {episode+1}/{num_episodes}, "
                  f"Reward: {episode_reward:.0f} (Avg10: {avg_reward:.0f}), "
                  f"Landmarks: {env.current_landmark_idx}/{len(env.landmark_points)} "
                  f"(Avg10: {avg_landmarks:.1f}, Max10: {max_landmarks}), "
                  f"Fuel: {env.vessel['fuel_tank_remaining_t']:.0f}t, "
                  f"Noise: {noise_scale:.3f}")
            
            # Check for improvement
            if avg_landmarks > best_avg_landmarks:
                best_avg_landmarks = avg_landmarks
                episodes_without_improvement = 0
            else:
                episodes_without_improvement += 5
            
            # Early stopping if stuck
            if episodes_without_improvement > 100 and episode > 200:
                print(f"\n⚠️ No improvement for {episodes_without_improvement} episodes. "
                      f"Best avg landmarks: {best_avg_landmarks:.1f}")
                print("Consider adjusting hyperparameters or reward function.")
        
        # Visualization
        if (episode + 1) % visualize_every_n_episodes == 0:
            plot_simulation_episode(episode, env, full_astar_path_latlon, 
                                  landmark_points, bathymetry_maze, 
                                  grid_params, weather_penalty_grid)
    
    return agent, episode_rewards


# --- Main execution ---
if __name__ == "__main__":
    full_astar_path_latlon, landmark_points, bathymetry_maze, grid_params, weather_penalty_grid = \
        generate_optimized_route_and_landmarks(vessel, num_sample_ports=3, random_state=50, 
                                               cell_size_m=10000, landmark_interval_km=120)

    if full_astar_path_latlon is None:
        print("Failed to generate optimized route and landmarks. Exiting.")
        exit()
    
    # Extract grid dimensions from grid_params for logging
    _, _, _, _, num_lat_cells, num_lon_cells = grid_params

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
    print(f"  - Landmarks: {len(landmark_points)} (60km intervals)") # Updated interval in print
    print(f"  - Grid cells: {num_lat_cells}x{num_lon_cells} (10km resolution)")
    print(f"  - Max steps per episode: 40")
    print(f"  - Time step: 2 hours")
    print(f"  - Obstacle handling: D* Lite for >3km, DRL for ≤3km")
    print(f"  - Segment-based optimization: Between consecutive landmarks")
    print("="*60 + "\n")
    
    start_time = time.time()
    
    trained_agent, episode_rewards = train_ddpg_agent(
        env, agent, replay_buffer, 
        num_episodes=1200,  # Keep 500 episodes for now
        batch_size=64,     # Keep 64 batch size for now
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
