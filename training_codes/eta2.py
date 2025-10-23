import pandas as pd
import numpy as np
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class MalaysiaMarineDataPreprocessor:
    def __init__(self, grid_size=5):
        self.grid_size = grid_size
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.grid_coordinates = []
        
    #using the grid to estimate the distance
    def create_grid_system(self, min_lat=1.0, max_lat=7.0, min_lon=99.0, max_lon=105.5):
        """Create 5km x 5km grid system for Malaysia region"""
        # Approximately 0.045 degrees = 5km (rough approximation)
        lat_range = np.arange(min_lat, max_lat, 0.045)
        lon_range = np.arange(min_lon, max_lon, 0.045)
        self.grid_coordinates = [(lat, lon) for lat in lat_range for lon in lon_range]
        print(f"Created Malaysia grid system with {len(self.grid_coordinates)} cells")
        
    def latlon_to_grid(self, lat, lon):
        """Convert latitude/longitude to grid cell index"""
        if not self.grid_coordinates:
            self.create_grid_system()
        
        # Find nearest grid cell
        distances = [np.sqrt((lat - grid_lat)**2 + (lon - grid_lon)**2) 
                    for grid_lat, grid_lon in self.grid_coordinates]
        return np.argmin(distances)
    
    def calculate_grid_distance(self, route_coords):
        """Calculate distance using grid system"""
        if len(route_coords) < 2:
            return 0
            
        total_distance = 0
        for i in range(len(route_coords)-1):
            # Calculate actual distance between consecutive points
            lat1, lon1 = route_coords[i]
            lat2, lon2 = route_coords[i+1]
            distance = self.haversine_distance(lat1, lon1, lat2, lon2)
            total_distance += distance
        return total_distance
    
    #lat/lon points
    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate great-circle distance between two points in km"""
        R = 6371  # Earth radius in km
        
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        
        a = (np.sin(dlat/2) * np.sin(dlat/2) + 
             np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * 
             np.sin(dlon/2) * np.sin(dlon/2))
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c
    
    #weather data
    def get_malaysia_weather_data(self, lat, lon, timestamp):
        """Simulate Malaysia weather data (monsoon patterns)"""
        # Deterministic seed by (lat, lon, ts). Remove if you want variability.
        np.random.seed(int(lat * 100 + lon * 100 + timestamp.timestamp()))
        
        # Malaysia has two monsoon seasons
        month = timestamp.month
        if month in [11, 12, 1, 2]:  # Northeast Monsoon
            wind_speed = np.random.uniform(5, 15)  # Stronger winds
            wave_height = np.random.uniform(0.5, 2.5)
        elif month in [5, 6, 7, 8]:  # Southwest Monsoon
            wind_speed = np.random.uniform(3, 10)  # Milder winds
            wave_height = np.random.uniform(0.3, 1.5)
        else:  # Inter-monsoon
            wind_speed = np.random.uniform(2, 8)
            wave_height = np.random.uniform(0.2, 1.0)
        
        return {
            'wind_speed': wind_speed,              # m/s
            'wind_direction': np.random.uniform(0, 360),  # degrees
            'wave_height': wave_height,            # meters
            'current_speed': np.random.uniform(0, 1.5),   # m/s (weaker currents in straits)
            'current_direction': np.random.uniform(0, 360),  # degrees
            'visibility': np.random.uniform(5, 15) # km
        }
    
    #label-encodes vessel_type/ports
    def preprocess_features(self, df):
        """Preprocess the dataset and create features"""
        df_processed = df.copy()
        
        # Encode categorical variables
        categorical_cols = ['vessel_type', 'depart_port', 'destination_port']
        for col in categorical_cols:
            if col in df_processed.columns:
                self.label_encoders[col] = LabelEncoder()
                df_processed[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df_processed[col])
        
        # Calculate grid-based distance
        print("Calculating grid distances for Malaysia routes...")
        df_processed['grid_distance'] = df_processed['route_coordinates'].apply(
            lambda x: self.calculate_grid_distance(x)
        )
        
        # Add Malaysia weather features
        print("Adding Malaysia weather features...")
        weather_data = df_processed.apply(
            lambda row: self.get_malaysia_weather_data(
                row['route_coordinates'][0][0],  # First point lat
                row['route_coordinates'][0][1],  # First point lon
                row['departure_time']
            ), axis=1
        )
        
        weather_df = pd.DataFrame(weather_data.tolist(), index=df_processed.index)
        df_processed = pd.concat([df_processed, weather_df], axis=1)
        
        # Temporal features
        df_processed['departure_time'] = pd.to_datetime(df_processed['departure_time'])
        df_processed['departure_hour'] = df_processed['departure_time'].dt.hour
        df_processed['departure_month'] = df_processed['departure_time'].dt.month
        df_processed['departure_dayofweek'] = df_processed['departure_time'].dt.dayofweek
        df_processed['departure_day'] = df_processed['departure_time'].dt.day
        df_processed['departure_minute'] = df_processed['departure_time'].dt.minute
        
        # Vessel performance features
        df_processed['speed_efficiency'] = df_processed['current_speed'] / df_processed['max_speed']
        df_processed['weather_impact'] = df_processed['wind_speed'] * df_processed['wave_height']
        df_processed['headwind_component'] = np.abs(
            df_processed['wind_direction'] - df_processed['current_direction']
        ) / 180  # Normalized headwind component
        
        print(f"Preprocessing complete. Final features: {df_processed.columns.tolist()}")
        return df_processed

#using XGBosst + LSTM
class MalaysiaHybridETAModel:
    def __init__(self, sequence_length=5):
        self.sequence_length = sequence_length
        self.xgb_model = None
        self.lstm_model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.is_trained = False

    def build_xgboost_model(self):
        """Build XGBoost model for static features"""
        return xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='mae'
        )

    # ---- Version-agnostic XGBoost fit helper ----
    def safe_xgb_fit(self, model, Xtr, ytr, Xva, yva, rounds=20, verbose=False):
        """
        Try callbacks (XGB 2.x) -> early_stopping_rounds (XGB 1.x) -> plain fit.
        """
        # Try callbacks API (XGBoost 2.x)
        try:
            from xgboost import callback
            return model.fit(
                Xtr, ytr,
                eval_set=[(Xva, yva)],
                callbacks=[callback.EarlyStopping(rounds=rounds, min_delta=0.0, save_best=True)],
                verbose=verbose
            )
        except TypeError:
            pass
        except Exception:
            pass

        # Try legacy early_stopping_rounds (XGBoost 1.x)
        try:
            return model.fit(
                Xtr, ytr,
                eval_set=[(Xva, yva)],
                early_stopping_rounds=rounds,
                verbose=verbose
            )
        except TypeError:
            pass
        except Exception:
            pass

        # Fallback: no early stopping
        return model.fit(
            Xtr, ytr,
            eval_set=[(Xva, yva)],
            verbose=verbose
        )
        
    def prepare_sequential_data(self, df):
        """Prepare sequential data for LSTM"""
        sequences = []
        targets = []
        
        # Sort by vessel and timestamp to create sequences
        df_sorted = df.sort_values(['vessel_id', 'departure_time']).reset_index(drop=True)
        vessel_groups = df_sorted.groupby('vessel_id')
        
        for vessel_id, group in vessel_groups:
            group = group.sort_values('departure_time')
            values = group[self.feature_columns].values
            
            # Create sequences
            for i in range(len(group) - self.sequence_length):
                sequence = values[i:(i + self.sequence_length)]
                target = group.iloc[i + self.sequence_length]['eta_hours']
                sequences.append(sequence)
                targets.append(target)
        
        if len(sequences) == 0 and len(df_sorted) > self.sequence_length:
            # Fallback: create sequences without grouping if no sequential data
            values = df_sorted[self.feature_columns].values
            for i in range(len(df_sorted) - self.sequence_length):
                sequence = values[i:(i + self.sequence_length)]
                target = df_sorted.iloc[i + self.sequence_length]['eta_hours']
                sequences.append(sequence)
                targets.append(target)
                
        return np.array(sequences), np.array(targets)
    
    def build_lstm_model(self, input_shape):
        """Build LSTM model for sequential patterns"""
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.3),
            LSTM(32, return_sequences=False),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dense(8, activation='relu'),
            Dense(1, activation='linear')
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def train_hybrid_model(self, df):
        """Train the hybrid model"""
        print("Starting hybrid model training for Malaysia routes...")
        
        # Define feature sets
        static_features = [
            'vessel_type_encoded', 'depart_port_encoded', 'destination_port_encoded',
            'grid_distance', 'max_speed', 'vessel_length', 'vessel_draft',
            'departure_hour', 'departure_month', 'departure_dayofweek'
        ]
        
        dynamic_features = [
            'wind_speed', 'wind_direction', 'wave_height', 
            'current_speed', 'current_direction', 'speed_efficiency', 
            'weather_impact', 'headwind_component', 'visibility'
        ]
        
        self.feature_columns = static_features + dynamic_features
        
        # Check if all features exist
        missing_features = [f for f in self.feature_columns if f not in df.columns]
        if missing_features:
            print(f"Warning: Missing features: {missing_features}")
            self.feature_columns = [f for f in self.feature_columns if f in df.columns]
        
        # Split data
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        
        print(f"Training XGBoost on {len(static_features)} static features...")
        # Train XGBoost on static features
        self.xgb_model = self.build_xgboost_model()

        # Version-agnostic early stopping
        self.safe_xgb_fit(
            self.xgb_model,
            train_df[static_features], train_df['eta_hours'],
            test_df[static_features],  test_df['eta_hours'],
            rounds=20,
            verbose=True  # set False if you want silence
        )

        # Get XGBoost predictions as features
        train_df['xgb_prediction'] = self.xgb_model.predict(train_df[static_features])
        test_df['xgb_prediction'] = self.xgb_model.predict(test_df[static_features])
        
        # Update feature columns to include XGBoost predictions
        self.feature_columns.append('xgb_prediction')
        
        # Prepare sequential data for LSTM
        print("Preparing sequential data for LSTM...")
        X_seq_train, y_seq_train = self.prepare_sequential_data(train_df)
        X_seq_test, y_seq_test = self.prepare_sequential_data(test_df)
        
        if len(X_seq_train) == 0 or len(X_seq_test) == 0:
            print("Insufficient sequential data. Using XGBoost only.")
            self.is_trained = True
            return None
        
        print(f"Sequential data shape - Train: {X_seq_train.shape}, Test: {X_seq_test.shape}")
        
        # Scale features for LSTM
        X_seq_train_reshaped = X_seq_train.reshape(-1, X_seq_train.shape[-1])
        X_seq_test_reshaped = X_seq_test.reshape(-1, X_seq_test.shape[-1])
        
        X_seq_train_scaled = self.scaler.fit_transform(X_seq_train_reshaped)
        X_seq_test_scaled = self.scaler.transform(X_seq_test_reshaped)
        
        X_seq_train_scaled = X_seq_train_scaled.reshape(X_seq_train.shape)
        X_seq_test_scaled = X_seq_test_scaled.reshape(X_seq_test.shape)
        
        # Train LSTM model
        print("Training LSTM model...")
        self.lstm_model = self.build_lstm_model(
            (self.sequence_length, len(self.feature_columns))
        )
        
        history = self.lstm_model.fit(
            X_seq_train_scaled, y_seq_train,
            validation_data=(X_seq_test_scaled, y_seq_test),
            epochs=50,
            batch_size=16,
            verbose=1
        )
        
        self.is_trained = True
        return history
    
    def predict_eta(self, journey_data):
        """Predict ETA for new journey (currently returns XGB prediction)"""
        if not self.is_trained and self.xgb_model is None:
            raise ValueError("Model must be trained before prediction")
        
        # Ensure we have a DataFrame
        if isinstance(journey_data, dict):
            journey_data = pd.DataFrame([journey_data])
        
        # Use XGBoost prediction
        static_features = [f for f in self.feature_columns 
                           if f != 'xgb_prediction' and f in journey_data.columns]
        
        xgb_pred = self.xgb_model.predict(journey_data[static_features])[0]
        return xgb_pred

class MalaysiaRoutePlanner:
    def __init__(self, eta_model, grid_system):
        self.eta_model = eta_model
        self.grid_system = grid_system
        self.port_database = self.load_malaysia_port_database()
        
    def load_malaysia_port_database(self):
        """Load Malaysia port database with coordinates"""
        ports = {
            'PORT_KELANG': (3.0000, 101.4000),      # Main port near Kuala Lumpur
            'PENANG': (5.4167, 100.3333),           # George Town
            'JOHOR_BAHRU': (1.4556, 103.7611),      # Southern Malaysia
            'MUAR': (2.0442, 102.5689),             # Muar Port
            'KUALA_TERENGGANU': (5.3302, 103.1408), # East Coast
            'KUANTAN': (3.8167, 103.3333),          # East Coast
            'MIRI': (4.3833, 113.9833),             # Sarawak
            'KOTA_KINABALU': (5.9833, 116.0667),    # Sabah
            'SANDAKAN': (5.8333, 118.1167),         # Sabah
            'BINTULU': (3.1667, 113.0333),          # Sarawak
            'MALACCA': (2.2000, 102.2500),          # Historical port
            'LUMUT': (4.2333, 100.6333),            # Naval base
            'KEMAMAN': (4.2333, 103.4167)           # East Coast
        }
        return ports

    # --- NEW: helper to break down hours into d/h/m ---
    def _breakdown_hours(self, hours: float):
        """Convert float hours -> (days, hours, minutes) with rounding to nearest minute."""
        total_minutes = int(round(hours * 60))
        days = total_minutes // (24 * 60)
        hours_rem = (total_minutes % (24 * 60)) // 60
        minutes_rem = total_minutes % 60
        return days, hours_rem, minutes_rem

    # --- Helpers: zone multipliers & per-segment speed model ---
    def _zone_multiplier(self, zone_type: str) -> float:
        # Multiplier on speed (straits slower, coastal a bit slower, open sea fastest)
        z = zone_type.upper()
        return {'STRAITS': 0.80, 'COASTAL': 0.90}.get(z, 1.00)  # OPEN_SEA->1.0

    def _time_of_day_multiplier(self, ts: pd.Timestamp, zone_type: str) -> float:
        # Congestion idea: daylight hours in straits a bit slower
        h = ts.hour
        if zone_type.upper() == 'STRAITS' and 6 <= h < 18:
            return 1.08   # 8% slower in daytime
        if zone_type.upper() == 'COASTAL' and 6 <= h < 18:
            return 1.03
        return 1.00

    def _effective_speed_kmh(self, base_knots: float, lat: float, lon: float,
                             ts: pd.Timestamp, zone_type: str) -> float:
        """
        Compute effective speed on THIS small leg using:
        - base vessel speed (knots)
        - local weather (wind, wave, surface current)
        - zone factor + time-of-day factor
        """
        # Base
        base_kmh = base_knots * 1.852

        # Local weather (from preprocessor)
        w = self.grid_system.get_malaysia_weather_data(lat, lon, ts)
        wind, wave = w['wind_speed'], w['wave_height']
        current_kmh = w['current_speed'] * 3.6

        # Weather penalty (reduce speed). Tune coefficients as needed.
        weather_mult = max(0.55, 1.0 - (0.030 * wind + 0.18 * wave))

        # Zone & time-of-day effects
        zone_mult = self._zone_multiplier(zone_type)
        tod_mult = self._time_of_day_multiplier(ts, zone_type)

        # A small fraction of surface current helps/hurts
        eff_kmh = base_kmh * weather_mult * zone_mult / tod_mult + 0.30 * current_kmh
        return max(eff_kmh, 3.0)  # floor at 3 km/h to avoid stalling

    # --- 5 km segmentation + ETA accumulation ---
    def _interp_point(self, a, b, t):
        """Linear lat/lon interpolation (OK for short marine legs)."""
        return (a[0] + (b[0] - a[0]) * t,
                a[1] + (b[1] - a[1]) * t)

    def _segmentize_leg(self, a, b, step_km=5.0):
        """Break one waypoint-to-waypoint leg into ~5 km sub-legs."""
        dist = self.grid_system.haversine_distance(a[0], a[1], b[0], b[1])
        if dist == 0:
            return []
        n = max(1, int(np.ceil(dist / step_km)))
        pts = []
        for i in range(n):
            t0 = i / n
            t1 = (i + 1) / n
            p0 = self._interp_point(a, b, t0)
            p1 = self._interp_point(a, b, t1)
            seg_km = self.grid_system.haversine_distance(p0[0], p0[1], p1[0], p1[1])
            pts.append((p0, p1, seg_km))
        return pts

    def compute_eta_for_route(self, route: dict, vessel_type: str,
                              departure_time: datetime, step_km=5.0):
        """
        Sum ETA over small segments along the provided polyline route.
        route = {'coordinates': [(lat,lon), ...], 'type': 'direct'|'straits'|'coastal', ...}
        Returns (eta_hours, details)
        """
        vinfo = self.get_vessel_info(vessel_type)
        base_knots = vinfo['max_speed']  # or use service speed if available

        coords = route['coordinates']
        ts = pd.to_datetime(departure_time)
        total_hours = 0.0
        details = []

        for i in range(len(coords) - 1):
            a, b = coords[i], coords[i + 1]
            small_legs = self._segmentize_leg(a, b, step_km=step_km)
            for (p0, p1, seg_km) in small_legs:
                # Use midpoint for local weather lookup
                mid = ((p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0)
                eff_kmh = self._effective_speed_kmh(base_knots, mid[0], mid[1], ts, route['type'])
                hours = seg_km / eff_kmh
                total_hours += hours
                ts = ts + pd.Timedelta(hours=hours)  # advance clock

                details.append({
                    "from": p0, "to": p1,
                    "segment_km": seg_km,
                    "eff_kmh": eff_kmh,
                    "hours": hours,
                    "cumulative_hours": total_hours,
                    "timestamp": ts
                })

        return total_hours, details

    def calculate_realistic_eta(self, depart_port, destination_port, vessel_type, departure_time):
        """
        (Legacy) single-shot ETA estimator — kept for reference.
        Not used once segment-based compute_eta_for_route() is wired in.
        """
        distance_matrix = {
            ('PORT_KELANG', 'MUAR'): 180,
            ('PORT_KELANG', 'PENANG'): 350,
            ('PORT_KELANG', 'JOHOR_BAHRU'): 320,
            ('PORT_KELANG', 'MALACCA'): 120,
            ('MUAR', 'PENANG'): 400,
            ('MUAR', 'JOHOR_BAHRU'): 150,
            ('MUAR', 'MALACCA'): 60,
            ('PENANG', 'JOHOR_BAHRU'): 670,
            ('PENANG', 'KUALA_TERENGGANU'): 450,
            ('JOHOR_BAHRU', 'KUANTAN'): 280,
        }
        key = (depart_port, destination_port)
        reverse_key = (destination_port, depart_port)
        
        if key in distance_matrix:
            distance = distance_matrix[key]
        elif reverse_key in distance_matrix:
            distance = distance_matrix[reverse_key]
        else:
            depart_coords = self.port_database[depart_port]
            dest_coords = self.port_database[destination_port]
            distance = self.grid_system.haversine_distance(
                depart_coords[0], depart_coords[1], 
                dest_coords[0], dest_coords[1]
            )
        
        vessel_speeds = {
            'CONTAINER': 18, 'TANKER': 14, 'BULK_CARRIER': 12,
            'GENERAL_CARGO': 15, 'FISHING': 8, 'PASSENGER': 20
        }
        
        speed_knots = vessel_speeds.get(vessel_type, 12)
        speed_kmh = speed_knots * 1.852
        base_hours = distance / speed_kmh
        
        adjustment_factors = {'MALACCA_STRAIT': 1.1, 'COASTAL': 1.0, 'OPEN_SEA': 0.95}
        if 'PORT_KELANG' in [depart_port, destination_port] and 'PENANG' in [depart_port, destination_port]:
            route_factor = adjustment_factors['MALACCA_STRAIT']
        elif 'MUAR' in [depart_port, destination_port]:
            route_factor = adjustment_factors['COASTAL']
        else:
            route_factor = adjustment_factors['OPEN_SEA']
        
        adjusted_hours = base_hours * route_factor
        operational_hours = 2
        total_hours = adjusted_hours + operational_hours
        return total_hours
    
    def format_eta_output(self, departure_time, eta_hours):
        """Format ETA output (days, hours, minutes) and arrival time strings."""
        departure_dt = pd.to_datetime(departure_time)
        arrival_dt = departure_dt + timedelta(hours=eta_hours)

        d, h, m = self._breakdown_hours(eta_hours)

        # e.g., "1 day 3 hours 12 minutes" (skip zero parts neatly)
        parts = []
        if d: parts.append(f"{d} day" + ("s" if d != 1 else ""))
        if h: parts.append(f"{h} hour" + ("s" if h != 1 else ""))
        if m or not parts: parts.append(f"{m} minute" + ("s" if m != 1 else ""))
        eta_str = " ".join(parts)

        depart_str = departure_dt.strftime("%d %B %Y (%A) %I.%M%p").lower()
        arrive_str = arrival_dt.strftime("%d %B %Y (%A) %I.%M%p").lower()

        return {
            'depart_at': depart_str,
            'arrive_at': arrive_str,
            'total_hours': eta_hours,
            'arrival_datetime': arrival_dt,
            'eta_days': d,
            'eta_hours': h,
            'eta_minutes': m,
            'eta_str': eta_str
        }
    
    def find_optimal_route(self, depart_port, destination_port, vessel_type, departure_time):
        """Find optimal route using segment-based ETA over ~5 km legs"""
        print(f"Finding optimal route from {depart_port} to {destination_port}")
        
        # Validate ports
        if depart_port not in self.port_database:
            raise ValueError(f"Unknown departure port: {depart_port}")
        if destination_port not in self.port_database:
            raise ValueError(f"Unknown destination port: {destination_port}")
            
        # Generate possible routes (distinct geometry)
        routes = self.generate_malaysia_routes(depart_port, destination_port)
        
        best_route = None
        best_eta = float('inf')
        route_predictions = []
        
        for i, route in enumerate(routes):
            eta_hours, details = self.compute_eta_for_route(
                route, vessel_type, departure_time, step_km=5.0
            )
            route_predictions.append((route, eta_hours, details))
            print(f"Route {i+1} ({route['type']}): {eta_hours:.2f} hours")

            if eta_hours < best_eta:
                best_eta = eta_hours
                best_route = route
        
        # Format output
        eta_output = self.format_eta_output(departure_time, best_eta)
        
        print(f"\nSelected route: {best_route['description']}")
        print(f"Depart at: {eta_output['depart_at']}")
        print(f"Arrive at {destination_port}: {eta_output['arrive_at']}")
        print(f"Total voyage time: {eta_output['eta_str']} "
              f"(~{eta_output['total_hours']:.2f} hours, {eta_output['total_hours']/24:.2f} days)")
        
        return best_route, eta_output, route_predictions
    
    def generate_malaysia_routes(self, depart_port, destination_port):
        """Generate route candidates with distinct geometry."""
        routes = []
        
        depart_coords = self.port_database[depart_port]
        dest_coords = self.port_database[destination_port]
        
        # Route 1: Direct (shortest line over water)
        routes.append({
            'coordinates': [depart_coords, dest_coords],
            'type': 'direct',
            'description': 'Direct great-circle approximation'
        })
        
        # Route 2: Straits — force a waypoint inside Malacca Strait corridor
        strait_point = (3.3, 100.8)  # illustrative waypoint
        routes.append({
            'coordinates': [depart_coords, strait_point, dest_coords],
            'type': 'straits',
            'description': 'Traffic separation–style path via Malacca Strait'
        })
        
        # Route 3: Coastal — bias a waypoint closer to the coastline
        coastal_point = (
            (depart_coords[0] + dest_coords[0]) / 2.0,
            (depart_coords[1] + dest_coords[1]) / 2.0 - 0.7  # slide longitude near shore
        )
        routes.append({
            'coordinates': [depart_coords, coastal_point, dest_coords],
            'type': 'coastal',
            'description': 'Coastal-hugging path'
        })
        
        return routes
    
    def get_vessel_info(self, vessel_type):
        """Get vessel characteristics based on type"""
        vessel_types = {
            'CONTAINER': {'max_speed': 18, 'length': 250, 'draft': 12},
            'TANKER': {'max_speed': 14, 'length': 200, 'draft': 14},
            'BULK_CARRIER': {'max_speed': 12, 'length': 180, 'draft': 11},
            'GENERAL_CARGO': {'max_speed': 15, 'length': 120, 'draft': 8},
            'FISHING': {'max_speed': 8, 'length': 30, 'draft': 4},
            'PASSENGER': {'max_speed': 20, 'length': 150, 'draft': 6}
        }
        return vessel_types.get(vessel_type, {'max_speed': 12, 'length': 100, 'draft': 5})

def generate_malaysia_sample_data(n_samples=1500):
    """Generate realistic sample data for Malaysia marine journeys"""
    np.random.seed(42)
    
    # Malaysia port pairs with realistic distances
    malaysia_routes = [
        ('PORT_KELANG', 'MUAR', 180),
        ('PORT_KELANG', 'PENANG', 350),
        ('PORT_KELANG', 'JOHOR_BAHRU', 320),
        ('PORT_KELANG', 'MALACCA', 120),
        ('MUAR', 'PENANG', 400),
        ('MUAR', 'JOHOR_BAHRU', 150),
        ('MUAR', 'MALACCA', 60),
        ('PENANG', 'JOHOR_BAHRU', 670),
        ('PENANG', 'KUALA_TERENGGANU', 450),
        ('JOHOR_BAHRU', 'KUANTAN', 280),
    ]
    
    data = []
    planner = MalaysiaRoutePlanner(None, None)  # only to access port DB & simple ETA
    ports = planner.port_database
    
    for i in range(n_samples):
        # Random Malaysia route
        depart_port, dest_port, base_distance = malaysia_routes[np.random.randint(len(malaysia_routes))]
        
        # Vessel characteristics
        vessel_type = np.random.choice(['CONTAINER', 'TANKER', 'BULK_CARRIER', 'GENERAL_CARGO', 'FISHING', 'PASSENGER'])
        vessel_info = planner.get_vessel_info(vessel_type)
        
        # Realistic ETA calculation (legacy method, just to build labels)
        departure_time = datetime(2024, 1, 1) + timedelta(hours=np.random.randint(0, 24*365))
        eta_hours = planner.calculate_realistic_eta(depart_port, dest_port, vessel_type, departure_time)
        
        # Add some random variation
        eta_hours *= np.random.uniform(0.9, 1.1)
        
        # Route coordinates
        route_coords = [
            ports[depart_port],
            ports[dest_port]
        ]
        
        data.append({
            'vessel_id': np.random.randint(1, 51),
            'vessel_type': vessel_type,
            'depart_port': depart_port,
            'destination_port': dest_port,
            'max_speed': vessel_info['max_speed'],
            'vessel_length': vessel_info['length'],
            'vessel_draft': vessel_info['draft'],
            'departure_time': departure_time,
            'route_coordinates': route_coords,
            'eta_hours': eta_hours
        })
    
    return pd.DataFrame(data)

def main():
    """Main function to demonstrate Malaysia marine ETA system"""
    print("=== Malaysia Marine ETA Prediction System ===")
    
    # Step 1: Initialize Malaysia grid system
    print("\n1. Initializing Malaysia grid system...")
    grid_system = MalaysiaMarineDataPreprocessor(grid_size=5)
    grid_system.create_grid_system()
    
    # Step 2: Generate Malaysia sample data
    print("\n2. Generating Malaysia sample data...")
    df = generate_malaysia_sample_data(1000)
    print(f"Generated {len(df)} sample Malaysia journeys")
    
    print("\n3. Preprocessing features...")
    df_processed = grid_system.preprocess_features(df)
    
    # Step 3: Train model
    print("\n4. Training Malaysia ETA model...")
    eta_model = MalaysiaHybridETAModel(sequence_length=3)
    
    try:
        history = eta_model.train_hybrid_model(df_processed)
    except Exception as e:
        print(f"Model training simplified due to: {e}")
        # Use simplified training
        static_features = ['vessel_type_encoded', 'depart_port_encoded', 'destination_port_encoded',
                          'grid_distance', 'max_speed', 'vessel_length', 'vessel_draft',
                          'departure_hour', 'departure_month', 'departure_dayofweek']
        
        train_df, test_df = train_test_split(df_processed, test_size=0.2, random_state=42)
        
        eta_model.xgb_model = eta_model.build_xgboost_model()
        # Version-agnostic early stopping
        eta_model.safe_xgb_fit(
            eta_model.xgb_model,
            train_df[static_features], train_df['eta_hours'],
            test_df[static_features],  test_df['eta_hours'],
            rounds=10,
            verbose=False
        )
        eta_model.is_trained = True
    
    # Step 4: Initialize Malaysia route planner
    print("\n5. Initializing Malaysia route planner...")
    route_planner = MalaysiaRoutePlanner(eta_model, grid_system)
    
    # Demonstrate with the specific example
    print("\n6. Demonstrating Malaysia route planning...")
    
    # Your specific example
    test_cases = [
        ('PORT_KELANG', 'MUAR', 'CONTAINER', datetime(2025, 9, 29, 8, 30)),  # 29 Sept 2025 8:30am
    ]
    
    for depart, dest, vessel_type, dep_time in test_cases:
        print(f"\n{'='*50}")
        print(f"ROUTE: {depart} → {dest}")
        print(f"VESSEL TYPE: {vessel_type}")
        print(f"DEPARTURE: {dep_time.strftime('%d %B %Y (%A) %I.%M%p').lower()}")
        print(f"{'='*50}")
        
        try:
            best_route, eta_output, all_predictions = route_planner.find_optimal_route(
                depart, dest, vessel_type, dep_time
            )
            
            print(f"\nRoute Details:")
            print(f"- Selected Type: {best_route['type']}")
            print(f"- Description: {best_route['description']}")
            print(f"- ETA: {eta_output['eta_str']} (~{eta_output['total_hours']:.2f} hours)")
            
        except Exception as e:
            print(f"Error in route planning: {e}")
    
    # Additional test cases
    print(f"\n{'='*50}")
    print("ADDITIONAL TEST CASES")
    print(f"{'='*50}")
    
    additional_cases = [
        ('PENANG', 'PORT_KELANG', 'PASSENGER', datetime(2025, 10, 1, 14, 0)),
        ('JOHOR_BAHRU', 'KUANTAN', 'TANKER', datetime(2025, 10, 2, 9, 0)),
    ]
    
    for depart, dest, vessel_type, dep_time in additional_cases:
        print(f"\n--- {depart} → {dest} ({vessel_type}) ---")
        try:
            best_route, eta_output, all_predictions = route_planner.find_optimal_route(
                depart, dest, vessel_type, dep_time
            )
            print(f"- ETA: {eta_output['eta_str']} (~{eta_output['total_hours']:.2f} hours)")
        except Exception as e:
            print(f"Error: {e}")
    
    print(f"\n{'='*50}")
    print("SYSTEM READY FOR MALAYSIA MARINE OPERATIONS")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
