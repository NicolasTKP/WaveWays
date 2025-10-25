import os
import math
import joblib
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# -----------------------
# Helpers
# -----------------------
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate great circle distance between two points (deg) in nautical miles.
    Returns distance in kilometers.
    """
    # convert degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    R_km = 6371.0
    return R_km * c

def parse_datetime_col(df, col='BaseDateTime'):
    df[col] = pd.to_datetime(df[col], utc=True)
    return df

# convert knots to km/h if needed: 1 knot = 1.852 km/h
def knots_to_kmh(knots):
    return knots * 1.852

# -----------------------
# TARGET SCALING HELPERS 
# -----------------------
def scale_target(tta_sec):
    """ Scales TTA from seconds to log(1 + TTA_hrs) """
    tta_hrs = tta_sec / 3600.0
    return np.log1p(tta_hrs) # log(1 + x)

def unscale_target(y_scaled):
    """ Unscales TTA from log(1 + TTA_hrs) back to seconds """
    tta_hrs = np.expm1(y_scaled) # exp(x) - 1
    tta_sec = tta_hrs * 3600.0
    return np.maximum(tta_sec, 0.0) # ensure non-negative

# -----------------------
# Feature engineering
# -----------------------
def engineer_features(df):
    """
    Add time features and simple derived features. Return df and list of feature column names.
    """
    df = df.copy()
    # time features (UTC aware)
    df['hour'] = df['BaseDateTime'].dt.hour
    df['dayofweek'] = df['BaseDateTime'].dt.dayofweek
    df['month'] = df['BaseDateTime'].dt.month

    # replace impossible heading values (some AIS uses 511 as not available)
    df['Heading'] = df['Heading'].replace(511.0, np.nan).fillna(0.0)
    df['SOG'] = df['SOG'].fillna(0.0) # SOG cleaned

    # difference between COG and Heading
    df['COG_diff'] = (df['COG'] - df['Heading']).fillna(0.0)
    df['COG_diff'] = ((df['COG_diff'] + 180) % 360) - 180  # normalize to [-180,180]

    # distance traveled estimate
    df['prev_LAT'] = df.groupby('MMSI')['LAT'].shift(1)
    df['prev_LON'] = df.groupby('MMSI')['LON'].shift(1)
    df['prev_time'] = df.groupby('MMSI')['BaseDateTime'].shift(1)
    df['delta_t_sec'] = (df['BaseDateTime'] - df['prev_time']).dt.total_seconds().fillna(0.0)
    df['delta_dist_km'] = haversine(df['LON'], df['LAT'], df['prev_LON'].fillna(df['LAT']), df['prev_LAT'].fillna(df['LON']))
    # estimated speed from positions (km/h)
    df['est_speed_kmh'] = np.where(df['delta_t_sec']>0, df['delta_dist_km'] / (df['delta_t_sec']/3600.0), 0.0)

    # SOG in km/h
    df['SOG_kmh'] = knots_to_kmh(df['SOG'])

    # relative speed indicator
    df['speed_diff_kmh'] = df['SOG_kmh'] - df['est_speed_kmh']

    # fill remaining numeric NaNs (should be minimal here)
    num_cols_to_fill = ['COG','SOG_kmh','est_speed_kmh','delta_dist_km','delta_t_sec','speed_diff_kmh']
    for c in num_cols_to_fill:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)

    # features for LSTM sequence (dynamic)
    dynamic_feature_cols = [
        'LAT','LON','SOG_kmh','COG','Heading','dist_to_end_km',
        'hour','dayofweek','month','est_speed_kmh','delta_dist_km','speed_diff_kmh'
    ]
    # features for XGBoost (last timestep dynamic + static)
    static_feature_cols = [
        'Length','Width','Draft','VesselType'
    ]
    
    # ensure present
    dynamic_feature_cols = [c for c in dynamic_feature_cols if c in df.columns]
    static_feature_cols = [c for c in static_feature_cols if c in df.columns]

    return df, dynamic_feature_cols, static_feature_cols

# -----------------------
# Models: LSTM + XGBoost hybrid
# -----------------------

# -----------------------
# Prediction function for route planner
# -----------------------
def predict_eta(recent_points, route_end_latlon, models):
    """
    recent_points: list of dicts or DataFrame rows with keys matching AIS columns
    route_end_latlon: (end_lat, end_lon)
    models: dict returned by train_hybrid_model or loaded models and scalers
    
    Returns: predicted arrival_datetime (UTC), TTA_seconds, components
    """
    # get meta
    dynamic_feature_cols = models['dynamic_feature_cols']
    static_feature_cols = models['static_feature_cols']
    seq_len = models['seq_len']
    scaler_seq = models['scaler_seq']
    scaler_last = models['scaler_last']
    lstm_model = models['lstm_model']
    xgb_model = models['xgb_model']

    # convert to DataFrame
    if isinstance(recent_points, list):
        rp = pd.DataFrame(recent_points)
    elif isinstance(recent_points, pd.DataFrame):
        rp = recent_points.copy()
    else:
        raise ValueError("recent_points must be list or DataFrame")

    rp = parse_datetime_col(rp, 'BaseDateTime')
    
    # ----------------------------------------------------
    # NEW: Ensure static features are present and filled
    # NOTE: In a real-time scenario, vessel static data must come from an external source (e.g. ship registry)
    rp['Length'] = rp['Length'].fillna(rp['Length'].iloc[-1] if 'Length' in rp.columns else 40)
    rp['Width'] = rp['Width'].fillna(rp['Width'].iloc[-1] if 'Width' in rp.columns else 10)
    rp['Draft'] = rp['Draft'].fillna(rp['Draft'].iloc[-1] if 'Draft' in rp.columns else 5)
    rp['VesselType'] = rp['VesselType'].fillna(0).astype('category').cat.codes
    # ----------------------------------------------------

    # require seq_len points
    if len(rp) < seq_len:
        # pad by repeating first row (better to fail or use interpolation for production)
        first = rp.iloc[[0]].copy()
        while len(rp) < seq_len:
            rp = pd.concat([first, rp], ignore_index=True)
        rp = rp.reset_index(drop=True)

    # take last seq_len rows
    rp = rp.sort_values('BaseDateTime').reset_index(drop=True).iloc[-seq_len:].copy()

    # compute engineered features (must match training)
    rp['hour'] = rp['BaseDateTime'].dt.hour
    rp['dayofweek'] = rp['BaseDateTime'].dt.dayofweek
    rp['month'] = rp['BaseDateTime'].dt.month
    rp['Heading'] = rp['Heading'].replace(511.0, np.nan).fillna(0.0)
    rp['SOG'] = rp['SOG'].fillna(0.0)
    rp['COG_diff'] = (rp['COG'] - rp['Heading']).fillna(0.0)
    rp['COG_diff'] = ((rp['COG_diff'] + 180) % 360) - 180
    rp['prev_LAT'] = rp['LAT'].shift(1)
    rp['prev_LON'] = rp['LON'].shift(1)
    rp['prev_time'] = rp['BaseDateTime'].shift(1)
    rp['delta_t_sec'] = (rp['BaseDateTime'] - rp['prev_time']).dt.total_seconds().fillna(0.0)
    rp['delta_dist_km'] = haversine(rp['LON'], rp['LAT'], rp['prev_LON'].fillna(rp['LAT']), rp['prev_LAT'].fillna(rp['LON']))
    rp['est_speed_kmh'] = np.where(rp['delta_t_sec']>0, rp['delta_dist_km'] / (rp['delta_t_sec']/3600.0), 0.0)
    rp['SOG_kmh'] = knots_to_kmh(rp['SOG'])
    rp['speed_diff_kmh'] = rp['SOG_kmh'] - rp['est_speed_kmh']
    
    # distance to route end
    end_lat, end_lon = route_end_latlon
    rp['dist_to_end_km'] = haversine(rp['LON'], rp['LAT'], end_lon, end_lat)  

    # 1. Dynamic Features for LSTM (Sequence)
    dyn_feat_df = rp[dynamic_feature_cols].fillna(0.0)
    nfeat_dyn = dyn_feat_df.shape[1]
    seq_arr = dyn_feat_df.values.reshape(1, seq_len, nfeat_dyn)
    seq_flat = seq_arr.reshape(-1, nfeat_dyn)
    seq_scaled = scaler_seq.transform(seq_flat).reshape(1, seq_len, nfeat_dyn)

    # 2. Combined Features for XGBoost (Last Step)
    last_dyn_feat = dyn_feat_df.values[-1]
    last_static_feat = rp[static_feature_cols].iloc[-1].values # Static features from the last point
    last_feat = np.concatenate([last_dyn_feat, last_static_feat]).reshape(1, -1)
    last_feat_scaled = scaler_last.transform(last_feat)

    # LSTM prediction (in log(1 + TTA_hrs) units)
    lstm_pred_scaled = float(lstm_model.predict(seq_scaled, verbose=0).ravel()[0])
    
    # XGBoost residual prediction (in log(1 + TTA_hrs) units)
    dmat = xgb.DMatrix(last_feat_scaled)
    xgb_residual_scaled = float(xgb_model.predict(dmat)[0])

    # Final prediction (in log(1 + TTA_hrs) units)
    final_tta_scaled = lstm_pred_scaled + xgb_residual_scaled

    # UN-SCALE to get TTA seconds
    final_tta_sec = unscale_target(final_tta_scaled)
    lstm_pred_sec = unscale_target(lstm_pred_scaled)
    
    # NOTE: Residual must be calculated in scaled space, but for reporting, 
    # we can show the magnitude of the correction in seconds.
    # The actual residual prediction (xgb_residual_scaled) is what's added.

    # convert tta seconds to arrival datetime using latest timestamp
    latest_time = rp['BaseDateTime'].iloc[-1]
    arrival_dt = latest_time + pd.Timedelta(seconds=final_tta_sec)

    return {
        'predicted_arrival_utc': arrival_dt.to_pydatetime(),
        'tta_seconds': final_tta_sec,
        'lstm_pred_sec': lstm_pred_sec,
        # Display the contribution of the XGBoost correction
        'xgb_correction_sec': final_tta_sec - lstm_pred_sec, 
        'latest_timestamp': latest_time.to_pydatetime()
    }

# -----------------------
# Utilities to save/load the ensemble
# -----------------------
def load_ensemble():
    # Load Keras model, XGBoost, scalers, meta
    lstm_model = tf.keras.models.load_model(
        "models/lstm_eta.h5",
        custom_objects={'mse': tf.keras.metrics.MeanSquaredError()}
    )
    xgb_model = xgb.Booster()
    xgb_model.load_model("models/xgb_residuals.json")
    scaler_seq = joblib.load("models/scaler_seq.joblib")
    scaler_last = joblib.load("models/scaler_last.joblib")
    meta = joblib.load("models/meta.joblib")
    return {
        'lstm_model': lstm_model,
        'xgb_model': xgb_model,
        'scaler_seq': scaler_seq,
        'scaler_last': scaler_last,
        'dynamic_feature_cols': meta['dynamic_feature_cols'],
        'static_feature_cols': meta['static_feature_cols'],
        'seq_len': meta['seq_len']
    }

# -----------------------
# Demo / Example usage (main)
# -----------------------
if __name__ == '__main__':
    print("Loading pre-trained models for ETA prediction...")
    try:
        models = load_ensemble()
        print("Successfully loaded pretrained models.")
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Please ensure the 'models/' directory exists and contains 'lstm_eta.h5', 'xgb_residuals.json', 'scaler_seq.joblib', 'scaler_last.joblib', and 'meta.joblib'.")
        raise SystemExit()

    # DEMO prediction: Create sample recent points for a vessel
    # This data should mimic the structure of AIS data points
    # Ensure 'seq_len' points are provided, as expected by predict_eta
    seq_len = models['seq_len']
    current_time = datetime.utcnow()
    
    # Example: A vessel moving from (3.0, 101.0) towards (3.1, 101.1)
    # We need 'seq_len' points. Let's create a simple linear path for demonstration.
    sample_recent_points = []
    start_lat, start_lon = 3.0, 101.0
    sog_knots = 10.0 # knots
    cog_deg = 90.0 # degrees
    heading_deg = 90.0 # degrees
    vessel_type = 70 # Cargo
    length = 180.0 # meters
    width = 30.0 # meters
    draft = 12.0 # meters

    for i in range(seq_len):
        # Simulate movement
        time_offset_minutes = (seq_len - 1 - i) * 5 # 5 minutes apart, last point is most recent
        point_time = current_time - timedelta(minutes=time_offset_minutes)
        
        # Simple linear progression for demo
        lat = start_lat + (i * 0.001)
        lon = start_lon + (i * 0.001)

        sample_recent_points.append({
            'MMSI': 367776660,
            'BaseDateTime': point_time.isoformat(),
            'LAT': lat,
            'LON': lon,
            'SOG': sog_knots,
            'COG': cog_deg,
            'Heading': heading_deg,
            'VesselType': vessel_type,
            'Length': length,
            'Width': width,
            'Draft': draft
        })
    
    # Define a plausible end point for the route
    DESTINATION_LAT = 1.35   # Example: Port of Tanjung Pelepas (PTP)
    DESTINATION_LON = 103.52 # Example: Port of Tanjung Pelepas (PTP)
    route_end = (DESTINATION_LAT, DESTINATION_LON)

    print("\n--- DEMO Prediction ---")
    print(f"Using {seq_len} recent points ending at {sample_recent_points[-1]['LAT']:.2f}, {sample_recent_points[-1]['LON']:.2f} at {sample_recent_points[-1]['BaseDateTime']}")
    print(f"Predicting ETA to destination: {DESTINATION_LAT:.2f}, {DESTINATION_LON:.2f}")
    
    pred = predict_eta(sample_recent_points, route_end, models)
    print("Prediction output:")
    print(pred)
