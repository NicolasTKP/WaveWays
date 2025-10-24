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
# Data loading + synthetic label creation (TTA)
# -----------------------
def load_and_prepare_ais(csv_path):
    """
    Loads AIS CSV, parses datetime, basic cleaning, groups into voyages, and
    creates a synthetic TTA (time-to-arrival) label by assuming the last record of
    a voyage is arrival time.
    """
    # Only load necessary columns to save memory and skip irrelevant data
    required_cols = ['MMSI', 'BaseDateTime', 'LAT', 'LON', 'SOG', 'COG', 'Heading', 'VesselType', 'Length', 'Width', 'Draft']
    df = pd.read_csv(csv_path, usecols=required_cols)
    df = parse_datetime_col(df, 'BaseDateTime')
    # drop rows with missing coords
    df = df.dropna(subset=['LAT', 'LON'])
    # sort
    df = df.sort_values(['MMSI', 'BaseDateTime']).reset_index(drop=True)

    # Create 'voyage_id' by grouping by MMSI and splitting when gap > threshold
    gap_thresh = pd.Timedelta(hours=2)

    def assign_voyages(g):
        g = g.copy()
        g['time_diff'] = g['BaseDateTime'].diff().fillna(pd.Timedelta(seconds=0))
        voyage_idx = 0
        voyage_ids = []
        for td in g['time_diff']:
            if td > gap_thresh:
                voyage_idx += 1
            voyage_ids.append(f"{g['MMSI'].iloc[0]}_{voyage_idx}")
        g['voyage_id'] = voyage_ids
        return g

    # Apply voyage assignment only if there are enough unique MMSI
    if df['MMSI'].nunique() > 1:
        df = df.groupby('MMSI', group_keys=False).apply(assign_voyages)
    else:
        # Handle case with very small sample
        df['voyage_id'] = f"{df['MMSI'].iloc[0]}_0" if not df.empty else None

    # for each voyage, set arrival_time = last timestamp
    arrival = df.groupby('voyage_id')['BaseDateTime'].max().rename('arrival_time')
    df = df.merge(arrival, on='voyage_id', how='left')

    # create time-to-arrival in seconds
    df['TTA_sec'] = (df['arrival_time'] - df['BaseDateTime']).dt.total_seconds()
    
    # -------------------------------------------------------------
    # NEW: SCALED TARGET VARIABLE
    df['TTA_scaled'] = scale_target(df['TTA_sec'])
    # -------------------------------------------------------------

    # remove negative or zero TTA (shouldn't happen) and very large TTA (filter incomplete voyages)
    df = df[(df['TTA_sec'] > 0) & (df['TTA_sec'] < 7*24*3600)]  # less than 7 days

    # compute distance to arrival point (using last recorded lat/lon of voyage)
    last_pos = df.groupby('voyage_id').agg({'LAT': 'last', 'LON': 'last'}).rename(columns={'LAT':'end_lat','LON':'end_lon'})
    df = df.merge(last_pos, on='voyage_id', how='left')
    df['dist_to_end_km'] = haversine(df['LON'], df['LAT'], df['end_lon'], df['end_lat'])

    # Fill NaNs in static features (crucial for XGBoost)
    df['Length'] = df['Length'].fillna(df['Length'].median() if not df['Length'].empty else 0)
    df['Width'] = df['Width'].fillna(df['Width'].median() if not df['Width'].empty else 0)
    df['Draft'] = df['Draft'].fillna(df['Draft'].median() if not df['Draft'].empty else 0)
    df['VesselType'] = df['VesselType'].fillna(0).astype('category').cat.codes # Simple encoding

    return df

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
# Sequence building for LSTM
# -----------------------
def build_sequences(df, dynamic_feature_cols, static_feature_cols, seq_len=8, min_points_per_voyage=10):
    """
    For each voyage_id, build sliding sequences of length seq_len.
    Label is TTA_scaled at the sequence's last timestep.
    Returns X_seq, y, X_last_dynamic, X_last_static
    """
    X_list, y_list, X_last_dynamic_list, X_last_static_list = [], [], [], []
    grouped = df.groupby('voyage_id')

    for vid, g in grouped:
        if len(g) < min_points_per_voyage:
            continue
        g = g.sort_values('BaseDateTime').reset_index(drop=True)
        
        dyn_feats = g[dynamic_feature_cols].values
        static_feats = g[static_feature_cols].iloc[0].values # Static features only need to be taken once per voyage (we use the first)
        
        tta_scaled = g['TTA_scaled'].values
        n = len(g)
        
        # sliding windows
        for i in range(seq_len-1, n):
            start = i - (seq_len-1)
            seq = dyn_feats[start:i+1]  # inclusive last
            
            if seq.shape[0] != seq_len:
                continue
            
            X_list.append(seq)
            y_list.append(tta_scaled[i])
            X_last_dynamic_list.append(seq[-1])  # dynamic features at last timestep
            X_last_static_list.append(static_feats) # static features appended for all windows in the voyage

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    
    # NEW: Combine dynamic and static features for XGBoost
    X_last_dynamic = np.array(X_last_dynamic_list, dtype=np.float32)
    X_last_static = np.array(X_last_static_list, dtype=np.float32)
    X_last = np.concatenate([X_last_dynamic, X_last_static], axis=1)

    return X, y, X_last, dynamic_feature_cols + static_feature_cols

# -----------------------
# Models: LSTM + XGBoost hybrid
# -----------------------
def build_lstm_model(input_shape, dropout=0.2):
    model = Sequential()
    # Masking layer is important if you intend to pad sequences to unequal lengths,
    # but here we use sequences of fixed length=seq_len. It can be kept for robustness.
    model.add(Masking(mask_value=0., input_shape=input_shape)) 
    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='linear'))  # predict log(1 + TTA_hrs)
    
    # Use a slightly lower learning rate for stability
    optimizer = Adam(learning_rate=0.001) 
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

# -----------------------
# Full training pipeline
# -----------------------
def train_hybrid_model(csv_path,
                         seq_len=10, # Increased sequence length
                         test_size=0.15,
                         random_state=42,
                         model_dir='models_malaysia'): # Specific model directory
    os.makedirs(model_dir, exist_ok=True)

    print("Loading and preparing AIS data...")
    df = load_and_prepare_ais(csv_path)
    print(f"Loaded {len(df)} AIS records after filtering.")
    if len(df) < 100:
        print("WARNING: Data size is very small. Training will likely fail/be meaningless.")
        return None

    print("Engineering features...")
    df, dynamic_feature_cols, static_feature_cols = engineer_features(df)
    feature_cols = dynamic_feature_cols + static_feature_cols
    print("Dynamic Feature columns:", dynamic_feature_cols)
    print("Static Feature columns (for XGBoost):", static_feature_cols)

    print("Building sequences for LSTM...")
    X_seq, y, X_last, combined_feature_cols = build_sequences(df, dynamic_feature_cols, static_feature_cols, seq_len=seq_len)
    if X_seq.shape[0] < 50:
         print(f"WARNING: Only {X_seq.shape[0]} sequences built. Not enough data for robust split/training.")
         return None
         
    print(f"Built sequences: {X_seq.shape[0]} samples, sequence length {X_seq.shape[1]}, features {X_seq.shape[2]}")

    # train/test split (random)
    X_seq_train, X_seq_test, X_last_train, X_last_test, y_train, y_test = train_test_split(
        X_seq, X_last, y, test_size=test_size, random_state=random_state
    )

    # --- Scaling Dynamic/Sequential Features for LSTM ---
    nsamples, seq_len_in, nfeat_dyn = X_seq_train.shape
    scaler_seq = StandardScaler()
    
    # Fit scaler on the flattened training sequences
    X_seq_train_flat = X_seq_train.reshape(-1, nfeat_dyn)
    scaler_seq.fit(X_seq_train_flat)
    
    # Transform and reshape back
    X_seq_train_scaled = scaler_seq.transform(X_seq_train_flat).reshape(-1, seq_len_in, nfeat_dyn)
    X_seq_test_flat = X_seq_test.reshape(-1, nfeat_dyn)
    X_seq_test_scaled = scaler_seq.transform(X_seq_test_flat).reshape(-1, seq_len_in, nfeat_dyn)
    
    # --- Scaling Combined Features for XGBoost ---
    scaler_last = StandardScaler()
    scaler_last.fit(X_last_train)
    X_last_train_scaled = scaler_last.transform(X_last_train)
    X_last_test_scaled = scaler_last.transform(X_last_test)

    # train LSTM
    input_shape = (seq_len_in, nfeat_dyn)
    print("Building LSTM model...")
    lstm_model = build_lstm_model(input_shape)
    lstm_path = os.path.join(model_dir, 'lstm_eta.h5')
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1), # Increased patience
        ModelCheckpoint(lstm_path, monitor='val_loss', save_best_only=True, verbose=1)
    ]
    print("Training LSTM...")
    hist = lstm_model.fit(
        X_seq_train_scaled, y_train,
        validation_split=0.12,
        epochs=100,
        batch_size=128, # Increased batch size
        callbacks=callbacks,
        verbose=2
    )
    
    # Load best weights
    lstm_model.load_weights(lstm_path)

    # LSTM predictions on train and test
    print("Predicting with LSTM...")
    y_pred_train_lstm = lstm_model.predict(X_seq_train_scaled, verbose=0).ravel()
    y_pred_test_lstm = lstm_model.predict(X_seq_test_scaled, verbose=0).ravel()

    # residuals: actual - lstm_pred (residuals are in log(1 + TTA_hrs) units)
    res_train = y_train - y_pred_train_lstm
    res_test = y_test - y_pred_test_lstm

    # Train XGBoost on last-step features to predict residuals
    print("Training XGBoost on residuals...")
    dtrain = xgb.DMatrix(X_last_train_scaled, label=res_train)
    dtest = xgb.DMatrix(X_last_test_scaled, label=res_test)
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 7, # Slightly deeper XGBoost
        'eta': 0.05,    # Slower learning rate
        'verbosity': 1
    }
    watchlist = [(dtrain, 'train'), (dtest, 'eval')]
    xgb_model = xgb.train(params, dtrain, num_boost_round=300, evals=watchlist, early_stopping_rounds=20, verbose_eval=30) # More rounds/patience

    # final ensemble predictions (in log(1 + TTA_hrs) units)
    y_pred_test_final_scaled = y_pred_test_lstm + xgb_model.predict(dtest)
    y_pred_train_final_scaled = y_pred_train_lstm + xgb_model.predict(dtrain)

    # UN-SCALE PREDICTIONS and TRUE LABELS
    y_train_unscaled = unscale_target(y_train)
    y_test_unscaled = unscale_target(y_test)
    y_pred_test_final = unscale_target(y_pred_test_final_scaled)
    y_pred_train_final = unscale_target(y_pred_train_final_scaled)
    
    # EVALUATE
    def evaluate(y_true, y_pred, label):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = math.sqrt(mean_squared_error(y_true, y_pred))
        print(f"{label} MAE: {mae:.2f} sec ({mae/3600:.2f} hrs), RMSE: {rmse:.2f} sec ({rmse/3600:.2f} hrs)")
        return mae, rmse

    print("\n--- FINAL EVALUATION (Unscaled TTA in Seconds) ---")
    evaluate(y_train_unscaled, y_pred_train_final, 'Train')
    evaluate(y_test_unscaled, y_pred_test_final, 'Test')

    # Save scalers, XGBoost model, and feature list
    joblib.dump(scaler_seq, os.path.join(model_dir, 'scaler_seq.joblib'))
    joblib.dump(scaler_last, os.path.join(model_dir, 'scaler_last.joblib'))
    xgb_model.save_model(os.path.join(model_dir, 'xgb_residuals.json'))
    joblib.dump({'dynamic_feature_cols': dynamic_feature_cols, 
                 'static_feature_cols': static_feature_cols,
                 'combined_feature_cols': combined_feature_cols,
                 'seq_len': seq_len}, os.path.join(model_dir, 'meta.joblib'))

    print("Saved models and scalers to:", model_dir)
    return {
        'lstm_model': lstm_model,
        'xgb_model': xgb_model,
        'scaler_seq': scaler_seq,
        'scaler_last': scaler_last,
        'dynamic_feature_cols': dynamic_feature_cols,
        'static_feature_cols': static_feature_cols,
        'seq_len': seq_len
    }

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
def load_ensemble(model_dir='models_malaysia'):
    # Load Keras model, XGBoost, scalers, meta
    lstm_model = tf.keras.models.load_model(os.path.join(model_dir, 'lstm_eta.h5'))
    xgb_model = xgb.Booster()
    xgb_model.load_model(os.path.join(model_dir, 'xgb_residuals.json'))
    scaler_seq = joblib.load(os.path.join(model_dir, 'scaler_seq.joblib'))
    scaler_last = joblib.load(os.path.join(model_dir, 'scaler_last.joblib'))
    meta = joblib.load(os.path.join(model_dir, 'meta.joblib'))
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
    # WARNING: This sample CSV will not allow successful training.
    # Replace 'path/to/your/malaysia_ais.csv' with your large dataset file path.
    sample_csv = "C:\\Users\\User\\Downloads\\AIS_2024_12_31.csv"
    
    # Create a slightly larger, but still insufficient, sample for structured testing
    if not os.path.exists(sample_csv):
        # Create a voyage of 10 points for MMSI 1 and 2 points for MMSI 2
        sample_text = """MMSI,BaseDateTime,LAT,LON,SOG,COG,Heading,VesselType,Length,Width,Draft
            367776660,2025-01-01T12:00:00,3.0,101.0,12.0,90.0,90.0,70,180,30,12.0
            367776660,2025-01-01T12:05:00,3.1,101.1,12.1,91.0,90.0,70,180,30,12.0
            367776660,2025-01-01T12:10:00,3.2,101.2,12.2,92.0,90.0,70,180,30,12.0
            367776660,2025-01-01T12:15:00,3.3,101.3,12.3,93.0,90.0,70,180,30,12.0
            367776660,2025-01-01T12:20:00,3.4,101.4,12.4,94.0,90.0,70,180,30,12.0
            367776660,2025-01-01T12:25:00,3.5,101.5,12.5,95.0,90.0,70,180,30,12.0
            367776660,2025-01-01T12:30:00,3.6,101.6,12.6,96.0,90.0,70,180,30,12.0
            367776660,2025-01-01T12:35:00,3.7,101.7,12.7,97.0,90.0,70,180,30,12.0
            367776660,2025-01-01T12:40:00,3.8,101.8,12.8,98.0,90.0,70,180,30,12.0
            367776660,2025-01-01T12:45:00,3.9,101.9,12.9,99.0,90.0,70,180,30,12.0
            368095340,2025-01-01T13:00:00,4.0,102.0,10.0,180.0,180.0,30,120,20,8.0
            368095340,2025-01-01T13:05:00,4.1,102.1,10.1,181.0,180.0,30,120,20,8.0
            """
        with open(sample_csv, 'w', newline='') as f:
            f.write(sample_text.replace(" ", ""))

    model_dir = 'models_malaysia'
    
    print(f"ATTENTION: Attempting training with dummy data '{sample_csv}'.")
    print("THIS WILL FAIL TO PRODUCE A USABLE MODEL. REPLACE WITH A LARGE DATASET.")

    models = None
    try:
        # Increased seq_len to 10 for better sequence modeling
        models = train_hybrid_model(sample_csv, seq_len=10, test_size=0.3, model_dir=model_dir)
    except Exception as e:
        print(f"\nTraining failed (Error: {e}). Trying to load pretrained models.")
        try:
            models = load_ensemble(model_dir=model_dir)
            print("Successfully loaded pretrained models.")
        except Exception as e2:
            print(f"No pretrained models found. Cannot run prediction demo. Error: {e2}")
            # raise SystemExit() # Commented out to allow demo to proceed if models loaded

    if models:
        # DEMO prediction: Use the last sequence of the dummy data
        df_full = load_and_prepare_ais(sample_csv)
        df_full, _, _ = engineer_features(df_full)

        def load_and_prepare_ais(csv_path):
            # ... (loading, parsing datetime, dropping NaN coords)
            
            # -------------------------------------------------------------
            # NEW: Geographic Filtering for Malaysian Marine Path (Straits Focus)
            # Define bounding box (LAT: 0.8N to 7.0N, LON: 98.0E to 105.0E)
            lat_min, lat_max = 0.8, 7.0
            lon_min, lon_max = 98.0, 105.0
            
            initial_count = len(df)
            
            df = df[
                (df['LAT'] >= lat_min) & (df['LAT'] <= lat_max) & 
                (df['LON'] >= lon_min) & (df['LON'] <= lon_max)
            ].reset_index(drop=True)
            
            filtered_count = len(df)
            print(f"Filtered {initial_count - filtered_count} records outside the Malaysia Straits bounding box.")
            # -------------------------------------------------------------
            
            
            return df
        
        # Take the last seq_len points of the longest voyage for the demo
        longest_voyage = df_full[df_full['MMSI']==367776660]
        recent = longest_voyage.iloc[-models['seq_len']:].to_dict('records')

        # Define a plausible end point near the last data point
        last_lat = longest_voyage['LAT'].iloc[-1]
        last_lon = longest_voyage['LON'].iloc[-1]
        route_end = (last_lat + 0.01, last_lon + 0.01)
        
        # Replace the line below with your desired fixed destination coordinates
        # Example: Port of Tanjung Pelepas (PTP) (1.35 N, 103.52 E)
        DESTINATION_LAT = 1.35   # Your desired Latitude
        DESTINATION_LON = 103.52 # Your desired Longitude
        
        route_end = (DESTINATION_LAT, DESTINATION_LON)


        print("\n--- DEMO Prediction ---")
        pred = predict_eta(recent, route_end, models)
        print("Prediction output:")
        print(pred)