import numpy as np
import pandas as pd
import xgboost as xgb

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
    # ... (omitted setup)
    # for each voyage, set arrival_time = last timestamp
    arrival = df.groupby('voyage_id')['BaseDateTime'].max().rename('arrival_time')
    df = df.merge(arrival, on='voyage_id', how='left')
    # create time-to-arrival in seconds
    df['TTA_sec'] = (df['arrival_time'] - df['BaseDateTime']).dt.total_seconds()
    
    # -------------------------------------------------------------
    # NEW: SCALED TARGET VARIABLE
    df['TTA_scaled'] = scale_target(df['TTA_sec'])
    # -------------------------------------------------------------

    # ... (omitted setup)

    return df

# -----------------------
# Prediction function for route planner
# -----------------------
def predict_eta(recent_points, route_end_latlon, models):
# ... (omitted setup and data prep) # 1. Dynamic Features for LSTM (Sequence) 
# ... (omitted scaling of sequence data) # 2. Combined Features for XGBoost (Last Step) 
# ... (omitted scaling of last step/static data)
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


if __name__ == '__main__':
    # ... (omitted setup and training attempt)
       
    if models:
        # DEMO prediction: Use the last sequence of the dummy data
        df_full = load_and_prepare_ais(sample_csv)
        df_full, _, _ = engineer_features(df_full)

        # Take the last seq_len points of the longest voyage for the demo
        longest_voyage = df_full[df_full['MMSI']==367776660]
        recent = longest_voyage.iloc[-models['seq_len']:].to_dict('records')

        # Define a plausible end point near the last data point
        last_lat = longest_voyage['LAT'].iloc[-1]
        last_lon = longest_voyage['LON'].iloc[-1]
        route_end = (last_lat + 0.01, last_lon + 0.01)
        print("\n--- DEMO Prediction ---")
        pred = predict_eta(recent, route_end, models)
        print("Prediction output:")
        print(pred)
