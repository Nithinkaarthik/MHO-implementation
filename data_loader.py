import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_client_data(dataset_dir, sample_size=5000):
    """
    Loads ToN-IoT datasets from the given directory.
    To simulate heterogeneity, each CSV is treated as a separate client domain.
    """
    files = [
        'IoT_Fridge.csv', 'IoT_Garage_Door.csv', 'IoT_Thermostat.csv', 
        'IoT_Motion_Light.csv', 'IoT_Weather.csv', 'IoT_Modbus.csv', 'IoT_GPS_Tracker.csv'
    ]
    
    clients_data = []
    
    for f in files:
        path = os.path.join(dataset_dir, f)
        if not os.path.exists(path):
            continue
            
        print(f"Loading {f}...")
        # Read random sample safely keeping Class Imbalance controlled
        df = pd.read_csv(path)
        
        # We need a smaller sample to let PyTorch train fully without taking 4 hours
        target_sample = min(sample_size, len(df))
        
        if 'label' in df.columns and len(df['label'].unique()) > 1:
            # Force exactly 50/50 split of Normal vs Attack via sampling with replacement
            try:
                g0 = df[df['label'] == 0].sample(n=int(sample_size/2), replace=True)
                g1 = df[df['label'] == 1].sample(n=int(sample_size/2), replace=True)
                df = pd.concat([g0, g1], ignore_index=True)
            except Exception:
                df = df.sample(n=target_sample, random_state=42, replace=True)
        else:
            df = df.sample(n=target_sample, random_state=42, replace=True)
            
        # Drop columns that leak target info or are non-numeric identifiers
        drop_cols = ['date', 'time', 'type', 'ts', 'src_ip', 'dst_ip', 'http.request.uri', 'http.user_agent', 'dns.qry.name']
        drop_cols_existing = [c for c in drop_cols if c in df.columns]
        df = df.drop(columns=drop_cols_existing)
        
        # Convert categoricals to numeric or drop them for simplicity in this MLP
        # Many ToN-IoT sub-datasets have boolean or string states like 'high', 'low' etc.
        # We will quickly label encode any remaining object columns
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype('category').cat.codes
            
        # Handle nan/inf
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        
        # We look for binary 'label' column
        if 'label' not in df.columns:
            print(f"Skipped {f} because 'label' not found.")
            continue
            
        y = df['label'].values.astype(np.int64)
        X = df.drop(columns=['label']).values.astype(np.float32)
        
        print(f"Appending {f} with shape {X.shape}")
        clients_data.append((X, y, f))
        
    return clients_data

def preprocess_and_split(clients_data, val_ratio=0.2):
    """
    Scales the data and splits into train/test per client.
    Because different CSVs might have different column counts (features),
    for a unified FL model, we actually need homogeneous feature sizes, or we pad them.
    Wait! The ToN-IoT original specific CSVs have DIFFERENT columns because device sensors differ!
    If we want a single global MLP, we need the SAME feature dimension!
    Let's find the common subset of columns, or pad with zeros.
    Actually, let's pad all clients to a maximum feature dimension.
    """
    # Find max feature dim
    max_dim = max([X.shape[1] for X, y, name in clients_data])
    
    processed_clients = []
    server_val_X_list = []
    server_val_y_list = []
    
    for X, y, name in clients_data:
        # Pad to max_dim
        if X.shape[1] < max_dim:
            pad_width = max_dim - X.shape[1]
            X = np.pad(X, ((0, 0), (0, pad_width)), mode='constant')

        # Shuffle to avoid label-order leakage into train/validation split.
        perm = np.random.permutation(len(X))
        X = X[perm]
        y = y[perm]
            
        # Scale
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Split train/val
        split_idx = int(len(X) * (1 - val_ratio))
        X_train, y_train = X[:split_idx], y[:split_idx]
        X_val, y_val = X[split_idx:], y[split_idx:]
        
        processed_clients.append({
            'name': name,
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val
        })
        
        server_val_X_list.append(X_val)
        server_val_y_list.append(y_val)
        
    # Create global validation set by concatenating all clients' validation data
    global_val_X = np.concatenate(server_val_X_list)
    global_val_y = np.concatenate(server_val_y_list)
    
    return processed_clients, global_val_X, global_val_y, max_dim

