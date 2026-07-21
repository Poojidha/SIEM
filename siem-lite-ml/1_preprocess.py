import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

def preprocess_data():
    base_path = Path(__file__).parent
    dataset_path = base_path / 'datasets' / 'Wednesday-WorkingHours.parquet'
    
    print(f"Loading dataset from {dataset_path}...")
    df = pd.read_parquet(dataset_path)
    
    # Clean column names (remove leading/trailing spaces)
    df.columns = df.columns.str.strip()
    
    # 28 features (example subset of common network flow features)
    features_28 = [
        'Destination Port', 'Flow Duration', 'Total Fwd Packets',
        'Total Backward Packets', 'Total Length of Fwd Packets',
        'Total Length of Bwd Packets', 'Fwd Packet Length Max',
        'Fwd Packet Length Min', 'Fwd Packet Length Mean',
        'Bwd Packet Length Max', 'Bwd Packet Length Min',
        'Bwd Packet Length Mean', 'Flow Bytes/s', 'Flow Packets/s',
        'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
        'Fwd IAT Total', 'Fwd IAT Mean', 'Bwd IAT Total', 'Bwd IAT Mean',
        'Fwd PSH Flags', 'Fwd Packets/s', 'Bwd Packets/s',
        'Min Packet Length', 'Max Packet Length', 'Packet Length Mean'
    ]
    
    # Check if features exist, and only select those that do
    available_features = [f for f in features_28 if f in df.columns]
    
    # Ensure Label exists
    if 'Label' not in df.columns:
        print("Error: 'Label' column not found in dataset!")
        return
        
    print(f"Selected {len(available_features)} features.")
    
    # Drop rows with NaN or Infinity
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=available_features, inplace=True)
    
    # Convert labels: 0 for Benign, 1 for Attack
    df['Label_Binary'] = df['Label'].apply(lambda x: 0 if x == 'Benign' else 1)
    
    X = df[available_features].values
    y = df['Label_Binary'].values
    
    # Split data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Fit StandardScaler exclusively on benign traffic in the training set
    print("Fitting scaler on benign training data...")
    benign_train_mask = (y_train == 0)
    X_train_benign = X_train[benign_train_mask]
    
    scaler = StandardScaler()
    scaler.fit(X_train_benign)
    
    # Transform all data
    X_train_benign_scaled = scaler.transform(X_train_benign)
    X_test_scaled = scaler.transform(X_test)
    
    # Save processed arrays and metadata
    processed_dir = base_path / 'processed'
    print("Saving processed data...")
    np.save(processed_dir / 'X_train_benign.npy', X_train_benign_scaled)
    np.save(processed_dir / 'X_test.npy', X_test_scaled)
    np.save(processed_dir / 'y_test.npy', y_test)
    
    joblib.dump(scaler, processed_dir / 'scaler.pkl')
    joblib.dump(available_features, processed_dir / 'feature_names.pkl')
    
    print("Preprocessing completed successfully!")

if __name__ == "__main__":
    preprocess_data()
