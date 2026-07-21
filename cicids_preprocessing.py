# cicids_preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

def load_and_merge_cicids():
    """Load all CICIDS2017 CSV files and merge them"""
    files = [
        'MachineLearningCSV/MachineLearningCVE/Monday-WorkingHours.pcap_ISCX.csv',
        'MachineLearningCSV/MachineLearningCVE/Tuesday-WorkingHours.pcap_ISCX.csv',
        'MachineLearningCSV/MachineLearningCVE/Wednesday-workingHours.pcap_ISCX.csv',
        'MachineLearningCSV/MachineLearningCVE/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
        'MachineLearningCSV/MachineLearningCVE/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
        'MachineLearningCSV/MachineLearningCVE/Friday-WorkingHours-Morning.pcap_ISCX.csv',
        'MachineLearningCSV/MachineLearningCVE/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
        'MachineLearningCSV/MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'
    ]
    
    dfs = []
    for file in files:
        print(f"Loading {file}...")
        df = pd.read_csv(file)
        # Clean column names (remove spaces)
        df.columns = df.columns.str.strip().str.replace(' ', '')
        dfs.append(df)
    
    full_df = pd.concat(dfs, ignore_index=True)
    print(f"Total records: {len(full_df)}")
    return full_df

def preprocess_cicids(df):
    """Preprocess CICIDS2017 for the SIEM framework"""
    
    # Handle missing/infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    
    # The 'Label' column contains: 'BENIGN' or attack names
    # Create binary target: 0 = BENIGN, 1 = ATTACK
    df['target'] = (df['Label'] != 'BENIGN').astype(int)
    
    # Drop non-numeric columns that we can't use for ML
    # Keep: FlowDuration, TotalFwdPackets, TotalBackwardPackets, 
    #       TotalLengthofFwdPackets, TotalLengthofBwdPackets, etc.
    drop_cols = ['FlowID', 'SourceIP', 'DestinationIP', 'SourcePort', 
                 'DestinationPort', 'Timestamp', 'Protocol', 'Label']
    
    # Select only numeric features (28 features as per your thesis)
    feature_cols = [
        'FlowDuration', 'TotalFwdPackets', 'TotalBackwardPackets',
        'TotalLengthofFwdPackets', 'TotalLengthofBwdPackets',
        'FwdPacketLengthMax', 'FwdPacketLengthMin', 'FwdPacketLengthMean',
        'FwdPacketLengthStd', 'BwdPacketLengthMax', 'BwdPacketLengthMin',
        'BwdPacketLengthMean', 'BwdPacketLengthStd', 'FlowBytes/s',
        'FlowPackets/s', 'FlowIATMean', 'FlowIATStd', 'FlowIATMax',
        'FlowIATMin', 'FwdIATTotal', 'FwdIATMean', 'FwdIATStd',
        'FwdIATMax', 'FwdIATMin', 'BwdIATTotal', 'BwdIATMean', 
        'BwdIATStd', 'BwdIATMax'
    ]
    
    X = df[feature_cols].copy()
    y = df['target'].values
    
    # Handle any remaining infinities in numeric columns
    X = X.replace([np.inf, -np.inf], 0)
    
    # Split: 80% train (benign only for training), 20% test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # For anomaly detection: train ONLY on benign traffic
    benign_mask = y_train == 0
    X_train_benign = X_train[benign_mask]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_benign)
    X_test_scaled = scaler.transform(X_test)
    
    # Save preprocessor
    pickle.dump(scaler, open('cicids_scaler.pkl', 'wb'))
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols

if __name__ == "__main__":
    df = load_and_merge_cicids()
    X_train, X_test, y_train, y_test, features = preprocess_cicids(df)
    print(f"Training samples (benign): {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Attack ratio in test: {y_test.mean():.2%}")