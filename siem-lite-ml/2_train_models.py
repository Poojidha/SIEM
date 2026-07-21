import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

class ShallowAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(ShallowAutoencoder, self).__init__()
        # PyTorch Autoencoder: encode to a smaller dimension and decode
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def train_models():
    base_path = Path(__file__).parent
    processed_dir = base_path / 'processed'
    models_dir = base_path / 'models'
    
    print("Loading preprocessed training data...")
    X_train_benign = np.load(processed_dir / 'X_train_benign.npy')
    
    # Optional: Subsample for OCSVM if data is too large to train quickly
    sample_size = min(50000, X_train_benign.shape[0])
    np.random.seed(42)
    indices = np.random.choice(X_train_benign.shape[0], sample_size, replace=False)
    X_train_sample = X_train_benign[indices]
    
    # 1. Train Isolation Forest
    print("Training Isolation Forest...")
    iso_forest = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
    iso_forest.fit(X_train_benign)
    joblib.dump(iso_forest, models_dir / 'isolation_forest.pkl')
    
    # 2. Train One-Class SVM
    print(f"Training One-Class SVM on {sample_size} samples...")
    ocsvm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.01)
    ocsvm.fit(X_train_sample)
    joblib.dump(ocsvm, models_dir / 'ocsvm.pkl')
    
    # 3. Train PyTorch Shallow Autoencoder
    print("Training Shallow Autoencoder...")
    input_dim = X_train_benign.shape[1]
    ae_model = ShallowAutoencoder(input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(ae_model.parameters(), lr=0.001)
    
    # Prepare DataLoader
    tensor_x = torch.Tensor(X_train_benign)
    dataset = TensorDataset(tensor_x, tensor_x)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    epochs = 5
    ae_model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x, _ in dataloader:
            optimizer.zero_grad()
            outputs = ae_model(batch_x)
            loss = criterion(outputs, batch_x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(dataloader):.4f}")
        
    torch.save(ae_model.state_dict(), models_dir / 'autoencoder.pt')
    
    # Calculate and save the 95th percentile reconstruction error threshold
    ae_model.eval()
    with torch.no_grad():
        reconstructions = ae_model(tensor_x)
        mse_loss = torch.mean((tensor_x - reconstructions) ** 2, dim=1).numpy()
    
    threshold = np.percentile(mse_loss, 95)
    np.save(models_dir / 'ae_threshold.npy', np.array([threshold]))
    print(f"Autoencoder 95th percentile threshold: {threshold:.4f}")
    
    print("Model training completed successfully!")

if __name__ == "__main__":
    train_models()
