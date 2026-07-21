import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class ShallowAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(ShallowAutoencoder, self).__init__()
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

def evaluate_models():
    base_path = Path(__file__).parent
    processed_dir = base_path / 'processed'
    models_dir = base_path / 'models'
    results_dir = base_path / 'results'
    figures_dir = base_path / 'figures'
    
    print("Loading test data and models...")
    X_test = np.load(processed_dir / 'X_test.npy')
    y_test = np.load(processed_dir / 'y_test.npy')
    
    iso_forest = joblib.load(models_dir / 'isolation_forest.pkl')
    ocsvm = joblib.load(models_dir / 'ocsvm.pkl')
    
    input_dim = X_test.shape[1]
    ae_model = ShallowAutoencoder(input_dim)
    ae_model.load_state_dict(torch.load(models_dir / 'autoencoder.pt', weights_only=True))
    ae_model.eval()
    ae_threshold = np.load(models_dir / 'ae_threshold.npy')[0]
    
    # 1. Predictions: Isolation Forest (-1 is anomaly, 1 is normal) -> Convert to 1 for Attack, 0 for Normal
    print("Running Isolation Forest...")
    if_preds = iso_forest.predict(X_test)
    if_preds = np.where(if_preds == -1, 1, 0)
    
    # 2. Predictions: One-Class SVM (-1 is anomaly, 1 is normal)
    print("Running One-Class SVM...")
    ocsvm_preds = ocsvm.predict(X_test)
    ocsvm_preds = np.where(ocsvm_preds == -1, 1, 0)
    
    # 3. Predictions: Autoencoder
    print("Running Autoencoder...")
    tensor_test = torch.Tensor(X_test)
    with torch.no_grad():
        reconstructions = ae_model(tensor_test)
        mse_loss = torch.mean((tensor_test - reconstructions) ** 2, dim=1).numpy()
    ae_preds = (mse_loss > ae_threshold).astype(int)
    
    # 4. Ensemble Voting (Majority Vote)
    print("Computing Majority Vote Ensemble...")
    # Sum the predictions (each is 0 or 1). If sum >= 2, it's an anomaly.
    ensemble_votes = if_preds + ocsvm_preds + ae_preds
    ensemble_preds = (ensemble_votes >= 2).astype(int)
    
    models = {
        'Isolation Forest': if_preds,
        'One-Class SVM': ocsvm_preds,
        'Autoencoder': ae_preds,
        'Ensemble (Voting)': ensemble_preds
    }
    
    results = []
    
    for name, preds in models.items():
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        
        tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0,1]).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'FPR': fpr
        })
        
        if name == 'Ensemble (Voting)':
            # Plot Confusion Matrix for Ensemble
            plt.figure(figsize=(6,5))
            sns.heatmap([[tn, fp], [fn, tp]], annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
            plt.title('Ensemble Model Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(figures_dir / 'confusion_matrix.png')
            plt.close()

    results_df = pd.DataFrame(results)
    results_df.to_csv(results_dir / 'detection_performance.csv', index=False)
    print("\nResults saved to results/detection_performance.csv")
    print(results_df.to_string(index=False))
    
    # Plot Performance Comparison
    results_melted = results_df.melt(id_vars='Model', value_vars=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                                     var_name='Metric', value_name='Score')
    plt.figure(figsize=(10,6))
    sns.barplot(x='Model', y='Score', hue='Metric', data=results_melted)
    plt.title('Model Performance Comparison')
    plt.ylim(0, 1)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(figures_dir / 'performance_comparison.png')
    plt.close()
    
    # Plot FPR Comparison
    plt.figure(figsize=(8,5))
    sns.barplot(x='Model', y='FPR', data=results_df, color='salmon')
    plt.title('False Positive Rate (FPR) Comparison')
    plt.tight_layout()
    plt.savefig(figures_dir / 'fpr_comparison.png')
    plt.close()
    
    print("Evaluation completed successfully!")

if __name__ == "__main__":
    evaluate_models()
