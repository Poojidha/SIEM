# SIEM Lite — Unsupervised ML Network Anomaly Detection

An end-to-end Security Information and Event Management (SIEM) anomaly detection pipeline built in Python and PyTorch. It detects Denial-of-Service (DoS) attacks (such as DoS Hulk, Slowloris, and GoldenEye) in network flow traffic using an unsupervised ensemble of three machine learning models trained on the **CICIDS2017** dataset.

## Architecture & Approach
Instead of relying on static signatures, this system learns normal ("Benign") network behavior across **28 flow-level features** (flow duration, packet sizes, inter-arrival times, and byte rates) and flags deviations using **Majority Vote Ensemble Learning**:

1. **Isolation Forest:** Isolates anomalous traffic flows through randomized feature partitioning.
2. **One-Class SVM:** Learns a tight decision boundary around normal traffic distributions.
3. **PyTorch Autoencoder:** Compresses and reconstructs benign flows, flagging traffic that exceeds the 95th-percentile reconstruction error threshold.
4. **Majority Vote Alerting:** Triggers a security alert only when at least **2 out of 3 models** agree a flow is malicious, reducing false positives.

## Repository Structure
```text
SIEM/
├── siem-lite-ml/                  # Core ML pipeline, models, and evaluation
│   ├── download_data.py           # Dataset preparation script
│   ├── 1_preprocess.py            # Feature selection, cleaning, and scaling
│   ├── 2_train_models.py          # Trains Isolation Forest, One-Class SVM & Autoencoder
│   ├── 3_evaluate.py              # Evaluates ensemble performance & generates plots
│   ├── datasets/                  # Raw CICIDS2017 subset
│   ├── processed/                 # Preprocessed feature matrices
│   ├── models/                    # Saved trained model weights
│   ├── results/                   # Quantitative evaluation metrics
│   └── figures/                   # Performance charts and confusion matrices
├── MachineLearningCSV/            # Network flow dataset files
└── cicids_preprocessing...        # Exploratory preprocessing scripts
```

## Quick Start
1. **Clone the repository:**
   ```bash
   git clone https://github.com
   cd SIEM/siem-lite-ml
   ```
2. **Prepare the dataset:**
   ```bash
   python download_data.py
   ```
3. **Preprocess the 28 network flow features:**
   ```bash
   python 1_preprocess.py
   ```
4. **Train the unsupervised models:**
   ```bash
   python 2_train_models.py
   ```
5. **Run evaluation and generate figures:**
   ```bash
   python 3_evaluate.py
   ```

## Tech Stack
* **Languages:** Python
* **ML & Deep Learning:** PyTorch, Scikit-Learn (Isolation Forest, One-Class SVM), Pandas, NumPy
* **Dataset:** CICIDS2017 Network Intrusion Dataset

## Future Roadmap
* Real-time packet capture integration via `Scapy` or `Zeek` logs.
* REST API endpoint (`FastAPI`) to score live incoming network flows.
* Docker containerization for reproducible deployment.

## License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
