# SIEM Lite Machine Learning Anomaly Detection

Welcome to the SIEM Lite Machine Learning experiment! This project demonstrates how we can use Machine Learning to detect cyber-attacks in network traffic (specifically, Denial of Service (DoS) attacks).

## 1. The Dataset
We use a subset of the famous **CICIDS2017** dataset. This dataset contains real-world network traffic captures, including normal ("Benign") traffic and various attacks like DoS Hulk, Slowloris, and GoldenEye.
The data is provided in a tabular format, where each row represents a single network "flow" (a sequence of packets between two computers).

### The Features
We look at 28 different characteristics (features) for each network flow to decide if it's an attack. Some of the most important ones include:
- **Flow Duration**: How long the connection lasted.
- **Total Fwd/Bwd Packets**: The number of packets sent forward (from source to destination) and backward.
- **Total Length of Fwd/Bwd Packets**: The total amount of data sent.
- **Packet Length (Min/Max/Mean)**: The sizes of the individual packets. Attackers often send many very small or very large packets.
- **Flow IAT (Inter-Arrival Time)**: The time between packets. Regular human traffic has natural pauses; automated bot traffic often has very consistent, small gaps.
- **Packets/s & Bytes/s**: The speed of data transfer. DoS attacks often have extremely high rates.

## 2. The Machine Learning Models

We are using an **Unsupervised Anomaly Detection** approach. This means we train our models *only on the normal, benign traffic*. The models learn what "normal" looks like. During testing, anything that looks significantly different from normal is flagged as an "anomaly" (an attack).

We use three different models to make our system robust:

1. **Isolation Forest**: Imagine a forest of decision trees. This model tries to isolate each data point by asking random yes/no questions about its features. Normal traffic points are clustered together and take many questions to isolate. Anomalies (attacks) are weird and far away, so they get isolated very quickly (few questions).
2. **One-Class SVM (Support Vector Machine)**: This model draws a tight invisible boundary around all the normal traffic data points. If a new traffic flow falls outside this boundary, it's flagged as an attack.
3. **PyTorch Autoencoder (Neural Network)**: An autoencoder is a neural network designed to compress data into a smaller representation and then reconstruct it back to the original. We train it to perfectly reconstruct normal traffic. When we feed it attack traffic, it hasn't learned how to compress/reconstruct it properly, resulting in a high "reconstruction error". If the error is above our 95th-percentile threshold, it's an attack.

### Ensemble Voting
Instead of relying on just one model, we use a **Majority Vote**. For every network flow, all three models make a guess (0 for Normal, 1 for Attack). If at least 2 out of the 3 models agree it's an attack, the system triggers an alert. This greatly reduces false alarms!

## 3. How to Run the Pipeline

Follow these steps to run the complete pipeline:

1. **Download & Prepare Data**: 
   Run `python download_data.py` (It copies our existing dataset into the `datasets/` folder).
2. **Preprocess Features**: 
   Run `python 1_preprocess.py`. This cleans the data, converts text labels to numbers, selects the features, and scales the values so the models can learn properly.
3. **Train Models**: 
   Run `python 2_train_models.py`. This trains our three models on the normal traffic and saves them.
4. **Evaluate Performance**: 
   Run `python 3_evaluate.py`. This tests the models on mixed traffic (normal + attacks) and generates performance charts in the `figures/` folder.

Check the `results/` and `figures/` folder to see how well the models caught the attacks!
