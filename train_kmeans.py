import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pickle
import time
import os

# --- CONFIGURATION ---
FEATURES_PATH = 'features.npy'
N_CLUSTERS = 10  # STL-10 has 10 classes
RANDOM_STATE = 42


def main():
    # 1. Load Features
    print("Loading features...")
    if not os.path.exists(FEATURES_PATH):
        print(f"❌ Error: {FEATURES_PATH} not found. Run extract_features.py first.")
        return

    features = np.load(FEATURES_PATH)
    print(f"Loaded shape: {features.shape}")

    # 2. Scale Features (Crucial for K-Means)
    # K-means is sensitive to the scale of input data.
    print("Scaling features...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # 3. Train K-Means
    print(f"Training K-Means ({N_CLUSTERS} clusters)...")
    start_time = time.time()

    # We use k-means++ initialization for better convergence
    kmeans = KMeans(
        n_clusters=N_CLUSTERS,
        init='k-means++',
        n_init=10,
        max_iter=300,
        random_state=RANDOM_STATE,
        verbose=0
    )

    # Fit the model and predict labels for the training set
    labels = kmeans.fit_predict(features_scaled)

    duration = time.time() - start_time
    print(f"Training complete in {duration:.2f} seconds.")
    print(f"   Inertia (Loss): {kmeans.inertia_:.2f}")

    # 4. Analyze Distribution
    # We want to check if the clusters are balanced.
    unique, counts = np.unique(labels, return_counts=True)
    print("\n📊 Cluster Distribution:")
    print("-" * 30)
    for k, count in zip(unique, counts):
        print(f"   Cluster {k}: {count} images")
    print("-" * 30)

    # 5. Save Artifacts
    # We save the model AND the scaler so we can process new user uploads later.
    print("Saving artifacts...")
    np.save('labels.npy', labels)

    with open('kmeans_model.pkl', 'wb') as f:
        pickle.dump(kmeans, f)

    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    print("   Saved: labels.npy, kmeans_model.pkl, scaler.pkl")


if __name__ == "__main__":
    main()