import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

# --- CONFIGURATION ---
# FORCE CPU for stability.
# The GPU driver causes SegFaults when interacting with the Gradio web server.
DEVICE = torch.device("cpu")
print(f"🚀 Initializing Search Engine on {DEVICE} (Stability Mode)...")


class SearchEngine:
    def __init__(self,
                 features_path='features.npy',
                 labels_path='labels.npy',
                 kmeans_path='kmeans_model.pkl',
                 scaler_path='scaler.pkl'):
        print(f"Initializing Search Engine on {DEVICE}...")

        # 1. Load Artifacts
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Missing {features_path}. Run extract_features.py first.")

        self.features = np.load(features_path)
        self.labels = np.load(labels_path)

        with open(kmeans_path, 'rb') as f:
            self.kmeans = pickle.load(f)

        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

        # 2. Load ResNet (Feature Extractor)
        # We must duplicate the exact structure used during training
        resnet = models.resnet18(pretrained=True)
        self.model = nn.Sequential(*list(resnet.children())[:-1])
        self.model.to(DEVICE)
        self.model.eval()

        # 3. Define Preprocessing (Must match training exactly)
        self.preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        print("Engine Ready.")

    def get_embedding(self, image):
        """
        Takes a PIL Image, pre-processes it, and returns the 512D feature vector.
        """
        # Ensure image is RGB (handles grayscale uploads)
        image = image.convert('RGB')

        # Preprocess
        img_tensor = self.preprocess(image).unsqueeze(0).to(DEVICE)

        # Extract
        with torch.no_grad():
            feature = self.model(img_tensor)
            # Flatten: [1, 512, 1, 1] -> [1, 512]
            feature = feature.flatten(start_dim=1).cpu().numpy()

        return feature

    def predict_cluster(self, feature_vector):
        """
        Predicts which cluster (0-9) the new image belongs to.
        """
        # 1. Scale the feature (Crucial! We trained on scaled data)
        feature_scaled = self.scaler.transform(feature_vector)

        # 2. Predict
        cluster_id = self.kmeans.predict(feature_scaled)[0]
        return cluster_id

    def search_neighbors(self, query_feature, top_k=5):
        """
        Finds the indices of the top_k most similar images in the database.
        """
        # Calculate similarity between Query and ALL database images
        # query_feature shape: (1, 512)
        # self.features shape: (5000, 512)
        similarities = cosine_similarity(query_feature, self.features)

        # Sort and get top indices
        # [0] because similarities returns shape (1, 5000)
        top_indices = np.argsort(similarities[0])[-top_k:][::-1]

        return top_indices