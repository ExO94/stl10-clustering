import os
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm

# --- CONFIGURATION ---
# Path to the directory containing the 'stl10_binary' folder
DATA_ROOT = '/home/ex0/Downloads/stl10'

# Output file name
SAVE_PATH = 'features.npy'

# CPU Optimization: Ryzen 3600X has 12 threads.
# We use a high worker count to ensure data preprocessing (resize/normalize)
# happens in parallel, keeping the main processing loop fed efficiently.
BATCH_SIZE = 64
NUM_WORKERS = os.cpu_count() or 8
SUBSET_SIZE = 5000  # Set to None to process the full dataset


def main():
    # Force CPU execution
    device = torch.device("cpu")
    print(f"System detected: {os.cpu_count()} logical cores.")
    print(f"Running on: {device} (Optimized mode)")

    # --- 1. PREPROCESSING ---
    # Standard ResNet18 input requirements
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # --- 2. DATASET SETUP ---
    print(f"Loading STL-10 dataset from: {DATA_ROOT}")

    try:
        dataset = datasets.STL10(
            root=DATA_ROOT,
            split='unlabeled',
            download=True,
            transform=preprocess
        )
    except RuntimeError as e:
        print(f"Error loading dataset: {e}")
        print("Ensure DATA_ROOT points to the folder containing 'stl10_binary'.")
        return

    # Create subset if specified
    if SUBSET_SIZE:
        print(f"Processing subset: {SUBSET_SIZE} images")
        indices = list(range(SUBSET_SIZE))
        dataset = Subset(dataset, indices)

    # CPU specific loader: Using multiple workers to maximize throughput
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True  # Optimizes memory transfer
    )

    # --- 3. MODEL SETUP ---
    print("Initializing ResNet18 model...")
    # Suppress potential torchvision warnings by accepting defaults implicitly
    model = models.resnet18(pretrained=True)

    # Remove the final classification layer (fc)
    # We retain the model up to the Global Average Pooling layer
    model = torch.nn.Sequential(*list(model.children())[:-1])

    model.to(device)
    model.eval()

    # --- 4. FEATURE EXTRACTION ---
    features_list = []
    print("Starting extraction pipeline...")

    # Disable gradient calculation to conserve memory and computation
    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="Processing Batches", unit="batch"):
            images = images.to(device)

            # Forward pass
            # Output shape: [Batch_Size, 512, 1, 1]
            embeddings = model(images)

            # Flatten to [Batch_Size, 512]
            embeddings = embeddings.flatten(start_dim=1)

            features_list.append(embeddings.numpy())

    # --- 5. SAVE ARTIFACTS ---
    if features_list:
        features = np.concatenate(features_list, axis=0)
        print(f"Extraction complete. Final tensor shape: {features.shape}")

        np.save(SAVE_PATH, features)
        print(f"Features saved successfully to: {os.path.abspath(SAVE_PATH)}")
    else:
        print("No features were extracted.")


if __name__ == '__main__':
    main()