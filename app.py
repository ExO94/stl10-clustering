import gradio as gr
from inference_utils import SearchEngine
from torchvision import datasets
import os

# --- CONFIGURATION ---
# Use the exact same path you used in extract_features.py
DATA_ROOT = '/home/ex0/Downloads/stl10'

# --- 1. INITIALIZE SYSTEM ---
print("Starting Server...")

# Initialize the logic engine (Loads ResNet + K-Means + Features)
engine = SearchEngine()

# Load the raw dataset so we can display the actual images in the gallery
# We use 'unlabeled' split to match the indices in features.npy
print(f"Linking to dataset at {DATA_ROOT}...")
dataset = datasets.STL10(root=DATA_ROOT, split='unlabeled', download=True)


def analyze_image(query_image):
    """
    The main function called by the UI.
    Args:
        query_image (PIL.Image): The image uploaded by the user.
    Returns:
        cluster_text (str): The predicted cluster ID.
        gallery_images (list): List of 5 similar PIL Images.
    """
    if query_image is None:
        return "No Image", []

    # 1. Get Embedding (ResNet18)
    feature_vector = engine.get_embedding(query_image)

    # 2. Predict Cluster (K-Means)
    cluster_id = engine.predict_cluster(feature_vector)

    # 3. Find Neighbors (Cosine Similarity)
    # This returns the INDICES of the matching images in our dataset
    neighbor_indices = engine.search_neighbors(feature_vector, top_k=5)

    # 4. Retrieve Actual Images
    # dataset[i] returns (Image, Label). We only need the Image [0].
    # We grab the images corresponding to the indices found above.
    neighbor_images = []
    for idx in neighbor_indices:
        img, _ = dataset[idx]  # Fetch from disk
        neighbor_images.append((img, f"Img #{idx}"))

    return f"Cluster {cluster_id}", neighbor_images


# --- 2. BUILD INTERFACE ---
# We use Gradio Blocks for a professional layout
with gr.Blocks(title="STL-10 Visual Search") as demo:
    gr.Markdown("#Semantic Image Clustering & Search")
    gr.Markdown("""
    **Architecture:** ResNet18 (Feature Extractor) + K-Means (Clustering)
    **Dataset:** STL-10 (Unsupervised)
    """)

    with gr.Row():
        # LEFT COLUMN: Input
        with gr.Column(scale=1):
            input_img = gr.Image(type="pil", label="Upload Query Image")
            run_btn = gr.Button("🔍 Analyze Image", variant="primary")

        # RIGHT COLUMN: Output
        with gr.Column(scale=2):
            cluster_text = gr.Textbox(label="Predicted Semantic Cluster",
                                      placeholder="Cluster ID will appear here...")
            gallery = gr.Gallery(label="Nearest Neighbors (Visual Search)",
                                 columns=3, height=300)

    # Connect the button to the function
    run_btn.click(
        fn=analyze_image,
        inputs=input_img,
        outputs=[cluster_text, gallery]
    )

    # Add some clickable examples from the dataset itself to test quickly
    # (We grab the first 3 images from the dataset as examples)
    gr.Examples(
        examples=[dataset[0][0], dataset[50][0], dataset[100][0]],
        inputs=input_img,
        label="Try these examples from the dataset:"
    )

# --- 3. LAUNCH ---
if __name__ == "__main__":
    # server_name="0.0.0.0" allows you to access it from other devices on your network
    demo.launch(server_name="0.0.0.0", server_port=7860)
