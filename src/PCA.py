import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from PIL import Image
import colorsys

# Streamlit interface
st.title("Color Palette Extractor with PCA, K-Means, and HSV Color Space")
st.write("Upload an image to extract dominant colors using PCA and K-Means clustering.")

# Upload image file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])


# Helper function to convert RGB to HSV and normalize for PCA
def rgb_to_hsv_normalized(rgb_array):
    hsv_array = np.apply_along_axis(
        lambda rgb: colorsys.rgb_to_hsv(*rgb), 1, rgb_array / 255.0)
    return np.array(hsv_array)


# Helper function to convert HSV to RGB for visualization
def hsv_to_rgb(hsv_array):
    rgb_array = np.apply_along_axis(
        lambda hsv: colorsys.hsv_to_rgb(*hsv), 1, hsv_array)
    return (np.array(rgb_array) * 255).astype(np.uint8)


if uploaded_file is not None:
    # Open and resize image
    image = Image.open(uploaded_file).convert('RGB')
    image = image.resize((100, 100))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert image to numpy array in RGB
    pixels_rgb = np.array(image).reshape(-1, 3)

    # Convert RGB to HSV and normalize
    pixels_hsv = rgb_to_hsv_normalized(pixels_rgb)

    # Standardize data for PCA
    scaler = StandardScaler()
    pixels_hsv_scaled = scaler.fit_transform(pixels_hsv)

    # Apply PCA to HSV-scaled data
    pca = PCA(n_components=3)
    pixels_reduced = pca.fit_transform(pixels_hsv_scaled)

    # Calculate variance retained by PCA
    explained_variance = np.sum(pca.explained_variance_ratio_)
    st.subheader("Variance Retained by PCA (3 Components)")
    st.write(f"PCA retains {explained_variance * 100:.2f}% of the original variance in HSV color space.")

    # K-Means clustering
    n_colors = st.slider("Select the number of dominant colors:", min_value=1, max_value=10, value=5)

    # K-Means on PCA-reduced data in HSV space
    kmeans_reduced = KMeans(n_clusters=n_colors)
    kmeans_reduced.fit(pixels_reduced)
    reduced_colors = kmeans_reduced.cluster_centers_
    dominant_colors_hsv = scaler.inverse_transform(pca.inverse_transform(reduced_colors))
    dominant_colors_rgb = hsv_to_rgb(dominant_colors_hsv)

    # Display dominant colors after PCA + K-Means
    st.subheader("Dominant Colors After PCA + K-Means in HSV Color Space")
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.imshow([dominant_colors_rgb / 255.0])
    ax.axis("off")
    plt.title("Dominant Colors After PCA + K-Means in HSV Color Space")
    st.pyplot(fig)

    # K-Means on original HSV color space without PCA
    kmeans_original = KMeans(n_clusters=n_colors)
    kmeans_original.fit(pixels_hsv_scaled)
    original_dominant_colors_hsv = scaler.inverse_transform(kmeans_original.cluster_centers_)
    original_dominant_colors_rgb = hsv_to_rgb(original_dominant_colors_hsv)

    # Display dominant colors without PCA
    st.subheader("Dominant Colors Without PCA")
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.imshow([original_dominant_colors_rgb / 255.0])
    ax.axis("off")
    plt.title("Dominant Colors Without PCA")
    st.pyplot(fig)

    # Inertia comparison
    st.subheader("K-Means Clustering Inertia Comparison")
    kmeans_original_inertia = kmeans_original.inertia_
    kmeans_reduced_inertia = kmeans_reduced.inertia_
    st.write(f"Inertia without PCA: {kmeans_original_inertia:.2f}")
    st.write(f"Inertia with PCA: {kmeans_reduced_inertia:.2f}")

    # Silhouette Score comparison (if needed)
    from sklearn.metrics import silhouette_score

    silhouette_original = silhouette_score(pixels_hsv_scaled, kmeans_original.labels_)
    silhouette_pca = silhouette_score(pixels_reduced, kmeans_reduced.labels_)
    st.subheader("Clustering Quality Comparison (Silhouette Score)")
    st.write(f"Silhouette Score without PCA: {silhouette_original:.2f}")
    st.write(f"Silhouette Score with PCA: {silhouette_pca:.2f}")

    st.write(
        "**Note**: Lower inertia and higher silhouette score indicate more compact clusters, suggesting PCA helps separate colors more effectively.")
