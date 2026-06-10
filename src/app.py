import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from PIL import Image
import io

# Streamlit interface
st.title("Enhanced Image-Based Color Palette Extractor")
st.write("Upload images to extract an accurate color palette with and without PCA.")

# Upload multiple image files
uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)


# Helper function to convert RGB to hex
def rgb_to_hex(rgb_array):
    return ['#%02x%02x%02x' % tuple(map(int, rgb)) for rgb in rgb_array]


if uploaded_files:
    # Display uploaded images
    st.subheader("Uploaded Images")
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        st.image(image, caption=f"Uploaded Image - {uploaded_file.name}", use_column_width=True)

    # Gather pixels from all images
    all_pixels_rgb = []
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert('RGB')
        image = image.resize((100, 100))  # Resize to make it more manageable
        pixels_rgb = np.array(image).reshape(-1, 3)
        all_pixels_rgb.append(pixels_rgb)

    all_pixels_rgb = np.vstack(all_pixels_rgb)  # Stack all pixel arrays into one

    # Number of dominant colors
    n_colors = st.slider("Select the number of dominant colors:", min_value=1, max_value=20, value=5)

    # Original K-Means clustering on RGB color space
    kmeans_original = KMeans(n_clusters=n_colors, init='k-means++', random_state=42)
    kmeans_original.fit(all_pixels_rgb)
    original_centers = kmeans_original.cluster_centers_.astype(int)
    original_hex_codes = rgb_to_hex(original_centers)

    # Display dominant colors without PCA
    st.subheader("Dominant Colors Without PCA")
    st.write("Hex Codes:", ", ".join(original_hex_codes))

    # Display palette without PCA
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.imshow([original_centers / 255.0])
    ax.axis("off")
    plt.title("Dominant Colors Without PCA")
    st.pyplot(fig)

    # Apply PCA with 2 components to reduce dimensionality
    pca = PCA(n_components=2)
    all_pixels_rgb_pca = pca.fit_transform(all_pixels_rgb)

    # K-Means on PCA-reduced data
    kmeans_pca = KMeans(n_clusters=n_colors, init='k-means++', random_state=42)
    kmeans_pca.fit(all_pixels_rgb_pca)
    pca_reduced_centers = pca.inverse_transform(kmeans_pca.cluster_centers_)
    pca_reduced_centers = np.clip(pca_reduced_centers, 0, 255).astype(int)
    pca_hex_codes = rgb_to_hex(pca_reduced_centers)

    # Display dominant colors with PCA
    st.subheader("Dominant Colors With PCA")
    st.write("Hex Codes:", ", ".join(pca_hex_codes))

    # Display palette with PCA
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.imshow([pca_reduced_centers / 255.0])
    ax.axis("off")
    plt.title("Dominant Colors With PCA")
    st.pyplot(fig)

    # Display before and after PCA scatter plots for comparison
    st.subheader("Scatter Plot Comparison: Before and After PCA")

    # Plot original RGB pixel data
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    ax[0].scatter(all_pixels_rgb[:, 0], all_pixels_rgb[:, 1], c=all_pixels_rgb / 255.0, s=1)
    ax[0].set_title("Original RGB Pixel Distribution")
    ax[0].set_xlabel("Red")
    ax[0].set_ylabel("Green")

    # Plot PCA-reduced RGB data
    ax[1].scatter(all_pixels_rgb_pca[:, 0], all_pixels_rgb_pca[:, 1], c=all_pixels_rgb / 255.0, s=1)
    ax[1].set_title("PCA-Reduced RGB Pixel Distribution")
    ax[1].set_xlabel("Principal Component 1")
    ax[1].set_ylabel("Principal Component 2")
    st.pyplot(fig)

    # Palette download as image
    palette_image = np.ones((50, len(original_centers) * 50, 3), dtype=np.uint8)
    for i, color in enumerate(original_centers):
        palette_image[:, i * 50:(i + 1) * 50] = color

    st.subheader("Download Original Palette")
    with io.BytesIO() as buffer:
        Image.fromarray(palette_image).save(buffer, format="PNG")
        st.download_button(
            label="Download Original Palette as PNG",
            data=buffer.getvalue(),
            file_name="original_palette.png",
            mime="image/png"
        )

    # PCA palette download as image
    palette_image_pca = np.ones((50, len(pca_reduced_centers) * 50, 3), dtype=np.uint8)
    for i, color in enumerate(pca_reduced_centers):
        palette_image_pca[:, i * 50:(i + 1) * 50] = color

    st.subheader("Download PCA Palette")
    with io.BytesIO() as buffer:
        Image.fromarray(palette_image_pca).save(buffer, format="PNG")
        st.download_button(
            label="Download PCA Palette as PNG",
            data=buffer.getvalue(),
            file_name="pca_palette.png",
            mime="image/png"
        )

    st.write(
        "Note: The scatter plots above show the difference in color distribution before and after applying PCA, which may help capture broad color patterns and reduce noise.")
