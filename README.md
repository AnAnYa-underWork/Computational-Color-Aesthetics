# Computational Color Aesthetics

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-green)
![Image Processing](https://img.shields.io/badge/Image-Processing-orange)
![PCA](https://img.shields.io/badge/PCA-Dimensionality%20Reduction-success)
![K-Means](https://img.shields.io/badge/K--Means-Clustering-purple)
![SVM](https://img.shields.io/badge/SVM-Classification-red)
![Research](https://img.shields.io/badge/Research-Published-success)
![Conference](https://img.shields.io/badge/Conference-INCOFT%202025-blue)
![DOI](https://img.shields.io/badge/DOI-10.5220%2F0013590600004664-blueviolet)

A machine learning framework for extracting dominant colors from images, generating aesthetically coherent color palettes, and classifying visual styles using Principal Component Analysis (PCA), K-Means Clustering, and Support Vector Machines (SVM).

---

## Publication

**Aesthetic of Colour: A Machine Learning Approach of Palette Generation and Aesthetic Classification**

**Conference:** International Conference on Futuristic Technology (INCOFT 2025)

**DOI:** https://doi.org/10.5220/0013590600004664

---

## Abstract

Color plays a fundamental role in visual communication, influencing perception, emotion, and decision-making. Traditional color palette design often relies on subjective intuition and artistic expertise, making it difficult to achieve consistency and scalability.

This research presents a computational framework that automates color palette generation and aesthetic classification through machine learning techniques. The proposed system extracts dominant colors from images, identifies meaningful color groupings, classifies palettes into aesthetic categories, and recommends visually similar palettes belonging to the same style.

The framework combines dimensionality reduction, clustering, and supervised learning to bridge the gap between computational image analysis and design-oriented aesthetic evaluation.

---

## Problem Statement

Creating visually appealing color palettes remains a largely subjective process that depends heavily on the experience and intuition of designers.

Existing image analysis systems focus primarily on object detection, recognition, and scene understanding, while aesthetic color analysis remains comparatively underexplored.

This project addresses that gap by developing an automated pipeline capable of:

- Extracting dominant colors from images
- Reducing high-dimensional color information
- Identifying representative color groups
- Classifying palettes into aesthetic styles
- Recommending visually coherent palettes

---

## Methodology

<p align="center">
  <img src="project%20architechture.png" width="800">
</p>

### 1. Image Preprocessing

- Input RGB image acquisition
- Pixel extraction
- Data normalization
- Feature preparation

### 2. Principal Component Analysis (PCA)

PCA is applied to reduce dimensionality while preserving the most significant chromatic information present within the image.

Benefits:

- Reduced computational complexity
- Noise reduction
- Improved clustering performance
- Better feature representation

### 3. K-Means Clustering

The transformed color data is clustered using K-Means to identify dominant color groups.

Output:

- Cluster centroids
- Representative dominant colors
- Base color palette

### 4. Feature Extraction

Each generated palette is converted into a numerical feature vector containing color information suitable for classification.

### 5. Support Vector Machine (SVM)

A multi-class SVM classifier is trained to categorize palettes into aesthetic styles such as:

- Pastel
- Neon
- Vintage
- Modern
- Monochrome

### 6. Palette Recommendation

After classification, the system recommends additional palettes sharing similar aesthetic characteristics.

---

## System Architecture

```text
Input Image
     │
     ▼
Pixel Extraction
     │
     ▼
Data Normalization
     │
     ▼
Principal Component Analysis
     │
     ▼
K-Means Clustering
     │
     ▼
Dominant Color Extraction
     │
     ▼
Feature Vector Generation
     │
     ▼
Support Vector Machine
     │
     ▼
Aesthetic Classification
     │
     ▼
Palette Recommendation
````

---

## Results

### Dominant Color Extraction

* PCA successfully reduced color-space dimensionality.
* K-Means effectively identified representative dominant colors.
* Improved clustering compactness was observed after PCA transformation.

### Classification Performance

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 94%   |
| Precision | 94%   |
| Recall    | 94%   |
| F1 Score  | 94%   |

### Key Findings

* Effective dimensionality reduction using PCA
* Strong aesthetic classification performance
* Improved silhouette scores after transformation
* Consistent palette recommendations across image categories

---

## Technologies Used

| Category                 | Technology   |
| ------------------------ | ------------ |
| Programming Language     | Python       |
| Data Analysis            | NumPy        |
| Machine Learning         | Scikit-Learn |
| Visualization            | Matplotlib   |
| Dimensionality Reduction | PCA          |
| Clustering               | K-Means      |
| Classification           | SVM          |


---

## Applications

* User Interface Design
* Graphic Design
* Branding and Marketing
* Visual Content Creation
* Design Recommendation Systems
* Computational Creativity Research
* Human-Centered Design

---

## Future Work

Potential improvements include:

* GAN-based palette generation
* Deep learning aesthetic prediction
* Texture-aware visual analysis
* Real-time design recommendation systems
* Interactive web deployment
* Expanded aesthetic categories

---

## Authors

**Ananya**
Department of Mechanical Engineering
Amrita School of Engineering, Bengaluru

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{rathore2025palette,
  title={Aesthetic of Colour: A Machine Learning Approach of Palette Generation and Aesthetic Classification},
  author={Rathore, Ananya and others},
  booktitle={International Conference on Futuristic Technology},
  year={2025},
  doi={10.5220/0013590600004664}
}
```

---

## License

This repository is intended for academic, educational, and research purposes.

Please refer to the published paper for citation requirements.

```
```

