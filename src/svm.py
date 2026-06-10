import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib

# Load the dataset
dataset = pd.read_csv("updated_color_palettes.csv")

# Check for class imbalance
print(dataset['Aesthetic'].value_counts())

# Split dataset for training and visualization
X = dataset.iloc[:, 1:]  # Exclude the 'Aesthetic' column
y = dataset['Aesthetic']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# Load your pre-trained model
def load_model(model_name="svm_color_aesthetics.pkl"):
    return joblib.load(model_name)

# Helper function for decision boundary in 2D
def plot_decision_boundary_2D(X, y, clf, ax, title="Decision Boundary"):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.8, cmap="coolwarm")
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k", cmap="coolwarm")
    ax.set_title(title)
    return scatter

# Plot decision boundary in 3D
def plot_decision_boundary_3D(X, y, clf, ax):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    z_min, z_max = X[:, 2].min() - 1, X[:, 2].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 20), np.linspace(y_min, y_max, 20))
    zz = np.linspace(z_min, z_max, 20)
    ZZ = np.zeros_like(xx)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            test_point = np.c_[xx[i, j], yy[i, j], zz.mean()]
            ZZ[i, j] = clf.decision_function(test_point.reshape(1, -1))
    ax.plot_surface(xx, yy, ZZ, cmap="coolwarm", alpha=0.6)
    scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap="coolwarm", edgecolor="k")
    return scatter

# Kernel transformation visualization
def plot_kernel_transformation(X, transformed_X, y, ax1, ax2):
    scatter = ax1.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolor="k")
    ax1.set_title("Original Space")
    scatter = ax2.scatter(transformed_X[:, 0], transformed_X[:, 1], c=y, cmap="coolwarm", edgecolor="k")
    ax2.set_title("Transformed Space")

# Confusion matrix
def plot_confusion_matrix(y_true, y_pred, ax):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm", ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

# Precision-Recall curve
def plot_precision_recall_curve(y_true, y_scores, ax):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ax.plot(recall, precision, color="b", lw=2)
    ax.set_title("Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")

# ROC curve
def plot_roc_curve(y_true, y_scores, ax):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color="b", lw=2, label=f"AUC = {roc_auc:.2f}")
    ax.plot([0, 1], [0, 1], "r--")
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")

# Margins visualization
def plot_svm_margins(X, y, clf, ax):
    plot_decision_boundary_2D(X, y, clf, ax, title="SVM Margins")
    ax.plot(X[:, 0], X[:, 1], linestyle='--', alpha=0.5)

# Example usage (replace with your data and model)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
svc = load_model("svm_color_aesthetics.pkl")
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
y_scores = svc.decision_function(X_test)

fig, axs = plt.subplots(4, 2, figsize=(16, 20))

# Decision Boundary 2D
plot_decision_boundary_2D(X_train[:, :2], y_train, svc, axs[0, 0])

# Decision Boundary 3D
ax_3d = fig.add_subplot(2, 2, 2, projection="3d")
plot_decision_boundary_3D(X_train[:, :3], y_train, svc, ax_3d)


# Kernel Transformation
# Apply a kernel transformation (e.g., polynomial or RBF)
X_transformed = svc._get_kernel(X_train)
plot_kernel_transformation(X_train[:, :2], X_transformed[:, :2], y_train, axs[1, 0], axs[1, 1])

# Confusion Matrix
plot_confusion_matrix(y_test, y_pred, axs[2, 0])

# Precision-Recall Curve
plot_precision_recall_curve(y_test, y_scores, axs[2, 1])

# ROC Curve
plot_roc_curve(y_test, y_scores, axs[3, 0])

# Margins Visualization
plot_svm_margins(X_train[:, :2], y_train, svc, axs[3, 1])

plt.tight_layout()
plt.show()
