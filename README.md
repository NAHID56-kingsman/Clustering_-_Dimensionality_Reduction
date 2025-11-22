# 🌸 Iris Dataset Clustering Analysis

> Exploring unsupervised learning techniques on the classic Iris dataset using K-Means, PCA, Hierarchical Clustering, and DBSCAN.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org)


---

## 🎯 About

This project applies four major clustering algorithms to the Iris dataset (150 samples, 4 features) and compares their performance against true species labels.

| Algorithm | Description |
|-----------|-------------|
| **K-Means** | Centroid-based partitioning |
| **PCA + K-Means** | Dimensionality reduction followed by clustering |
| **Hierarchical** | Agglomerative clustering with dendrogram |
| **DBSCAN** | Density-based spatial clustering |

---

## 📁 Project Structure

```
iris-clustering/
│
├── iris_clustering.ipynb    # Main Jupyter notebook
├── script.py                # Standalone Python script
├── README.md
└── requirements.txt
```

---

## ⚡ Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/iris-clustering.git
cd iris-clustering

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```txt
numpy
pandas
matplotlib
scikit-learn
scipy
```

---

## 🔬 Methods & Implementation

### 1️⃣ Data Loading

```python
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = iris.target
```

### 2️⃣ K-Means Clustering

```python
from sklearn.cluster import KMeans

km = KMeans(n_clusters=3, random_state=42, n_init=10)
label_km = km.fit_predict(x)
```

### 3️⃣ Elbow Method (Optimal K)

```python
inertias = []
for k in range(1, 11):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(x)
    inertias.append(model.inertia_)
```

### 4️⃣ PCA Dimensionality Reduction

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
x_pca = pca.fit_transform(x)

# Check explained variance
print(pca.explained_variance_ratio_)
```

### 5️⃣ Hierarchical Clustering

```python
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

linked = linkage(x, method='ward')
hc_labels = fcluster(linked, t=6, criterion='maxclust')
```

### 6️⃣ DBSCAN

```python
from sklearn.cluster import DBSCAN

db = DBSCAN(eps=1.0, min_samples=5)
label_db = db.fit_predict(x)
```

---

## 📊 Visualizations

| Visualization | Purpose |
|---------------|---------|
| K-Means Scatter | Cluster assignments on 2D features |
| Elbow Plot | Inertia vs. K to find optimal clusters |
| PCA Projection | 4D → 2D transformation |
| Dendrogram | Hierarchical cluster merging structure |
| DBSCAN Results | Density-based cluster detection |

---

## 📈 Evaluation

Compare clustering results against true labels using crosstabs:

```python
pd.crosstab(label_km, df["target"])
pd.crosstab(label_kpca, df["target"])
pd.crosstab(hc_labels, df["target"])
```

---

## 🧪 Key Findings

- **K-Means** effectively separates Setosa but struggles with Versicolor/Virginica overlap
- **PCA** captures ~95% variance in first 2 components
- **Hierarchical** provides interpretable cluster hierarchy via dendrogram
- **DBSCAN** is sensitive to `eps` parameter; optimal value requires tuning

---

## 🤝 Contributing

Contributions are welcome! Feel free to:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

<p align="center">
  Made with ❤️ for Machine Learning enthusiasts
</p>
