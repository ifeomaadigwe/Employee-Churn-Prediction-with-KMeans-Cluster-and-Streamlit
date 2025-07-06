import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pipeline import create_preprocessing_pipeline, preprocess_data

# Load and clean the dataset
df = pd.read_csv(r'C:\Users\IfeomaAugustaAdigwe\Desktop\Employee Churn Prediction\data\hr_data_cleaned.csv')
df.columns = df.columns.str.strip()

# Create and apply preprocessing pipeline
preprocessor, features = create_preprocessing_pipeline()
X, _, _ = preprocess_data(df, preprocessor, features)

# Range of k values to test
k_values = range(2, 11)
silhouette_scores = []

# Evaluate silhouette score for each k
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)
    print(f"k = {k}, Silhouette Score = {score:.4f}")

# Plot the silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(k_values, silhouette_scores, marker='o')
plt.title('Silhouette Score vs. Number of Clusters (k)')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.xticks(k_values)
plt.grid(True)
plt.tight_layout()
plt.savefig('artifacts/silhouette_scores.png')
plt.show()
