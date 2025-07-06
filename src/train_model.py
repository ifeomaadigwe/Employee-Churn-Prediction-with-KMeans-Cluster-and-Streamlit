import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from pipeline import create_preprocessing_pipeline, preprocess_data
import mlflow
import mlflow.sklearn
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

def get_feature_names(preprocessor):
    """Extract full feature names from the column transformer after fitting."""
    num_features = preprocessor.named_transformers_['num'].get_feature_names_out()
    cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out()
    return np.concatenate([num_features, cat_features])

def train_and_evaluate():
    """Train a KMeans clustering model and evaluate its performance."""

    # Load the dataset
    df = pd.read_csv(r'C:\Users\IfeomaAugustaAdigwe\Desktop\Employee Churn Prediction\data\hr_data_cleaned.csv')

    # Create preprocessing pipeline
    preprocessor, features = create_preprocessing_pipeline()

    # Preprocess the data
    X, preprocessor, _ = preprocess_data(df, preprocessor, features)

    # Get proper feature names
    full_feature_names = get_feature_names(preprocessor)

    # Convert transformed array into DataFrame
    X_df = pd.DataFrame(X, columns=full_feature_names)

    # Initialize MLflow experiment
    mlflow.set_experiment("Employee_Churn_Clustering")

    with mlflow.start_run():
        # Train KMeans model
        kmeans = KMeans(n_clusters=6, random_state=42)
        kmeans.fit(X_df)

        # Predict clusters
        df['cluster'] = kmeans.predict(X_df)

        # Evaluate with silhouette score
        silhouette_avg = silhouette_score(X_df, df['cluster'])
        print(f"Silhouette Score: {silhouette_avg}")

        # Log parameters and metrics
        mlflow.log_param("n_clusters", 6)
        mlflow.log_metric("silhouette_score", float(silhouette_avg))

        # Log models
        mlflow.sklearn.log_model(kmeans, "kmeans_model")
        mlflow.sklearn.log_model(preprocessor, "preprocessor")

        # Save locally
        os.makedirs('artifacts', exist_ok=True)
        joblib.dump(preprocessor, 'artifacts/preprocessor.pkl')
        joblib.dump(kmeans, 'artifacts/kmeans_model.pkl')

        # Cluster Visualization using real feature names
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=X_df.iloc[:, 0], y=X_df.iloc[:, 1], hue=df['cluster'], palette='viridis', s=50)
        plt.title('KMeans Clustering Results (First Two Features)')
        plt.xlabel(X_df.columns[0])
        plt.ylabel(X_df.columns[1])
        plt.legend(title='Cluster')
        plt.tight_layout()
        plt.savefig('artifacts/kmeans_clustering_feature_view.png')
        plt.close()

        # PCA-based 2D visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_df)

        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['cluster'], palette='viridis', s=50)
        plt.title('KMeans Clustering Results (PCA)')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.legend(title='Cluster')
        plt.tight_layout()
        plt.savefig('artifacts/kmeans_clustering_pca.png')
        plt.close()

    print("✅ Training and evaluation completed. Results saved in 'artifacts' directory.")

if __name__ == "__main__":
    train_and_evaluate()
