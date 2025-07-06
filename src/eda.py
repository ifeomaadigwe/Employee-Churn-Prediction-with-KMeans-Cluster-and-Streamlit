import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

def perform_eda(df, save_path):
    """Perform exploratory data analysis on the given DataFrame and save plots and cleaned CSV."""
    os.makedirs('artifacts', exist_ok=True)

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Rename columns to exactly match your preferred format
    df.rename(columns={
        'satisfaction_level': 'Satisfaction_Level',
        'last_evaluation': 'Last_Evaluation_Score',
        'number_project': 'Number_of_Projects',
        'average_montly_hours': 'Average_Monthly_Hours',
        'time_spend_company': 'Years_at_Company',
        'Work_accident': 'Work_Accident',
        'left': 'Left_Company',
        'promotion_last_5years': 'Promotion_in_Last_5_Years',
        'Departments': 'Departments',
        'salary': 'Salary_Level'
    }, inplace=True)

    # Save the cleaned DataFrame with updated column names
    df.to_csv(save_path, index=False)
    print(f"✅ Cleaned data saved to: {save_path}")

    # Display basic information about the DataFrame
    print("\n📊 DataFrame Info:")
    print(df.info())
    print("\n📈 DataFrame Description:")
    print(df.describe())

    # Display the first and last few rows
    print("\n🔍 First few rows of the DataFrame:")
    print(df.head())
    print("\n🔍 Last few rows of the DataFrame:")
    print(df.tail())

    # Correlation matrix for numeric features
    numeric_df = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(12, 8))
    sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='coolwarm', square=True)
    plt.title('Correlation Matrix')
    plt.savefig('artifacts/correlation_matrix.png')
    plt.close()

    # Feature distributions
    for column in numeric_df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[column], kde=True, bins=30)
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.savefig(f'artifacts/distribution_{column}.png')
        plt.close()

    # Elbow method for optimal number of clusters
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)
    inertia = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(pca_data)
        inertia.append(kmeans.inertia_)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), inertia, marker='o')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.xticks(range(1, 11))
    plt.savefig('artifacts/elbow_method.png')
    plt.close()

    # Additional visualizations grouped by 'Left_Company'
    categorical_plots = {
        'Departments': 'Employees by Department and Exit Status',
        'Salary_Level': 'Employees by Salary Level and Exit Status',
        'Promotion_in_Last_5_Years': 'Employees by Promotion Status and Exit Status',
        'Number_of_Projects': 'Employees by Number of Projects and Exit Status',
        'Years_at_Company': 'Employees by Tenure and Exit Status'
    }

    print("\n📁 Generating grouped bar plots...")
    for column, title in categorical_plots.items():
        if column in df.columns:
            plt.figure(figsize=(12, 6))
            sns.countplot(data=df, x=column, hue='Left_Company', palette='Set2')
            plt.title(title)
            plt.xlabel(column)
            plt.ylabel('Number of Employees')
            plt.xticks(rotation=45)
            plt.tight_layout()
            filename = f'artifacts/grouped_by_{column}.png'
            plt.savefig(filename)
            plt.close()
        else:
            print(f"⚠️ Column '{column}' not found in DataFrame. Skipping plot.")

    print("\n✅ EDA completed and all plots saved in the 'artifacts' directory.")

if __name__ == "__main__":
    # Load dataset
    input_path = r'C:\Users\IfeomaAugustaAdigwe\Desktop\Employee Churn Prediction\data\hr_data.csv'
    output_path = r'C:\Users\IfeomaAugustaAdigwe\Desktop\Employee Churn Prediction\data\hr_data_cleaned.csv'
    df = pd.read_csv(input_path)

    # Clean column names before renaming
    df.columns = df.columns.str.strip()

    perform_eda(df, output_path)
