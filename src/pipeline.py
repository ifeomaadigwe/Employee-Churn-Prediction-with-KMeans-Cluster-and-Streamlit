from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd

def create_preprocessing_pipeline():
    """Create a preprocessing pipeline for the dataset."""
    
    numeric_features = [
        'Satisfaction_Level',
        'Last_Evaluation_Score',
        'Number_of_Projects',
        'Average_Monthly_Hours',
        'Years_at_Company',
        'Work_Accident',
        'Promotion_in_Last_5_Years'
    ]
    
    categorical_features = ['Departments', 'Salary_Level']

    # Define transformers
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')


    # Combine into a column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    return preprocessor, numeric_features + categorical_features

def preprocess_data(df, preprocessor, features):
    """Preprocess the data using the provided preprocessor and features."""
    X = df[features]
    X_transformed = preprocessor.fit_transform(X)
    return X_transformed, preprocessor, features

print("\n✅ Pipeline completed.")
# This module defines the preprocessing pipeline for the employee churn dataset.
# It includes scaling numeric features and one-hot encoding categorical features.