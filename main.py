from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
from pathlib import Path
from typing import List, Literal
import pandas as pd

# Initialize FastAPI app
app = FastAPI(
    title="Employee Churn Predictor API",
    description="API for predicting employee behavioral clusters using an unsupervised KMeans model.",
    version="1.0.0"
)

# Define input data model
class EmployeeInput(BaseModel):
    satisfaction_level: float
    evaluation_score: float
    number_of_projects: int
    average_monthly_hours: int
    years_at_company: int
    work_accident: Literal[0, 1]
    promotion_last_5_years: Literal[0, 1]
    department: str
    salary: Literal["low", "medium", "high"]

# Define response data model
class ClusterResponse(BaseModel):
    cluster: int
    description: str

# File paths
CURRENT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = CURRENT_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "kmeans_model.pkl"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.pkl"

# Load model and preprocessor
try:
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
except Exception as e:
    raise Exception(f"Error loading artifacts: {e}")

# Human-readable cluster descriptions
cluster_descriptions = {
    0: "High performers: happy and productive — Action: Retain, reward and develop for leadership",
    1: "Underperforming but stable: needs closer inspection — Action: Provide support, mentorship, or role realignment",
    2: "Very disengaged and unproductive: At-risk employees likely to churn — Action: Investigate dissatisfaction, consider exit interviews or improvement plans",
    3: "Mixed performance: possibly inconsistent — Action: Provide feedback, training, or reassess fit",
    4: "Generally positive but not exceptional: Solid contributors, possibly steady team players — Action: Engage and motivate to reach full potential",
    5: "Highly dissatisfied and underperforming: Critical churn risk, may be overworked or misaligned — Action: Immediate intervention required — identify root causes"
}

@app.get("/")
async def root():
    return {"message": "Welcome to the Employee Churn Prediction API. Access /docs to explore available endpoints."}

@app.post("/predict", response_model=ClusterResponse)
async def predict_cluster(employee: EmployeeInput):
    try:
        # Wrap input into a DataFrame to preserve column names
        input_df = pd.DataFrame([[
            employee.satisfaction_level,
            employee.evaluation_score,
            employee.number_of_projects,
            employee.average_monthly_hours,
            employee.years_at_company,
            employee.work_accident,
            employee.promotion_last_5_years,
            employee.department,
            employee.salary
        ]], columns=[
            'Satisfaction_Level',
            'Last_Evaluation_Score',
            'Number_of_Projects',
            'Average_Monthly_Hours',
            'Years_at_Company',
            'Work_Accident',
            'Promotion_in_Last_5_Years',
            'Departments',
            'Salary_Level'
        ])

        # Preprocess input
        transformed = preprocessor.transform(input_df)
        cluster = int(model.predict(transformed)[0])
        description = cluster_descriptions.get(cluster, "No description available.")

        return ClusterResponse(cluster=cluster, description=description)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

