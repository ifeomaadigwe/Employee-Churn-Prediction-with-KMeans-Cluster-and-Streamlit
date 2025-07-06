import streamlit as st
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import plotly.express as px

# --- Page Configuration ---
st.set_page_config(page_title="Employee Churn Predictor", layout="wide")
st.title("💼 Employee Churn Prediction App")
st.markdown("Input employee attributes to understand their churn cluster and suggested action plan.")

# --- Define file paths ---
CURRENT_DIR = Path(__file__).parent
MODEL_PATH = CURRENT_DIR.parent / "artifacts" / "kmeans_model.pkl"
PREPROCESSOR_PATH = CURRENT_DIR.parent / "artifacts" / "preprocessor.pkl"

# --- Load model and preprocessor ---
@st.cache_resource(show_spinner=False)
def load_artifacts():
    try:
        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        return model, preprocessor
    except Exception as e:
        st.error(f"❌ Error loading model or preprocessor: {e}")
        return None, None

model, preprocessor = load_artifacts()
if model is None or preprocessor is None:
    st.stop()

# --- Sidebar guidance ---
st.sidebar.title("📝 Input Tips")
st.sidebar.markdown("""
- Adjust the sliders and dropdowns to describe an employee's profile  
- The model will classify them into a behavioral cluster  
- You'll receive a recommendation based on the assigned cluster
""")

# --- Input form ---
st.subheader("📋 Enter Employee Profile")
col1, col2 = st.columns(2)

with col1:
    satisfaction = st.slider("Satisfaction Level", 0.0, 1.0, 0.6)
    evaluation = st.slider("Last Evaluation Score", 0.0, 1.0, 0.7)
    projects = st.slider("Number of Projects", 1, 10, 4)
    hours = st.slider("Average Monthly Hours", 90, 310, 150)
    tenure = st.slider("Years at Company", 1, 10, 3)

with col2:
    accident = st.selectbox("Work Accident", [0, 1])
    promotion = st.selectbox("Promotion in Last 5 Years", [0, 1])
    department = st.selectbox("Departments", [
        'Sales', 'Technical', 'Support', 'IT', 'HR',
        'Accounting', 'Management', 'Marketing', 'Product Mng'
    ])
    salary = st.selectbox("Salary_Level", ['low', 'medium', 'high'])

# --- Input formatting ---
input_df = pd.DataFrame([[
    satisfaction,
    evaluation,
    projects,
    hours,
    tenure,
    accident,
    promotion,
    department,
    salary
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

# --- Cluster descriptions ---
cluster_descriptions = {
    0: "High performers: happy and productive — Action: Retain, reward, and develop for leadership",
    1: "Underperforming but stable: needs closer inspection — Action: Provide support, mentorship, or role realignment",
    2: "Very disengaged and unproductive: at-risk employees likely to churn — Action: Investigate dissatisfaction, consider exit interviews or improvement plans",
    3: "Mixed performance: possibly inconsistent — Action: Provide feedback, training, or reassess fit",
    4: "Generally positive but not exceptional: solid contributors, possibly steady team players — Action: Engage and motivate to reach full potential",
    5: "Highly dissatisfied and underperforming: critical churn risk, may be overworked or misaligned — Action: Immediate intervention required — identify root causes"

}

# --- Run prediction ---
try:
    transformed = preprocessor.transform(input_df)
    cluster = model.predict(transformed)[0]
except Exception as e:
    st.error(f"Prediction error: {e}")
    st.stop()

# --- Display result ---
if st.button("🔍 Predict Cluster"):
    st.markdown("---")
    message = cluster_descriptions.get(cluster, f"Cluster #{cluster}")
    st.success(f"🧠 This employee belongs to: **{message}**")

    fig = px.scatter(
        x=[satisfaction],
        y=[projects],
        color=[salary],
        size=[hours],
        labels={'x': 'Satisfaction Level', 'y': 'Number of Projects'},
        title="Employee Position: Satisfaction vs. Projects",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    st.plotly_chart(fig)

st.markdown("---")
st.caption("Developed by Ifeoma Adigwe • Powered by Streamlit & scikit-learn")
