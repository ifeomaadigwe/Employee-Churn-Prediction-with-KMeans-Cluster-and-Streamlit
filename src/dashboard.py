# dashboard.py

from taipy.gui import Gui
import app

# Load HTML-rendered charts from visuals.py
fig_churn_html = app.fig_churn_html
fig_dept_html = app.fig_dept_html
fig_tenure_html = app.fig_tenure_html
fig_edu_html = app.fig_edu_html
fig_role_html = app.fig_role_html
fig_location_html = app.fig_location_html
fig_perf_html = app.fig_perf_html
fig_satisfaction_html = app.fig_satisfaction_html
fig_overtime_html = app.fig_overtime_html
fig_distance_html = app.fig_distance_html

# Define dashboard layout
page = """
# 🎯 Dashboard is loading

Welcome to the **Employee Churn Analysis Dashboard** powered by Taipy and Plotly.  
Explore insights across departments, satisfaction levels, overtime, and more.

<|layout|columns=1|gap=30px|

<|{fig_churn_html}|html|>
<|{fig_dept_html}|html|>
<|{fig_tenure_html}|html|>
<|{fig_edu_html}|html|>
<|{fig_role_html}|html|>
<|{fig_location_html}|html|>
<|{fig_perf_html}|html|>
<|{fig_satisfaction_html}|html|>
<|{fig_overtime_html}|html|>
<|{fig_distance_html}|html|>

|>
"""

# Launch the Taipy application
Gui(page).run(port=5050)
