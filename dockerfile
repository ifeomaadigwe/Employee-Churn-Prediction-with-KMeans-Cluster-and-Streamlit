# ================================
# 📦 Dockerfile - Employee Churn Prediction
# ================================

# ✅ Use lightweight Python image
FROM python:3.11-slim

# ✅ Environment configuration
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ✅ Set working directory
WORKDIR /app

# ✅ Install OS-level dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ✅ Copy all project files into container
COPY . /app

# ✅ Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# ✅ Expose Streamlit's default port
EXPOSE 8501

# ✅ Start the Streamlit app
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.enableCORS=false"]


