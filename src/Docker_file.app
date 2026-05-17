# Use a lightweight official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file from the root into the container
COPY requirements.txt .

# Install the software dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project structure into the container
COPY . .

# Expose port 8501 for the Streamlit interface
EXPOSE 8501

# Command to run Streamlit
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]