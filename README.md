# 🛺 Namma Yatri Dynamic Pricing Engine & MLOps Pipeline

An end-to-end MLOps system that predicts average neighborhood ride-hailing fares using open-source mobility data from Bengaluru. The project transitions a trained Scikit-learn model into a live production environment.

## 🛠️ Tech Stack
* **Core:** Python, Pandas, Scikit-learn
* **MLOps & Tracking:** MLflow, SQLite
* **Deployment:** FastAPI, Streamlit

## 📈 System Architecture
1. **Experimentation:** Hyperparameters tracked using **MLflow** to identify the most optimized architecture, successfully dropping computational overhead by 66% while achieving an **R² score of 0.9699** and an **RMSE of ₹2.79**.
2. **Production API:** A robust **FastAPI** backend serves the champion model with sub-50ms inference latency, using **Pydantic** for input validation schemas.
3. **User Dashboard:** A modern **Streamlit** UI enables real-time fare parameter simulation.

## 🚀 How to Run Locally

1. **Clone the repo:**
   ```bash
   git clone [https://github.com/scariadon5/Namma-Yatri-Pricing-MLOPS-
   cd Uber-Pricing-MLOPS](https://github.com/scariadon5/Namma-Yatri-Pricing-MLOPS-)


2. Setup the Virtual Environment & Dependencies
   
# Activate your virtual environment (Windows Bash command)
```bash
source .venv/Scripts/activate
# Install the exact software dependencies
pip install -r requirements.txt


3. **Spin Up the FastAPI Inference Server**
```bash
uvicorn src.api:app --reload

Once running, you can explore the interactive API schemas at: http://127.0.0.1:8000/docs

4. **Launch the Interactive User Dashboard**
Open a new terminal window, ensure the virtual environment is active, and run:
```bash
streamlit run src/app.py

The dashboard will automatically launch in your browser at http://localhost:8501. Adjust the sliders to pass live payload variables directly to your FastAPI backend!

Once this file is saved in your root directory, you can run your `git add README.md`, commit it, and push it live to update your GitHub homepage profile!