import streamlit as st
import requests

st.set_page_config(
    page_title="Namma Yatri Dynamic Pricing",
    page_icon="🛺",
    layout="centered"
)

st.title("🛺 Namma Yatri Dynamic Pricing Engine")
st.markdown("""
This dashboard predicts the average neighborhood ride-hailing fare based on real-time macro data. 
It queries a live **FastAPI** backend serving a **Gradient Boosting Regressor** trained on Bangalore mobility data.
""")

st.divider()

st.subheader("Simulate Ride Dynamics")
col1, col2 = st.columns(2)

with col1:
    distance_km = st.slider("Average Distance (km)", min_value=5.0, max_value=10.0, value=7.5, step=0.10)
    cancellation_rate = st.slider("Cancellation Rate (%)", min_value=0.0, max_value=100.0, value=35.0, step=1.0)

with col2:
    completed_trips = st.number_input("Completed Trips in Area", min_value=10000, max_value=2000000, value=450000, step=10000)

st.write("")
if st.button("Predict Average Fare", type="primary", use_container_width=True):
    
    payload = {
        "distance_km": float(distance_km),
        "completed_trips": float(completed_trips),
        "cancellation_rate": float(cancellation_rate / 100)  # API expects a decimal, not a percentage
    }
    
    API_URL = "http://api:8000/predict"
    
    try:
        with st.spinner("Querying Model API..."):
            response = requests.post(API_URL, json=payload)
            
        if response.status_code == 200:
            result = response.json()
            predicted_fare = result["predicted_fare_inr"]
            
            st.success("Prediction Successful!")
            st.metric(label="Predicted Neighborhood Fare", value=f"₹ {predicted_fare}")
            st.caption(f"Powered by: {result['model_version']}")
            
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the API. Check that the FastAPI server is running on port 8000.")
