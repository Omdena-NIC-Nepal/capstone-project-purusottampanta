import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.express as px
import os

st.title("Temperature and Precipitation Prediction")

# Load data 
@st.cache_data
def load_data():
    try:
        return pd.read_csv("data/climate_data.csv")
    except FileNotFoundError:
        st.error("Error: 'climate_data.csv' not found in 'data/' directory. Run 'generate_sample_data.py' to create sample data.")
        return None

climate_data = load_data()

if climate_data is not None:
    # User input options
    st.subheader("Prediction Settings")
    variable = st.selectbox("Select variable to predict", ["Temperature", "Precipitation"])
    forecast_years = st.slider("Number of years to forecast", min_value=1, max_value=10, value=5)

    # Prepare data for Prophet based on selected variable
    if variable == "Temperature":
        prophet_data = climate_data[["year", "temperature"]].rename(columns={"year": "ds", "temperature": "y"})
        y_label = "Temperature (°C)"
    else:
        prophet_data = climate_data[["year", "precipitation"]].rename(columns={"year": "ds", "precipitation": "y"})
        y_label = "Precipitation (mm)"

    # Convert 'ds' to datetime
    prophet_data["ds"] = pd.to_datetime(prophet_data["ds"].astype(str) + "-01-01")

    # Train Prophet model
    try:
        model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        model.fit(prophet_data)
    except Exception as e:
        st.error(f"Error training model: {e}. Ensure data in 'climate_data.csv' is valid.")
        st.stop()

    # Create future dataframe with user-specified forecast horizon
    future = model.make_future_dataframe(periods=forecast_years, freq="YS")
    forecast = model.predict(future)

    # Display predictions
    st.subheader(f"Predicted {y_label} (Next {forecast_years} Years)")
    st.write(f"The table below lists the predicted {y_label} values for each year in the forecast period, along with the lower and upper bounds of the 95% confidence interval.")
    historical_max = climate_data["year"].max()
    future_predictions = forecast[forecast["ds"].dt.year > historical_max]
    st.dataframe(future_predictions[["ds", "yhat", "yhat_lower", "yhat_upper"]])

    # Create interactive Plotly chart
    st.subheader(f"{y_label} Forecast")
    fig = px.line(forecast, x="ds", y="yhat", title=f"{y_label} Forecast for Nepal")
    fig.add_scatter(x=forecast["ds"], y=forecast["yhat_lower"], mode="lines", name="Lower Bound", line=dict(color="gray"))
    fig.add_scatter(x=forecast["ds"], y=forecast["yhat_upper"], mode="lines", name="Upper Bound", line=dict(color="gray"))
    # Add historical data
    historical = prophet_data.rename(columns={"y": "actual"})
    fig.add_scatter(x=historical["ds"], y=historical["actual"], mode="lines", name="Historical", line=dict(color="blue"))
    # Update layout
    fig.update_layout(xaxis_title="Year", yaxis_title=y_label)
    st.plotly_chart(fig)

    # Additional explanation
    st.write(f"The above plot shows the historical {y_label} data up to {climate_data['year'].max()} and the forecasted {y_label} for the next {forecast_years} years using the Prophet model.")