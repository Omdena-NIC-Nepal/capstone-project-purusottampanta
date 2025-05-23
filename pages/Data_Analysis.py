import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os

st.title("Data Analysis")

# Load data 
@st.cache_data
def load_data():
    try:
        climate_data = pd.read_csv("data/climate_data.csv")
        crop_data = pd.read_csv("data/crop_yield_data.csv")
        # print(climate_data)
        return climate_data, crop_data
    except FileNotFoundError as e:
        st.error(f"Error: {e}. Please ensure 'climate_data.csv' and 'crop_yield_data.csv' are in the 'data/' directory. Run 'generate_sample_data.py' to create sample data.")
        return None, None

climate_data, crop_data = load_data()

if climate_data is not None and crop_data is not None:
    # Merge data
    data = pd.merge(climate_data, crop_data, on="year", how="inner")
    
    # Sidebar filters
    st.sidebar.header("Filters")
    year_range = st.sidebar.slider("Select Year Range", int(data["year"].min()), int(data["year"].max()), (int(data["year"].min()), int(data["year"].max())))
    data_filtered = data[(data["year"] >= year_range[0]) & (data["year"] <= year_range[1])]
    
    # Display data previews
    st.subheader("Climate Data Preview")
    st.dataframe(climate_data.head())
    
    st.subheader("Crop Yield Data Preview")
    st.dataframe(crop_data.head())
    
    # Visualizations
    st.subheader("Temperature Trend")
    fig_temp = px.line(data_filtered, x="year", y="temperature", title="Annual Average Temperature in Nepal (°C)")
    st.plotly_chart(fig_temp, use_container_width=True)
    
    st.subheader("Precipitation Trend")
    fig_precip = px.line(data_filtered, x="year", y="precipitation", title="Annual Total Precipitation in Nepal (mm)")
    st.plotly_chart(fig_precip, use_container_width=True)
    
    st.subheader("Crop Yield Trend")
    fig_yield = px.line(data_filtered, x="year", y="total_cereal_production", title="Annual Cereal Production in Nepal (tons)")
    st.plotly_chart(fig_yield, use_container_width=True)
    
    # Correlation Analysis
    st.subheader("Correlation Analysis")
    correlation = data_filtered[["temperature", "precipitation", "total_cereal_production"]].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(correlation, annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Matrix")
    plt.savefig("correlation_matrix.png")
    st.image("correlation_matrix.png")