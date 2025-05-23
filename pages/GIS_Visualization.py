import streamlit as st
import geopandas as gpd
import pandas as pd
import plotly.express as px
import os

st.title("GIS Visualization")

# Load data 
@st.cache_data
def load_data():
    try:
        gdf = gpd.read_file("data/nepal_districts.shp")
        crop_data = pd.read_csv("data/crop_yield_data.csv")
        predicted_yield = pd.read_csv("data/predicted_crop_yields.csv")

        crop_data["total_cereal_production"] = pd.to_numeric(crop_data["total_cereal_production"], errors="coerce").fillna(0)
        predicted_yield["total_cereal_production"] = pd.to_numeric(predicted_yield["total_cereal_production"], errors="coerce").fillna(0)
        return gdf, crop_data, predicted_yield
    except Exception as e:
        st.error(f"Error loading data: {e}. Ensure 'nepal_districts.shp', 'crop_yield_data.csv', and 'predicted_crop_yields.csv' are in 'data/'. Run 'generate_sample_data.py' for sample data.")
        return None, None, None

gdf, crop_data, predicted_yield = load_data()

if gdf is not None and crop_data is not None and predicted_yield is not None:
    # Sidebar filter for data type
    st.sidebar.header("Map Settings")
    data_type = st.sidebar.selectbox("Select Data Type", ["Historical Crop Yield", "Predicted Crop Yield"])
    
    if data_type == "Historical Crop Yield":
        # Filter historical data
        available_years = crop_data["year"].unique()
        selected_year = st.sidebar.selectbox("Select Year", available_years, index=len(available_years)-1)
        crop_data_filtered = crop_data[crop_data["year"] == selected_year]
        title = f"Historical Crop Yield by District ({selected_year})"
    else:
        # Filter predicted data
        scenario = st.sidebar.selectbox("Select Scenario", ["RCP4.5", "RCP8.5"])
        available_years = predicted_yield["year"].unique()
        selected_year = st.sidebar.selectbox("Select Year", available_years)
        crop_data_filtered = predicted_yield[(predicted_yield["scenario"] == scenario) & (predicted_yield["year"] == selected_year)]
        title = f"Predicted Crop Yield by District ({selected_year}, {scenario})"

    # Merge with shapefile
    try:
        district_columns = ["DISTRICT", "DISTRICT_NA", "NAME"]
        district_col = next((col for col in district_columns if col in gdf.columns), None)
        if district_col is None:
            st.error("No suitable district column found in shapefile. Expected one of: DISTRICT, DISTRICT_NA, NAME")
        else:
            gdf_merged = gdf.merge(crop_data_filtered, left_on=district_col, right_on="district", how="left")
            

            gdf_merged["total_cereal_production"] = gdf_merged["total_cereal_production"].fillna(0)
            
            # Create choropleth map
            st.subheader(title)
            fig = px.choropleth(gdf_merged,
                                geojson=gdf_merged.geometry.__geo_interface__,
                                locations=gdf_merged.index,
                                color="total_cereal_production",
                                color_continuous_scale="Viridis",
                                labels={"total_cereal_production": "Crop Production (tons)"},
                                title=title)
            fig.update_geos(fitbounds="locations", visible=False)
            fig.update_layout(margin={"r": 0, "t": 50, "l": 0, "b": 0})
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating map: {e}. Check if 'total_cereal_production' contains valid numeric data.")

    # Download option
    st.subheader("Download Crop Yield Data")
    if data_type == "Historical Crop Yield":
        st.download_button("Download Historical Crop Yield Data", crop_data.to_csv(index=False), "crop_yield_data.csv")
    else:
        st.download_button("Download Predicted Crop Yield Data", predicted_yield.to_csv(index=False), "predicted_crop_yields.csv")