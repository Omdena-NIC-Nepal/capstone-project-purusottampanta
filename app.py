import streamlit as st

st.set_page_config(page_title="Climate Change Impact Assessment and Prediction System for Nepal", layout="wide")

st.title("Climate Change Impact Assessment and Prediction System for Nepal")
st.markdown("""
This application assesses climate change impacts on crop yield in Nepal and Flood risk analysis in Nepal using real data from the Copernicus Climate Data Store and Open Data Nepal. It offers interactive visualizations, temperature predictions, and GIS mapping.

**Features**:
- **Data Analysis**: Explore trends in temperature, precipitation, and crop yields.
- **Flood Risk**: Flood risk analysis by districts and criticality.
- **GIS Visualization**: View crop yields by district on an interactive map.
- **Prediction**: Forecast future temperatures using Prophet.
- **SVM Classification**: Extreme event prediction using SVM classification.
- **Text Analysis**: Analyze text for flood realted news using Spacy.

Use the sidebar to navigate through the sections.
""")