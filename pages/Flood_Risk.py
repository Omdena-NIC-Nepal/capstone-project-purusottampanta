import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import json
from urllib.request import urlopen
from sklearn.ensemble import RandomForestClassifier

# Load data
@st.cache_data
def load_data():
    with urlopen("https://raw.githubusercontent.com/mesaugat/geoJSON-Nepal/master/nepal-districts.geojson") as response:
        geojson = json.load(response)
    
    districts = [feature['properties']['DISTRICT'] for feature in geojson['features']]
    
    # Generate synthetic flood data
    flood_data = pd.DataFrame({
        "DISTRICT": districts,
        "FLOOD_RISK": np.random.randint(1, 5, size=len(districts)),
        "ELEVATION": np.random.uniform(100, 3000, len(districts)),
        "RIVER_DENSITY": np.random.uniform(0.1, 2.5, len(districts)),
        "RAINFALL": np.random.uniform(800, 2500, len(districts)),
        "HISTORICAL_EVENTS": np.random.randint(0, 10, len(districts))
    })
    
    return geojson, flood_data

# Prediction model
@st.cache_resource
def load_model():
    model = RandomForestClassifier()
    X = np.random.rand(100, 3)
    y = np.random.randint(0, 2, 100)
    model.fit(X, y)
    return model

def app():
    st.title("🌊 Flood Risk Analysis and Prediction")
    
    # Load data
    geojson, flood_data = load_data()
    model = load_model()
    
    # Sidebar Filters
    with st.sidebar:
        st.header("Filters")
        
        # District filter 
        districts = ['All'] + sorted(flood_data['DISTRICT'].unique().tolist())
        selected_district = st.selectbox(
            "Select District",
            options=districts,
            index=0 
        )
        
        # Risk level filter 
        risk_levels = st.multiselect(
            "Risk Levels",
            options=[1, 2, 3, 4],
            default=[1, 2, 3, 4],
            format_func=lambda x: f"Level {x}"
        )
        
        # Historical events filter 
        min_events, max_events = st.slider(
            "Historical Flood Events Range",
            min_value=int(flood_data['HISTORICAL_EVENTS'].min()),
            max_value=int(flood_data['HISTORICAL_EVENTS'].max()),
            value=(int(flood_data['HISTORICAL_EVENTS'].min()), 
                   int(flood_data['HISTORICAL_EVENTS'].max()))
        )
    
    # Apply filters
    filtered_data = flood_data.copy()
    
    # District filter
    if selected_district != 'All':
        filtered_data = filtered_data[filtered_data['DISTRICT'] == selected_district]
    
    # Other filters
    filtered_data = filtered_data[
        (filtered_data['FLOOD_RISK'].isin(risk_levels)) &
        (filtered_data['HISTORICAL_EVENTS'].between(min_events, max_events))
    ]
    
    # Main display
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Interactive map
        st.subheader("District Flood Risk Map")
        fig = px.choropleth(
            filtered_data,
            geojson=geojson,
            locations='DISTRICT',
            featureidkey="properties.DISTRICT",
            color='FLOOD_RISK',
            color_continuous_scale="YlOrRd",
            range_color=(1, 4),
            hover_data=['ELEVATION', 'RIVER_DENSITY', 'HISTORICAL_EVENTS']
        )
        fig.update_geos(fitbounds="locations", visible=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Current district info
        st.subheader("Selected District Info")
        if selected_district == 'All':
            st.write("Select a district to view details")
        else:
            district_info = flood_data[flood_data['DISTRICT'] == selected_district].iloc[0]
            st.metric("Flood Risk Level", f"Level {district_info['FLOOD_RISK']}")
            st.metric("Elevation (m)", f"{district_info['ELEVATION']:.0f}")
            st.metric("River Density", f"{district_info['RIVER_DENSITY']:.2f} km/km²")
            st.metric("Historical Events", district_info['HISTORICAL_EVENTS'])
    
    # Prediction Section
    st.header("Flood Risk Prediction")
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=1200.0)
        with col2:
            river_level = st.number_input("River Level (m)", min_value=0.0, value=3.5)
        with col3:
            elevation = st.number_input("Elevation (m)", min_value=0.0, value=1500.0)
        
        if st.form_submit_button("Predict Flood Risk"):
            prediction = model.predict([[rainfall, river_level, elevation]])
            probability = model.predict_proba([[rainfall, river_level, elevation]])
            
            if prediction[0] == 1:
                st.error(f"High Flood Risk ({probability[0][1]*100:.1f}% probability)")
            else:
                st.success(f"Low Flood Risk ({probability[0][0]*100:.1f}% probability)")
    
    # Data table
    st.subheader("Filtered District Data")
    st.dataframe(
        filtered_data,
        column_config={
            "FLOOD_RISK": st.column_config.NumberColumn(
                "Risk Level",
                help="1 (Low) - 4 (High)",
                format="%d ⚠️"
            )
        },
        hide_index=True,
        use_container_width=True
    )

if __name__ == "__main__":
    app()