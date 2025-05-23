# 🌏 Climate Change Impact Assessment and Prediction System for Nepal

## Access the live deployed App using link https://capstone-project-purusottampanta.streamlit.app/

This Streamlit-based application evaluates the impact of climate change on agriculture and flood risks in Nepal using real datasets from the Copernicus Climate Data Store and Open Data Nepal. It combines data analysis, machine learning, and GIS tools to present interactive visualizations and predictions.

## 🔍 Features

- **📊 Data Analysis**  
  Analyze historical temperature, precipitation, and crop yield trends across Nepal.

- **🌊 Flood Risk Assessment**  
  Visualize district-level flood risk using historical flood data and criticality factors.

- **🗺️ GIS Visualization**  
  Interactive maps displaying crop yield and climate data per district.

- **📈 Prediction (Prophet)**  
  Forecast future temperature trends using Facebook Prophet.

- **🤖 SVM Classification**  
  Predict extreme weather events using Support Vector Machine classification.

- **📰 Text Analysis**  
  Extract insights from flood-related news articles using spaCy NLP.

> Use the **sidebar** to navigate through the sections of the application.

---

## 🧑‍💻 Author

**Purusottam Panta**  
MSc. Computer Science  
Kathmandu, Nepal

---

## 🚀 Getting Started

### Clone the Repository

```bash
git clone https://github.com/Omdena-NIC-Nepal/capstone-project-purusottampanta.git
cd climate-change-nepal
```

### ✅ Install Dependencies
Ensure you have Python 3.8+ installed. Then install all required libraries using pip:

```bash
pip install -r requirements.txt
```

### ▶️ Run the App
Start the application using Streamlit:

```bash
streamlit run app.py
```

### Project Structure

The project have follwoing folder structures
```bash
├── app.py
├── requirements.txt
├── data/
│   ├── climate_data.csv
│   ├── crop_yield_data.csv
│   ├── predict_crop_yields.csv
│   ├── nepal-districts.geojson
│   ├── nepal_districts.prj
│   ├── nepal_districts.sbn
│   ├── nepal_districts.sbx
│   ├── nepal_districts.shp
│   ├── nepal_districts.shx
├── pages/
│   ├── Data Analysis.py
│   ├── Flood Risk.py
│   ├── GIS Visualization.py
│   ├── Prediction.py
│   ├── SVM Classification.py
│   ├── Text Analysis.py
├── utils/
│   └── generate_sample_data.py

```
### 📜 License
This project is open-source. Feel free to fork and modify as needed. Please include attribution to the author.

### 🙌 Acknowledgements
- **Desmond sir and Joseph sir**
- **Omdena and NIC Nepal Family**
- **Copernicus Climate Data Store**
- **Open Data Nepal**
- **Streamlit**
- **Facebook Prophet**
- **spaCy**