import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Generate sample climate data
def generate_climate_data():
    years = range(1980, 2021)
    np.random.seed(42)
    data = pd.DataFrame({
        "year": years,
        "temperature": np.random.normal(loc=20, scale=2, size=len(years)),
        "precipitation": np.random.normal(loc=1200, scale=200, size=len(years))
    })
    data.to_csv("data/climate_data.csv", index=False)
    return data

# Generate sample crop yield data
def generate_crop_yield_data(climate_data):
    districts = ["TAPLEJUNG","PANCHTHAR","ILAM","JHAPA","MORANG","SUNSARI","DHANKUTA","TERHATHUM","SANKHUWASABHA","BHOJPUR","SOLUKHUMBU","OKHALDHUNGA","KHOTANG","UDAYAPUR","SAPTARI","SIRAHA","DHANUSHA","MAHOTTARI","SARLAHI","SINDHULI","RAMECHHAP","DOLAKHA","SINDHUPALCHOK","KABHREPALANCHOK","LALITPUR","BHAKTAPUR","KATHMANDU","NUWAKOT","RASUWA","DHADING","MAKAWANPUR","RAUTAHAT","BARA","PARSA","CHITAWAN","GORKHA","LAMJUNG","TANAHU","SYANGJA","KASKI","MANANG","MUSTANG","MYAGDI","PARBAT","BAGLUNG","GULMI","PALPA","NAWALPARASI_W","RUPANDEHI","KAPILBASTU","ARGHAKHANCHI","PYUTHAN","ROLPA","RUKUM_W","SALYAN","DANG","BANKE","BARDIYA","SURKHET","DAILEKH","JAJARKOT","DOLPA","JUMLA","KALIKOT","MUGU","HUMLA","BAJURA","BAJHANG","ACHHAM","DOTI","KAILALI","KANCHANPUR","DADELDHURA","BAITADI","DARCHULA","NAWALPARASI_E","RUKUM_E"]
    years = range(1980, 2021)
    data = []
    for year in years:
        for district in districts:
            data.append({
                "year": year,
                "district": district,
                "total_cereal_production": np.random.normal(loc=5000, scale=1000)
            })
    df = pd.DataFrame(data)
    df.to_csv("data/crop_yield_data.csv", index=False)
    return df

# Generate sample predicted crop yields
def generate_predicted_yields(climate_data, crop_data):
    # Train a simple linear regression model
    merged_data = pd.merge(climate_data, crop_data, on="year")
    X = merged_data[["temperature", "precipitation"]]
    y = merged_data["total_cereal_production"]
    model = LinearRegression()
    model.fit(X, y)

    # Generate future climate data for predictions
    future_years = [2030, 2050, 2070, 2100]
    districts = crop_data["district"].unique()
    scenarios = ["RCP4.5", "RCP8.5"]
    predictions = []
    np.random.seed(42)
    for year in future_years:
        for district in districts:
            for scenario in scenarios:
                # Simulate future climate data (adjust based on scenario)
                temp_adjust = 1.0 if scenario == "RCP4.5" else 2.0
                precip_adjust = 0.9 if scenario == "RCP4.5" else 0.8
                temp = np.random.normal(loc=20 + temp_adjust, scale=2)
                precip = np.random.normal(loc=1200 * precip_adjust, scale=200)
                # Create a DataFrame for prediction with correct column names
                X_new = pd.DataFrame([[temp, precip]], columns=["temperature", "precipitation"])
                predicted_yield = model.predict(X_new)[0]
                predictions.append({
                    "district": district,
                    "year": year,
                    "scenario": scenario,
                    "total_cereal_production": max(predicted_yield, 0)  # Ensure non-negative
                })
    predicted_df = pd.DataFrame(predictions)
    predicted_df.to_csv("data/predicted_crop_yields.csv", index=False)
    return predicted_df

if __name__ == "__main__":
    # Create data directory if it doesn't exist
    import os
    os.makedirs("data", exist_ok=True)
    climate_data = generate_climate_data()
    crop_data = generate_crop_yield_data(climate_data)
    generate_predicted_yields(climate_data, crop_data)