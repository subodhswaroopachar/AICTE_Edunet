import pandas as pd
import numpy as np
import joblib
import streamlit as st

# Load the trained model and feature structure
model = joblib.load("pollution_model.pkl")
model_cols = joblib.load("model_columns.pkl")

# Pollutants used in training
pollutants = ['O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL']

# Acceptable limits (mg/L)
limits = {
    'O2': 5.0,      # Minimum acceptable
    'NO3': 10.0,
    'NO2': 0.1,
    'SO4': 250.0,
    'PO4': 0.1,
    'CL': 250.0
}

# Streamlit page settings
st.set_page_config(page_title="Water Pollutants Predictor", layout="centered")
st.title("💧 Water Pollutants Predictor & Drinking Water Safety Check")
st.write("Predict pollutant levels for a monitoring station and assess water quality for drinking based on WHO/EPA standards.")

# User inputs
year_input = st.number_input("Enter Year", min_value=2000, max_value=2100, value=2024)
station_id = st.number_input("Enter Station ID", min_value=1, max_value=22, value=1)

if st.button("Predict"):
    # Prepare input
    input_df = pd.DataFrame({'year': [year_input], 'id': [station_id]})
    input_encoded = pd.get_dummies(input_df, columns=['id'])

    # Align columns with training set
    for col in model_cols:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[model_cols]

    try:
        # Predict
        predicted = model.predict(input_encoded)[0]

        # Prepare results table
        results = []
        for p, val in zip(pollutants, predicted):
            val_rounded = round(val, 4)
            limit = limits[p]
            if p == "O2":
                status = "OK" if val_rounded >= limit else "Low O₂ – Not Safe"
                condition = f">= {limit}"
            else:
                status = "OK" if val_rounded <= limit else f"High {p} – Not Safe"
                condition = f"<= {limit}"
            results.append({
                "Pollutant": p,
                "Predicted (mg/L)": val_rounded,
                "Acceptable Limit (mg/L)": condition,
                "Status": "🟢 OK" if status == "OK" else "🔴 " + status
            })

        # Display predictions and safety results
        st.subheader(f"📊 Predicted Pollutant Levels and Safety Check for Station {station_id} in {year_input}:")
        st.dataframe(pd.DataFrame(results))

        # Final water safety verdict
        if all("OK" in row["Status"] for row in results):
            st.success("✅ Water is likely **safe for drinking** based on predicted pollutant levels.")
        else:
            st.error("🚫 Water is **not safe for drinking** due to one or more critical pollutants.")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# Informational expander
with st.expander("ℹ️ What do these parameters mean?"):
    st.markdown("""
| Pollutant | Description | Environmental Significance |
|-----------|-------------|-----------------------------|
| **O₂ (Dissolved Oxygen)** | Oxygen in water | <5 mg/L stresses/kills fish |
| **NO₃ (Nitrate)** | From fertilizers/sewage | Causes algae growth, health risk |
| **NO₂ (Nitrite)** | Nitrogen cycle intermediate | Toxic even at low levels |
| **SO₄ (Sulfate)** | From mining/detergents | Taste issue, corrosion |
| **PO₄ (Phosphate)** | From sewage/fertilizer | Eutrophication, fish kills |
| **Cl (Chloride)** | From salt/sewage | Affects taste, harms freshwater life |
""")
