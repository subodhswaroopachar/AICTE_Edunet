import pandas as pd
import joblib

# Load model and column structure
model = joblib.load("pollution_model.pkl")
model_cols = joblib.load("model_columns.pkl")

# Pollutants and limits
pollutants = ['O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL']
limits = {
    'O2': 5.0,      # Minimum acceptable
    'NO3': 10.0,
    'NO2': 0.1,
    'SO4': 250.0,
    'PO4': 0.1,
    'CL': 250.0
}

safe_records = []

for year in range(2000, 2024):  # Adjust years as needed
    for station_id in range(1, 23):  # Assuming 22 stations
        input_df = pd.DataFrame({'year': [year], 'id': [station_id]})
        input_encoded = pd.get_dummies(input_df, columns=['id'])

        for col in model_cols:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[model_cols]

        predicted = model.predict(input_encoded)[0]

        # Round and check limits
        status_ok = True
        for p, val in zip(pollutants, predicted):
            val = round(val, 4)
            if p == "O2":
                if val < limits[p]:
                    status_ok = False
                    break
            else:
                if val > limits[p]:
                    status_ok = False
                    break

        if status_ok:
            safe_records.append({'year': year, 'station_id': station_id})

# Show results
safe_df = pd.DataFrame(safe_records)
print("✅ Stations and Years with Safe Water:")
print(safe_df)

# Optional: save to CSV
safe_df.to_csv("safe_water_stations.csv", index=False)
