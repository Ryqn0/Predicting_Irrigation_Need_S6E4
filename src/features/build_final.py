






def add_threshold_distances(df):
    """Signed distances from key domain thresholds."""
    df["soil_lt_25"]     = (df["Soil_Moisture"] < 25).astype(int)
    df["wind_gt_10"]     = (df["Wind_Speed_kmh"] > 10).astype(int)
    df["temp_gt_30"]     = (df["Temperature_C"] > 30).astype(int)
    df["rain_lt_300"]    = (df["Rainfall_mm"] < 300).astype(int)

    # domain
    df["moist_rain"]     = df["Soil_Moisture"] / (df["Rainfall_mm"] + 1)
    df["moist_temp"]     = df["Soil_Moisture"] / (df["Temperature_C"] + 1)
    df["moist_wind"]    = df["Soil_Moisture"] / (df["Wind_Speed_kmh"] + 1)
    df["ET_proxy"]       = (df["Temperature_C"] * df["Wind_Speed_kmh"] * df["Sunlight_Hours"]) / (df["Humidity"] + 1)
    df["heat_stress"]    = df["Temperature_C"] * df["Sunlight_Hours"]
    df["dfrying_force"]  = df["Wind_Speed_kmh"] * df["Temperature_C"] / (df["Humidity"] + 1)
    df["water_supply"]   = df["Rainfall_mm"] + df["Previous_Irrigation_mm"]
    df["water_dfeficit"] = df["Soil_Moisture"] - df["water_supply"] * 0.1
    df["soil_quality"]   = df["Organic_Carbon"] / (df["Electrical_Conductivity"] + 0.1)
    df["moist_x_temp"]   = df["Soil_Moisture"] * df["Temperature_C"]
    df["windf_x_temp"]   = df["Wind_Speed_kmh"] * df["Temperature_C"]
    
    return df

def add_formula_features(df):
    df['high_score'] = (
        (df['Soil_Moisture'] < 25) * 2 + 
        (df['Rainfall_mm'] < 300) * 2 + 
        (df['Temperature_C'] > 30) * 1 + 
        (df['Wind_Speed_kmh'] > 10) * 1
    )
    
    df['low_score'] = (
        (df['Crop_Growth_Stage'].isin(['Harvest', 'Sowing'])) * 2 + 
        (df['Mulching_Used'] == 'Yes') * 1
    )
    
    df['formula_score'] = df['high_score'] - df['low_score']

    df['formula_pred'] = 0 # Low
    df.loc[(df['formula_score'] > 0) & (df['formula_score'] <= 3), 'formula_pred'] = 1 # Medium
    df.loc[df['formula_score'] > 3, 'formula_pred'] = 2 # High
    return df