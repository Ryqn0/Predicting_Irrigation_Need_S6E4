'''
Taken from a notebook in Kaggle, adapted for our pipeline.
Build final features for modeling. This includes:
- Adding domain-inspired features based on key thresholds and interactions.
- Adding a simple formula-based feature that combines key indicators into a score.

'''

import itertools
import numpy as np
import pandas as pd

target = "Irrigation_Need"

target_map = {'Low':0, 'Medium':1, 'High':2}

cat_cols = ['Soil_Type', 'Crop_Type', 'Crop_Growth_Stage', 'Season',
       'Irrigation_Type', 'Water_Source', 'Mulching_Used', 'Region']

num_cols = ['Soil_pH', 'Soil_Moisture', 'Organic_Carbon',
       'Electrical_Conductivity', 'Temperature_C', 'Humidity', 'Rainfall_mm',
       'Sunlight_Hours', 'Wind_Speed_kmh', 'Field_Area_hectare',
       'Previous_Irrigation_mm']

top_cat_cols = ['Crop_Growth_Stage',  'Mulching_Used', 'Crop_Type']

top_num_cols = ['Soil_Moisture', 'Temperature_C', 'Wind_Speed_kmh', 'Rainfall_mm']

top_cols = ['Soil_Moisture', 'Crop_Growth_Stage', 'Temperature_C', 
            'Mulching_Used', 'Wind_Speed_kmh', 'Rainfall_mm']

round_config = {        
     'Soil_Moisture'            : [0,-1],
     'Temperature_C'            : [-1],
     'Rainfall_mm'              : [0,-1,-2,-3],
     'Wind_Speed_kmh'           : [0,-1]
}

digit_config = {        
     'Soil_Moisture'            : [-1,0,1,2],
     'Temperature_C'            : [-1,0,1,2],
     'Rainfall_mm'              : [-3,-2,-1,0,1,2],
     'Wind_Speed_kmh'           : [-1,0,1,2]
}

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

def ngram_features(df, degrees=(2, 3)):
    """Categorical n-gram interactions for each degree in `degrees`."""
    bigrams = []
    trigrams = []
    for n in degrees:
        for cols in itertools.combinations(top_cat_cols, n):
            new_col = "_x_".join(cols)
            df[new_col] = df[list(cols)].astype(str).agg("_".join, axis=1)
            if n == 2:
                bigrams.append(new_col)
            elif n == 3:
                trigrams.append(new_col)
    
    ngrams = bigrams + trigrams
    df[ngrams] = df[ngrams].astype("category")

    return df

def binning_features(df):
    """Binning numerical features based on domain-inspired thresholds."""
    
    for col in top_num_cols:
        if col == "Soil_Moisture":
            bins = [0, 25, 50, 75, 100]
            labels = ['Very Low', 'Low', 'Moderate', 'High']
        elif col == "Temperature_C":
            bins = [0, 15, 25, 30, 50]
            labels = ['Cold', 'Cool', 'Warm', 'Hot']
        elif col == "Wind_Speed_kmh":
            bins = [0, 5, 10, 20, 100]
            labels = ['Calm', 'Breezy', 'Windy', 'Stormy']
        elif col == "Rainfall_mm":
            bins = [0, 100, 300, 500, np.inf]
            labels = ['Dry', 'Light Rain', 'Moderate Rain', 'Heavy Rain']
        
        new_col = f"{col}_binned"
        df[new_col] = pd.cut(df[col], bins=bins, labels=labels)
    
    return df

def numeric_features(df):
    """Interactions between top numerical and categorical features."""
    for col, value in round_config.items():
        for r in value:
            new_col = f"{col}_round_{r}"
            df[new_col] = df[col].round(r)
    
    for col, value in digit_config.items():
        for d in value:
            new_col = f"{col}_digit_{d}"
            df[new_col] = ((df[col] * 10**d) % 10).astype(int)

    for col in top_num_cols:
        new_col = f"{col}_decimal"
        df[new_col] = (df[col] % 1).round(2)

    return df

def pairwise_interactions(df):
    """Pairwise interactions between top numerical features."""
    for col1, col2 in itertools.combinations(top_num_cols, 2):
        df[f"{col1}_x_{col2}"] = df[col1] * df[col2]
        df[f"{col1}_div_{col2}"] = df[col1] / (df[col2] + 1e-5)
    return df

def build_final_features(df):
    print("Building final features...")
    print("Adding threshold distance features...")
    df = add_threshold_distances(df)
    print("Threshold distance features added.")
    print("Adding formula-based features...")
    df = add_formula_features(df)
    print("Formula-based features added.")
    print("Adding n-gram features...")
    df = ngram_features(df)
    print("N-gram features added.")
    print("Adding binning features...")
    df = binning_features(df)
    print("Binning features added.")
    print("Adding numeric interaction features...")
    df = numeric_features(df)
    print("Numeric interaction features added.")
    print("Adding pairwise interactions...")
    df = pairwise_interactions(df)
    print("Pairwise interactions added.")
    print("Final feature building complete.")
    
    return df