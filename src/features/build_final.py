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

def encode_cat_cols(df, encodings=None):
    """Label-encode cat_cols to integers.

    Pass encodings=None to fit on train — returns (df, encodings).
    Pass the returned dict to transform test. Unseen values get -1.
    """
    fitted = {} if encodings is None else encodings
    for col in cat_cols:
        known = fitted.get(col)
        cat = pd.Categorical(df[col], categories=known)
        if encodings is None:
            fitted[col] = cat.categories
        df[col] = cat.codes
    return df, fitted

def ngram_features(df, degrees=(2, 3), categories=None):
    """Categorical n-gram interactions for each degree in `degrees`.

    Pass categories=None (default) to fit on train — returns (df, categories).
    Pass the returned categories dict when transforming test data.
    Unseen combinations in test get code -1.
    """
    fitted = {} if categories is None else categories
    for n in degrees:
        for cols in itertools.combinations(top_cat_cols, n):
            new_col = "_x_".join(cols)
            combined = df[list(cols)].astype(str).agg("_".join, axis=1)
            known = fitted.get(new_col)
            cat = pd.Categorical(combined, categories=known)
            if categories is None:
                fitted[new_col] = cat.categories
            df[new_col] = cat.codes

    return df, fitted

_bin_config = {
    'Soil_Moisture':  ([0, 25, 50, 75, 100],      ['Very Low', 'Low', 'Moderate', 'High']),
    'Temperature_C':  ([0, 15, 25, 30, 50],        ['Cold', 'Cool', 'Warm', 'Hot']),
    'Wind_Speed_kmh': ([0, 5, 10, 20, 100],        ['Calm', 'Breezy', 'Windy', 'Stormy']),
    'Rainfall_mm':    ([0, 100, 300, 500, np.inf], ['Dry', 'Light Rain', 'Moderate Rain', 'Heavy Rain']),
}

def binning_features(df):
    """Bins numerical features into integer codes. Bins are fixed so no fit step needed."""
    for col, (bins, labels) in _bin_config.items():
        df[f"{col}_binned"] = pd.cut(df[col], bins=bins, labels=labels).cat.codes
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

def one_hot_encode(df):
    """Label-encode any remaining object columns to integers."""
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df

def build_final_features(df, encodings=None):
    """Build all features. Returns (df, encodings).

    Train:  df_train, enc = build_final_features(df_train)
    Test:   df_test,  _   = build_final_features(df_test, encodings=enc)

    encodings holds both cat_col label maps and ngram category maps,
    so every string column is consistently integer-coded across splits.
    """
    cat_enc    = encodings.get("cat")    if encodings else None
    ngram_enc  = encodings.get("ngram")  if encodings else None

    print("Building final features...")
    print("Adding threshold distance features...")
    df = add_threshold_distances(df)
    print("Finished adding threshold distance features.")
    print("Adding formula-based features...")
    df = add_formula_features(df)
    print("Finished adding formula-based features.")
    print("Encoding categorical features...")
    df, cat_enc   = encode_cat_cols(df, encodings=cat_enc)
    print("Finished encoding categorical features.")
    print("Adding n-gram interaction features...")
    df, ngram_enc = ngram_features(df, categories=ngram_enc)
    print("Finished adding n-gram interaction features.")
    print("Adding binning features...")
    df = binning_features(df)
    print("Finished adding binning features.")
    print("Adding numeric interaction features...")
    df = numeric_features(df)
    print("Finished adding numeric interaction features.")
    print("Adding pairwise interaction features...")
    df = pairwise_interactions(df)
    print("Finished adding pairwise interaction features.")
    if target in df.columns:
        print("Mapping target variable to integers...")
        df[target] = df[target].map(target_map)
        print("Finished mapping target variable.")
    print("Applying one-hot encoding to remaining categorical features...")
    df = one_hot_encode(df)
    print("Finished applying one-hot encoding.")
    # Catch any remaining category or object columns (e.g. from future feature additions)
    for col in df.select_dtypes(include=["category"]).columns:
        print(f"Encoding remaining categorical column: {col}...")
        df[col] = df[col].cat.codes
        print(f"Finished encoding column: {col}.")
    
    print("All features built successfully. Final shape:", df.shape)

    return df, {"cat": cat_enc, "ngram": ngram_enc}