import pandas as pd
import numpy as np

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    
    # `ET_proxy` — evapotranspiration driver
    df['ET_proxy'] = (df['Temperature_C'] * df['Wind_Speed_kmh'] * (1 - df['Humidity']/100) * df['Sunlight_Hours'])
    
    # `Is_Active_Growth` — binary peak-demand flag
    df['Is_Active_Growth'] = df['Crop_Growth_Stage'].isin(['Vegetative','Flowering']).astype(int)

    # `ET_after_mulch` — mulching reduces evaporation
    mulch_flag = (df['Mulching_Used'] == 'Yes').astype(int)
    df['ET_after_mulch'] = df['ET_proxy'] * (1 - 0.3 * mulch_flag)

    # `Moisture_Deficit` — distance from field capacity
    df['Moisture_Deficit'] = 50 - df['Soil_Moisture']

    # `Dryness_Index` — demand/supply ratio
    df['Dryness_Index'] = df['ET_after_mulch'] / (df['Soil_Moisture'] + 1)

    # `Kc` — crop coefficient by Crop × Growth Stage
    KC_LOOKUP = {
    ('Rice',      'Sowing'):     1.05, ('Rice',      'Vegetative'): 1.15,
    ('Rice',      'Flowering'):  1.20, ('Rice',      'Harvest'):    0.90,
    ('Wheat',     'Sowing'):     0.35, ('Wheat',     'Vegetative'): 0.75,
    ('Wheat',     'Flowering'):  1.15, ('Wheat',     'Harvest'):    0.35,
    ('Maize',     'Sowing'):     0.30, ('Maize',     'Vegetative'): 0.80,
    ('Maize',     'Flowering'):  1.20, ('Maize',     'Harvest'):    0.45,
    ('Cotton',    'Sowing'):     0.35, ('Cotton',    'Vegetative'): 0.80,
    ('Cotton',    'Flowering'):  1.18, ('Cotton',    'Harvest'):    0.60,
    ('Sugarcane', 'Sowing'):     0.40, ('Sugarcane', 'Vegetative'): 0.85,
    ('Sugarcane', 'Flowering'):  1.25, ('Sugarcane', 'Harvest'):    0.75,
    ('Potato',    'Sowing'):     0.50, ('Potato',    'Vegetative'): 0.85,
    ('Potato',    'Flowering'):  1.15, ('Potato',    'Harvest'):    0.75,
    }
    df['Kc'] = [KC_LOOKUP[(c,s)] for c,s in zip(df['Crop_Type'], df['Crop_Growth_Stage'])]

    # Soil water-holding: `Avail_Water`, `Sat_Deficit`, `Rel_Moisture`
    SOIL_PWP = {'Sandy':5, 'Loamy':10, 'Silt':12, 'Clay':22}   # permanent wilting point
    SOIL_FC  = {'Sandy':15,'Loamy':30, 'Silt':35, 'Clay':45}   # field capacity
    df['Soil_PWP'] = df['Soil_Type'].map(SOIL_PWP)
    df['Soil_FC']  = df['Soil_Type'].map(SOIL_FC)
    df['Avail_Water']  = (df['Soil_Moisture'] - df['Soil_PWP']).clip(lower=0)
    df['Sat_Deficit']  = (df['Soil_FC']       - df['Soil_Moisture']).clip(lower=0)
    df['Rel_Moisture'] = df['Soil_Moisture']  / df['Soil_FC']

    # `VPD` — vapor pressure deficit (Magnus formula)
    es = 0.6108 * np.exp(17.27 * df['Temperature_C'] / (df['Temperature_C'] + 237.3))
    df['VPD'] = es * (1 - df['Humidity']/100)

    # `ET_crop` — crop-specific water demand
    df['ET_crop'] = df['ET_proxy'] * df['Kc']

    # `Water_Stress` — composite demand/supply index
    mulch_flag = (df['Mulching_Used'] == 'Yes').astype(int)
    df['Water_Stress'] = (df['ET_crop'] * df['VPD'] * (1 - 0.3*mulch_flag)) / (df['Avail_Water'] + 1)

    # Percentile rank features
    for c in ['Soil_Moisture','Temperature_C','Wind_Speed_kmh','Rainfall_mm']:
        df[c+'_rank'] = df[c].rank(pct=True)

    # Group statistics — deviation from peers
    g = df.groupby(['Crop_Type','Crop_Growth_Stage'])['Soil_Moisture']
    df['SM_dev_from_group']  = df['Soil_Moisture'] - g.transform('mean')
    df['SM_zscore_in_group'] = (df['Soil_Moisture'] - g.transform('mean')) / (g.transform('std') + 1e-3)

    # Binning at physical thresholds
    df['Temp_bin']  = pd.cut(df['Temperature_C'], bins=[-1, 15, 25, 32, 50],  labels=False)
    df['Moist_bin'] = pd.cut(df['Soil_Moisture'], bins=[-1, 15, 30, 45, 100], labels=False)
    df['Wind_bin']  = pd.cut(df['Wind_Speed_kmh'],bins=[-1,  5, 12, 20, 100], labels=False)



    # `Crop_Stage` — K-fold target encoding
    df['Crop_Stage'] = (df['Crop_Type'].astype(str) + '_' + df['Crop_Growth_Stage'].astype(str))

    return df
