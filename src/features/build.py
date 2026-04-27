"""
Feature engineering for Kaggle Playground Series S6E4 — Irrigation Need Prediction (CLAUDE CODE)
====================================================================================

Usage
-----
    import pandas as pd
    from feature_engineering import build_features

    train = pd.read_csv('train.csv')
    test  = pd.read_csv('test.csv')
    y_train = train['Irrigation_Need'].map({'Low': 0, 'Medium': 1, 'High': 2})

    X_train, X_test = build_features(train, test, y_train)

Everything is split into three stages you can run independently:
    add_physical_features(df)       — Stage A: physical intuition
    add_domain_features(df)         — Stage B: FAO-56 agronomy
    add_distribution_features(df)   — Stage C: ranks / group stats / bins
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold


# =============================================================================
# Lookup tables — FAO-56 / agronomy domain knowledge
# =============================================================================

# Kc: crop coefficient by (Crop_Type, Crop_Growth_Stage). From FAO-56 literature.
# Flowering = peak water demand, Sowing = minimum, Harvest = declining.
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

# Soil texture → rough Field Capacity (%) and Permanent Wilting Point (%)
SOIL_FC  = {'Sandy': 15, 'Loamy': 30, 'Silt': 35, 'Clay': 45}
SOIL_PWP = {'Sandy':  5, 'Loamy': 10, 'Silt': 12, 'Clay': 22}

# Categorical columns in the raw data
CAT_COLS = ['Soil_Type', 'Crop_Type', 'Crop_Growth_Stage', 'Season',
            'Irrigation_Type', 'Water_Source', 'Mulching_Used', 'Region']


# =============================================================================
# Stage A: physical intuition features
# =============================================================================

def add_physical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Water demand / supply features derived from raw physical quantities.

    Why: irrigation need = crop demand − water already available. These features
    express that balance directly so the model doesn't have to rediscover it.
    """
    df = df.copy()

    # Evapotranspiration proxy. Hot + windy + dry air + long sunny days = high loss.
    # Multiplicative so the model captures the joint effect without deep splits.
    df['ET_proxy'] = (
        df['Temperature_C'] * df['Wind_Speed_kmh']
        * (1 - df['Humidity'] / 100) * df['Sunlight_Hours']
    )

    # Binary flag: crop is actively growing (peak water demand).
    # Crosstab on target shows Sowing/Harvest ≈ 87% Low, Veg/Flower ≈ 63% Medium+6% High.
    df['Is_Active_Growth'] = df['Crop_Growth_Stage'].isin(
        ['Vegetative', 'Flowering']
    ).astype(int)

    # Mulching reduces evaporation by ~30% (agronomy literature).
    mulch_flag = (df['Mulching_Used'] == 'Yes').astype(int)
    df['ET_after_mulch'] = df['ET_proxy'] * (1 - 0.3 * mulch_flag)

    # Moisture deficit vs. rough field capacity.
    df['Moisture_Deficit'] = 50 - df['Soil_Moisture']

    # Single demand/supply ratio. +1 in denominator guards against division by zero.
    df['Dryness_Index'] = df['ET_after_mulch'] / (df['Soil_Moisture'] + 1)

    return df


# =============================================================================
# Stage B: FAO-56 domain features
# =============================================================================

def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agronomy-inspired features using FAO-56 style lookups and formulas.

    Why: these encode crop × stage × soil physics relationships that would
    otherwise require deep trees and many interactions to learn implicitly.
    """
    df = df.copy()

    # Crop coefficient from lookup table
    df['Kc'] = [KC_LOOKUP[(c, s)]
                for c, s in zip(df['Crop_Type'], df['Crop_Growth_Stage'])]

    # Soil water-holding: Permanent Wilting Point (below this, plants can't extract water)
    df['Soil_PWP'] = df['Soil_Type'].map(SOIL_PWP)
    df['Soil_FC']  = df['Soil_Type'].map(SOIL_FC)

    # Plant-available water. clip(0) guards against physically impossible negatives.
    df['Avail_Water']  = (df['Soil_Moisture'] - df['Soil_PWP']).clip(lower=0)
    df['Sat_Deficit']  = (df['Soil_FC'] - df['Soil_Moisture']).clip(lower=0)
    df['Rel_Moisture'] = df['Soil_Moisture'] / df['Soil_FC']

    # Vapor Pressure Deficit via Magnus formula — the physically correct driver
    # of transpiration. Saturated vapor pressure grows exponentially with temperature.
    es = 0.6108 * np.exp(17.27 * df['Temperature_C'] / (df['Temperature_C'] + 237.3))
    df['VPD'] = es * (1 - df['Humidity'] / 100)

    # Crop-specific evapotranspiration (requires Stage A first)
    if 'ET_proxy' in df.columns:
        df['ET_crop'] = df['ET_proxy'] * df['Kc']

    # Composite water-stress index: demand × air-dryness × mulch-adjustment / supply
    mulch_flag = (df['Mulching_Used'] == 'Yes').astype(int)
    demand = df.get('ET_crop', df.get('ET_proxy', pd.Series(1, index=df.index)))
    df['Water_Stress'] = (
        demand * df['VPD'] * (1 - 0.3 * mulch_flag)
    ) / (df['Avail_Water'] + 1)

    return df


# =============================================================================
# Stage C: distribution / statistical features
# =============================================================================

def add_distribution_features(
    df: pd.DataFrame,
    group_stats: dict | None = None,
):
    """Rank, binning, and group-statistics features.

    group_stats: pass None on train (computed in place); pass the returned
    dict on test so that group means/stds come from train only (no leakage).
    """
    df = df.copy()

    # Combined Crop × Growth_Stage key (24 categories). The single most useful
    # new categorical once target-encoded.
    df['Crop_Stage'] = (df['Crop_Type'].astype(str) + '_'
                        + df['Crop_Growth_Stage'].astype(str))

    # Percentile ranks — scale-free, robust to outliers, linearise tree splits.
    for c in ['Soil_Moisture', 'Temperature_C', 'Wind_Speed_kmh', 'Rainfall_mm']:
        df[c + '_rank'] = df[c].rank(pct=True)

    # Group statistics of Soil_Moisture by (Crop_Type, Crop_Growth_Stage).
    # Train: compute and return. Test: use precomputed to prevent leakage.
    if group_stats is None:
        g = df.groupby(['Crop_Type', 'Crop_Growth_Stage'])['Soil_Moisture']
        group_mean = g.transform('mean')
        group_std  = g.transform('std')
        group_stats = {
            'sm_mean': g.mean().to_dict(),
            'sm_std':  g.std().to_dict(),
        }
    else:
        keys = list(zip(df['Crop_Type'], df['Crop_Growth_Stage']))
        global_mean = np.mean(list(group_stats['sm_mean'].values()))
        global_std  = np.mean(list(group_stats['sm_std'].values()))
        group_mean = pd.Series(
            [group_stats['sm_mean'].get(k, global_mean) for k in keys],
            index=df.index)
        group_std = pd.Series(
            [group_stats['sm_std'].get(k, global_std) for k in keys],
            index=df.index)

    df['SM_dev_from_group']  = df['Soil_Moisture'] - group_mean
    df['SM_zscore_in_group'] = (df['Soil_Moisture'] - group_mean) / (group_std + 1e-3)

    # Binning at physically meaningful thresholds (not data-dependent quantiles,
    # so train and test get identical bin edges — no leakage).
    df['Temp_bin']  = pd.cut(df['Temperature_C'],   bins=[-1, 15, 25, 32, 50],  labels=False)
    df['Moist_bin'] = pd.cut(df['Soil_Moisture'],   bins=[-1, 15, 30, 45, 100], labels=False)
    df['Wind_bin']  = pd.cut(df['Wind_Speed_kmh'],  bins=[-1,  5, 12, 20, 100], labels=False)

    return df, group_stats


# =============================================================================
# Target encoding (K-fold, leakage-safe)
# =============================================================================

def kfold_target_encode(
    train_col: pd.Series,
    test_col: pd.Series,
    y_train: pd.Series,
    n_splits: int = 5,
    random_state: int = 0,
):
    """Out-of-fold target encoding. Train values are encoded using folds OTHER
    than the one they belong to; test values use the overall training mean per category.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    tr_enc = np.zeros(len(train_col))
    global_mean = y_train.mean()

    train_col = train_col.reset_index(drop=True)
    y_train   = y_train.reset_index(drop=True)

    for i_tr, i_val in kf.split(train_col):
        fold_means = y_train.iloc[i_tr].groupby(train_col.iloc[i_tr].values).mean()
        tr_enc[i_val] = train_col.iloc[i_val].map(fold_means).fillna(global_mean).values

    overall_means = y_train.groupby(train_col.values).mean()
    te_enc = test_col.reset_index(drop=True).map(overall_means).fillna(global_mean).values

    return tr_enc, te_enc


# =============================================================================
# Categorical encoding
# =============================================================================

def encode_categoricals(train, test, cat_cols=CAT_COLS):
    """Label encode categoricals consistently across train and test."""
    train = train.copy()
    test  = test.copy()
    for c in cat_cols:
        le = LabelEncoder()
        le.fit(pd.concat([train[c], test[c]]).astype(str))
        train[c + '_enc'] = le.transform(train[c].astype(str))
        test[c + '_enc']  = le.transform(test[c].astype(str))
    return train, test


# =============================================================================
# Full pipeline
# =============================================================================

def build_features(train, test, y_train, target_encode_cols=('Crop_Stage',)):
    """End-to-end. Returns (X_train, X_test) as numeric DataFrames ready for modelling.

    y_train must be numeric (e.g. Low→0, Medium→1, High→2).
    """
    # Stage A
    train = add_physical_features(train)
    test  = add_physical_features(test)

    # Stage B
    train = add_domain_features(train)
    test  = add_domain_features(test)

    # Stage C (group_stats computed on train only, applied to test)
    train, group_stats = add_distribution_features(train, group_stats=None)
    test,  _           = add_distribution_features(test,  group_stats=group_stats)

    # Categorical label encoding
    train, test = encode_categoricals(train, test)

    # Map the original target to numeric if it's still in string form (e.g. Low/Medium/High → 0/1/2)
    if 'Irrigation_Need' in train.columns and train['Irrigation_Need'].dtype == 'object':
        train['Irrigation_Need'] = train['Irrigation_Need'].map({'Low': 0, 'Medium': 1, 'High': 2})
    
    if 'Irrigation_Need' in test.columns and test['Irrigation_Need'].dtype == 'object':
        test['Irrigation_Need'] = test['Irrigation_Need'].map({'Low': 0, 'Medium': 1, 'High': 2})

    # K-fold target encoding
    for col in target_encode_cols:
        tr_enc, te_enc = kfold_target_encode(train[col].astype(str),
                                              test[col].astype(str),
                                              y_train)
        train[col + '_TE'] = tr_enc
        test[col + '_TE']  = te_enc

    # Drop raw string columns and id/target
    drop = list(CAT_COLS) + ['Crop_Stage', 'id', 'Irrigation_Need']
    X_train = train.drop(columns=[c for c in drop if c in train.columns])
    X_test  = test.drop(columns=[c for c in drop if c in test.columns])
    X_test  = X_test[X_train.columns]  # align

    return X_train, X_test

