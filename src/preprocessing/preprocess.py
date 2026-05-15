"""
Multi-Year Data Preprocessing & Cleaning Pipeline
==================================================
Aggregates monthly raw CSVs into multi-year datasets, applies
physical constraints, Global IQR outlier detection, and derives
LULC classification features.

Inputs:  data/processed/samples_YYYY_with_month.csv
Outputs: data/processed/samples_clean_consistent_LULC_YYYY.csv
"""

import pandas as pd
import numpy as np
import sys
import os

# --- 1. Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "..", "data", "processed")

DATA_FILES = {
    '2021': os.path.join(DATA_DIR, 'samples_2021_with_month.csv'),
    '2022': os.path.join(DATA_DIR, 'samples_2022_with_month.csv'),
    '2023': os.path.join(DATA_DIR, 'samples_2023_with_month.csv')
}

# Physical limits for normalized indices
PHYSICAL_LIMITS = {
    'NDVI': (-1.0, 1.0),
    'NDBI': (-1.0, 1.0),
    'NDWI': (-1.0, 1.0),
    'SAVI': (-1.0, 1.0),
    'Albedo': (0.0, 1.0)
}

# LST physical bounds (conservative for urban areas)
LST_PHYSICAL_BOUNDS = (10.0, 50.0)  # More realistic than 0-100°C

# IQR multiplier for consistent outlier detection
LST_IQR_MULTIPLIER = 1.5

# --- 2. Calculate GLOBAL IQR Bounds ---
def calculate_global_iqr_bounds():
    """
    Calculate consistent IQR bounds across ALL years to ensure data consistency
    """
    print("📊 Calculating GLOBAL IQR bounds from ALL years...")
    
    all_lst_values = []
    
    for year_str, filepath in DATA_FILES.items():
        try:
            df = pd.read_csv(filepath)
            # Apply basic physical filter first
            mask = (df['LST'] >= 0) & (df['LST'] <= 100)
            lst_values = df[mask]['LST'].values
            all_lst_values.extend(lst_values)
            print(f"   {year_str}: {len(lst_values):,} samples")
        except FileNotFoundError:
            print(f"   Warning: {filepath} not found, skipping")
            continue
    
    if not all_lst_values:
        print("❌ ERROR: No data found for IQR calculation!")
        return None, None
    
    all_lst_series = pd.Series(all_lst_values)
    
    Q1 = all_lst_series.quantile(0.25)
    Q3 = all_lst_series.quantile(0.75)
    IQR = Q3 - Q1
    
    global_lower = Q1 - (LST_IQR_MULTIPLIER * IQR)
    global_upper = Q3 + (LST_IQR_MULTIPLIER * IQR)
    
    # Ensure bounds are within physical reality
    global_lower = max(global_lower, LST_PHYSICAL_BOUNDS[0])
    global_upper = min(global_upper, LST_PHYSICAL_BOUNDS[1])
    
    print(f"✅ GLOBAL IQR Bounds Calculated:")
    print(f"   Q1: {Q1:.2f}°C, Q3: {Q3:.2f}°C, IQR: {IQR:.2f}°C")
    print(f"   Lower Bound: {global_lower:.2f}°C")
    print(f"   Upper Bound: {global_upper:.2f}°C")
    print(f"   Samples used: {len(all_lst_values):,}")
    
    return global_lower, global_upper

# --- 3. LULC Classification Function ---
def classify_lulc(row):
    """
    Creates the LULC feature.
    """
    if row['NDWI'] > 0.1:
        return 1  # Water
    elif row['NDVI'] > 0.3:
        return 2  # Vegetation
    elif row['NDBI'] > 0.0:
        return 3  # Built-up
    else:
        return 4  # Bare/Other

# --- 4. Enhanced Cleaning Function with GLOBAL Consistency ---
def clean_data_consistent(filepath, year, global_lower, global_upper):
    """
    Applies consistent cleaning across ALL years using global bounds
    """
    print(f"\n--- Cleaning {year} with GLOBAL Bounds ---")
    
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"❌ Error: Cannot find file {filepath}")
        return None
        
    original_count = len(df)
    print(f"📥 Original samples for {year}: {original_count:,}")
    
    # --- Step 1: Drop True Duplicates ---
    df.drop_duplicates(inplace=True)
    print(f"✅ Removed {original_count - len(df)} duplicates")
    
    # --- Step 2: Handle NaN/Inf ---
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # --- Step 3: Physical Limits for Indices ---
    print("\n🔬 Applying physical limits...")
    for feature, (low, high) in PHYSICAL_LIMITS.items():
        if feature in df.columns:
            mask = (df[feature] < low) | (df[feature] > high)
            df.loc[mask, feature] = np.nan
            if mask.sum() > 0:
                print(f"   Marked {mask.sum():,} bad {feature} values")
    
    # --- Step 4: CONSISTENT LST Cleaning ---
    print(f"\n🌡️ Applying GLOBAL LST bounds: {global_lower:.1f}°C to {global_upper:.1f}°C")
    
    # Apply global bounds (same for all years)
    lst_mask = (df['LST'] >= global_lower) & (df['LST'] <= global_upper)
    
    # Additional physical sanity check
    lst_mask = lst_mask & (df['LST'] >= LST_PHYSICAL_BOUNDS[0]) & (df['LST'] <= LST_PHYSICAL_BOUNDS[1])
    
    bad_lst_count = (~lst_mask).sum()
    df.loc[~lst_mask, 'LST'] = np.nan
    print(f"   Marked {bad_lst_count:,} LST values outside global bounds")
    
    # --- Step 5: Final Cleanup ---
    df.dropna(inplace=True)
    final_count = len(df)
    
    print(f"\n📊 {year} Cleaning Summary:")
    print(f"   Original: {original_count:,} samples")
    print(f"   Final: {final_count:,} samples")
    print(f"   Removed: {original_count - final_count:,} samples ({((original_count - final_count)/original_count*100):.1f}%)")
    
    # Basic statistics
    if final_count > 0:
        print(f"   LST - Mean: {df['LST'].mean():.2f}°C, Std: {df['LST'].std():.2f}°C")
        print(f"   NDVI - Mean: {df['NDVI'].mean():.3f}, Std: {df['NDVI'].std():.3f}")
    
    df['year'] = year
    return df

# --- 5. Data Distribution Validation ---
def validate_data_consistency(clean_dfs):
    """
    Validate that cleaned datasets have consistent distributions
    """
    print("\n" + "="*60)
    print("🔍 VALIDATING DATA CONSISTENCY ACROSS YEARS")
    print("="*60)
    
    years = list(clean_dfs.keys())
    
    print("📊 LST Statistics by Year:")
    print("-" * 40)
    for year in years:
        df = clean_dfs[year]
        print(f"   {year}: Mean = {df['LST'].mean():6.2f}°C, Std = {df['LST'].std():5.2f}°C, Samples = {len(df):,}")
    
    # Check feature correlations consistency
    print(f"\n📈 Key Feature-LST Correlations:")
    print("-" * 45)
    
    key_features = ['solar_angle_sin', 'net_radiation', 'NDBI', 'NDVI']
    for feature in key_features:
        if feature in clean_dfs[years[0]].columns:
            corrs = []
            for year in years:
                df = clean_dfs[year]
                if feature in df.columns:
                    corr = df[feature].corr(df['LST'])
                    corrs.append(corr)
                else:
                    corrs.append(np.nan)
            
            corr_std = np.std(corrs)
            status = "✅" if corr_std < 0.1 else "⚠️" if corr_std < 0.2 else "❌"
            print(f"   {feature:20}: {[f'{c:.3f}' for c in corrs]} | Std: {corr_std:.3f} {status}")

# --- 6. Main Execution ---
def main():
    print("🚀 STARTING CONSISTENT MULTI-YEAR DATA PREPROCESSING")
    print("=" * 60)
    
    # Step 1: Calculate GLOBAL bounds once
    global_lower, global_upper = calculate_global_iqr_bounds()
    
    if global_lower is None:
        print("❌ Failed to calculate global bounds. Exiting.")
        return
    
    # Step 2: Clean each year with SAME bounds
    clean_dfs = {}
    
    for year_str, filepath in DATA_FILES.items():
        year = int(year_str)
        clean_df = clean_data_consistent(filepath, year, global_lower, global_upper)
        
        if clean_df is None or clean_df.empty:
            print(f"❌ No clean data for {year_str}")
            continue
            
        # Add LULC classification
        print(f"🏙️  Adding LULC classification for {year_str}...")
        clean_df['LULC'] = clean_df.apply(classify_lulc, axis=1)
        
        # Save cleaned file
        output_filename = os.path.join(DATA_DIR, f"samples_clean_consistent_LULC_{year_str}.csv")
        clean_df.to_csv(output_filename, index=False)
        print(f"💾 Saved {len(clean_df):,} samples to {output_filename}")
        
        clean_dfs[year_str] = clean_df
    
    # Step 3: Validate consistency
    if len(clean_dfs) >= 2:
        validate_data_consistency(clean_dfs)
        
        # Final summary
        print("\n" + "="*60)
        print("✅ PREPROCESSING COMPLETED SUCCESSFULLY!")
        print("="*60)
        for year_str, df in clean_dfs.items():
            print(f"   {year_str}: {len(df):,} consistent samples")
        
        print(f"\n🎯 All years cleaned with SAME global bounds:")
        print(f"   LST Range: {global_lower:.1f}°C to {global_upper:.1f}°C")
        print(f"   Expected improvement: More balanced model performance!")
    else:
        print("❌ Insufficient data for validation")

if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    main()
