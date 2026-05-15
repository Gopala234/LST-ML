"""
Landsat 9 (2022-2023) Raster Sampling Script
=============================================
Samples random, unique pixels from monthly GeoTIFF composites
exported via Google Earth Engine. Extracts NDVI, NDBI, NDWI, SAVI,
Albedo, and LST values with geographic coordinates (EPSG:4326).

Output: samples_2022_with_month.csv, samples_2023_with_month.csv
"""

import rasterio
import random
import numpy as np
import pandas as pd
import glob
import os
import re
from tqdm import tqdm
from pyproj import Transformer

# --- 1. Configuration ---
# Relative path from this script's location to the raw data folder
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "..", "data", "raw", "landsat9_2022_2023")

# Your list of bands, in the *exact* order they appear in the GeoTIFFs
BAND_NAMES = ['NDVI', 'NDBI', 'NDWI', 'SAVI', 'Albedo', 'LST']

# Define the final CSV header
CSV_HEADERS = ['lon', 'lat', 'month'] + BAND_NAMES

# Target samples per file (to get ~700k per year)
SAMPLES_PER_FILE = 58500

NUM_BANDS = len(BAND_NAMES)

# Regex to find the month (e.g., "2022-01")
MONTH_REGEX = re.compile(r"(\d{4})-(\d{2})")

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "..", "data", "processed")

# --- 2. Sampling Function (with all corrections) ---

def sample_rasters(file_list, num_samples_per_file):
    """
    Samples random, unique pixels (lon, lat, month, and band values) from GeoTIFFs.
    Filters all NaN and nodata values at the source.
    """
    all_samples = []
    
    for filepath in tqdm(file_list, desc="Sampling files"):
        try:
            filename = os.path.basename(filepath)
            
            # --- Extract month ---
            match = MONTH_REGEX.search(filename)
            if not match:
                print(f"Warning: Skipping {filename}. Could not find YYYY-MM pattern.")
                continue
            
            month = int(match.group(2))
            
            with rasterio.open(filepath) as src:
                if src.count != NUM_BANDS:
                    print(f"Warning: Skipping {filename}. Expected {NUM_BANDS} bands, found {src.count}")
                    continue
                
                height, width = src.height, src.width
                nodata_val = src.nodata
                
                # --- The "EXPERT" Fix: Use a set to track unique pixels ---
                # This prevents sampling the same (row, col) twice *from this file*
                sampled_pixels_in_file = set()
                
                transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
                
                samples_collected = 0
                attempts = 0
                
                while samples_collected < num_samples_per_file:
                    attempts += 1
                    # Stop if we try too many times (e.g., raster is mostly nodata)
                    if attempts > num_samples_per_file * 50: 
                        print(f"Warning: Could only find {samples_collected} valid samples in {filename} after {attempts} attempts. Moving on.")
                        break
                        
                    row, col = random.randint(0, height - 1), random.randint(0, width - 1)
                    
                    # --- Check for Unique Pixel ---
                    if (row, col) in sampled_pixels_in_file:
                        continue  # This pixel was already sampled, try again
                        
                    # Add to set so we don't sample it again
                    sampled_pixels_in_file.add((row, col))
                    
                    window = rasterio.windows.Window(col, row, 1, 1)
                    pixel_data = src.read(window=window)
                    vals = pixel_data[:, 0, 0]
                    
                    # --- The "Gatekeeper" Check ---
                    # Removes "corrupt" data (NaN, nodata) at the source
                    is_nodata = False
                    if nodata_val is not None and np.any(vals == nodata_val):
                        is_nodata = True
                    is_nan = np.any(np.isnan(vals))
                    
                    if not is_nodata and not is_nan:
                        native_x, native_y = src.xy(row, col)
                        lon, lat = transformer.transform(native_x, native_y)
                        
                        sample_row = [lon, lat, month] + list(vals)
                        
                        all_samples.append(sample_row)
                        samples_collected += 1
                        
        except Exception as e:
            print(f"Error processing {filepath}: {e}")

    df = pd.DataFrame(all_samples, columns=CSV_HEADERS)
    return df

# --- 3. Main Execution ---

def main():
    print("Starting raster sampling process (with lon/lat AND month)...")
    
    file_pattern = os.path.join(DATA_DIR, "*.tif*")  # Find .tif and .tiff
    all_files = glob.glob(file_pattern)
    
    if not all_files:
        print(f"Error: No .tif or .tiff files found in '{DATA_DIR}'.")
        return

    # --- Separate files by year ---
    files_2022 = sorted([f for f in all_files if "2022" in os.path.basename(f) and not f.endswith('.xml')])
    files_2023 = sorted([f for f in all_files if "2023" in os.path.basename(f) and not f.endswith('.xml')])

    print(f"Found {len(files_2022)} files for 2022.")
    print(f"Found {len(files_2023)} files for 2023.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Process 2022 ---
    if files_2022:
        print(f"\nProcessing 2022 data...")
        df_2022 = sample_rasters(files_2022, SAMPLES_PER_FILE)
        output_file_2022 = os.path.join(OUTPUT_DIR, "samples_2022_with_month.csv")
        df_2022.to_csv(output_file_2022, index=False)
        print(f"Successfully saved {len(df_2022)} samples to {output_file_2022}")
    
    # --- Process 2023 ---
    if files_2023:
        print(f"\nProcessing 2023 data...")
        df_2023 = sample_rasters(files_2023, SAMPLES_PER_FILE)
        output_file_2023 = os.path.join(OUTPUT_DIR, "samples_2023_with_month.csv")
        df_2023.to_csv(output_file_2023, index=False)
        print(f"Successfully saved {len(df_2023)} samples to {output_file_2023}")

    print("\nSampling complete for 2022 and 2023.")

if __name__ == "__main__":
    main()
