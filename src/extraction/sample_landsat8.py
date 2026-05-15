"""
Landsat 8 (2021) Raster Sampling Script
========================================
Samples random, unique pixels from monthly GeoTIFF composites
exported via Google Earth Engine. Extracts NDVI, NDBI, NDWI, SAVI,
Albedo, and LST values with geographic coordinates (EPSG:4326).

Output: samples_2021_with_month.csv (~70 MB, ~700k samples)
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
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "..", "data", "raw", "landsat8_2021")

BAND_NAMES = ['NDVI', 'NDBI', 'NDWI', 'SAVI', 'Albedo', 'LST']
CSV_HEADERS = ['lon', 'lat', 'month'] + BAND_NAMES
SAMPLES_PER_FILE = 58500  # ~700k samples across 12 months
NUM_BANDS = len(BAND_NAMES)
MONTH_REGEX = re.compile(r"(\d{4})-(\d{2})")

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "..", "data", "processed")

# --- 2. Sampling Function ---

def sample_rasters(file_list, num_samples_per_file):
    """
    Samples random, unique pixels (lon, lat, month, and band values) from GeoTIFFs.
    """
    all_samples = []
    
    for filepath in tqdm(file_list, desc="Sampling 2021 files"):
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
                
                # --- Bug Fix: Use correct EPSG:4326 ---
                transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
                
                samples_collected = 0
                attempts = 0
                
                # --- Unique Pixel Set ---
                # This ensures you don't sample the same pixel twice from one file
                sampled_pixels_in_file = set()
                
                while samples_collected < num_samples_per_file:
                    attempts += 1
                    if attempts > num_samples_per_file * 50: 
                        print(f"Warning: Could only find {samples_collected} valid samples in {filename} after {attempts} attempts. Moving on.")
                        break
                        
                    row, col = random.randint(0, height - 1), random.randint(0, width - 1)
                    
                    # --- Check for unique pixel ---
                    if (row, col) in sampled_pixels_in_file:
                        continue  # Skip if already sampled
                    
                    sampled_pixels_in_file.add((row, col))

                    window = rasterio.windows.Window(col, row, 1, 1)
                    pixel_data = src.read(window=window)
                    vals = pixel_data[:, 0, 0]
                    
                    # This logic block handles "corrupt" data (NaN, nodata)
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

# --- 3. Main Execution (2021 ONLY) ---

def main():
    print("Starting raster sampling process for 2021...")
    
    file_pattern = os.path.join(DATA_DIR, "*.tif")
    all_files = glob.glob(file_pattern)
    file_pattern_tiff = os.path.join(DATA_DIR, "*.tiff")
    all_files.extend(glob.glob(file_pattern_tiff))
    
    if not all_files:
        print(f"Error: No .tif or .tiff files found in '{DATA_DIR}'.")
        return

    # --- Only find 2021 files ---
    files_2021 = sorted([f for f in all_files if "2021" in os.path.basename(f)])

    if not files_2021:
        print(f"Error: No files containing '2021' found in '{DATA_DIR}'.")
        return
        
    print(f"Found {len(files_2021)} files for 2021.")

    # --- Process 2021 ---
    if files_2021:
        df_2021 = sample_rasters(files_2021, SAMPLES_PER_FILE)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_file_2021 = os.path.join(OUTPUT_DIR, "samples_2021_with_month.csv")
        df_2021.to_csv(output_file_2021, index=False)
        print(f"Successfully saved {len(df_2021)} samples to {output_file_2021}")

    print("\nStep 2 (2021): Sampling complete.")

if __name__ == "__main__":
    main()
