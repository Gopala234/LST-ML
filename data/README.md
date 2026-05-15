# Data Directory

Large CSV and GeoTIFF files are excluded from this repository via `.gitignore` to keep the code lightweight. 

## Structure
```
data/
├── raw/
│   ├── landsat8_2021/            # Put 2021 .tif files here
│   └── landsat9_2022_2023/       # Put 2022-2023 .tif files here
├── processed/
│   ├── samples_*_with_month.csv  # Output of extraction scripts
│   └── samples_clean_*.csv       # Output of preprocessing scripts
└── shapefiles/
    └── Mangaluru.zip             # Geographic bounds for GEE
```

## How to Reproduce the Data

1. **Extract from Google Earth Engine:**
   Copy `src/extraction/Collectdatagee.js` into the Google Earth Engine Code Editor. Upload `shapefiles/Mangaluru.zip` as your AOI asset and run the script. It will export 24 monthly GeoTIFF composites to your Google Drive. Place these in `raw/landsat9_2022_2023/` (and modify the dates for `landsat8_2021/`).

2. **Sample the Rasters:**
   Run the python extraction scripts to randomly sample ~700,000 pixels into tabular format.
   ```bash
   python src/extraction/sample_landsat8.py
   python src/extraction/sample_landsat9.py
   ```
   *These will output `samples_202x_with_month.csv` into the `processed/` directory.*

3. **Preprocess and Clean:**
   Run the global IQR cleaning algorithm to constrain physical values and generate the LULC categorical targets.
   ```bash
   python src/preprocessing/preprocess.py
   ```
   *This outputs the final `samples_clean_consistent_LULC_202x.csv` files used by the ML notebooks.*
