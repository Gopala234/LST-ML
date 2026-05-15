# 📊 Report: Data Extraction, Preprocessing, and Cleaning Pipeline

This report provides a comprehensive, technical breakdown of the machine learning data pipeline designed to extract, clean, and standardize satellite imagery data from Landsat 8 (2021) and Landsat 9 (2022-2023). 

---

## Part 1: Data Extraction Pipeline (`Sample2021.py` & `Sample22n23.py`)

The extraction scripts are responsible for reading directly from raw month-by-month Google Earth Engine (GEE) GeoTIFFs and transforming visual pixels into tabular CSV data. The output datasets generated in this stage are:
*   `samples_2021_with_month.csv`
*   `samples_2022_with_month.csv`
*   `samples_2023_with_month.csv`

### 1. File & Metadata Handling
*   **Targeting:** The script iterates through a folder of `.tif` images, using regex `(\d{4})-(\d{2})` to accurately extract the specific year and month directly from the filename (e.g., `L8_median_2021-01.tif`).
*   **Band Validation:** It verifies that each image contains exactly 6 expected bands (`NDVI`, `NDBI`, `NDWI`, `SAVI`, `Albedo`, `LST`), skipping corrupt or incomplete files immediately.

### 2. Intelligent Pixel Sampling
*   **Target Quota:** The goal is generating a high-volume dataset by randomly sampling `58,500` pixels per monthly raster, resulting in approximately ~700,000 samples per year.
*   **Randomization & Uniqueness:** The sampler selects random `(row, col)` coordinates. To avoid data leakage and bias, it introduces an "expert system" safeguard: `sampled_pixels_in_file = set()`. If the script attempts to sample a pixel coordinate it has already gathered, it skips it.
*   **Fail-Safe Thresholds:** The while-loop restricts sampling attempts to `num_samples_per_file * 50`. If an image is largely composed of empty borders or water, the script intelligently exits the loop rather than hanging infinitely.

### 3. Spatial & Value Integrity
*   **Coordinate Reprojection:** Satellite arrays use native spatial coordinate references. `pyproj.Transformer` correctly maps the abstract Cartesian grid `(row, col)` into accurate Geographic Coordinates `[Longitude, Latitude]` mapping to `WGS84` (`EPSG:4326`).
*   **Immediate "Gatekeeper" Filtering:** Before moving any pixel to working memory, it checks for `NaN` (Not a Number) limits and internal `nodata_val` masks. If a pixel hits these masks, it is immediately discarded at the source, vastly reducing downstream memory bloat.

---

## Part 2: Data Preprocessing & Cleaning (`Pv2.py`)

The preprocessing phase aggregates the monthly raw data into multi-year datasets, standardizes limits, filters extreme outliers, and derives additional metadata.

### 1. Hard Physical Constraints
The script aggressively prunes values that fall outside the bounds of physical reality. If any index value breaches its natural thresholds, it is marked as `NaN` and dropped:
*   **Normalized Difference Indices:** `NDVI`, `NDBI`, `NDWI`, `SAVI` are strictly enforced to remain between `[-1.0, 1.0]`. 
*   **Albedo:** Constrained strictly between `[0.0, 1.0]` (representing 0% to 100% solar reflectance).
*   **Basic Thermal Limits:** Any Land Surface Temperature (`LST`) values outside an aggressive bound of `10°C` to `50°C` are dropped to prevent anomalies (e.g., malfunctioning sensors or unexpected ice/fire) from skewing the urban dataset.

### 2. Advanced Outlier Detection: The "Global IQR" Algorithm
*   **The Problem:** Calculating Interquartile Range (IQR) independently per year runs the risk of standardizing out legitimate global warming trends, or letting an unusually hot year artificially raise the "allowed" ceiling.
*   **The Solution:** The pipeline evaluates `LST` data across **all three years (2021-2023) concurrently**. It calculates a **Global Q1 and Q3**, and enforces a conservative multiplier (`IQR * 1.5`) across the board. By establishing a shared floor and ceiling, the data remains rigorously standard. 

### 3. Feature Engineering: LULC Classification
A deterministic algorithm assigns a Land Use Land Cover (`LULC`) classification feature based directly on existing spectral bands for every single coordinate point:
1.  **Water (1):** Defined if the Water Index `NDWI > 0.1`
2.  **Vegetation (2):** Defined if Vegetation Index `NDVI > 0.3`
3.  **Built-up (3):** Defined if Built-up Index `NDBI > 0.0`
4.  **Bare/Other (4):** Any pixel that fails to meet the above criteria.
*(Note: These indices are evaluated sequentially, meaning water takes precedence over vegetation, and vegetation over artificial structure).*

### 4. Final Aggregation
*   Duplicate rows and leftover `NaN/Inf` dataframes are entirely dropped out.
*   The system exports the finished, clean datasets. The final preprocessed files generated are:
    *   `samples_clean_consistent_LULC_2021.csv`
    *   `samples_clean_consistent_LULC_2022.csv`
    *   `samples_clean_consistent_LULC_2023.csv`

---

> [!TIP]
> **Summary on Integrity**
> By handling physical and geographical extraction cleanly at the source (Part 1) and standardizing outlier rules synchronously across all observed years (Part 2), this pipeline ensures your machine learning model is identifying true underlying trends, rather than learning artificial data faults.
