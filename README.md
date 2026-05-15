# 🌡️ LST Urban Heat Prediction — Mangaluru

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)

[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-FF6600?style=for-the-badge)](https://xgboost.readthedocs.io/)
[![GEE](https://img.shields.io/badge/Data-Google%20Earth%20Engine-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://earthengine.google.com/)

> **End-to-end Machine Learning pipeline to predict Land Surface Temperature (LST) and map Urban Heat Islands over Mangaluru, India — using three years of multi-temporal Landsat satellite imagery (2021–2023).**



---

## 📌 Overview

Urban Heat Islands (UHIs) are a growing consequence of rapid urbanization, where built-up surfaces absorb and re-emit significantly more heat than surrounding vegetated areas. This project builds a **physics-informed, data-driven ML pipeline** to predict LST at pixel-level resolution across **Mangaluru, Dakshina Kannada** — a rapidly urbanizing coastal city with a tropical monsoon climate (~75% avg. humidity).

The pipeline combines:
- **Remote Sensing** — Monthly Landsat 8 (2021) and Landsat 9 (2022–2023) GeoTIFF composites exported from Google Earth Engine, clipped to Mangaluru's boundary at **30m resolution**.
- **Robust Data Engineering** — Custom **Global IQR** outlier detection that standardizes temperature bounds across all three years simultaneously.
- **Physics-Informed Feature Engineering** — Cyclic temporal encoding, a Vegetation Cooling Effect, and an Urban Heat Composite interaction feature derived from EDA findings.
- **Strict Temporal Validation** — Models trained on 2021–2022 are evaluated exclusively on **2023** (a fully unseen future year), simulating a real operational forecasting scenario.

---

## 🏗️ System Architecture

```
[Google Earth Engine]
  Landsat 8 (2021) ──┐
  Landsat 9 (2022)   ├─► 36 Monthly GeoTIFF Composites (30m, EPSG:4326)
  Landsat 9 (2023) ──┘   [NDVI · NDBI · NDWI · SAVI · Albedo · LST]
                                      │
                                      ▼
                        [Python Raster Sampler]
                   58,500 random pixels × 12 months/yr
                   → ~700,000 tabular samples/year
                                      │
                                      ▼
                   [Global IQR Preprocessing Pipeline]
            Physical limits + cross-year outlier removal
            LULC classification (Water/Veg/Built-up/Bare)
                                      │
                         ┌────────────┴────────────┐
                         │   Train: 2021 + 2022     │  300,000 samples
                         │   Validate: 2023 only    │  150,000 samples
                         └────────────┬────────────┘
                                      │
                             Feature Engineering
                    season_cos · NDBI · Albedo · Effect_veg
                               Composite_UHI
                                      │
                                      ▼
                  [XGBoost — Conservative Configuration]
                   max_depth=5 · n_estimators=300 · lr=0.05
                    subsample=0.7 · L1=2.0 · L2=2.0
                                      │
                                      ▼
                       LST Prediction Map (°C) 🗺️
```

---

## 🚀 Key Results

All models were evaluated on the **fully held-out 2023 validation set**.

### Phase 1 — Algorithm Comparison (5 engineered features)

| Algorithm | Train R² | Val R² | Val RMSE | Val MAE | Gen. Gap |
|---|---|---|---|---|---|
| Ridge Regression (Baseline) | 0.2859 | 0.3495 | 3.585 °C | 2.891 °C | -0.064 |
| AdaBoost | 0.4007 | 0.3288 | 3.642 °C | 2.927 °C | +0.072 |
| Random Forest | 0.6481 | 0.4669 | 3.246 °C | 2.493 °C | +0.181 |
| **XGBoost** | **0.5546** | **0.5040** | **3.131 °C** | **2.390 °C** | **+0.051** |

XGBoost achieved the best validation R² **and** the smallest generalization gap, confirming it learned stable physical relationships rather than memorizing training noise.

### Phase 2 — XGBoost Variant Optimization

| Variant | Val R² | Val RMSE | Val MAE |
|---|---|---|---|
| XGB\_Original | 0.4895 | 3.176 °C | 2.425 °C |
| XGB\_DART | 0.5061 | 3.124 °C | 2.383 °C |
| **XGB\_Conservative** ✅ | **0.5101** | **3.111 °C** | **2.375 °C** |

The **Conservative** configuration (shallower trees, stronger regularization) was selected as the final model. Restricting depth prevents the model from memorizing noise in noisy satellite data; increasing the number of estimators compensates for the reduced complexity per tree.

---

## 🔧 Feature Engineering

Five physics-informed features were derived from EDA findings to replace the original 6 raw, correlated spectral bands.

| Feature | Formula | Physical Meaning |
|---|---|---|
| `season_cos` | `cos(2π × (month − 1) / 12)` | Cyclic seasonal encoding — peaks at winter, troughs at summer |
| `NDBI` | `(SWIR1 − NIR) / (SWIR1 + NIR)` | Impervious surface density (asphalt, concrete) |
| `Albedo` | `0.356·B2 + 0.130·B3 + 0.373·B4 + 0.085·B5 + 0.072·B6 + 0.0018·B7` *(Liang, 2001)* | Surface reflectivity — low albedo absorbs more solar radiation |
| `Effect_veg` | `NDVI × SAVI` | Merged vegetation cooling proxy; resolves NDVI/SAVI multicollinearity (r=0.98) |
| `Composite_UHI` | `NDBI × (1 − Albedo) × (1 − NDVI)` | Interaction term: simultaneous built-up + dark + unvegetated = peak UHI |

> **Why these 5?** EDA revealed NDVI and SAVI are nearly perfectly correlated (r = 0.98), making them redundant individually. The `Composite_UHI` term acts as a mathematical "AND gate" — it only activates strongly when all three UHI conditions are true at once.

> **Feature Importance (from final XGBoost model):** `Composite_UHI` is the top predictor, followed by `season_cos` (~52% combined seasonal contribution), validating the physics-informed approach.

---

## 📂 Repository Structure

```
LST-ML/
│
├── data/
│   ├── raw/
│   │   ├── landsat8_2021/          # 12 monthly GeoTIFFs — git-ignored
│   │   └── landsat9_2022_2023/     # 24 monthly GeoTIFFs — git-ignored
│   ├── processed/                  # Sampled & cleaned CSVs — git-ignored
│   │   ├── samples_YYYY_with_month.csv
│   │   └── samples_clean_consistent_LULC_YYYY.csv
│   ├── shapefiles/
│   │   └── Mangaluru.zip           # AOI polygon for GEE clipping
│   └── README.md                   # Data reproduction guide
│
├── docs/
│   ├── data_pipeline_report.md     # Technical report: extraction & preprocessing
│   ├── LST_Mini_Project_Report.pdf # Full project report
│   ├── LST_Prediction.pptx         # Presentation slides
│   └── LST_Prediction_PPT.pdf      # Presentation (PDF)
│
├── notebooks/
│   ├── LST_Prediction_Pipeline.ipynb  # ⭐ Main ML pipeline (run this)
│   └── LST_Full_Analysis.ipynb        # Full EDA & experimental playground
│
├── src/
│   ├── extraction/
│   │   ├── Collectdatagee.js          # GEE script: exports monthly GeoTIFFs
│   │   ├── sample_landsat8.py         # Raster sampler — 2021 Landsat 8
│   │   └── sample_landsat9.py         # Raster sampler — 2022–2023 Landsat 9
│   └── preprocessing/
│       ├── preprocess.py              # Global IQR cleaning + LULC labels
│       ├── analysis.py                # Data quality & distribution validation
│       └── hyperparameter_tuning.py   # RandomizedSearchCV for XGBoost
│
├── results/                        # Model artifacts (.joblib) — git-ignored
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/LST-ML.git
cd LST-ML

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
```

### 2. Reproduce the Data (Optional)

Skip if you already have the processed CSVs.

**Step A — Export GeoTIFFs from Google Earth Engine:**
1. Open [code.earthengine.google.com](https://code.earthengine.google.com/)
2. Upload `data/shapefiles/Mangaluru.zip` as a GEE asset; set it as `aoi`
3. Paste `src/extraction/Collectdatagee.js` and run — creates 24 export tasks (Landsat 9, 2022–2023)
4. Download outputs to `data/raw/landsat9_2022_2023/`; repeat with modified dates for 2021 → `data/raw/landsat8_2021/`

**Step B — Sample the rasters:**
```bash
python src/extraction/sample_landsat8.py
python src/extraction/sample_landsat9.py
```
Each script samples 58,500 unique random pixels per monthly GeoTIFF → ~700,000 rows/year → `data/processed/`.

**Step C — Preprocess & clean:**
```bash
python src/preprocessing/preprocess.py
```
Applies Global IQR bounds, physical constraints (LST 10–50°C, indices ±1.0), and LULC labels.

**Step D — Validate data quality (optional):**
```bash
python src/preprocessing/analysis.py
```
Runs KS-tests for cross-year consistency and feature-LST correlation stability.

### 3. Train the Model

Open `notebooks/LST_Prediction_Pipeline.ipynb` in Jupyter and run all cells. The notebook trains all four algorithms, performs XGBoost variant optimization, and outputs performance metrics + a spatial LST prediction map.

---

## 🧠 Technical Notes

### Global IQR Outlier Cleaning
`preprocess.py` pools all three years' LST values into one series to calculate a **single shared Q1/Q3/IQR**, then applies a `1.5×IQR` fence identically across 2021, 2022, and 2023. Per-year IQR would risk removing real climate trends (e.g., a hotter 2023 vs 2021).

### LULC Classification

| Code | Class | Rule |
|---|---|---|
| 1 | Water | `NDWI ≥ 0.1` |
| 2 | Vegetation | `NDVI ≥ 0.3` |
| 3 | Built-up | `NDBI ≥ 0.0` |
| 4 | Bare/Other | All others |

Rules evaluated in priority order. The January 2021 full-pixel analysis (653,348 valid pixels) found: Vegetation 89.87% · Urban 4.03% · Other 4.04% · Water 1.97% · Bare 0.08%.

### Monsoon Data Gap
Cloud cover during the peak monsoon (June–August) significantly reduces valid pixel counts, especially in July. This is expected for a tropical coastal region and is handled by monthly median compositing during the GEE export phase.

### Temporal Leakage Prevention
Pixel coordinates are cross-checked between the train and validation sets; any overlapping `(lon, lat, month)` triplet is removed from the validation set before evaluation.

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Satellite Data | Landsat 8 C2 L2, Landsat 9 C2 L2, Google Earth Engine |
| Raster Processing | `rasterio`, `pyproj` |
| Data Engineering | `pandas`, `numpy`, `scipy` |
| Machine Learning | `scikit-learn`, `xgboost` |
| Visualization | `matplotlib`, `seaborn` |
| Notebooks | `jupyter` |
| Persistence | `joblib` |

---

## 📄 Documentation

| Document | Description |
|---|---|
| [`docs/data_pipeline_report.md`](docs/data_pipeline_report.md) | Technical breakdown of extraction & preprocessing |
| [`docs/LST_Mini_Project_Report.pdf`](docs/LST_Mini_Project_Report.pdf) | Full project report with methodology and results |
| [`data/README.md`](data/README.md) | Step-by-step guide to reproduce the dataset |

---

