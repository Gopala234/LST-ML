import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp

print("🔍 DEBUG ANALYSIS OF PREPROCESSED DATA")
print("=" * 60)

# Load the newly preprocessed data
files = {
    '2021': 'samples_clean_consistent_LULC_2021.csv',
    '2022': 'samples_clean_consistent_LULC_2022.csv', 
    '2023': 'samples_clean_consistent_LULC_2023.csv'
}

# Load all data
dfs = {}
for year, file in files.items():
    try:
        dfs[year] = pd.read_csv(file)
        print(f"✅ Loaded {year}: {len(dfs[year]):,} samples")
    except FileNotFoundError:
        print(f"❌ Could not load {file}")

print(f"\n📊 TOTAL SAMPLES: {sum(len(df) for df in dfs.values()):,}")

# ============================================================================
# 1. TARGET VARIABLE ANALYSIS
# ============================================================================

print("\n" + "="*60)
print("🎯 TARGET VARIABLE (LST) ANALYSIS")
print("="*60)

lst_stats = {}
for year, df in dfs.items():
    lst_stats[year] = {
        'mean': df['LST'].mean(),
        'std': df['LST'].std(),
        'min': df['LST'].min(),
        'max': df['LST'].max(),
        'q1': df['LST'].quantile(0.25),
        'q3': df['LST'].quantile(0.75)
    }

print("📈 LST Statistics by Year:")
print("-" * 70)
print(f"{'Year':6} {'Mean':8} {'Std':8} {'Min':8} {'Max':8} {'Q1':8} {'Q3':8} {'Samples':10}")
print("-" * 70)
for year, stats in lst_stats.items():
    print(f"{year:6} {stats['mean']:8.2f} {stats['std']:8.2f} {stats['min']:8.2f} {stats['max']:8.2f} "
          f"{stats['q1']:8.2f} {stats['q3']:8.2f} {len(dfs[year]):10,}")

# Statistical tests for LST distribution
print(f"\n📊 LST Distribution Statistical Tests:")
print("-" * 50)
years = list(dfs.keys())
for i in range(len(years)):
    for j in range(i+1, len(years)):
        stat, p_value = ks_2samp(dfs[years[i]]['LST'], dfs[years[j]]['LST'])
        status = "✅ SIMILAR" if p_value > 0.05 else "❌ DIFFERENT"
        print(f"   {years[i]} vs {years[j]}: p-value = {p_value:.6f} - {status}")

# ============================================================================
# 2. FEATURE DISTRIBUTION ANALYSIS
# ============================================================================

print("\n" + "="*60)
print("📊 FEATURE DISTRIBUTION ANALYSIS")
print("="*60)

key_features = ['NDVI', 'NDBI', 'SAVI', 'Albedo', 'net_radiation']

for feature in key_features:
    if feature in dfs['2021'].columns:
        print(f"\n🔍 {feature} Distribution:")
        print("-" * 40)
        
        # Basic stats
        stats = {}
        for year in years:
            stats[year] = {
                'mean': dfs[year][feature].mean(),
                'std': dfs[year][feature].std(),
                'min': dfs[year][feature].min(),
                'max': dfs[year][feature].max()
            }
        
        # Print stats
        for year in years:
            print(f"   {year}: Mean={stats[year]['mean']:.3f}, Std={stats[year]['std']:.3f}, "
                  f"Range=[{stats[year]['min']:.3f}, {stats[year]['max']:.3f}]")
        
        # Statistical tests
        different_pairs = 0
        total_pairs = 0
        for i in range(len(years)):
            for j in range(i+1, len(years)):
                stat, p_value = ks_2samp(dfs[years[i]][feature], dfs[years[j]][feature])
                total_pairs += 1
                if p_value < 0.05:
                    different_pairs += 1
        
        consistency = (total_pairs - different_pairs) / total_pairs * 100
        status = "✅ EXCELLENT" if consistency >= 80 else "✅ GOOD" if consistency >= 60 else "⚠️  MODERATE" if consistency >= 40 else "❌ POOR"
        print(f"   Distribution Consistency: {consistency:.1f}% ({status})")

# ============================================================================
# 3. FEATURE-TARGET RELATIONSHIP CONSISTENCY
# ============================================================================

print("\n" + "="*60)
print("📈 FEATURE-LST RELATIONSHIP CONSISTENCY")
print("="*60)

# Key features to check correlations with LST
correlation_features = ['NDVI', 'NDBI', 'SAVI', 'Albedo', 'net_radiation', 
                       'solar_angle_sin', 'solar_angle_cos', 'season_sin', 'season_cos']

print("\n🔗 Correlation with LST by Year:")
print("-" * 70)
print(f"{'Feature':20} {'2021':8} {'2022':8} {'2023':8} {'Std':8} {'Status':10}")
print("-" * 70)

correlation_analysis = {}
for feature in correlation_features:
    if feature in dfs['2021'].columns:
        corrs = []
        for year in years:
            corr = dfs[year][feature].corr(dfs[year]['LST'])
            corrs.append(corr)
        
        corr_std = np.std(corrs)
        mean_corr = np.mean(corrs)
        
        # Status based on consistency
        if corr_std < 0.05:
            status = "✅ EXCELLENT"
        elif corr_std < 0.10:
            status = "✅ GOOD" 
        elif corr_std < 0.15:
            status = "⚠️  MODERATE"
        else:
            status = "❌ POOR"
        
        correlation_analysis[feature] = {
            'correlations': corrs,
            'std': corr_std,
            'status': status
        }
        
        print(f"{feature:20} {corrs[0]:8.3f} {corrs[1]:8.3f} {corrs[2]:8.3f} {corr_std:8.3f} {status:10}")

# ============================================================================
# 4. LULC DISTRIBUTION ANALYSIS
# ============================================================================

print("\n" + "="*60)
print("🏙️  LULC DISTRIBUTION ANALYSIS")
print("="*60)

lulc_labels = {1: 'Water', 2: 'Vegetation', 3: 'Built-up', 4: 'Bare/Other'}

print("\n📊 LULC Distribution by Year (%):")
print("-" * 50)
print(f"{'Year':6} {'Water':8} {'Veg':8} {'Built-up':10} {'Bare':8} {'Total':10}")
print("-" * 50)

for year in years:
    lulc_counts = dfs[year]['LULC'].value_counts().sort_index()
    total = lulc_counts.sum()
    percentages = [lulc_counts.get(i, 0) / total * 100 for i in range(1, 5)]
    
    print(f"{year:6} {percentages[0]:7.1f}% {percentages[1]:7.1f}% {percentages[2]:9.1f}% {percentages[3]:7.1f}% {total:10,}")

# ============================================================================
# 5. TEMPORAL PATTERN ANALYSIS (FIXED)
# ============================================================================

print("\n" + "="*60)
print("📅 TEMPORAL PATTERN ANALYSIS")
print("="*60)

print("\n🌡️  Monthly LST Patterns:")
print("-" * 40)

monthly_stats = {}
for year in years:
    monthly_means = dfs[year].groupby('month')['LST'].mean()
    monthly_stats[year] = monthly_means
    print(f"   {year}: {[f'{x:.1f}' for x in monthly_means.values]}")

# Check monthly pattern consistency (FIXED)
print(f"\n📈 Monthly Pattern Consistency:")
monthly_corrs = []
for i in range(len(years)):
    for j in range(i+1, len(years)):
        # Align months that exist in both years
        common_months = set(monthly_stats[years[i]].index) & set(monthly_stats[years[j]].index)
        if len(common_months) >= 2:  # Need at least 2 months for correlation
            series1 = monthly_stats[years[i]].loc[list(common_months)].sort_index()
            series2 = monthly_stats[years[j]].loc[list(common_months)].sort_index()
            corr = np.corrcoef(series1, series2)[0,1]
            monthly_corrs.append(corr)
            print(f"   {years[i]} vs {years[j]}: r = {corr:.3f} (based on {len(common_months)} months)")
        else:
            print(f"   {years[i]} vs {years[j]}: Insufficient common months")

if monthly_corrs:
    avg_monthly_corr = np.mean(monthly_corrs)
    print(f"   Average monthly correlation: {avg_monthly_corr:.3f}")
else:
    avg_monthly_corr = 0
    print(f"   Could not calculate monthly correlation")
# ============================================================================
# 6. DATA QUALITY SUMMARY
# ============================================================================

print("\n" + "="*60)
print("🏆 OVERALL DATA QUALITY ASSESSMENT")
print("="*60)

# Calculate overall consistency scores
lst_consistency = 1 - (np.std([lst_stats[y]['mean'] for y in years]) / 5)  # Normalize
feature_consistency = np.mean([correlation_analysis[f]['std'] for f in correlation_analysis]) / 0.1  # Normalize
monthly_consistency = avg_monthly_corr

overall_quality = (lst_consistency + (1 - feature_consistency) + monthly_consistency) / 3

print(f"📊 Quality Metrics:")
print(f"   LST Mean Consistency:    {lst_consistency:.3f} ({'✅ GOOD' if lst_consistency > 0.8 else '⚠️  MODERATE' if lst_consistency > 0.6 else '❌ POOR'})")
print(f"   Feature Correlation Std: {feature_consistency:.3f} ({'✅ GOOD' if feature_consistency < 0.1 else '⚠️  MODERATE' if feature_consistency < 0.2 else '❌ POOR'})")
print(f"   Monthly Pattern Corr:    {monthly_consistency:.3f} ({'✅ GOOD' if monthly_consistency > 0.8 else '⚠️  MODERATE' if monthly_consistency > 0.6 else '❌ POOR'})")
print(f"   OVERALL QUALITY SCORE:   {overall_quality:.3f} / 1.0")

if overall_quality > 0.8:
    print(f"\n🎉 EXCELLENT! Data is highly consistent across years - Ready for modeling!")
elif overall_quality > 0.6:
    print(f"\n✅ GOOD! Data is reasonably consistent - Should work well for modeling")
elif overall_quality > 0.4:
    print(f"\n⚠️  MODERATE! Some inconsistencies present - Monitor model performance")
else:
    print(f"\n❌ POOR! Significant inconsistencies - Consider additional preprocessing")

print(f"\n📋 RECOMMENDATIONS:")
if lst_consistency < 0.8:
    print("   • Consider temperature normalization across years")
if feature_consistency > 0.1:
    print("   • Monitor feature importance stability in the model")
if monthly_consistency < 0.8:
    print("   • Consider seasonal normalization or month-specific models")

print(f"\n✅ DEBUG ANALYSIS COMPLETED!")