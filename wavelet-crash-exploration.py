# -*- coding: utf-8 -*-
# %% [markdown]
# # Wavelet Analysis of Synthetic Traffic Crash and Weather Data
# 
# This experiment demonstrates:
# 1. Creating synthetic crash and weather data
# 2. Building a spatio-temporal grid
# 3. Wavelet analysis using PyWavelets
# 4. Cross-correlation, coherence, and phase analysis

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pywt
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# %% [markdown]
# ## 1. Create Synthetic Crash and Weather Data

# %%
# Parameters
n_days = 365 * 2  # 2 years of daily data
n_grid_cells = 9   # 3x3 grid
start_date = datetime(2022, 1, 1)

# Create date range
dates = [start_date + timedelta(days=i) for i in range(n_days)]

# %%
def generate_synthetic_weather_data(dates):
    """Generate synthetic weather data with seasonal patterns"""
    n_days = len(dates)
    
    # Base temperature with seasonal pattern
    day_of_year = np.array([d.timetuple().tm_yday for d in dates])
    base_temp = 15 + 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    
    # Add some random variation
    temp_variation = np.random.normal(0, 3, n_days)
    temperature = base_temp + temp_variation
    
    # Precipitation probability (higher in winter)
    precip_prob = 0.3 + 0.2 * np.sin(2 * np.pi * (day_of_year - 300) / 365)
    
    # Rainfall amount (mm)
    rainfall = np.zeros(n_days)
    for i in range(n_days):
        if np.random.random() < precip_prob[i]:
            rainfall[i] = np.random.gamma(2, 2)  # Gamma distribution for rainfall
    
    # Snow (when temperature < 0°C)
    snow = np.zeros(n_days)
    snow_mask = temperature < 0
    snow[snow_mask] = np.random.exponential(1, np.sum(snow_mask))
    
    # Ice probability (complex function of temp and recent precip)
    ice_prob = 0.1 * (temperature < 2) * (temperature > -5) * (np.roll(rainfall, 1) > 0)
    ice = np.random.binomial(1, ice_prob)
    
    return pd.DataFrame({
        'date': dates,
        'temperature': temperature,
        'rainfall': rainfall,
        'snow': snow,
        'ice': ice
    })

# %%
def generate_synthetic_crash_data(dates, weather_df, n_grid_cells=9):
    """Generate synthetic crash data influenced by weather and temporal patterns"""
    n_days = len(dates)
    
    # Base crash rate with seasonal and weekly patterns
    day_of_year = np.array([d.timetuple().tm_yday for d in dates])
    day_of_week = np.array([d.weekday() for d in dates])
    
    # Seasonal pattern (more crashes in winter)
    seasonal_pattern = 0.5 * np.sin(2 * np.pi * (day_of_year - 300) / 365)
    
    # Weekly pattern (more crashes on weekends)
    weekly_pattern = 0.3 * np.sin(2 * np.pi * day_of_week / 7)
    
    # Base crash rate for each grid cell (urban vs rural simulation)
    base_rates = np.array([2.0, 1.5, 1.0,  # Urban core
                          1.5, 1.2, 0.8,   # Suburban
                          1.0, 0.7, 0.5])  # Rural
    
    crash_data = []
    
    for grid_id in range(n_grid_cells):
        base_rate = base_rates[grid_id]
        
        # Combine patterns
        temporal_pattern = base_rate * (1 + seasonal_pattern + weekly_pattern)
        
        # Weather effects
        rain_effect = 0.1 * weather_df['rainfall'].values
        snow_effect = 0.3 * weather_df['snow'].values
        ice_effect = 0.5 * weather_df['ice'].values
        
        # Expected crash count
        expected_crashes = temporal_pattern * (1 + rain_effect + snow_effect + ice_effect)
        
        # Add some spatial correlation (neighborhood effects)
        if grid_id > 0:
            spatial_corr = 0.1 * np.roll(expected_crashes, 1)
        else:
            spatial_corr = 0
        
        expected_crashes += spatial_corr
        
        # Generate actual crash counts (Poisson distribution)
        crash_counts = np.random.poisson(np.maximum(0.1, expected_crashes))
        
        for i, date in enumerate(dates):
            crash_data.append({
                'date': date,
                'grid_id': grid_id,
                'crash_count': crash_counts[i],
                'x_coord': grid_id % 3,
                'y_coord': grid_id // 3
            })
    
    return pd.DataFrame(crash_data)

# %%
# Generate the data
print("Generating synthetic weather data...")
weather_df = generate_synthetic_weather_data(dates)

print("Generating synthetic crash data...")
crash_df = generate_synthetic_crash_data(dates, weather_df, n_grid_cells)

print(f"Weather data shape: {weather_df.shape}")
print(f"Crash data shape: {crash_df.shape}")

# %%
# Visualize sample data
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Weather data
axes[0, 0].plot(weather_df['date'][:100], weather_df['temperature'][:100])
axes[0, 0].set_title('Temperature (First 100 Days)')
axes[0, 0].set_ylabel('°C')

axes[0, 1].plot(weather_df['date'][:100], weather_df['rainfall'][:100])
axes[0, 1].set_title('Rainfall (First 100 Days)')
axes[0, 1].set_ylabel('mm')

# Crash data for one grid cell
grid_0_crashes = crash_df[crash_df['grid_id'] == 0]
axes[1, 0].plot(grid_0_crashes['date'][:100], grid_0_crashes['crash_count'][:100])
axes[1, 0].set_title('Crash Counts - Grid 0 (First 100 Days)')
axes[1, 0].set_ylabel('Crash Count')

# Spatial distribution
spatial_avg = crash_df.groupby('grid_id')['crash_count'].mean().values.reshape(3, 3)
im = axes[1, 1].imshow(spatial_avg, cmap='Reds')
axes[1, 1].set_title('Average Crash Count by Grid Cell')
plt.colorbar(im, ax=axes[1, 1])

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2. Create Spatio-Temporal Grid

# %%
def create_spatio_temporal_grid(crash_df, weather_df):
    """Create a unified spatio-temporal grid dataset"""
    
    # Aggregate crash data by date and grid
    grid_data = crash_df.groupby(['date', 'grid_id']).agg({
        'crash_count': 'sum',
        'x_coord': 'first',
        'y_coord': 'first'
    }).reset_index()
    
    # Merge with weather data
    merged_data = pd.merge(grid_data, weather_df, on='date', how='left')
    
    # Add temporal features
    merged_data['day_of_year'] = merged_data['date'].dt.dayofyear
    merged_data['day_of_week'] = merged_data['date'].dt.dayofweek
    merged_data['month'] = merged_data['date'].dt.month
    merged_data['is_weekend'] = merged_data['day_of_week'].isin([5, 6]).astype(int)
    
    return merged_data

# %%
# Create the grid
print("Creating spatio-temporal grid...")
grid_df = create_spatio_temporal_grid(crash_df, weather_df)
print(f"Grid data shape: {grid_df.shape}")
print("\nGrid data sample:")
print(grid_df.head())

# %% [markdown]
# ## 3. Wavelet Analysis Functions

# %%
def wavelet_analysis(time_series, scales=None, wavelet='morl'):
    """Perform continuous wavelet transform on a time series"""
    if scales is None:
        scales = np.arange(1, 128)
    
    coefficients, frequencies = pywt.cwt(time_series, scales, wavelet)
    return coefficients, frequencies, scales

def wavelet_cross_correlation(x, y, scales=None, wavelet='morl'):
    """Calculate wavelet cross-correlation between two time series"""
    if scales is None:
        scales = np.arange(1, 128)
    
    # Perform CWT on both series
    coeff_x, _ = pywt.cwt(x, scales, wavelet)
    coeff_y, _ = pywt.cwt(y, scales, wavelet)
    
    # Calculate cross-correlation for each scale
    xcorr = np.zeros(len(scales))
    for i, scale in enumerate(scales):
        # Normalize cross-correlation
        corr = np.corrcoef(coeff_x[i], coeff_y[i])[0, 1]
        xcorr[i] = corr if not np.isnan(corr) else 0
    
    return xcorr, scales

def wavelet_coherence(x, y, scales=None, wavelet='morl'):
    """Calculate wavelet coherence between two time series"""
    if scales is None:
        scales = np.arange(1, 128)
    
    # Perform CWT
    coeff_x, _ = pywt.cwt(x, scales, wavelet)
    coeff_y, _ = pywt.cwt(y, scales, wavelet)
    
    # Calculate wavelet coherence
    Wxy = coeff_x * np.conj(coeff_y)
    Wxx = np.abs(coeff_x) ** 2
    Wyy = np.abs(coeff_y) ** 2
    
    # Smoothing (simple moving average)
    def smooth(w, window_size=5):
        return np.array([np.convolve(w[i], np.ones(window_size)/window_size, mode='same') 
                        for i in range(len(w))])
    
    smooth_Wxy = smooth(Wxy)
    smooth_Wxx = smooth(Wxx)
    smooth_Wyy = smooth(Wyy)
    
    # Coherence
    coherence = np.abs(smooth_Wxy) ** 2 / (smooth_Wxx * smooth_Wyy)
    
    # Phase difference
    phase = np.angle(Wxy)
    
    return coherence, phase, scales

# %% [markdown]
# ## 4. Apply Wavelet Analysis to Grid Cell Data

# %%
# Focus on one grid cell for demonstration
grid_id = 4  # Central grid cell
cell_data = grid_df[grid_df['grid_id'] == grid_id].sort_values('date')

# Extract time series
crash_ts = cell_data['crash_count'].values
rain_ts = cell_data['rainfall'].values
temp_ts = cell_data['temperature'].values

# Normalize the time series
crash_norm = (crash_ts - np.mean(crash_ts)) / np.std(crash_ts)
rain_norm = (rain_ts - np.mean(rain_ts)) / np.std(rain_ts)
temp_norm = (temp_ts - np.mean(temp_ts)) / np.std(temp_ts)

# %%
# Perform individual wavelet transforms
print("Performing wavelet transforms...")
scales = np.arange(1, 128)

# Crash data wavelet transform
coeff_crash, freq_crash, scales_crash = wavelet_analysis(crash_norm, scales)
print(f"Crash coefficients shape: {coeff_crash.shape}")

# Rainfall wavelet transform
coeff_rain, freq_rain, scales_rain = wavelet_analysis(rain_norm, scales)
print(f"Rain coefficients shape: {coeff_rain.shape}")

# %% [markdown]
# ## 5. Visualization and Analysis

# %%
# Create comprehensive visualization
fig = plt.figure(figsize=(20, 15))

# 1. Original Time Series
ax1 = plt.subplot(3, 3, 1)
ax1.plot(cell_data['date'], crash_norm, label='Crashes (norm)', alpha=0.7)
ax1.plot(cell_data['date'], rain_norm, label='Rainfall (norm)', alpha=0.7)
ax1.plot(cell_data['date'], temp_norm, label='Temperature (norm)', alpha=0.7)
ax1.set_title('Normalized Time Series\n(Grid Cell {})'.format(grid_id))
ax1.set_xlabel('Date')
ax1.set_ylabel('Normalized Value')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Wavelet Power Spectrum - Crashes
ax2 = plt.subplot(3, 3, 2)
im2 = ax2.contourf(cell_data['date'], scales, np.abs(coeff_crash), 100, cmap='jet')
ax2.set_title('Wavelet Power Spectrum - Crashes')
ax2.set_xlabel('Date')
ax2.set_ylabel('Scale')
plt.colorbar(im2, ax=ax2)

# 3. Wavelet Power Spectrum - Rainfall
ax3 = plt.subplot(3, 3, 3)
im3 = ax3.contourf(cell_data['date'], scales, np.abs(coeff_rain), 100, cmap='jet')
ax3.set_title('Wavelet Power Spectrum - Rainfall')
ax3.set_xlabel('Date')
ax3.set_ylabel('Scale')
plt.colorbar(im3, ax=ax3)

# 4. Wavelet Cross-Correlation
ax4 = plt.subplot(3, 3, 4)
xcorr_rain, scales_xcorr = wavelet_cross_correlation(crash_ts, rain_ts, scales)
xcorr_temp, _ = wavelet_cross_correlation(crash_ts, temp_ts, scales)

ax4.plot(scales_xcorr, xcorr_rain, 'b-', label='Crashes vs Rainfall', linewidth=2)
ax4.plot(scales_xcorr, xcorr_temp, 'r-', label='Crashes vs Temperature', linewidth=2)
ax4.set_title('Wavelet Cross-Correlation')
ax4.set_xlabel('Scale (days)')
ax4.set_ylabel('Correlation Coefficient')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Wavelet Coherence - Crashes vs Rainfall
ax5 = plt.subplot(3, 3, 5)
coherence_rain, phase_rain, scales_coh = wavelet_coherence(crash_ts, rain_ts, scales[:64])
im5 = ax5.contourf(cell_data['date'], scales_coh, coherence_rain, 100, cmap='jet')
ax5.set_title('Wavelet Coherence\nCrashes vs Rainfall')
ax5.set_xlabel('Date')
ax5.set_ylabel('Scale')
plt.colorbar(im5, ax=ax5)

# 6. Wavelet Phase - Crashes vs Rainfall
ax6 = plt.subplot(3, 3, 6)
im6 = ax6.contourf(cell_data['date'], scales_coh, phase_rain, 100, cmap='hsv')
ax6.set_title('Wavelet Phase Difference\nCrashes vs Rainfall')
ax6.set_xlabel('Date')
ax6.set_ylabel('Scale')
plt.colorbar(im6, ax=ax6)

# 7. Scale-Averaged Power
ax7 = plt.subplot(3, 3, 7)
# Short-term scales (1-30 days)
short_term_power = np.mean(np.abs(coeff_crash)[:30], axis=0)
# Long-term scales (30-90 days)
long_term_power = np.mean(np.abs(coeff_crash)[30:90], axis=0)

ax7.plot(cell_data['date'], short_term_power, 'r-', label='Short-term (1-30 days)', alpha=0.8)
ax7.plot(cell_data['date'], long_term_power, 'b-', label='Long-term (30-90 days)', alpha=0.8)
ax7.set_title('Scale-Averaged Wavelet Power - Crashes')
ax7.set_xlabel('Date')
ax7.set_ylabel('Average Power')
ax7.legend()
ax7.grid(True, alpha=0.3)

# 8. Dominant Scales
ax8 = plt.subplot(3, 3, 8)
dominant_scales = scales[np.argmax(np.abs(coeff_crash), axis=0)]
ax8.hist(dominant_scales, bins=50, alpha=0.7, color='green')
ax8.set_title('Distribution of Dominant Scales')
ax8.set_xlabel('Scale (days)')
ax8.set_ylabel('Frequency')

# 9. Spatial Correlation of Dominant Patterns
ax9 = plt.subplot(3, 3, 9)
# Calculate average power for seasonal band (80-100 days)
seasonal_band = (scales >= 80) & (scales <= 100)
seasonal_power = np.mean(np.abs(coeff_crash)[seasonal_band], axis=0)

# Compare with other grid cells
other_grid_data = grid_df[grid_df['grid_id'] != grid_id]
correlations = []
for other_id in range(n_grid_cells):
    if other_id != grid_id:
        other_data = grid_df[grid_df['grid_id'] == other_id].sort_values('date')
        other_crashes = other_data['crash_count'].values
        if len(other_crashes) == len(seasonal_power):
            corr = np.corrcoef(seasonal_power, other_crashes[:len(seasonal_power)])[0, 1]
            correlations.append(corr)

ax9.bar(range(len(correlations)), correlations, alpha=0.7)
ax9.set_title('Spatial Correlation of Seasonal Pattern')
ax9.set_xlabel('Other Grid Cells')
ax9.set_ylabel('Correlation with Grid {}'.format(grid_id))
ax9.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Statistical Summary and Interpretation

# %%
# Calculate summary statistics
print("=" * 60)
print("WAVELET ANALYSIS SUMMARY - GRID CELL {}".format(grid_id))
print("=" * 60)

# Dominant periodicities
total_power = np.sum(np.abs(coeff_crash), axis=1)
dominant_scale_idx = np.argmax(total_power)
dominant_scale = scales[dominant_scale_idx]
print(f"\n1. DOMINANT PERIODICITIES:")
print(f"   Primary periodicity: {dominant_scale:.1f} days")
print(f"   Interpretation: This corresponds to {'seasonal' if dominant_scale > 60 else 'weekly/monthly' if dominant_scale > 20 else 'short-term'} patterns")

# Cross-correlation peaks
rain_peak_scale = scales[np.argmax(np.abs(xcorr_rain))]
rain_max_corr = np.max(np.abs(xcorr_rain))
temp_peak_scale = scales[np.argmax(np.abs(xcorr_temp))]
temp_max_corr = np.max(np.abs(xcorr_temp))

print(f"\n2. WEATHER RELATIONSHIPS:")
print(f"   Rainfall: Peak correlation at {rain_peak_scale:.1f} days scale (r = {rain_max_corr:.3f})")
print(f"   Temperature: Peak correlation at {temp_peak_scale:.1f} days scale (r = {temp_max_corr:.3f})")

# Coherence analysis
mean_coherence = np.mean(coherence_rain, axis=1)
high_coherence_scales = scales_coh[mean_coherence > 0.7]
if len(high_coherence_scales) > 0:
    print(f"   High coherence with rainfall at scales: {high_coherence_scales[:3]} days")

# Phase relationship
mean_phase = np.mean(phase_rain[:, len(phase_rain[0])//2:], axis=1)  # Focus on latter half
dominant_phase_scale_idx = np.argmax(mean_coherence)
dominant_phase = mean_phase[dominant_phase_scale_idx]

print(f"\n3. PHASE RELATIONSHIP (Crashes vs Rainfall):")
print(f"   Dominant phase: {dominant_phase:.2f} radians")
if abs(dominant_phase) < 0.5:
    print("   Interpretation: In-phase relationship (crashes occur during/just after rain)")
elif dominant_phase > 0:
    print("   Interpretation: Crashes lag rainfall")
else:
    print("   Interpretation: Crashes lead rainfall (less common)")

# Spatial analysis
print(f"\n4. SPATIAL PATTERNS:")
print(f"   Average correlation with neighboring cells: {np.mean(correlations):.3f}")
print(f"   Spatial consistency: {'High' if np.mean(correlations) > 0.5 else 'Moderate' if np.mean(correlations) > 0.2 else 'Low'}")

# %%
# Additional analysis: Compare multiple grid cells
print("\n" + "=" * 60)
print("MULTI-GRID COMPARISON")
print("=" * 60)

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
for grid_id in range(9):
    ax = axes[grid_id // 3, grid_id % 3]
    
    cell_data = grid_df[grid_df['grid_id'] == grid_id].sort_values('date')
    crash_ts = cell_data['crash_count'].values
    
    if len(crash_ts) > 0:
        coeff, _, _ = wavelet_analysis(crash_ts, scales)
        total_power = np.sum(np.abs(coeff), axis=1)
        
        ax.plot(scales, total_power)
        ax.set_title(f'Grid {grid_id} - Total Power by Scale')
        ax.set_xlabel('Scale (days)')
        ax.set_ylabel('Total Power')
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Key Insights from the Analysis:
# 
# 1. **Multi-scale Patterns**: The wavelet analysis reveals how crash patterns operate at different temporal scales (daily, weekly, seasonal).
# 
# 2. **Weather Relationships**: Cross-correlation shows how strongly weather variables influence crashes at different time scales.
# 
# 3. **Phase Analysis**: Reveals whether crashes occur simultaneously with weather events or have a lagged response.
# 
# 4. **Spatial Consistency**: Shows how patterns vary across different geographic areas.
# 
# 5. **Localized Events**: The wavelet power spectrum can identify specific periods of unusually high crash activity.

# %%
# Save the processed data for future use
grid_df.to_csv('synthetic_spatio_temporal_grid.csv', index=False)
print("\nAnalysis complete! Data saved to 'synthetic_spatio_temporal_grid.csv'")
