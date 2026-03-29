# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 12:30:05 2025

@author: smoke
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import pywt
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

# For spatial operations
import geopandas as gpd
from shapely.geometry import Point, box
import folium

# Set random seed for reproducibility
np.random.seed(42)

# %% [markdown]
# ## 1. Create Synthetic Data

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

# Generate dates
n_days = 365 * 2
start_date = datetime(2022, 1, 1)
dates = [start_date + timedelta(days=i) for i in range(n_days)]

# Generate data
print("Generating synthetic weather data...")
weather_df = generate_synthetic_weather_data(dates)

print("Generating synthetic crash data...")
crash_df = generate_synthetic_crash_data(dates, weather_df, n_grid_cells=9)

print(f"Weather data shape: {weather_df.shape}")
print(f"Crash data shape: {crash_df.shape}")
print(f"Date range: {crash_df['date'].min()} to {crash_df['date'].max()}")

# %% [markdown]
# ## 2. Create Geospatial Grid

# %%
def create_geospatial_grid(bounds, grid_size=(3,3)):
    """
    Create a geospatial grid for mapping
    
    bounds: (min_lon, min_lat, max_lon, max_lat)
    grid_size: (n_cols, n_rows)
    """
    min_lon, min_lat, max_lon, max_lat = bounds
    grid_cells = []
    
    # Calculate cell dimensions
    lon_step = (max_lon - min_lon) / grid_size[0]
    lat_step = (max_lat - min_lat) / grid_size[1]
    
    # Create GeoDataFrame with grid cells
    grid_id = 0
    for row in range(grid_size[1]):
        for col in range(grid_size[0]):
            # Calculate cell bounds
            cell_min_lon = min_lon + col * lon_step
            cell_max_lon = min_lon + (col + 1) * lon_step
            cell_min_lat = min_lat + row * lat_step
            cell_max_lat = min_lat + (row + 1) * lat_step
            
            # Create polygon
            geometry = box(cell_min_lon, cell_min_lat, cell_max_lon, cell_max_lat)
            
            grid_cells.append({
                'grid_id': grid_id,
                'geometry': geometry,
                'x_coord': col,
                'y_coord': row
            })
            grid_id += 1
    
    return gpd.GeoDataFrame(grid_cells, crs='EPSG:4326')

# Create geographic bounds (e.g., for a city)
city_bounds = (-74.025, 40.70, -73.95, 40.80)  # NYC area example
print("Creating geospatial grid...")
grid_gdf = create_geospatial_grid(city_bounds, grid_size=(3,3))
print(f"Grid created with {len(grid_gdf)} cells")
print(f"Grid columns: {grid_gdf.columns.tolist()}")

# %% [markdown]
# ## 3. Prepare Crash Data for Mapping

# %%
# Filter crash data for target date
target_date = datetime(2022, 6, 1)
date_str = target_date.strftime('%Y-%m-%d')

print(f"\n=== Preparing crash data for {date_str} ===")

# Get crashes for the target date
day_crashes = crash_df[crash_df['date'].dt.strftime('%Y-%m-%d') == date_str].copy()

print(f"Found {len(day_crashes)} crash records")
print(f"Total crashes on {date_str}: {day_crashes['crash_count'].sum()}")

# Create crash points within appropriate grid cells
crash_points = []
for _, row in day_crashes.iterrows():
    grid_id = row['grid_id']
    grid_cell = grid_gdf[grid_gdf['grid_id'] == grid_id]
    
    if len(grid_cell) > 0:
        bounds = grid_cell.iloc[0].geometry.bounds
        # Create points proportional to crash count
        num_points = min(int(row['crash_count']), 10)  # Cap at 10 points per record
        for _ in range(num_points):
            lon = np.random.uniform(bounds[0], bounds[2])
            lat = np.random.uniform(bounds[1], bounds[3])
            crash_points.append({
                'geometry': Point(lon, lat),
                'grid_id': grid_id,
                'date': date_str,
                'point_id': len(crash_points)  # Unique ID for each point
            })

# Create GeoDataFrame for crashes
if crash_points:
    crash_gdf = gpd.GeoDataFrame(crash_points, crs='EPSG:4326')
    print(f"Created {len(crash_gdf)} crash points for mapping")
else:
    # Create empty GeoDataFrame if no crashes
    crash_gdf = gpd.GeoDataFrame({
        'geometry': [],
        'grid_id': [],
        'date': [],
        'point_id': []
    }, crs='EPSG:4326')
    print("No crashes found for this date")

print(f"crash_gdf columns: {crash_gdf.columns.tolist()}")
print(f"crash_gdf shape: {crash_gdf.shape}")

# %% [markdown]
# ## 4. Fixed Mapping Function (Returns grid with crash counts)

# %%
def create_crash_map_with_data(grid_gdf, crash_data, date_filter=None):
    """
    Create an interactive map with crash data and return the grid with crash counts
    
    Returns:
    - folium.Map: Interactive map
    - gpd.GeoDataFrame: Grid with 'crash_count' column added
    """
    
    # Create a copy of the grid
    grid_with_counts = grid_gdf.copy()
    
    print(f"\n=== Creating crash map ===")
    print(f"Date filter: {date_filter}")
    
    # Initialize crash_count column
    grid_with_counts['crash_count'] = 0
    
    # Filter crash data if needed
    if date_filter and len(crash_data) > 0:
        if 'date' in crash_data.columns:
            # Convert date_filter to string
            if isinstance(date_filter, datetime):
                date_str = date_filter.strftime('%Y-%m-%d')
            else:
                date_str = str(date_filter)
            
            filtered_crashes = crash_data[crash_data['date'] == date_str].copy()
            print(f"Filtered to {len(filtered_crashes)} crashes for date {date_str}")
        else:
            print("Warning: 'date' column not found, using all data")
            filtered_crashes = crash_data.copy()
    else:
        filtered_crashes = crash_data.copy()
    
    # Perform spatial analysis if we have crash points
    if len(filtered_crashes) > 0 and 'geometry' in filtered_crashes.columns:
        print(f"Performing spatial analysis with {len(filtered_crashes)} crash points...")
        
        # Ensure we have a GeoDataFrame
        if not isinstance(filtered_crashes, gpd.GeoDataFrame):
            filtered_crashes = gpd.GeoDataFrame(
                filtered_crashes, 
                geometry='geometry',
                crs='EPSG:4326'
            )
        
        # Count crashes in each grid cell
        for idx, grid_row in grid_with_counts.iterrows():
            # Find points within this grid cell
            points_in_cell = filtered_crashes[filtered_crashes.within(grid_row['geometry'])]
            crash_count = len(points_in_cell)
            grid_with_counts.at[idx, 'crash_count'] = crash_count
        
        print(f"Total crashes mapped: {grid_with_counts['crash_count'].sum()}")
    else:
        print("No valid crash data available")
    
    # Create interactive map
    center_lat = (grid_with_counts.geometry.bounds.miny.mean() + 
                  grid_with_counts.geometry.bounds.maxy.mean()) / 2
    center_lon = (grid_with_counts.geometry.bounds.minx.mean() + 
                  grid_with_counts.geometry.bounds.maxx.mean()) / 2
    
    print(f"Map center: {center_lat:.4f}, {center_lon:.4f}")
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
    
    # Add choropleth layer for crash counts
    if 'crash_count' in grid_with_counts.columns:
        # Create choropleth
        folium.Choropleth(
            geo_data=grid_with_counts.__geo_interface__,
            data=grid_with_counts,
            columns=['grid_id', 'crash_count'],
            key_on='feature.properties.grid_id',
            fill_color='YlOrRd',
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name='Crash Count',
            bins=5
        ).add_to(m)
        
        # Add tooltips
        for _, row in grid_with_counts.iterrows():
            folium.GeoJson(
                row['geometry'],
                style_function=lambda x: {
                    'color': 'black',
                    'weight': 1,
                    'fillOpacity': 0  # Transparent fill to show choropleth
                },
                tooltip=f"Grid {row['grid_id']}: {int(row['crash_count'])} crashes"
            ).add_to(m)
    
    # Add crash points if available
    if len(filtered_crashes) > 0 and 'geometry' in filtered_crashes.columns:
        print(f"Adding {len(filtered_crashes)} crash points to map")
        for _, crash in filtered_crashes.iterrows():
            if crash.geometry and not crash.geometry.is_empty:
                folium.CircleMarker(
                    location=[crash.geometry.y, crash.geometry.x],
                    radius=3,
                    color='red',
                    fill=True,
                    fill_color='red',
                    fill_opacity=0.7,
                    popup=f"Grid: {crash.get('grid_id', 'N/A')}"
                ).add_to(m)
    
    print("=== Map creation complete ===\n")
    return m, grid_with_counts

# %% [markdown]
# ## 5. Create Interactive Map and Get Grid with Counts

# %%
# Create interactive map and get grid with crash counts
print("Creating interactive map...")
map_obj, grid_with_counts = create_crash_map_with_data(grid_gdf, crash_gdf, date_filter=date_str)

# Save the map
map_obj.save('grid_crash_map.html')
print("Map saved as 'grid_crash_map.html'")

# Display the grid with crash counts
print("\nGrid with crash counts:")
print(grid_with_counts[['grid_id', 'x_coord', 'y_coord', 'crash_count']])

# %% [markdown]
# ## 6. Create Static Visualizations (FIXED)

# %%
# Create static visualizations using grid_with_counts
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Grid cells with crash counts
ax1 = axes[0, 0]

# Make sure we have crash_count column
if 'crash_count' not in grid_with_counts.columns:
    grid_with_counts['crash_count'] = 0

grid_with_counts.boundary.plot(ax=ax1, linewidth=1, edgecolor='black')
grid_with_counts.plot(ax=ax1, column='crash_count', cmap='Reds', 
                     legend=True, alpha=0.6, legend_kwds={'label': 'Crash Count'})

# Add grid IDs
for idx, row in grid_with_counts.iterrows():
    centroid = row.geometry.centroid
    ax1.text(centroid.x, centroid.y, f"Grid {row['grid_id']}\n({row['crash_count']})", 
            ha='center', va='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

if len(crash_gdf) > 0:
    crash_gdf.plot(ax=ax1, color='red', markersize=20, alpha=0.5, 
                  label='Crash locations', marker='o')

ax1.set_title(f'Crash Distribution - {date_str}')
ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latitude')
ax1.legend()

# Plot 2: Time series of total crashes
ax2 = axes[0, 1]
total_crashes_by_date = crash_df.groupby('date')['crash_count'].sum().reset_index()
ax2.plot(total_crashes_by_date['date'], total_crashes_by_date['crash_count'], 
        'b-', alpha=0.7, linewidth=1)
ax2.axvline(x=target_date, color='r', linestyle='--', alpha=0.5, 
           label=f'Selected date: {date_str}')
ax2.set_title('Total Daily Crashes (All Grids)')
ax2.set_xlabel('Date')
ax2.set_ylabel('Number of Crashes')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Crash distribution by grid cell (all time)
ax3 = axes[1, 0]
grid_totals = crash_df.groupby('grid_id')['crash_count'].sum().reset_index()
grid_totals = grid_totals.merge(grid_gdf[['grid_id', 'x_coord', 'y_coord']], on='grid_id')
grid_totals['location'] = grid_totals.apply(lambda x: f"({x['x_coord']},{x['y_coord']})", axis=1)

bars = ax3.bar(grid_totals['location'], grid_totals['crash_count'], color='skyblue')
ax3.set_title('Total Crashes by Grid Cell (All Time)')
ax3.set_xlabel('Grid Coordinates (x,y)')
ax3.set_ylabel('Total Crashes')
ax3.set_xticklabels(grid_totals['location'], rotation=45)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}', ha='center', va='bottom')

# Plot 4: Daily crash pattern for selected grid
ax4 = axes[1, 1]
selected_grid = 4
grid_4_data = crash_df[crash_df['grid_id'] == selected_grid]

if len(grid_4_data) > 0:
    # Calculate weekly pattern
    grid_4_data['day_of_week'] = grid_4_data['date'].dt.dayofweek
    weekly_avg = grid_4_data.groupby('day_of_week')['crash_count'].mean()
    
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    ax4.bar(range(7), weekly_avg, color='lightcoral', alpha=0.7)
    ax4.set_title(f'Average Daily Pattern - Grid {selected_grid}')
    ax4.set_xlabel('Day of Week')
    ax4.set_ylabel('Average Crashes')
    ax4.set_xticks(range(7))
    ax4.set_xticklabels(days)
    ax4.grid(True, alpha=0.3, axis='y')
else:
    ax4.text(0.5, 0.5, f'No data for Grid {selected_grid}', 
            ha='center', va='center', transform=ax4.transAxes)
    ax4.set_title(f'Average Daily Pattern - Grid {selected_grid}')

plt.tight_layout()
plt.savefig('crash_analysis_visualization.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 7. Additional Analysis: Wavelet Analysis on Crash Data

# %%
def analyze_grid_with_wavelets(grid_data, grid_id):
    """Perform wavelet analysis on crash data for a specific grid"""
    
    # Get crash time series for this grid
    grid_crashes = grid_data[grid_data['grid_id'] == grid_id].sort_values('date')
    
    if len(grid_crashes) < 30:  # Need enough data
        print(f"Insufficient data for Grid {grid_id}")
        return None
    
    crash_series = grid_crashes['crash_count'].values
    
    # Perform wavelet transform
    scales = np.arange(1, 128)
    coefficients, frequencies = pywt.cwt(crash_series, scales, 'morl')
    
    # Create wavelet analysis plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 1. Original time series
    axes[0, 0].plot(grid_crashes['date'], crash_series, 'b-', linewidth=1)
    axes[0, 0].set_title(f'Grid {grid_id} - Crash Time Series')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Crashes')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Wavelet power spectrum
    power = np.abs(coefficients) ** 2
    im = axes[0, 1].contourf(grid_crashes['date'], scales, power, 100, cmap='jet')
    axes[0, 1].set_title(f'Grid {grid_id} - Wavelet Power Spectrum')
    axes[0, 1].set_xlabel('Date')
    axes[0, 1].set_ylabel('Scale (days)')
    plt.colorbar(im, ax=axes[0, 1])
    
    # 3. Global wavelet spectrum
    global_power = np.mean(power, axis=1)
    axes[1, 0].plot(scales, global_power, 'b-', linewidth=2)
    axes[1, 0].set_title('Global Wavelet Spectrum')
    axes[1, 0].set_xlabel('Scale (days)')
    axes[1, 0].set_ylabel('Average Power')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Scale-averaged time series
    # Short-term (1-14 days)
    short_scales = scales <= 14
    short_power = np.mean(power[short_scales], axis=0)
    
    # Long-term (>60 days)
    long_scales = scales > 60
    if np.any(long_scales):
        long_power = np.mean(power[long_scales], axis=0)
        axes[1, 1].plot(grid_crashes['date'], long_power, 'b-', label='Long-term (>60d)', alpha=0.7)
    
    axes[1, 1].plot(grid_crashes['date'], short_power, 'r-', label='Short-term (1-14d)', alpha=0.7)
    axes[1, 1].set_title('Scale-Averaged Power')
    axes[1, 1].set_xlabel('Date')
    axes[1, 1].set_ylabel('Average Power')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'wavelet_analysis_grid_{grid_id}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return coefficients

# Analyze a specific grid with wavelets
print("\nPerforming wavelet analysis on Grid 4...")
wavelet_coeff = analyze_grid_with_wavelets(crash_df, 4)

# %% [markdown]
# ## 8. Summary and Export

# %%
print("\n" + "="*60)
print("ANALYSIS COMPLETE - SUMMARY")
print("="*60)
print(f"Total crash records: {len(crash_df):,}")
print(f"Date range: {crash_df['date'].min().strftime('%Y-%m-%d')} to {crash_df['date'].max().strftime('%Y-%m-%d')}")
print(f"Total crashes: {crash_df['crash_count'].sum():,}")
print(f"Average daily crashes: {crash_df['crash_count'].mean():.2f}")

print(f"\nSelected date: {date_str}")
print(f"Crashes on selected date: {day_crashes['crash_count'].sum()}")
print(f"Grid cells with crashes: {(grid_with_counts['crash_count'] > 0).sum()} out of {len(grid_with_counts)}")

print(f"\nTop 3 grid cells on {date_str}:")
top_grids = grid_with_counts.nlargest(3, 'crash_count')[['grid_id', 'crash_count']]
for _, row in top_grids.iterrows():
    print(f"  Grid {row['grid_id']}: {row['crash_count']} crashes")

print(f"\nFiles created:")
print(f"  1. grid_crash_map.html - Interactive map")
print(f"  2. crash_analysis_visualization.png - Static visualizations")
if wavelet_coeff is not None:
    print(f"  3. wavelet_analysis_grid_4.png - Wavelet analysis")

print(f"\nTo view the interactive map:")
print(f"  Open 'grid_crash_map.html' in any web browser")

print("="*60)

# Export data for further analysis
grid_with_counts.to_file('grid_with_crash_counts.geojson', driver='GeoJSON')
crash_df.to_csv('synthetic_crash_data.csv', index=False)
print("\nData exported:")
print("  - grid_with_crash_counts.geojson")
print("  - synthetic_crash_data.csv")